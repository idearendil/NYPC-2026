#!/usr/bin/env python3
"""Background random-map generation for training.

The reference generator (``testing-tool.generate_map``) is pure Python and costs
~25 ms per map. Every finished episode needs a fresh one, so at the batch sizes
training uses (B >= 1024, an episode ending every few steps in every game slot)
map generation alone can eat a third of the rollout wall-clock -- all of it on
one core, while the GPU idles.

This module moves that work off the critical path: a pool of worker PROCESSES
keeps a bounded queue of ready maps topped up, and ``MapFactory.get()`` just pops
one. Workers deliberately never import torch (they only need the reference
generator), so they start in well under a second and cost no GPU memory.

Maps cross the process boundary as the generator's raw TEXT LINES, not as
``MapData``: that dataclass lives in a module loaded by path (``tt_reference``),
which the children cannot unpickle by name. Parsing the lines back
(``tt.read_map``) is microseconds.

Determinism: with workers > 0 the ORDER in which maps come out of the queue
depends on process scheduling, so a run is no longer reproducible from the seed
alone (the set of maps is still drawn from the same distribution, and each
worker is seeded deterministically). Pass ``workers=0`` for the old fully
deterministic in-process behaviour.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue as _queue
import random

_TT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing-tool.py")


def _load_tt():
    """Import testing-tool.py (its name isn't a valid module name) by path."""
    spec = importlib.util.spec_from_file_location("tt_reference", _TT_PATH)
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
    return tt


_TT = None


def tt():
    global _TT
    if _TT is None:
        _TT = _load_tt()
    return _TT


def random_map_lines(rng: random.Random):
    """One uniformly random legal map, as the generator's raw text lines.

    Mirrors ``FastEnv._random_map``'s draw (NP in [25,54], KP uniform over the
    legal range for that N) and retries the map-placement failures the generator
    raises for unlucky stronghold counts.
    """
    t = tt()
    while True:
        NP = rng.randint(25, 54)
        N = 2 * NP + 1
        klo = (3 * N + 19) // 20
        khi = N // 5
        klo = klo + 1 if klo % 2 == 0 else klo
        khi = khi - 1 if khi % 2 == 0 else khi
        KP = rng.randint((klo - 1) // 2, (khi - 1) // 2)
        try:
            return t.generate_map(t.XoShiro256(rng.getrandbits(63)), NP, KP)
        except (ValueError, RuntimeError):
            continue


def _worker(q, seed, stop):
    """Generate maps forever, blocking whenever the queue is full."""
    rng = random.Random(seed)
    try:
        while not stop.is_set():
            lines = random_map_lines(rng)
            while not stop.is_set():
                try:
                    q.put(lines, timeout=0.5)
                    break
                except _queue.Full:
                    continue
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass


class MapFactory:
    """A pool of worker processes that keeps ``depth`` maps ready to hand out.

    ``get()`` returns a parsed ``MapData``; it falls back to generating inline if
    the queue happens to be empty, so it is always correct, just occasionally
    slow. ``workers=0`` disables the pool entirely (deterministic, in-process).
    """

    def __init__(self, workers: int = 4, seed: int = 0, depth: int = 256):
        self.workers = max(0, int(workers))
        self._rng = random.Random(seed)
        self.procs, self.q, self._stop = [], None, None
        self.hits = self.misses = 0
        if self.workers == 0:
            return
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=depth)
        self._stop = ctx.Event()
        for i in range(self.workers):
            p = ctx.Process(target=_worker, args=(self.q, seed * 7919 + i + 1, self._stop),
                            daemon=True)
            p.start()
            self.procs.append(p)

    def get(self):
        """One fresh random map (``MapData``).

        The small timeout (rather than ``get_nowait``) is deliberate: a
        multiprocessing.Queue hands items over through a feeder thread and a
        pipe, so an item that a worker has already produced can still read as
        Empty for a moment. Waiting a couple of milliseconds for one is far
        cheaper than the ~25 ms of generating a map inline instead.
        """
        if self.q is not None:
            try:
                lines = self.q.get(timeout=0.002)
                self.hits += 1
                return tt().read_map(lines)
            except _queue.Empty:
                self.misses += 1
        return tt().read_map(random_map_lines(self._rng))

    def get_many(self, n: int):
        return [self.get() for _ in range(n)]

    def close(self):
        if self._stop is not None:
            self._stop.set()
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self.procs = []
        if self.q is not None:
            self.q.cancel_join_thread()
            self.q.close()
            self.q = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_maps(n: int, seed: int = 0, workers: int = 0):
    """``n`` random maps for the initial batch.

    With ``workers > 0`` they are generated in parallel with a short-lived Pool
    (startup would otherwise be n * 25 ms of serial Python -- a minute at
    B = 2048); the map *distribution* is identical either way.
    """
    if workers <= 0 or n < 16:
        rng = random.Random(seed)
        t = tt()
        return [t.read_map(random_map_lines(rng)) for _ in range(n)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, n)) as pool:
        out = pool.map(_gen_one, [seed * 104729 + i + 1 for i in range(n)],
                       chunksize=max(1, n // (4 * workers)))
    t = tt()
    return [t.read_map(lines) for lines in out]


def _gen_one(seed):
    return random_map_lines(random.Random(seed))
