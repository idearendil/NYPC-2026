"""Background generation of per-episode instances (maps, scenarios, seeds).

The problem this solves is easy to miss until it costs you a third of your
wall-clock: if generating one episode's map is tens of milliseconds of pure Python,
and every finished game needs a fresh one, then at a large batch size map
generation becomes a serial single-core stall in the MIDDLE of the rollout, with
the GPU idle. Worker processes fix it; a thread pool would not, because the
generator is Python and holds the GIL.

Two rules the API is shaped around:

* Payloads cross the process boundary as something trivially PICKLABLE (text
  lines, tuples, arrays) and are turned into the env's real object in the PARENT
  via ``postprocess``. A dataclass defined in a module that was loaded by path
  cannot be unpickled by name in a child, and workers should not import torch at
  all -- they start in milliseconds and cost no VRAM that way.
* ``get()`` never blocks for long and never fails: an empty queue just means it
  generates one inline. Correct always, occasionally slow, and it counts the
  misses so you can see when you are under-provisioned.

Determinism: with ``workers > 0`` the ORDER instances come out depends on process
scheduling, so a run is no longer reproducible from the seed alone (the
distribution is identical, and each worker is seeded deterministically). Use
``workers=0`` when you need bit-reproducibility.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import random


def _worker(q, gen, seed, stop):
    """Generate forever, blocking while the queue is full."""
    rng = random.Random(seed)
    try:
        while not stop.is_set():
            payload = gen(rng)
            while not stop.is_set():
                try:
                    q.put(payload, timeout=0.5)
                    break
                except _queue.Full:
                    continue
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass


def _gen_one(args):
    gen, seed = args
    return gen(random.Random(seed))


class InstanceFactory:
    """A pool of worker processes keeping ``depth`` instances ready to hand out.

    ``gen(rng)`` must be a module-level callable (it is pickled by reference) that
    returns a picklable payload. ``postprocess(payload)`` runs in this process.
    """

    def __init__(self, gen, workers=4, seed=0, depth=256, postprocess=None):
        self.gen = gen
        self.post = postprocess or (lambda x: x)
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
            p = ctx.Process(target=_worker,
                            args=(self.q, gen, seed * 7919 + i + 1, self._stop),
                            daemon=True)
            p.start()
            self.procs.append(p)

    def get(self):
        """One fresh instance.

        The small timeout (rather than ``get_nowait``) is deliberate: a
        multiprocessing.Queue hands items over through a feeder thread and a pipe,
        so an item a worker has ALREADY produced can read as Empty for a moment.
        Waiting a couple of milliseconds for one beats generating a whole instance
        inline instead.
        """
        if self.q is not None:
            try:
                payload = self.q.get(timeout=0.002)
                self.hits += 1
                return self.post(payload)
            except _queue.Empty:
                self.misses += 1
        return self.post(self.gen(self._rng))

    def get_many(self, n):
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
            # cancel_join_thread BEFORE close: the feeder thread would otherwise
            # block shutdown waiting to flush a queue nobody is draining.
            self.q.cancel_join_thread()
            self.q.close()
            self.q = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_instances(n, gen, seed=0, workers=0, postprocess=None):
    """``n`` instances for the initial batch, optionally generated in parallel.

    Startup is otherwise n x (generation cost) of serial Python -- a minute or more
    at B = 2048 -- before the first step of training runs.
    """
    post = postprocess or (lambda x: x)
    if workers <= 0 or n < 16:
        rng = random.Random(seed)
        return [post(gen(rng)) for _ in range(n)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, n)) as pool:
        out = pool.map(_gen_one, [(gen, seed * 104729 + i + 1) for i in range(n)],
                       chunksize=max(1, n // (4 * max(workers, 1))))
    return [post(p) for p in out]
