"""Validate the multi-rank logic WITHOUT a working NCCL/gloo build.

Why this exists: the data-parallel code has one property that must hold or the run
is quietly wrong -- every rank must end each iteration with bit-identical weights
AND an identical opponent pool, because each rank decides on its own when to
snapshot and evict. A bug there does not crash; it produces two divergent trainers
writing one checkpoint.

Windows has neither backend available (`makeDeviceForHostname(): unsupported gloo
device`, no nccl), and even on Linux a 2-process test is awkward inside a test
suite. So this drives N ``train()`` calls in THREADS of one process and swaps
``Dist`` for an in-process implementation of the same collectives. Every
distributed code path is exercised -- which quantities get reduced, where, and
whether the ranks stay in lockstep -- and only the wire protocol is stubbed.

    from rlkit.testing import run_threaded_ranks
    out = run_threaded_ranks(lambda rank: my_train(rank), world=2)
    assert_params_identical(out[0], out[1])
"""
from __future__ import annotations

import threading

import torch

from . import dist as _dist


class Shared:
    """Rendezvous state shared by the fake ranks."""

    def __init__(self, world):
        self.world = world
        self.bar = threading.Barrier(world)
        self.slots = [None] * world


class FakeDist:
    """``Dist`` with the collectives implemented over a threading.Barrier."""

    def __init__(self, sh, rank):
        self.sh, self.rank, self.world = sh, rank, sh.world
        self.local_rank, self.enabled = 0, True

    @property
    def is_main(self):
        return self.rank == 0

    def device_for(self, device=None):
        if device is not None:
            return device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def split(self, total, name):
        if total % self.world:
            raise ValueError(f"{name}={total} not divisible by {self.world}")
        return total // self.world

    def barrier(self):
        self.sh.bar.wait()

    def _reduce(self, t):
        self.sh.slots[self.rank] = t.detach().to("cpu").clone()
        self.sh.bar.wait()
        out = sum(self.sh.slots)
        self.sh.bar.wait()
        return out

    def all_reduce_(self, t, op="sum"):
        r = self._reduce(t).to(t.device, t.dtype)
        if op == "mean":
            r = r / self.world
        t.copy_(r)
        return t

    def broadcast_params(self, modules):
        ts = [t for m in modules for t in list(m.parameters()) + list(m.buffers())]
        if self.rank == 0:
            self.sh.slots[0] = [t.detach().cpu().clone() for t in ts]
        self.sh.bar.wait()
        if self.rank != 0:
            for t, s in zip(ts, self.sh.slots[0]):
                t.data.copy_(s)
        self.sh.bar.wait()

    def all_reduce_grads(self, params):
        gs = [p.grad for p in params if p.grad is not None]
        if not gs:
            return
        flat = torch._utils._flatten_dense_tensors(gs)
        self.all_reduce_(flat)
        flat /= self.world
        for g, s in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
            g.copy_(s)

    def sum_scalars(self, values, device):
        t = torch.tensor([float(v) for v in values], dtype=torch.float64)
        return self._reduce(t).tolist()

    def shutdown(self):
        pass


def run_threaded_ranks(entry, world=2):
    """Run ``entry(rank)`` in ``world`` threads with ``Dist.init`` faked.

    Returns ``{rank: return value}``. Exceptions in a rank are re-raised here so a
    failure cannot look like a pass with a missing rank.
    """
    sh = Shared(world)
    fakes = {r: FakeDist(sh, r) for r in range(world)}
    tls = threading.local()
    real_init = _dist.Dist.init
    # Per-thread, because the ranks run concurrently and a class-level patch would
    # otherwise hand whichever rank started last to everybody.
    _dist.Dist.init = staticmethod(lambda: fakes[tls.rank])
    out, errs = {}, {}

    def body(rank):
        tls.rank = rank
        try:
            out[rank] = entry(rank)
        except BaseException as e:      # noqa: BLE001 - re-raised below
            errs[rank] = e

    try:
        ths = [threading.Thread(target=body, args=(r,)) for r in range(world)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    finally:
        _dist.Dist.init = real_init
    if errs:
        rank, err = sorted(errs.items())[0]
        raise RuntimeError(f"rank {rank} failed: {err!r}") from err
    return out


def flat_params(policy):
    """Every parameter of a policy, on the CPU, in a stable order."""
    return [p.detach().cpu().clone()
            for _, m in sorted(policy.modules().items())
            for p in m.parameters()]


def assert_params_identical(a, b, label="ranks"):
    """Bit-identity check with a useful message when it fails."""
    assert len(a) == len(b), f"{label}: different parameter counts {len(a)}/{len(b)}"
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]
    if bad:
        i = bad[0]
        raise AssertionError(
            f"{label}: {len(bad)}/{len(a)} tensors diverged (first idx {i}, "
            f"max abs diff {float((a[i] - b[i]).abs().max()):.3e})")
    return len(a)
