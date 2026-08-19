"""Backend tuning and multi-GPU data parallelism.

The split is over GAMES: rank r simulates ``cfg.B // world`` of the batch in its
own env against its own opponent draws, and the PPO gradients are averaged across
ranks -- so both halves of an iteration (rollout AND update) are parallelised and
the data collected per iteration is identical to a single-GPU run with the same
config.

Nothing here is game-specific, and ``world == 1`` makes every method a no-op, so
the single-GPU path is exactly the serial one.
"""
from __future__ import annotations

import os

import torch


def tune_backend(threads=None):
    """Speed-only global knobs, applied once per process.

    TF32 matmuls (Ampere/Ada tensor cores) are a free ~1.5x on transformer GEMMs;
    their 10-bit mantissa is far more precision than log1p'd game features carry.

    Deliberately NOT enabled: bf16/fp16 autocast. PPO compares a log-prob stored
    during collection against one recomputed in the update, and bf16 over a sum of
    ~20 log-probs injects percent-level noise straight into the importance ratio.
    TF32's ~1e-3 does not.

    The CPU thread cap matters because a host-resident rollout buffer is indexed
    and copied on the CPU every minibatch; past ~8 threads those small copies just
    pay more synchronisation.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if threads is None:
        threads = min(8, max(1, (os.cpu_count() or 8) // 2))
    torch.set_num_threads(max(1, int(threads)))


class Dist:
    """Data-parallel state for one process.

    Everything the ranks must AGREE on is kept in lockstep explicitly: weights are
    broadcast once at startup and stay identical because every rank applies the
    same averaged gradient, and the opponent pool's win rates are all-reduced once
    per iteration (see ``OpponentPool.apply_tally``) so the add/evict decisions
    can never diverge. Anything else -- which games a rank simulates, which
    opponents it drew -- is deliberately different per rank.

    Nets are NOT wrapped in ``nn.parallel.DistributedDataParallel``: a policy is
    free to call its submodules directly (a second head applied to an
    already-computed encoding, a sub-network run over a flattened subset of the
    rows), which DDP's forward-hook reducer does not support. For the ~1e5..1e6
    parameters a game like this needs, one explicit flat all-reduce per optimizer
    step costs microseconds.
    """

    def __init__(self):
        self.rank, self.world, self.local_rank = 0, 1, 0
        self.enabled = False

    @classmethod
    def init(cls):
        d = cls()
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world <= 1:
            return d
        import torch.distributed as dist
        d.rank = int(os.environ["RANK"])
        d.local_rank = int(os.environ.get("LOCAL_RANK", d.rank))
        d.world = world
        n_gpu = torch.cuda.device_count()
        # nccl needs one distinct GPU per rank; anything else (CPU debugging, more
        # ranks than GPUs) falls back to gloo rather than deadlocking.
        use_nccl = n_gpu >= world
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if use_nccl else "gloo")
        if n_gpu:
            torch.cuda.set_device(min(d.local_rank, n_gpu - 1))
        d.enabled = True
        return d

    # ---- properties ------------------------------------------------------- #
    @property
    def is_main(self):
        return self.rank == 0

    def device_for(self, device=None):
        """The device this rank should use ('cuda:<local_rank>' when available)."""
        if device is not None:
            return device
        n = torch.cuda.device_count()
        return f"cuda:{min(self.local_rank, n - 1)}" if n else "cpu"

    def split(self, total, name):
        """Per-rank share of a TOTAL (B, minibatch), erroring on a bad divisor."""
        if total % self.world:
            raise ValueError(f"{name}={total} must be divisible by world size "
                             f"{self.world}")
        return total // self.world

    # ---- collectives ------------------------------------------------------ #
    def barrier(self):
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()

    def broadcast_params(self, modules):
        """Make every rank's weights bit-identical to rank 0's."""
        if not self.enabled:
            return
        import torch.distributed as dist
        for m in modules:
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t.data, src=0)

    def all_reduce_grads(self, params):
        """Average gradients in ONE flat all-reduce (there are many tiny ones)."""
        if not self.enabled:
            return
        import torch.distributed as dist
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= self.world
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)

    def all_reduce_(self, t, op="sum"):
        """In-place sum/mean all-reduce of a tensor (returns it)."""
        if not self.enabled:
            return t
        import torch.distributed as dist
        dist.all_reduce(t)
        if op == "mean":
            t /= self.world
        return t

    def sum_scalars(self, values, device):
        """All-reduce a list of python floats; returns a list of floats."""
        t = torch.tensor([float(v) for v in values], device=device,
                         dtype=torch.float64)
        self.all_reduce_(t)
        return t.tolist()

    def shutdown(self):
        if self.enabled:
            import torch.distributed as dist
            dist.destroy_process_group()


def launch(entry, n_gpu=None, args=(), port=None, verbose=True):
    """Run ``entry(*args)`` on N GPUs, spawning one process per GPU.

    Already inside a torchrun/elastic launch (WORLD_SIZE in the environment)? Runs
    inline and lets ``Dist.init`` pick the rank up. Otherwise spawns, so
    ``--gpus 2`` is all a user has to type. ``entry`` must be importable by name
    (a module-level function) and is called as ``entry(*args)`` in the children,
    with RANK/LOCAL_RANK/WORLD_SIZE set.

    IMPORTANT: nothing may have touched CUDA in the parent before this call --
    initialising a CUDA context pre-fork/spawn is what makes multi-process launch
    fail in confusing ways.
    """
    if "WORLD_SIZE" in os.environ:
        return entry(*args)
    if n_gpu is None:
        n_gpu = torch.cuda.device_count()
    n_gpu = max(1, min(int(n_gpu), max(1, torch.cuda.device_count())))
    if n_gpu <= 1:
        return entry(*args)
    if verbose:
        print(f"launching data-parallel training on {n_gpu} GPUs")
    import torch.multiprocessing as tmp
    port = port or str(29500 + (os.getpid() % 2000))
    tmp.spawn(_worker, args=(n_gpu, port, entry, args), nprocs=n_gpu, join=True)


def _worker(rank, world, port, entry, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    entry(*args)
