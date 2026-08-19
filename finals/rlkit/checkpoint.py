"""Crash-safe checkpointing and a deliberately forgiving resume.

A checkpoint is written at the START of every iteration, so a run killed anywhere
resumes at most one iteration behind. Only rank 0 writes: every rank holds
identical weights and an identical pool, and the state that does differ (which
games it is simulating) is regenerated on resume anyway.

The resume path is lenient ON PURPOSE. During a long training run you will add a
feature to the observation, a head to a network, a scripted opponent to the pool --
and each time, a strict load would force you to either throw the run away or write
a migration. So: parameters load by name where the shape still matches, optimizer
moments are rebuilt if the parameter set changed, and the pool re-aligns its
scripted slots by name. What must NOT be papered over is a change that silently
reinterprets learned weights (an action head that changed meaning); pass
``required`` for those and get a hard error with an instruction instead.
"""
from __future__ import annotations

import os

import torch


def save_atomic(path, obj):
    """Write a checkpoint via temp file + replace, so a crash mid-write cannot
    leave a truncated file where the resume path expects a valid one."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lenient_load(module, sd, verbose=True, name=""):
    """Load the shape-compatible subset of ``sd`` into ``module``.

    Returns the number of parameters left at their fresh initialisation.
    """
    cur = module.state_dict()
    keep = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    res = module.load_state_dict(keep, strict=False)
    n_fresh = len(res.missing_keys)
    if n_fresh and verbose:
        print(f"  {name or 'module'}: {n_fresh} tensor(s) kept fresh "
              f"(not in the checkpoint, or a different shape)")
    return n_fresh


class Checkpointer:
    """Assembles / restores the whole training state.

    ``guards`` lets a game declare a shape that MUST match, e.g.::

        guards = {'actor_t2': {'head.2.weight': 2}}   # out_features of that layer

    which turns "this checkpoint predates the second action head" from a silent
    re-initialisation of the entire move policy into a clear error.
    """

    def __init__(self, path, dist=None, device="cpu", guards=None):
        self.path = path
        self.dist = dist
        self.device = device
        self.guards = guards or {}

    @property
    def exists(self):
        return bool(self.path) and os.path.exists(self.path)

    # ---- save ------------------------------------------------------------- #
    def save(self, it, *, policy, ppo, pool, task=None, cfg=None, extra=None):
        on_cuda = torch.device(self.device).type == "cuda"
        obj = {
            "iter": it,
            "policy": policy.state_dict(),
            "opt": ppo.state_dict(),
            "pool": pool.state_dict(),
            "task": task.state_dict() if task is not None else None,
            "cfg": cfg,
            "torch_rng": torch.get_rng_state(),
            # only THIS rank's device: a multi-GPU run must not stamp its peers'
            "cuda_rng": [torch.cuda.get_rng_state(self.device)] if on_cuda else None,
        }
        if extra:
            obj.update(extra)
        save_atomic(self.path, obj)

    # ---- load ------------------------------------------------------------- #
    def load(self, *, policy, ppo, pool, make_policy, task=None, B=None,
             seed=0, verbose=True):
        """Restore in place; returns the iteration to resume at."""
        ck = torch.load(self.path, map_location=self.device, weights_only=False)
        rank = self.dist.rank if self.dist is not None else 0
        main = self.dist is None or self.dist.is_main

        self._check_guards(policy, ck["policy"])
        for name, module in policy.modules().items():
            if name in ck["policy"]:
                lenient_load(module, ck["policy"][name],
                             verbose=verbose and main, name=name)
            elif verbose and main:
                print(f"  {name}: absent from the checkpoint, left fresh")
        ppo.load_state_dict(ck.get("opt", {}), verbose=verbose and main)
        pool.load_state_dict(ck["pool"], make_policy, B=B or pool.assign.numel(),
                             rank=rank, verbose=verbose and main)
        if task is not None and ck.get("task") is not None:
            task.load_state_dict(ck["task"])

        # RNG states must be CPU ByteTensors (map_location may have moved them).
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"][0].cpu(), device=self.device)
        it = ck["iter"]
        # Every rank just restored the SAME streams; re-diverge the others or they
        # would all simulate rank 0's games from here on.
        if rank:
            torch.manual_seed(int(torch.randint(1 << 30, (1,)).item())
                              + 7919 * rank)
            pool.reseed(seed + 777 + 104729 * rank + it, B=B)
        if verbose and main:
            print(f"resumed from {self.path} at iter {it} "
                  f"(pool size {pool.size})")
        return it

    def _check_guards(self, policy, sd):
        for mod_name, checks in self.guards.items():
            msd = sd.get(mod_name)
            if not msd:
                continue
            for key, want in checks.items():
                got = msd.get(key)
                if got is not None and got.shape[0] != want:
                    raise RuntimeError(
                        f"{self.path} is incompatible: {mod_name}.{key} has "
                        f"leading dim {got.shape[0]}, this build needs {want}. "
                        f"Loading it would silently keep weights that no longer "
                        f"mean what they meant. Archive the file and train fresh "
                        f"(--no-resume, or move it aside).")
