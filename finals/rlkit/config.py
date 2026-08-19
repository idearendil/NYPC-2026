"""Configuration: a dataclass of game-independent knobs, YAML loading, and the
training-phase schedule.

A game adds its own fields by SUBCLASSING ``BaseConfig``::

    @dataclass
    class Config(rlkit.BaseConfig):
        d_model: int = 64
        n_layers: int = 3

``load_config(path, Config)`` then accepts those keys too, and any of them can be
phase-scheduled (see ``PhaseSchedule``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class BaseConfig:
    # ---- rollout / batch -------------------------------------------------- #
    # B is the TOTAL number of parallel games. Under --gpus N each rank
    # simulates B/N of them, so an N-GPU run collects exactly the same data per
    # iteration as a 1-GPU run with the same config. The rollout of a batched GPU
    # env is normally kernel-LAUNCH bound rather than FLOP bound, which makes B
    # the single biggest throughput lever: raise it until either VRAM or the GAE
    # horizon (steps_per_iter / B, which wants to stay well above
    # 1/(1-gamma*lam)) stops you.
    B: int = 1024
    steps_per_iter: int = 200_000      # env-steps per iteration; horizon = /B
    iters: int = 1000

    # ---- PPO -------------------------------------------------------------- #
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 3
    minibatch: int = 4096              # TOTAL; split across ranks like B
    # KL early stop: abandon an iteration's remaining epochs once the policy has
    # drifted this far from the collection policy. None disables it (and skips
    # the per-epoch device sync it needs).
    target_kl: Optional[float] = None
    ent_coef: float = 5e-3
    # Weight on the value loss. Only used when the policy exposes a SINGLE
    # parameter group (shared trunk); with separate actor/critic groups each has
    # its own optimizer and the value loss needs no relative weight.
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # ---- rollout buffer --------------------------------------------------- #
    # "cpu" (host RAM), "cuda", or "auto" (VRAM when the estimated buffer fits in
    # store_vram_frac of what is free). Host RAM is usually just as fast -- the
    # per-minibatch gather and H2D copy overlap with GPU compute -- and leaves
    # the VRAM for a bigger B, which does matter.
    store_device: str = "cpu"
    store_vram_frac: float = 0.45      # flatten() briefly needs ~2x, hence < 0.5

    # ---- throughput ------------------------------------------------------- #
    # Worker PROCESSES pre-generating per-episode instances (maps/scenarios). If
    # generating one costs milliseconds of pure Python and every finished episode
    # needs one, this is otherwise a serial stall in the middle of the rollout.
    # Counted PER RANK. 0 = generate inline (deterministic, slower).
    instance_workers: int = 0
    instance_queue: int = 512          # ready instances buffered
    torch_threads: Optional[int] = None    # None -> min(8, cpu_count // 2)

    # ---- opponent pool ---------------------------------------------------- #
    opp_ema_alpha: float = 0.02        # EMA rate for per-opponent win rate
    pool_add_threshold: float = 0.6    # snapshot when the MIN win rate exceeds this
    pool_max_size: int = 7             # initial total cap (incl. scripted bots)
    pool_snapshot_every: int = 0       # every N iters add a PERMANENT snapshot (0 = off)
    opp_sample_floor: float = 0.05     # min sampling weight per opponent

    # ---- training phases -------------------------------------------------- #
    # Iterations are grouped into blocks of phase_iters; entering block k applies
    # phases[k] on top of the flat values. The LAST entry is held for every later
    # block, and the phase is resolved from the iteration number so a resumed run
    # lands in the right one. phases=None -> flat values throughout.
    phase_iters: int = 250
    phases: Optional[list] = None

    # ---- logging / checkpointing ------------------------------------------ #
    use_wandb: bool = False
    wandb_project: str = "rlkit"
    ckpt_path: str = "checkpoint.pt"   # written at the START of every iteration
    resume: bool = True                # resume from ckpt_path if it exists
    log_every: int = 1


# Fields a phase may NOT override: they either size a tensor/buffer that is
# allocated once, or belong to the process setup rather than the optimization.
_STRUCTURAL = frozenset({
    "B", "iters", "store_device", "store_vram_frac", "instance_workers",
    "instance_queue", "torch_threads", "phases", "phase_iters", "use_wandb",
    "wandb_project", "ckpt_path", "resume", "log_every",
})


def load_config(path, cls=BaseConfig):
    """Load ``cls`` from a YAML file, keeping defaults for anything missing.

    Unknown keys are a hard error rather than a silent no-op: a typo in a
    hyperparameter file is otherwise invisible until the run finishes wrong.
    """
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}\n"
                         f"known: {sorted(known)}")
    return cls(**d)


def dump_config(cfg):
    """Config as a plain dict (for wandb / checkpoints)."""
    return dataclasses.asdict(cfg)


class PhaseSchedule:
    """Resolves the hyperparameters in force at a given iteration.

    ``cfg.phases`` is a list of dicts; iteration ``it`` uses entry
    ``min(it // cfg.phase_iters, len(phases) - 1)`` layered on top of the flat
    config, so the last entry is the steady state. Any non-structural config
    field may appear in a phase -- including fields your own Config subclass
    added, which is how a game's own knob can be annealed alongside lr.

    Everything downstream reads the RESOLVED config, so nothing else in the
    trainer has to know that phases exist.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        known = {f.name for f in fields(type(cfg))}
        for k, p in enumerate(cfg.phases or []):
            if not isinstance(p, dict):
                raise ValueError(f"phase {k + 1} must be a dict, got {type(p)}")
            bad = set(p) - known
            if bad:
                raise ValueError(f"phase {k + 1} has unknown keys {sorted(bad)}")
            structural = set(p) & _STRUCTURAL
            if structural:
                raise ValueError(
                    f"phase {k + 1} may not override {sorted(structural)}: these "
                    f"size buffers or set up the process and are read once")

    @property
    def enabled(self):
        return bool(self.cfg.phases)

    def at(self, it):
        """-> (phase number (0 when no schedule), resolved config)"""
        if not self.cfg.phases:
            return 0, self.cfg
        k = min(it // max(1, self.cfg.phase_iters), len(self.cfg.phases) - 1)
        return k + 1, dataclasses.replace(self.cfg, **self.cfg.phases[k])

    def max_steps_per_iter(self):
        """Largest rollout any phase will ask for -- used to size the buffer once
        so its host/VRAM placement cannot flip mid-run."""
        return max([p.get("steps_per_iter", self.cfg.steps_per_iter)
                    for p in (self.cfg.phases or [])] + [self.cfg.steps_per_iter])
