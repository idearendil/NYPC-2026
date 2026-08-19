"""The training loop: the one function that ties everything together.

    train(cfg, setup)

``setup`` is the only game-specific thing it takes -- a callable that builds the
task, the policy and the scripted opponents for THIS rank. Everything else
(iteration structure, phases, checkpointing, the pool's evolution, logging,
multi-GPU) is here and is reusable as-is.

Iteration structure, and the ordering constraints that matter:

    1. resolve the phase                (so a resumed run lands in the right one)
    2. checkpoint                        BEFORE any work, so a crash loses <= 1 iter
    3. rollout                           -> buffer + on-device tally
    4. advance the pool's EMA win rates  ONCE, from all-reduced tallies
    5. bootstrap value, GAE, flatten, whiten (global statistics)
    6. PPO epochs
    7. grow/snapshot the pool            AFTER the update, so a snapshot is of the
                                         policy that the win rates were measured on
    8. log
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .buffer import RolloutBuffer, whiten_
from .checkpoint import Checkpointer
from .config import PhaseSchedule, dump_config
from .dist import Dist, tune_backend
from .logger import Logger, console_line
from .pool import OpponentPool
from .ppo import PPO
from .rollout import bootstrap_value, run_rollout


@dataclass
class SetupCtx:
    """What ``setup`` is told about the process it is building for."""
    cfg: object
    device: str
    B: int                  # games THIS rank simulates (cfg.B // world)
    minibatch: int          # rows THIS rank takes per optimizer step
    rank: int
    world: int
    seed: int               # already offset per rank where it should be
    dist: object


@dataclass
class Setup:
    """What ``setup`` hands back."""
    task: object
    policy: object
    # A fresh, untrained policy of the same architecture. Used to rebuild the
    # pool's frozen snapshots from a checkpoint, so it must be cheap and must not
    # touch the env.
    make_policy: Callable[[], object]
    scripted: List = field(default_factory=list)
    # Per-module {state_dict key: required leading dim} that must match on resume;
    # see Checkpointer. Use it for heads whose MEANING changed.
    guards: Optional[Dict] = None
    run_name: Optional[str] = None


def train(cfg, setup, *, device=None, seed=0, log_every=None, verbose=True,
          on_metrics=None):
    """Run PPO self-play. Returns ``(policy, pool, task)``.

    ``on_metrics(it, metrics)`` is called once per iteration on every rank
    after the update, with the same dict that goes to wandb -- the hook for
    scripted experiments (a batch-size sweep, an early-stopping harness)
    without having to scrape stdout.
    """
    dd = Dist.init()
    tune_backend(cfg.torch_threads)
    device = dd.device_for(device)
    log_every = log_every if log_every is not None else cfg.log_every
    main = dd.is_main and verbose

    # cfg.B and cfg.minibatch are TOTALS; each rank takes its share, so an N-GPU
    # run sees the same horizon, the same number of optimizer steps and the same
    # effective batch as a 1-GPU run with the same config.
    B_loc = dd.split(cfg.B, "B")
    mb_loc = dd.split(cfg.minibatch, "minibatch")

    # Build IDENTICAL nets on every rank but simulate DIFFERENT games: seed the
    # construction from `seed` alone (weights are broadcast from rank 0 right
    # after, so this is belt-and-braces), and offset per rank afterwards.
    torch.manual_seed(seed)
    su = setup(SetupCtx(cfg=cfg, device=device, B=B_loc, minibatch=mb_loc,
                        rank=dd.rank, world=dd.world,
                        seed=seed + 7919 * dd.rank, dist=dd))
    task, policy = su.task, su.policy
    dd.broadcast_params(list(policy.modules().values()))
    torch.manual_seed(seed + 104729 * dd.rank)

    pool = OpponentPool(su.scripted, policy, B=B_loc, device=device,
                        ema_alpha=cfg.opp_ema_alpha,
                        add_threshold=cfg.pool_add_threshold,
                        max_size=cfg.pool_max_size,
                        snapshot_every=cfg.pool_snapshot_every,
                        sample_floor=cfg.opp_sample_floor,
                        seed=seed + 777 + 7919 * dd.rank)
    ppo = PPO(policy, cfg, dd, device)
    phases = PhaseSchedule(cfg)

    # One buffer for the whole run, sized from the LARGEST rollout any phase asks
    # for so its host/VRAM placement cannot flip mid-run.
    max_steps = max(1, phases.max_steps_per_iter() // cfg.B)
    buffer = RolloutBuffer(cfg.store_device, compute_device=device,
                           expected_steps=max_steps,
                           vram_frac=cfg.store_vram_frac, verbose=main)

    ckpt = Checkpointer(cfg.ckpt_path, dd, device, guards=su.guards)
    start_iter = 0
    if cfg.resume and ckpt.exists:
        start_iter = ckpt.load(policy=policy, ppo=ppo, pool=pool,
                               make_policy=su.make_policy, task=task, B=B_loc,
                               seed=seed, verbose=verbose)

    logger = Logger(enabled=cfg.use_wandb, project=cfg.wandb_project,
                    config=dump_config(cfg), is_main=dd.is_main, name=su.run_name)

    cur_phase = None
    try:
        for it in range(start_iter, cfg.iters):
            phase_no, c = phases.at(it)
            steps = max(1, c.steps_per_iter // cfg.B)
            if phase_no != cur_phase:
                cur_phase = phase_no
                ppo.set_lr(c.lr)
                if phase_no and main:
                    print(f"== phase {phase_no} (iter {it}): lr {c.lr:g} "
                          f"epochs {c.epochs} ent_coef {c.ent_coef:g} "
                          f"steps/iter {c.steps_per_iter:,} (horizon {steps})")

            if dd.is_main:
                ckpt.save(it, policy=policy, ppo=ppo, pool=pool, task=task,
                          cfg=dump_config(cfg))

            t0 = time.time()
            # ---- collect ---------------------------------------------------- #
            policy.train_mode(False)
            tally = run_rollout(task, policy, pool, buffer, steps,
                                device=device, dist=dd)
            episodes, reward_sum = pool.apply_tally(tally, dd)
            last_val = bootstrap_value(task, policy)

            # ---- advantages ------------------------------------------------- #
            buffer.compute_gae(last_val, c.gamma, c.lam)
            flat = buffer.flatten()
            whiten_(flat, "adv", dd, device)

            # ---- update ----------------------------------------------------- #
            policy.train_mode(True)
            m = ppo.update(flat, c, mb_size=mb_loc)
            del flat

            # ---- pool evolution (after the update: a snapshot should be of the
            #      policy whose win rates we just measured) --------------------- #
            added = pool.maybe_snapshot(it, policy)
            added = pool.maybe_grow(policy) or added

            # ---- log --------------------------------------------------------- #
            dt = time.time() - t0
            m.update(pool.metrics())
            m.update(task.metrics())
            m.update({
                "iter": it,
                "phase": phase_no,
                "episodes": episodes,
                "avg_ep_R": reward_sum / max(episodes, 1),
                "lr": c.lr,
                "ent_coef": c.ent_coef,
                "epochs": c.epochs,
                "steps_per_iter": c.steps_per_iter,
                "horizon": steps,
                "pool_added": int(added),
                "iter_seconds": dt,
                "steps_per_s": steps * cfg.B / max(dt, 1e-9),
                "world_size": dd.world,
            })
            line = (console_line(it, phase_no, m, task.log_extra())
                    if (it % max(log_every, 1) == 0 and main) else None)
            logger.log(it, m, line)
            if on_metrics is not None:
                on_metrics(it, m)
    finally:
        logger.close()
        task.close()
        dd.shutdown()
    return policy, pool, task


def add_cli_args(ap):
    """The standard flags. Kept here so every game's entry point has the same ones.

    Returns the parser; apply the result with ``apply_cli_args``.
    """
    ap.add_argument("--config", default="config.yaml",
                    help="YAML hyperparameter file (used if it exists)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="steps_per_iter; also DISABLES the phase schedule")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--minibatch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint path")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-phases", action="store_true",
                    help="use the flat config values throughout")
    ap.add_argument("--gpus", type=int, default=None,
                    help="data-parallel training across N GPUs (default: all "
                         "visible; 1 disables). B and minibatch are TOTALS and "
                         "are split, so the data per iteration is unchanged.")
    ap.add_argument("--workers", type=int, default=None,
                    help="background instance-generation processes (0 = inline)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (correctness, not speed)")
    return ap


def apply_cli_args(cfg, args, verbose=True):
    """Overlay the standard flags onto a config."""
    if args.B is not None:
        cfg.B = args.B
    if args.steps is not None:
        cfg.steps_per_iter = args.steps
        if cfg.phases:
            if verbose:
                print("--steps given: phase schedule disabled (flat values)")
            cfg.phases = None
    if args.no_phases and cfg.phases:
        if verbose:
            print("--no-phases: using the flat config values throughout")
        cfg.phases = None
    if args.iters is not None:
        cfg.iters = args.iters
    if args.minibatch is not None:
        cfg.minibatch = args.minibatch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.ckpt is not None:
        cfg.ckpt_path = args.ckpt
    if args.workers is not None:
        cfg.instance_workers = args.workers
    if args.no_wandb:
        cfg.use_wandb = False
    if args.no_resume:
        cfg.resume = False
    return cfg
