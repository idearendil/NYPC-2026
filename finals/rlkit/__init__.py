"""rlkit -- the game-INDEPENDENT half of the self-play PPO trainer.

Everything in this package is reusable across games. It contains no knowledge of
a board, an action space, an observation layout or a reward rule: those live
behind the three small interfaces in ``rlkit.interfaces`` (``Task``, ``Policy``,
``ScriptedOpponent``), which are the only things you write for a new game.

What you get for free
---------------------
* PPO update (clip, entropy bonus, separate or shared actor/critic optimizers,
  grad clipping, KL early stop, explained variance, clip fraction).
* GAE + rollout buffer with a host/VRAM placement decision, global advantage
  whitening, and a minibatch iterator.
* An opponent pool: fixed scripted bots + frozen policy snapshots, EMA win
  rates, win-rate-inverse sampling, periodic permanent snapshots, growth on a
  win-rate threshold, FIFO eviction, stable per-opponent ids for logging.
* Multi-GPU data parallelism (manual flat gradient all-reduce) where the ranks
  are kept bit-identical, so their pool decisions never diverge.
* Crash-safe checkpoint/resume including the pool, the optimizers and the RNGs.
* A training-phase schedule (lr / epochs / entropy / rollout size by iteration).
* Background worker processes generating per-episode instances (maps).
* Console + wandb logging.
* An in-process fake of the collectives so the multi-rank logic can be tested
  without NCCL (``rlkit.testing``).
* A parity harness for checking a batched env against a reference simulator
  turn by turn (``rlkit.parity``).

Start here
----------
``examples/toy_duel.py``   -- a complete, runnable 200-line game + Task + Policy.
                              Copy it and replace the game.
``examples/nypc2026.py``   -- the 2026 preliminary game plugged in, showing how a
                              big real task (a factored action space, scripted
                              bots, a turn that needs two inferences) fits.
``FINALS_PLAYBOOK.md``     -- the 6-hour checklist.
"""
from __future__ import annotations

from .config import BaseConfig, PhaseSchedule, load_config
from .dist import Dist, launch, tune_backend
from .interfaces import (ActorOut, CriticOut, Policy, ScriptedOpponent, Task,
                         TwoPlayerTask)
from .buffer import RolloutBuffer, minibatch_iter, whiten_
from .pool import OpponentPool, Tally
from .ppo import PPO
from .rollout import RolloutContext, run_rollout
from .trainer import Setup, SetupCtx, train
from .checkpoint import Checkpointer, lenient_load, save_atomic
from .logger import Logger
from .factory import InstanceFactory, make_instances
from . import parity, testing, utils

__all__ = [
    "BaseConfig", "PhaseSchedule", "load_config",
    "Dist", "launch", "tune_backend",
    "Task", "TwoPlayerTask", "Policy", "ScriptedOpponent", "ActorOut", "CriticOut",
    "RolloutBuffer", "minibatch_iter", "whiten_",
    "OpponentPool", "Tally", "PPO",
    "RolloutContext", "run_rollout",
    "Setup", "SetupCtx", "train",
    "Checkpointer", "lenient_load", "save_atomic",
    "Logger", "InstanceFactory", "make_instances", "parity", "testing", "utils",
]
