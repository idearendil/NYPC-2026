"""The three interfaces that separate a GAME from the trainer.

This file is the contract. If a new game implements ``Task``, ``Policy`` and
(optionally) some ``ScriptedOpponent``s, everything else in rlkit works unchanged:
PPO, GAE, the opponent pool, multi-GPU, checkpoints, phases, logging.

    Task      -- the batched environment and its turn structure. Owns the board,
                 the action space, the observation, the reward and the episode
                 reset. Anything weird and game-specific goes HERE.
    Policy    -- the trainable networks, plus how to sample an action from them
                 (collection) and how to re-evaluate a stored action (update).
    ScriptedOpponent -- a hand-written bot, batched, used as a fixed pool member.

Two conventions that the whole framework depends on:

1. A "store" dict is whatever ``Policy.evaluate`` will need later to recompute the
   log-prob and entropy of the action that was actually taken -- observations,
   masks, sampled outcomes, and ``old_logp``. The trainer treats it as an opaque
   dict of [B, ...] tensors, concatenates it over the rollout, and hands
   minibatches of it back to you. Add whatever you need; there is no schema.
2. Every tensor you put in the store must be FRESHLY ALLOCATED, never a view of
   mutable env state (see ``rlkit.utils.to_device``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import copy

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# what Policy.evaluate / evaluate_value return
# --------------------------------------------------------------------------- #
@dataclass
class ActorOut:
    """Re-evaluation of a stored action under the CURRENT policy.

    ``logp`` and ``entropy`` are per-sample [n]; for a factored action space they
    are the SUM over the factors (the joint log-prob and the summed entropy),
    exactly as they were summed when the action was sampled -- the ratio
    ``exp(logp - old_logp)`` only means anything if both sides count the same
    factors with the same masks.

    ``extra_loss`` is added straight to the actor loss -- anything your policy
    wants to train that is not the surrogate itself. Leave it at 0.0 if you have
    none. ``metrics`` are scalars to log; they are averaged over minibatches
    on-device, so pass 0-dim tensors, not floats read with .item().
    """
    logp: torch.Tensor
    entropy: torch.Tensor
    extra_loss: Any = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticOut:
    """Value prediction for a minibatch, plus any extra critic-side loss."""
    value: torch.Tensor
    extra_loss: Any = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
class Policy(ABC):
    """The trainable networks and the three ways they are used.

    A Policy must hold ONLY modules and plain config -- no env, no task, no open
    files -- because ``clone_frozen()`` deep-copies it to make pool snapshots.

    ``act`` and ``evaluate`` are two views of the SAME decision: one samples it,
    the other recomputes its log-prob and entropy under the updated weights. Keep
    the two in sync -- that pair is what PPO's ratio is built from.
    """

    # ---- what the trainer needs to know about the parameters --------------- #
    @abstractmethod
    def modules(self) -> Dict[str, nn.Module]:
        """Named modules -- used for checkpointing and for the startup broadcast.
        The names are the checkpoint keys, so keep them stable."""

    @abstractmethod
    def param_groups(self) -> Dict[str, list]:
        """Optimizer groups. Either

            {'actor': [...], 'critic': [...]}   -- two Adams, two backward passes
                (the value loss then needs no relative weight, and a value-loss
                spike cannot blow up the policy's gradient), or
            {'all': [...]}                      -- one Adam over a shared trunk,
                loss = policy + vf_coef * value + your extra losses.

        Any other single-group name is treated like 'all'.
        """

    # ---- collection ------------------------------------------------------- #
    @abstractmethod
    def act(self, obs, **kw) -> Tuple[Dict, Dict, Dict]:
        """Sample an action. Called under ``no_grad`` during the rollout.

        Returns ``(action, store, extra)``:
          action -- what ``Task.env_step`` consumes (a dict of tensors).
          store  -- everything ``evaluate`` will need, INCLUDING ``old_logp``.
          extra  -- anything the task itself needs to carry to the next turn
                    (say, a multi-turn commitment the action started). May be
                    ``{}``; the trainer never looks inside.

        ``**kw`` is passed through from the task untouched, which is how per-turn
        modifiers (relaxed rules for an opponent, a masked-out target) reach here.
        """

    @abstractmethod
    def value(self, obs) -> torch.Tensor:
        """Critic value [B] for the transition being stored."""

    # ---- update ----------------------------------------------------------- #
    @abstractmethod
    def evaluate(self, mb) -> ActorOut:
        """Re-evaluate a minibatch of stored actions under the current policy."""

    @abstractmethod
    def evaluate_value(self, mb) -> CriticOut:
        """Re-run the critic on a minibatch (separate forward from ``evaluate``)."""

    # ---- provided --------------------------------------------------------- #
    def clone_frozen(self):
        """An inference-only deep copy, for the opponent pool.

        Override if a subclass holds anything that must not be deep-copied.
        """
        c = copy.deepcopy(self)
        for m in c.modules().values():
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return c

    def state_dict(self):
        return {k: m.state_dict() for k, m in self.modules().items()}

    def load_state_dict(self, sd, strict=True):
        for k, m in self.modules().items():
            if k in sd:
                m.load_state_dict(sd[k], strict=strict)

    def train_mode(self, flag=True):
        for m in self.modules().values():
            m.train(flag)

    def all_params(self):
        return [p for g in self.param_groups().values() for p in g]


# --------------------------------------------------------------------------- #
# ScriptedOpponent
# --------------------------------------------------------------------------- #
class ScriptedOpponent(ABC):
    """A hand-written batched bot occupying a FIXED pool slot (never evicted).

    Scripted opponents matter more than they look: a pool of nothing but your own
    snapshots is homogeneous, and a policy can drift into beating only itself. One
    or two committed strategies (an all-in rush, a greedy expander) keep an
    absolute yardstick in the pool and in the win-rate curves.

    ``full_batch`` says whether ``act`` computes all B games (usually cheaper for
    a vectorised bot -- the trainer keeps only the rows assigned to it) or just its
    own rows.
    """

    name: str = "scripted"
    full_batch: bool = True

    @abstractmethod
    def act(self, task, obs, rows, **kw) -> Tuple[Dict, Dict]:
        """Return ``(action, extra)``. ``rows`` are the games assigned to this bot;
        with ``full_batch`` the tensors must still be full-batch. A scripted bot
        normally ignores ``obs`` and reads the task's env state directly, and must
        tolerate unknown ``**kw``."""

    def reset_rows(self, rows):
        """Clear per-game state for episodes that just restarted. No-op by
        default (a stateless bot needs nothing)."""


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
class Task(ABC):
    """The game. The trainer only ever calls the four methods below.

    Prefer subclassing ``TwoPlayerTask``, which implements ``rollout_turn`` for the
    ordinary "both sides act once, then the env advances" structure and leaves you
    ``observe`` / ``env_step`` / ``empty_opponent_out``.
    """

    B: int          # games simulated in THIS process (already the per-rank share)

    @abstractmethod
    def rollout_turn(self, ctx) -> Dict[str, torch.Tensor]:
        """Advance the batch by one env step and return the transition to store.

        ``ctx`` (a ``RolloutContext``) is how you reach the networks:
            ctx.agent_act(obs, **kw)     -> (action, store, extra)
            ctx.agent_value(obs)         -> [B] value
            ctx.opponent_act(obs, **kw)  -> (action, extra), dispatched over the
                                            pool assignment for you
        Call ``agent_act`` exactly ONCE for the transition you return -- that is
        the decision PPO will train on. You may call it again for further
        inferences (a first-turn re-decision, a look-ahead) as long as the store
        you return is the one you want trained.

        The returned dict is stored verbatim; ``reward`` and ``done`` are added by
        the trainer, and ``value`` must be in it.
        """

    @abstractmethod
    def reward_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(reward [B] float, done [B] bool)`` for the step just taken, from the
        AGENT's perspective. A sparse terminal reward (+/-1, +/-10) is usually
        right; the discount and the critic spread it back over the episode."""

    @abstractmethod
    def reset_finished(self, done, rows) -> None:
        """Restart the finished games: regenerate their instance/map, reset the
        env rows, and clear any per-game state you carry across turns (feature
        history, commitments, per-episode flags)."""

    # ---- optional hooks ---------------------------------------------------- #
    # Define this only if "the value of the state the rollout ended in" is not
    # simply policy.value(self.observe(AGENT)) -- GAE needs it for the last step.
    #
    #   def bootstrap_value(self, policy) -> torch.Tensor: ...

    def tally_mask(self) -> Optional[torch.Tensor]:
        """[B] bool: which games' results may feed the opponent-pool win rate.
        Return None (default) for all of them. Use it to exclude games played
        under modified rules, which are not a fair yardstick."""
        return None

    def log_extra(self) -> str:
        """Extra text appended to the console line for one iteration."""
        return ""

    def metrics(self) -> Dict[str, float]:
        """Extra scalars to log for one iteration (game-specific diagnostics)."""
        return {}

    def state_dict(self) -> Optional[Dict]:
        """Anything the task must carry across a restart. Usually None: episodes
        do not resume across runs (the env is rebuilt), so per-game state is
        deliberately NOT checkpointed -- a fresh state always matches a fresh env.
        """
        return None

    def load_state_dict(self, sd) -> None:
        pass

    def close(self) -> None:
        """Release worker processes / files at the end of training."""


class TwoPlayerTask(Task):
    """``rollout_turn`` for the common symmetric two-player turn structure.

    Both sides observe the same state from their own perspective, act
    simultaneously, and the env resolves the turn. Implement ``observe``,
    ``env_step`` and ``empty_opponent_out``; override ``rollout_turn`` itself if
    your game needs something the skeleton cannot express.
    """

    AGENT, OPPONENT = 0, 1

    @abstractmethod
    def observe(self, side: int) -> Dict[str, torch.Tensor]:
        """The observation dict for one side, from that side's perspective.

        Canonicalise the perspective here (mirror the board, swap 'mine'/'theirs')
        so ONE network can play either side -- that is what makes self-play work
        with a single policy.
        """

    @abstractmethod
    def env_step(self, agent_action, opp_action) -> None:
        """Apply both sides' actions and advance one turn."""

    @abstractmethod
    def empty_opponent_out(self, B) -> Tuple[Dict, Dict]:
        """Zero/neutral ``(action, extra)`` templates of full batch size.

        The pool merges several opponents' outputs into these, so the neutral
        value must be the real no-op for each field -- e.g. ``-1`` for "no move",
        not 0, which would order every idle unit to region 0.
        """

    def labels(self) -> Dict[str, torch.Tensor]:
        """Extra per-step tensors to store alongside the transition, computed
        AFTER the step has resolved. Default: none."""
        return {}

    def act_kwargs(self, side: int) -> Dict:
        """Per-turn extra kwargs for that side's ``act`` (rule modifiers, masks)."""
        return {}

    def rollout_turn(self, ctx):
        obs = self.observe(self.AGENT)
        action, store, extra = ctx.agent_act(obs, **self.act_kwargs(self.AGENT))
        store["value"] = ctx.agent_value(obs)
        obs_op = self.observe(self.OPPONENT)
        opp_action, opp_extra = ctx.opponent_act(obs_op,
                                                **self.act_kwargs(self.OPPONENT))
        self.on_actions(action, extra, opp_action, opp_extra)
        self.env_step(action, opp_action)
        store.update(self.labels())
        return store

    def on_actions(self, action, extra, opp_action, opp_extra) -> None:
        """Hook between sampling and stepping: carry ``extra`` into env state,
        stash a prediction to feed back next turn, rewrite the actions."""
