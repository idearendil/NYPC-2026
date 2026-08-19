#!/usr/bin/env python3
"""THE TEMPLATE: a complete rlkit integration of a tiny two-player game.

Copy this file, replace the game, keep the shape. It is deliberately small enough
to read in one sitting and deliberately exercises the parts you will actually need
under time pressure:

  * a batched simultaneous-move game on the GPU (``Duel``)
  * a FACTORED action space -- a masked categorical plus a conditional Bernoulli --
    with the log-prob summed the same way at sampling and at re-evaluation
  * two scripted opponents for the pool
  * per-episode INSTANCE generation through ``rlkit.InstanceFactory``
  * the standard CLI, config file, phases, checkpoint/resume and multi-GPU

Run it:
    python -m examples.toy_duel --smoke
    python -m examples.toy_duel --config examples/toy_duel.yaml
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlkit

# --------------------------------------------------------------------------- #
# 1. the game
# --------------------------------------------------------------------------- #
# Two duellists. Each turn both simultaneously GROW (economy +1), STRIKE (damage =
# economy, doubled if they go all-in at the cost of economy) or GUARD (absorb 3).
GROW, STRIKE, GUARD = 0, 1, 2
N_CAT = 3
GUARD_ABSORB = 3
ALLIN_ECON_COST = 2
STRIKE_MIN_ECON = 2        # you need force to attack -> an action MASK
OBS_DIM = 6


class Duel:
    """The batched env. Plain tensors, no autograd, one step per call.

    Everything is [B, ...] and every rule is a masked tensor expression: this is
    what makes a rollout cost one kernel launch per rule instead of one per game.
    """

    def __init__(self, B, device, hp0=20, econ0=1, max_days=40):
        self.B, self.device = B, device
        self.hp0 = torch.full((B, 2), hp0, dtype=torch.long, device=device)
        self.econ0 = torch.full((B, 2), econ0, dtype=torch.long, device=device)
        self.limit = torch.full((B,), max_days, dtype=torch.long, device=device)
        self.hp = self.hp0.clone()
        self.econ = self.econ0.clone()
        self.day = torch.zeros(B, dtype=torch.long, device=device)

    # ---- instances ------------------------------------------------------- #
    def set_instances(self, rows, hp0, econ0, limit):
        self.hp0[rows], self.econ0[rows], self.limit[rows] = hp0, econ0, limit

    def reset(self, rows):
        self.hp[rows] = self.hp0[rows]
        self.econ[rows] = self.econ0[rows]
        self.day[rows] = 0

    # ---- dynamics -------------------------------------------------------- #
    def step(self, cat, allin):
        """``cat`` [B,2] long, ``allin`` [B,2] bool. Resolves both sides at once."""
        strike = cat == STRIKE
        dmg = torch.where(strike, self.econ * torch.where(allin, 2, 1),
                          torch.zeros_like(self.econ))
        absorb = torch.where(cat == GUARD, GUARD_ABSORB, 0)
        # side 0's damage lands on side 1 and vice versa -> flip along dim 1
        incoming = (dmg.flip(1) - absorb).clamp(min=0)
        self.hp = self.hp - incoming
        self.econ = self.econ + (cat == GROW).long()
        self.econ = torch.where(strike & allin,
                                (self.econ - ALLIN_ECON_COST).clamp(min=1), self.econ)
        self.day = self.day + 1

    def outcome(self):
        """``(reward [B] for side 0, done [B])`` -- +/-10 terminal, 0 for a draw."""
        dead = self.hp <= 0
        over = dead.any(1) | (self.day >= self.limit)
        win = dead[:, 1] & ~dead[:, 0]
        lose = dead[:, 0] & ~dead[:, 1]
        timeout = (self.day >= self.limit) & ~dead.any(1)
        r = torch.zeros(self.B, device=self.device)
        r = torch.where(win, 10.0, r)
        r = torch.where(lose, -10.0, r)
        r = torch.where(timeout & (self.hp[:, 0] > self.hp[:, 1]), 10.0, r)
        r = torch.where(timeout & (self.hp[:, 0] < self.hp[:, 1]), -10.0, r)
        return r, over


# --------------------------------------------------------------------------- #
# 2. per-episode instances (via worker processes)
# --------------------------------------------------------------------------- #
def gen_instance(rng: random.Random):
    """One episode's starting conditions, as a plain picklable tuple.

    Module-level and torch-free ON PURPOSE: ``InstanceFactory`` pickles this by
    reference into worker processes, and a worker that does not import torch starts
    in milliseconds and costs no VRAM. Anything expensive to construct is built
    back in the parent by ``postprocess``.
    """
    return (rng.randint(14, 26), rng.randint(1, 3), rng.randint(30, 50))


# --------------------------------------------------------------------------- #
# 3. the policy
# --------------------------------------------------------------------------- #
def bern_logp(p, x):
    p = p.clamp(1e-6, 1 - 1e-6)
    return x * torch.log(p) + (1 - x) * torch.log(1 - p)


def bern_entropy(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log1p(-p))


def mlp(d_in, d, d_out):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                         nn.Linear(d, d_out))


class DuelPolicy(rlkit.Policy):
    """Actor and critic as two separate nets -> two optimizer groups.

    The action is FACTORED: a categorical over the three moves and a Bernoulli
    "all-in" that only exists when the move is STRIKE. Two invariants make PPO
    valid over a factored space like this, and both are easy to break:

      1. the stored log-prob is the SUM over factors, with the same masks, as the
         one recomputed in ``evaluate`` -- otherwise the ratio is meaningless;
      2. a factor that was MASKED OFF contributes neither log-prob nor entropy, so
         a decision the policy never made cannot be reinforced.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        d = cfg.d_model
        self.actor = mlp(OBS_DIM, d, N_CAT + 1).to(device)   # cat logits + all-in
        self.critic = mlp(OBS_DIM, d, 1).to(device)

    # ---- plumbing --------------------------------------------------------- #
    def modules(self):
        return {"actor": self.actor, "critic": self.critic}

    def param_groups(self):
        return {"actor": list(self.actor.parameters()),
                "critic": list(self.critic.parameters())}

    # ---- collection ------------------------------------------------------- #
    def act(self, obs, **kw):
        out = self.actor(obs["obs"])
        cat_logits = out[:, :N_CAT].masked_fill(~obs["cat_mask"], -1e9)
        logp_cat = F.log_softmax(cat_logits, dim=1)
        cat = torch.multinomial(logp_cat.exp(), 1).squeeze(1)
        p_allin = torch.sigmoid(out[:, N_CAT])
        allin_mask = cat == STRIKE
        allin = torch.bernoulli(p_allin) * allin_mask.float()
        old_logp = (logp_cat.gather(1, cat[:, None]).squeeze(1)
                    + allin_mask.float() * bern_logp(p_allin, allin))
        store = dict(obs=obs["obs"], cat_mask=obs["cat_mask"], cat=cat, allin=allin,
                     allin_mask=allin_mask, old_logp=old_logp)
        action = dict(cat=cat, allin=allin > 0.5)
        return action, store, {}          # this game carries nothing between turns

    def value(self, obs):
        return self.critic(obs["obs"])[:, 0]

    # ---- update ----------------------------------------------------------- #
    def evaluate(self, mb):
        out = self.actor(mb["obs"])
        logp_cat = F.log_softmax(out[:, :N_CAT].masked_fill(~mb["cat_mask"], -1e9),
                                 dim=1)
        p_allin = torch.sigmoid(out[:, N_CAT])
        m = mb["allin_mask"].float()
        logp = (logp_cat.gather(1, mb["cat"][:, None]).squeeze(1)
                + m * bern_logp(p_allin, mb["allin"]))
        ent = (-(logp_cat.exp() * logp_cat).sum(1) + m * bern_entropy(p_allin))
        return rlkit.ActorOut(logp=logp, entropy=ent)

    def evaluate_value(self, mb):
        return rlkit.CriticOut(value=self.critic(mb["obs"])[:, 0])


# --------------------------------------------------------------------------- #
# 4. scripted opponents
# --------------------------------------------------------------------------- #
class Rusher(rlkit.ScriptedOpponent):
    """Strike as soon as it can; go all-in when hurt. The aggression yardstick."""

    name = "rusher"

    def act(self, task, obs, rows, **kw):
        env, side = task.env, task.OPPONENT
        econ, hp = env.econ[:, side], env.hp[:, side]
        cat = torch.where(econ >= STRIKE_MIN_ECON,
                          torch.full_like(econ, STRIKE), torch.full_like(econ, GROW))
        return dict(cat=cat, allin=(cat == STRIKE) & (hp < 8)), {}


class Turtle(rlkit.ScriptedOpponent):
    """Grow to 5, then alternate strike/guard. The greedy-economy yardstick."""

    name = "turtle"

    def act(self, task, obs, rows, **kw):
        env, side = task.env, task.OPPONENT
        econ = env.econ[:, side]
        alt = torch.where(env.day % 2 == 0, torch.full_like(econ, STRIKE),
                          torch.full_like(econ, GUARD))
        cat = torch.where(econ >= 5, alt, torch.full_like(econ, GROW))
        cat = torch.where((cat == STRIKE) & (econ < STRIKE_MIN_ECON),
                          torch.full_like(cat, GROW), cat)
        return dict(cat=cat, allin=torch.zeros_like(cat, dtype=torch.bool)), {}


# --------------------------------------------------------------------------- #
# 5. the task
# --------------------------------------------------------------------------- #
class DuelTask(rlkit.TwoPlayerTask):
    """Glue: perspective-canonical observations, the turn, and episode restarts."""

    def __init__(self, B, device, cfg, seed=0, factory=None):
        self.B, self.device, self.cfg = B, device, cfg
        self.env = Duel(B, device)
        self.factory = factory
        self.rng = random.Random(seed)
        self._draw_instances(torch.arange(B, device=device))
        self.env.reset(torch.arange(B, device=device))

    # ---- observation ------------------------------------------------------ #
    def observe(self, side):
        env = self.env
        opp = 1 - side
        hp_me = env.hp[:, side].float()
        hp_op = env.hp[:, opp].float()
        prog = env.day.float() / env.limit.float() - 0.5
        # Perspective-canonical: every feature is "mine" vs "theirs", never
        # "player 0" vs "player 1" -- which is what lets ONE network play both
        # sides, and therefore lets self-play work at all.
        obs = torch.stack([
            torch.log1p(hp_me.clamp(min=0) / 5), torch.log1p(hp_op.clamp(min=0) / 5),
            torch.log1p(env.econ[:, side].float()),
            torch.log1p(env.econ[:, opp].float()),
            prog, (hp_me - hp_op) / 20.0,
        ], dim=1)
        return dict(obs=obs, cat_mask=self._cat_mask(env.econ[:, side]))

    def _cat_mask(self, econ):
        mask = torch.ones(self.B, N_CAT, dtype=torch.bool, device=self.device)
        mask[:, STRIKE] = econ >= STRIKE_MIN_ECON
        return mask

    # ---- the turn --------------------------------------------------------- #
    def empty_opponent_out(self, B):
        return (dict(cat=torch.full((B,), GROW, dtype=torch.long, device=self.device),
                     allin=torch.zeros(B, dtype=torch.bool, device=self.device)),
                {})

    def env_step(self, agent_action, opp_action):
        cat = torch.stack([agent_action["cat"], opp_action["cat"]], dim=1)
        allin = torch.stack([agent_action["allin"], opp_action["allin"]], dim=1)
        self.env.step(cat, allin)

    def reward_done(self):
        return self.env.outcome()

    # ---- episode boundaries ----------------------------------------------- #
    def _draw_instances(self, rows):
        n = rows.numel()
        if n == 0:
            return
        if self.factory is not None:
            inst = [self.factory.get() for _ in range(n)]
        else:
            inst = [gen_instance(self.rng) for _ in range(n)]
        t = torch.tensor(inst, dtype=torch.long, device=self.device)
        self.env.set_instances(rows, t[:, 0:1].expand(n, 2), t[:, 1:2].expand(n, 2),
                               t[:, 2])

    def reset_finished(self, done, rows):
        self._draw_instances(rows)
        self.env.reset(rows)

    def metrics(self):
        m = {"mean_econ": float(self.env.econ.float().mean()),
             "mean_day": float(self.env.day.float().mean())}
        if self.factory is not None:
            m["instance_queue_misses"] = self.factory.misses
        return m

    def log_extra(self):
        return f"econ {float(self.env.econ.float().mean()):.1f}"

    def close(self):
        if self.factory is not None:
            self.factory.close()


# --------------------------------------------------------------------------- #
# 6. config + wiring
# --------------------------------------------------------------------------- #
@dataclass
class Config(rlkit.BaseConfig):
    """Game-specific knobs on top of the framework's."""
    d_model: int = 64


def build(ctx: rlkit.SetupCtx) -> rlkit.Setup:
    """Everything game-specific the trainer needs, for THIS rank.

    ``ctx.B`` is already this rank's share of the total batch and ``ctx.seed`` is
    already offset per rank -- so a multi-GPU run simulates different games without
    this function knowing anything about ranks.
    """
    cfg = ctx.cfg
    factory = None
    if cfg.instance_workers > 0:
        factory = rlkit.InstanceFactory(gen_instance, workers=cfg.instance_workers,
                                        seed=ctx.seed + 31, depth=cfg.instance_queue)
    task = DuelTask(ctx.B, ctx.device, cfg, seed=ctx.seed + 5, factory=factory)
    policy = DuelPolicy(cfg, ctx.device)
    return rlkit.Setup(task=task, policy=policy,
                       make_policy=lambda: DuelPolicy(cfg, ctx.device),
                       scripted=[Rusher(), Turtle()])


def main():
    ap = rlkit.trainer.add_cli_args(argparse.ArgumentParser())
    ap.set_defaults(config=os.path.join(os.path.dirname(__file__), "toy_duel.yaml"))
    args = ap.parse_args()
    if args.smoke:
        # A smoke run is a CORRECTNESS check, not a speed one: tiny everything, no
        # wandb, no resume, its own checkpoint file, and a pool forced to grow
        # every iteration so the add/evict paths actually run.
        cfg = Config(B=64, steps_per_iter=4096, iters=2, minibatch=1024, d_model=32,
                     use_wandb=False, resume=False, phases=None,
                     ckpt_path="checkpoint_toy_smoke.pt", instance_workers=0,
                     pool_snapshot_every=2, pool_add_threshold=0.0)
    else:
        cfg = (rlkit.load_config(args.config, Config)
               if os.path.exists(args.config) else Config())
        rlkit.trainer.apply_cli_args(cfg, args)
    # Multi-GPU must go through launch() BEFORE anything touches CUDA.
    if args.gpus and args.gpus > 1:
        rlkit.launch(_dist_entry, args.gpus, args=(cfg,))
    else:
        rlkit.train(cfg, build, device=args.device)


def _dist_entry(cfg):
    """Module-level so the spawned children can import it."""
    rlkit.train(cfg, build)


if __name__ == "__main__":
    main()
