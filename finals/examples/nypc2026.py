#!/usr/bin/env python3
"""The 2026 preliminary game, plugged into rlkit.

This exists as PROOF that the interfaces fit a big real task, and as the reference
to read next to your own adapter. It deliberately reuses the existing, already
validated game code from ``ppo_selfplay.py`` (feature extraction, the factored
action sampler, the networks, the scripted bots) rather than reimplementing it, so
what you see here is exactly and only the GLUE:

    NypcPolicy  ~ 60 lines   wraps ActorT1 / ActorT2 / Critic
    RusherBot / JapperBot    ~ 10 lines each
    NypcTask    ~ 120 lines  observations, the turn (incl. the opening split),
                             reward, episode restart

Everything hard about this game -- a hidden-information estimate fed to the actor
while the critic sees the truth, auxiliary heads whose prediction is fed back as an
input feature, a factored action space with gold-affordability masking, a turn-1
double inference, opponents playing under relaxed rules whose results are excluded
from the win rate -- lands inside those three classes. The trainer does not know
about any of it.

    python -m examples.nypc2026 --smoke
    python -m examples.nypc2026 --config examples/nypc2026.yaml --gpus 2

NOTE: this writes its OWN checkpoint (checkpoint_rlkit.pt) in rlkit's format. It
does not read the legacy ``checkpoint.pt`` produced by ppo_selfplay.py -- that file
stays where it is, and ppo_selfplay.py keeps working unchanged.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import torch

# The PREVIOUS game's env, nets and scripted bots live in the repository root, one
# level above this kit -- this example is the only thing in `finals/` that depends
# on anything outside it.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import rlkit

try:
    import fast_env as fe
    import map_gen
    import ppo_selfplay as P
except ImportError as e:                            # pragma: no cover
    raise ImportError(
        f"examples/nypc2026.py is a REFERENCE integration of the 2026 PRELIMINARY "
        f"game and needs that game's code (fast_env.py, ppo_selfplay.py, "
        f"map_gen.py, testing-tool.py) in {_ROOT!r}. It is not needed for a new "
        f"game -- read it as an example of a large Task/Policy, or delete it. "
        f"({e})") from e


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Config(rlkit.BaseConfig):
    d_model: int = 64
    aux_coef: float = 0.15
    # turn-1 opening split: infer twice on the first turn so the HQ's two spare
    # warriors head to DIFFERENT strongholds (a faster 2-base opening, and what the
    # submission bot does).
    opening_split: bool = True
    # "full mobilisation" games: a fraction where the OPPONENT may ignore the
    # work-cap move restriction. Their results are excluded from the win rate --
    # useful training data, but not a fair yardstick.
    opp_relax_frac: float = 0.20
    opp_relax_prob: float = 0.10
    # env capacity: reserve room for the largest map so per-episode regeneration
    # never has to reallocate.
    n_cap: int = 109
    t_cap: int = 23


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class NypcPolicy(rlkit.Policy):
    """ActorT1 + ActorT2 (policy) and Critic (value), as separate optimizer groups.

    The critic reads ``t1_crit``/``glob_crit`` -- the privileged view, with the
    opponent's exact hidden gold and its actually-committed arrivals -- while the
    actor reads ``t1``/``glob``, which only contain what a submission could compute.
    Same tensors, two views, built once per turn by ``extract``.
    """

    def __init__(self, cfg, device, n_regions):
        self.cfg, self.N = cfg, n_regions
        self.t1 = P.ActorT1(d=cfg.d_model).to(device)
        self.t2 = P.ActorT2(cfg.d_model + P.T2_EXTRA, d=cfg.d_model).to(device)
        self.critic = P.Critic(d=cfg.d_model).to(device)

    def modules(self):
        return {"actor_t1": self.t1, "actor_t2": self.t2, "critic": self.critic}

    def param_groups(self):
        return {"actor": list(self.t1.parameters()) + list(self.t2.parameters()),
                "critic": list(self.critic.parameters())}

    def act(self, obs, **kw):
        action, store, _logp, commit, gold_pred = P.sample_policy(
            self.t1, self.t2, obs, self.N, **kw)
        # `extra` carries what the TASK needs next turn: the HQ-saving commitment
        # (a multi-turn macro) and this net's own prediction of the opponent's gold,
        # which becomes an input feature on the following turn.
        return action, store, dict(hq_commit=commit, gold_pred=gold_pred)

    def value(self, obs):
        return self.critic.value(obs["t1_crit"], obs["glob_crit"], obs["tmask"])

    def evaluate(self, mb):
        logp, ent, aux_pred = P.evaluate_policy(self.t1, self.t2, mb)
        aux = P.aux_losses(aux_pred, mb["gold_aux"], mb["gold_glob"], mb["tmask"])
        decided = mb["mob_mask"].sum()
        return rlkit.ActorOut(
            logp=logp, entropy=ent, extra_loss=self.cfg.aux_coef * aux,
            metrics={"aux_actor": aux.detach(),
                     # how often full mobilisation was chosen among the moves where
                     # the bit could change anything
                     "mobilize_rate": mb["mob_bit"].sum() / decided.clamp(min=1),
                     "mobilize_decisions": decided})

    def evaluate_value(self, mb):
        value, aux_pred = self.critic.value_aux(mb["t1_crit"], mb["glob_crit"],
                                               mb["tmask"])
        aux = P.aux_losses(aux_pred, mb["gold_aux"], mb["gold_glob"], mb["tmask"])
        return rlkit.CriticOut(value=value, extra_loss=self.cfg.aux_coef * aux,
                               metrics={"aux_critic": aux.detach()})


# --------------------------------------------------------------------------- #
# scripted opponents
# --------------------------------------------------------------------------- #
class RusherBot(rlkit.ScriptedOpponent):
    """Mass warriors, launch one wave, then switch to economy. All-in aggression."""

    name = "rusher"

    def __init__(self, B, device):
        self.state = P.RusherState(B, device)

    def act(self, task, obs, rows, **kw):
        return P.rusher_action(task.env, task.OPPONENT, self.state), {}

    def reset_rows(self, rows):
        self.state.reset_rows(rows)


class JapperBot(rlkit.ScriptedOpponent):
    """Expand to two bases, gather at a rally, send waves of five. Greedy economy."""

    name = "japper"

    def __init__(self, B, device):
        self.state = P.JapperState(B, device)

    def act(self, task, obs, rows, **kw):
        return P.japper_action(task.env, task.OPPONENT, self.state), {}

    def reset_rows(self, rows):
        self.state.reset_rows(rows)


# --------------------------------------------------------------------------- #
# task
# --------------------------------------------------------------------------- #
class NypcTask(rlkit.TwoPlayerTask):
    """The env, the per-turn feature history, and the turn structure.

    ``rollout_turn`` is overridden rather than using the TwoPlayerTask skeleton,
    because this game's first turn needs TWO agent inferences (see the opening
    split) -- which is exactly the case the escape hatch exists for.
    """

    AGENT, OPPONENT = 0, 1

    def __init__(self, B, device, cfg, seed=0):
        self.B, self.device, self.cfg = B, device, cfg
        maps = map_gen.make_maps(B, seed, workers=cfg.instance_workers)
        self.env = fe.FastEnv(maps, device=device, n_cap=cfg.n_cap, t_cap=cfg.t_cap)
        self.env._map_rng = random.Random(seed + 12345)
        self.N, T = self.env.N, self.env.mb.T
        # Background map generation: one map is ~25 ms of single-core Python and
        # every finished episode needs one, so at a large B this is otherwise a
        # serial stall in the middle of the rollout with the GPU idle.
        self.factory = None
        if cfg.instance_workers > 0:
            self.factory = rlkit.InstanceFactory(
                map_gen.random_map_lines, workers=cfg.instance_workers,
                seed=seed + 31, depth=cfg.instance_queue,
                postprocess=map_gen.tt().read_map)
            self.env.map_factory = self.factory
        self._misses = 0
        # per-side feature history: last turn's raw enemy-reachability (for the
        # per-turn delta) and each side's own last opp-gold prediction. Both are
        # zeroed at an episode boundary -- a fresh map has no previous turn.
        self.prev_reach = [torch.zeros(B, T, 5, device=device) for _ in range(2)]
        self.prev_gold = [torch.zeros(B, device=device) for _ in range(2)]
        # which games let the opponent ignore the work-cap move restriction
        self.relaxed = torch.rand(B, device=device) < cfg.opp_relax_frac
        self._relax_reg = None

    # ---- observation ------------------------------------------------------ #
    def observe(self, side):
        return P.extract(self.env, side, self.prev_reach[side], self.prev_gold[side])

    def empty_opponent_out(self, B):
        dev = self.device
        action = dict(
            build=torch.zeros(B, self.N, dtype=torch.bool, device=dev),
            # -1 = "no move": zero would order every idle region to region 0
            move=torch.full((B, self.N), -1, dtype=torch.long, device=dev),
            train=torch.zeros(B, dtype=torch.long, device=dev),
            force_build=torch.zeros(B, self.N, dtype=torch.bool, device=dev),
            mobilize=torch.zeros(B, self.N, dtype=torch.bool, device=dev))
        extra = dict(hq_commit=torch.zeros(B, dtype=torch.bool, device=dev),
                     gold_pred=torch.zeros(B, device=dev))
        return action, extra

    # ---- the turn --------------------------------------------------------- #
    def rollout_turn(self, ctx):
        env, cfg = self.env, self.cfg
        obs = self.observe(self.AGENT)
        action, store, extra = ctx.agent_act(obs)
        store["value"] = ctx.agent_value(obs)

        obs_op = self.observe(self.OPPONENT)
        # per-region full-mobilisation mask for the opponent this turn: only in
        # flagged games, each region independently.
        self._relax_reg = (self.relaxed[:, None]
                           & (torch.rand(self.B, self.N, device=self.device)
                              < cfg.opp_relax_prob))
        opp_action, opp_extra = ctx.opponent_act(obs_op, relax_reg=self._relax_reg)

        exec_action = action
        if cfg.opening_split:
            exec_action = self._opening_split(ctx, action, opp_action)

        # this turn's reach / prediction become next turn's features
        self.prev_reach = [obs["reach_raw"].clone(), obs_op["reach_raw"].clone()]
        self.prev_gold = [extra["gold_pred"], opp_extra["gold_pred"]]

        self.env_step(exec_action, opp_action)
        # carry the multi-turn HQ-upgrade commitment (reset for finished games in
        # reset_finished, via regen)
        env.hq_commit[:, self.AGENT] = extra["hq_commit"]
        env.hq_commit[:, self.OPPONENT] = opp_extra["hq_commit"]

        # auxiliary labels are read from the POST-step state: a genuine one-step
        # prediction task for the aux heads.
        tok_tgt, opp_gold = env.aux_label(self.AGENT)
        store["gold_aux"] = torch.log1p(tok_tgt)
        store["gold_glob"] = torch.log1p(opp_gold / 100.0)
        return store

    def _opening_split(self, ctx, action, opp_action):
        """Turn 1 only: re-infer with warrior #1 already en route and its target
        masked, so warrior #2 picks a DIFFERENT stronghold.

        The region-move action space can only send one target per source, so a
        single inference would send both of the HQ's spare warriors to the same
        stronghold and delay the second base by many turns. The STORED transition
        stays the FIRST inference -- the second move is folded into the executed
        action, exactly how the submission bot behaves.
        """
        env, exec_action = self.env, action
        turn1 = env.day == 0
        hq0 = env.hq_region[:, self.AGENT]
        a_reg0 = action["move"].gather(1, hq0[:, None]).squeeze(1)
        split0 = turn1 & (a_reg0 >= 0)
        if bool(split0.any()):
            env.opening_premove(self.AGENT, hq0, a_reg0, split0)
            forbid = torch.where(split0,
                                 P.region_to_token(env.mb.token_ids, a_reg0),
                                 torch.full_like(a_reg0, -1))
            act2, _s2, _e2 = ctx.agent_act(self.observe(self.AGENT),
                                           forbid_tgt=forbid)
            mv = action["move"].clone()
            mob = action["mobilize"].clone()
            r0 = split0.nonzero(as_tuple=True)[0]
            mv[r0, hq0[r0]] = act2["move"].gather(1, hq0[:, None]).squeeze(1)[r0]
            mob[r0, hq0[r0]] = act2["mobilize"].gather(1, hq0[:, None]).squeeze(1)[r0]
            exec_action = {**action, "move": mv, "mobilize": mob}

        # net opponents only: the scripted bots keep their own region moves
        hq1 = env.hq_region[:, self.OPPONENT]
        a_reg1 = opp_action["move"].gather(1, hq1[:, None]).squeeze(1)
        split1 = turn1 & (ctx.assign >= ctx.pool.n_scripted) & (a_reg1 >= 0)
        if bool(split1.any()):
            env.opening_premove(self.OPPONENT, hq1, a_reg1, split1)
            forbid = torch.where(split1,
                                 P.region_to_token(env.mb.token_ids, a_reg1),
                                 torch.full_like(a_reg1, -1))
            act2, _e = ctx.opponent_act(self.observe(self.OPPONENT),
                                        relax_reg=self._relax_reg,
                                        forbid_tgt=forbid)
            r1 = split1.nonzero(as_tuple=True)[0]
            opp_action["move"][r1, hq1[r1]] = \
                act2["move"].gather(1, hq1[:, None]).squeeze(1)[r1]
            opp_action["mobilize"][r1, hq1[r1]] = \
                act2["mobilize"].gather(1, hq1[:, None]).squeeze(1)[r1]
        return exec_action

    def env_step(self, agent_action, opp_action):
        # relax_right applies the modified move rules INSIDE the env for the
        # flagged games, matching the mask the opponent policy sampled under
        self.env.step({"left": agent_action, "right": opp_action},
                      relax_right=self._relax_reg)

    def reward_done(self):
        return P.reward_done(self.env)

    # ---- episode boundaries ----------------------------------------------- #
    def reset_finished(self, done, rows):
        # re-roll the modified-rules flag, then regenerate the map and reset
        self.relaxed[rows] = (torch.rand(rows.numel(), device=self.device)
                              < self.cfg.opp_relax_frac)
        self.env.regen(done)
        for s in range(2):
            self.prev_reach[s][rows] = 0
            self.prev_gold[s][rows] = 0

    def tally_mask(self):
        # games played under relaxed rules train fine but are not a fair yardstick
        return ~self.relaxed

    # ---- reporting -------------------------------------------------------- #
    def metrics(self):
        if self.factory is None:
            return {}
        misses, self._misses = self.factory.misses - self._misses, self.factory.misses
        return {"instance_queue_misses": misses}

    def close(self):
        if self.factory is not None:
            self.factory.close()


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def build(ctx: rlkit.SetupCtx) -> rlkit.Setup:
    cfg = ctx.cfg
    task = NypcTask(ctx.B, ctx.device, cfg, seed=ctx.seed)
    policy = NypcPolicy(cfg, ctx.device, task.N)
    return rlkit.Setup(
        task=task, policy=policy,
        make_policy=lambda: NypcPolicy(cfg, ctx.device, task.N),
        scripted=[RusherBot(ctx.B, ctx.device), JapperBot(ctx.B, ctx.device)],
        # T2's head emits 2 values per target (move logit + full-mobilisation
        # logit). Silently re-initializing it would throw away the entire move
        # policy, so an older checkpoint must fail loudly instead.
        guards={"actor_t2": {"head.2.weight": 2}})


def main():
    ap = rlkit.trainer.add_cli_args(argparse.ArgumentParser())
    # not the top-level config.yaml -- that one belongs to the legacy
    # ppo_selfplay.py entry point and uses its key names
    ap.set_defaults(config=os.path.join(os.path.dirname(__file__), "nypc2026.yaml"))
    args = ap.parse_args()
    if args.smoke:
        cfg = Config(B=8, steps_per_iter=400, iters=2, minibatch=256, d_model=32,
                     use_wandb=False, resume=False, phases=None,
                     ckpt_path="checkpoint_rlkit_smoke.pt", instance_workers=0)
    else:
        cfg = (rlkit.load_config(args.config, Config)
               if os.path.exists(args.config) else Config())
        rlkit.trainer.apply_cli_args(cfg, args)
    if args.gpus and args.gpus > 1:
        rlkit.launch(_dist_entry, args.gpus, args=(cfg,))
    else:
        rlkit.train(cfg, build, device=args.device)


def _dist_entry(cfg):
    rlkit.train(cfg, build)


if __name__ == "__main__":
    main()
