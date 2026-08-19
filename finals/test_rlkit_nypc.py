#!/usr/bin/env python3
"""End-to-end test of the REAL game driven by rlkit (examples/nypc2026.py).

Separate from test_rlkit.py on purpose: that suite must keep passing with no game
at all in the repository, since the whole point of rlkit is that it survives the
game being replaced. This one is the integration proof for the current game.

    python test_rlkit_nypc.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

import rlkit

try:
    from examples import nypc2026 as G
except ImportError as e:
    # The adapter imports the PREVIOUS game (fast_env / ppo_selfplay), which lives
    # in the repository root above this kit. Without it there is nothing to
    # integrate, and that is not a failure of the kit.
    print(f"SKIP: the previous game is not importable from here ({e})")
    sys.exit(0)

TMP = None


def _cfg(**kw):
    cfg = G.Config(B=8, steps_per_iter=400, iters=2, minibatch=256, d_model=32,
                   use_wandb=False, resume=False, phases=None, instance_workers=0,
                   log_every=99, ckpt_path=os.path.join(TMP, "nypc.pt"))
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_train_two_iters():
    torch.manual_seed(0)
    policy, pool, task = rlkit.train(_cfg(), G.build, verbose=False)
    for name, m in policy.modules().items():
        for k, p in m.state_dict().items():
            assert torch.isfinite(p).all(), f"non-finite parameter {name}.{k}"
    assert pool.size >= 3 and pool.ids[:2] == ["rusher", "japper"]
    assert task.env.day.max() > 0, "no turns were played"
    ck = torch.load(_cfg().ckpt_path, weights_only=False)
    assert set(ck["policy"]) == {"actor_t1", "actor_t2", "critic"}


def test_store_has_what_ppo_needs():
    """The stored transition must carry everything evaluate()/evaluate_value() read
    -- a missing key here is a KeyError deep in the first PPO minibatch."""
    torch.manual_seed(0)
    cfg = _cfg()
    ctx = rlkit.SetupCtx(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu",
                         B=cfg.B, minibatch=cfg.minibatch, rank=0, world=1, seed=0,
                         dist=None)
    su = G.build(ctx)
    pool = rlkit.OpponentPool(su.scripted, su.policy, B=cfg.B, device=ctx.device,
                              max_size=5, seed=0)
    buf = rlkit.RolloutBuffer("cpu", compute_device=ctx.device, expected_steps=3)
    rlkit.run_rollout(su.task, su.policy, pool, buf, 3, device=ctx.device)
    buf.compute_gae(rlkit.rollout.bootstrap_value(su.task, su.policy), 0.99, 0.95)
    flat = buf.flatten()
    need = {"t1", "glob", "t1_crit", "glob_crit", "tmask", "extra4", "tok_dist",
            "normx", "normy", "build_mask", "build_outcome", "train_mask",
            "train_cat", "valid_src", "tgt", "surplus_pb", "tgt_allowed",
            "mob_mask", "mob_bit", "old_logp", "gold_aux", "gold_glob",
            "adv", "ret", "value"}
    missing = need - set(flat)
    assert not missing, f"missing from the rollout store: {sorted(missing)}"
    n = flat["old_logp"].shape[0]
    assert n == 3 * cfg.B, n
    mb = {k: v.to(ctx.device) for k, v in flat.items()}
    a = su.policy.evaluate(mb)
    c = su.policy.evaluate_value(mb)
    assert a.logp.shape == (n,) and c.value.shape == (n,)
    assert torch.isfinite(a.logp).all() and torch.isfinite(c.value).all()
    su.task.close()


def test_relaxed_games_excluded_from_winrate():
    torch.manual_seed(0)
    cfg = _cfg(opp_relax_frac=1.0)          # every game plays under relaxed rules
    ctx = rlkit.SetupCtx(cfg=cfg, device="cpu", B=cfg.B, minibatch=cfg.minibatch,
                         rank=0, world=1, seed=0, dist=None)
    task = G.NypcTask(cfg.B, "cpu", cfg, seed=0)
    assert bool(task.tally_mask().sum()) == 0, \
        "relaxed games must be excluded from the win-rate tally"
    task.close()


def test_guard_rejects_a_pre_mobilisation_checkpoint():
    """A T2 head that emitted one value per target meant something different. The
    guard must refuse it loudly instead of re-initializing the whole move policy."""
    cfg = _cfg()
    dev = "cpu"
    pol = G.NypcPolicy(cfg, dev, 109)
    ppo = rlkit.PPO(pol, cfg, None, dev)
    pool = rlkit.OpponentPool([], pol, B=2, device=dev, max_size=3, seed=0)
    path = os.path.join(TMP, "old.pt")
    ck = rlkit.Checkpointer(path, None, dev, guards={"actor_t2": {"head.2.weight": 2}})
    ck.save(3, policy=pol, ppo=ppo, pool=pool)
    # rewrite the file as if it came from the one-output era
    obj = torch.load(path, weights_only=False)
    obj["policy"]["actor_t2"]["head.2.weight"] = \
        obj["policy"]["actor_t2"]["head.2.weight"][:1]
    torch.save(obj, path)
    try:
        ck.load(policy=pol, ppo=ppo, pool=pool,
                make_policy=lambda: G.NypcPolicy(cfg, dev, 109), B=2, verbose=False)
    except RuntimeError as e:
        assert "incompatible" in str(e), e
    else:
        raise AssertionError("the guard did not fire")


TESTS = [test_train_two_iters, test_store_has_what_ppo_needs,
         test_relaxed_games_excluded_from_winrate,
         test_guard_rejects_a_pre_mobilisation_checkpoint]


def main():
    global TMP
    TMP = tempfile.mkdtemp(prefix="rlkit_nypc_")
    failed = []
    try:
        for t in TESTS:
            try:
                t()
                print(f"[ok]   {t.__name__}")
            except Exception as e:            # noqa: BLE001 - reported below
                import traceback
                print(f"[FAIL] {t.__name__}: {e}")
                traceback.print_exc()
                failed.append(t.__name__)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {len(TESTS) - len(failed)}/{len(TESTS)} passed"
          + (f"  FAILED: {failed}" if failed else "  ALL GOOD"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
