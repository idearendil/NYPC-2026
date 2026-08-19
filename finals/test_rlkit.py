#!/usr/bin/env python3
"""Unit + end-to-end tests for rlkit (the game-independent trainer).

These are the tests you want to have PASSING before a contest starts, because
they cover the machinery you will not have time to debug: the phase schedule, GAE,
the opponent pool's win-rate maths and its eviction index shuffling, the
checkpoint round trip, and a real 2-iteration training run of the toy game.

    python test_rlkit.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import torch
import torch.nn as nn

import rlkit
from examples import toy_duel as toy

TMP = None          # scratch dir for checkpoints, per run


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class DummyPolicy(rlkit.Policy):
    """The smallest thing that satisfies the Policy contract."""

    def __init__(self, n=3):
        self.net = nn.Linear(n, n)

    def modules(self):
        return {"net": self.net}

    def param_groups(self):
        return {"all": list(self.net.parameters())}

    def act(self, obs, **kw):
        return {}, {}, {}

    def value(self, obs):
        return torch.zeros(1)

    def evaluate(self, mb):
        raise NotImplementedError

    def evaluate_value(self, mb):
        raise NotImplementedError


class DummyBot(rlkit.ScriptedOpponent):
    def __init__(self, name):
        self.name = name

    def act(self, task, obs, rows, **kw):
        return {}, {}


# --------------------------------------------------------------------------- #
# 1. config / phases
# --------------------------------------------------------------------------- #
def test_phases():
    @dataclass
    class C(rlkit.BaseConfig):
        aux_coef: float = 0.5

    cfg = C(lr=1e-3, epochs=2, phase_iters=10, phases=[
        dict(lr=5e-4, epochs=5),
        dict(lr=1e-4, ent_coef=0.001, aux_coef=0.1),
    ])
    ps = rlkit.PhaseSchedule(cfg)
    n0, c0 = ps.at(0)
    n9, c9 = ps.at(9)
    n10, c10 = ps.at(10)
    n99, c99 = ps.at(99)
    assert (n0, n9, n10, n99) == (1, 1, 2, 2), (n0, n9, n10, n99)
    assert c0.lr == 5e-4 and c0.epochs == 5
    # unspecified keys fall back to the flat config
    assert c10.epochs == 2 and c10.lr == 1e-4
    # a subclass's own field can be phase-scheduled
    assert c0.aux_coef == 0.5 and c10.aux_coef == 0.1
    # the LAST entry is held forever
    assert c99.lr == c10.lr
    # the original config is never mutated
    assert cfg.lr == 1e-3

    # no schedule -> phase 0, flat values
    flat = rlkit.PhaseSchedule(C(phases=None))
    assert flat.at(5) == (0, flat.cfg)

    for bad, why in [({"B": 8}, "structural"), ({"nope": 1}, "unknown")]:
        try:
            rlkit.PhaseSchedule(C(phases=[bad]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"a {why} phase key must be rejected: {bad}")

    assert rlkit.PhaseSchedule(C(steps_per_iter=100, phases=[
        dict(steps_per_iter=50), dict(steps_per_iter=300)])).max_steps_per_iter() == 300


# --------------------------------------------------------------------------- #
# 2. buffer: GAE, flatten, whitening
# --------------------------------------------------------------------------- #
def test_gae():
    torch.manual_seed(0)
    B, S, gamma, lam = 4, 7, 0.9, 0.8
    buf = rlkit.RolloutBuffer("cpu", compute_device="cpu", expected_steps=S)
    rew, done, val = [], [], []
    for _ in range(S):
        r = torch.randn(B)
        d = (torch.rand(B) < 0.25).float()
        v = torch.randn(B)
        rew.append(r); done.append(d); val.append(v)
        buf.add(dict(reward=r, done=d, value=v, x=torch.randn(B, 3)))
    last = torch.randn(B)
    buf.compute_gae(last, gamma, lam)

    # independent reference
    ref = [None] * S
    gae = torch.zeros(B)
    for t in reversed(range(S)):
        nt = 1.0 - done[t]
        nv = last if t == S - 1 else val[t + 1]
        gae = (rew[t] + gamma * nv * nt - val[t]) + gamma * lam * nt * gae
        ref[t] = gae.clone()
    for t in range(S):
        assert torch.allclose(buf.buf[t]["adv"], ref[t], atol=1e-6), f"step {t}"
        assert torch.allclose(buf.buf[t]["ret"], ref[t] + val[t], atol=1e-6)
    # a terminal step must not bootstrap through the boundary
    for t in range(S):
        term = done[t] > 0
        if term.any():
            assert torch.allclose(buf.buf[t]["adv"][term],
                                  (rew[t] - val[t])[term], atol=1e-6), \
                "advantage leaked across an episode boundary"

    flat = buf.flatten()
    assert flat["x"].shape == (S * B, 3)
    assert "reward" not in flat and "done" not in flat, "dropped keys leaked"
    assert len(buf) == 0, "flatten() must release the per-step tensors"
    rlkit.whiten_(flat, "adv")
    a = flat["adv"]
    assert abs(float(a.mean())) < 1e-5 and abs(float(a.std(unbiased=False)) - 1) < 1e-4


# --------------------------------------------------------------------------- #
# 3. pool: EMA maths, growth, eviction, migration
# --------------------------------------------------------------------------- #
def _pool(B=8, **kw):
    opts = dict(B=B, device="cpu", max_size=5, add_threshold=0.6, ema_alpha=0.1,
                seed=1)
    opts.update(kw)
    return rlkit.OpponentPool([DummyBot("rusher"), DummyBot("turtle")],
                              DummyPolicy(), **opts)


def _feed(pool, assign, rewards):
    tally = pool.new_tally()
    for i in range(len(assign)):
        tally.update(rewards[i:i + 1], torch.ones(1, dtype=torch.bool),
                     assign[i:i + 1])
    return pool.apply_tally(tally)


def test_pool_ema():
    """The closed-form EMA is (a) exactly n iterative steps when the n results
    agree, and (b) ORDER-INDEPENDENT. (b) is the property that lets several ranks
    all-reduce their tallies and reach bit-identical win rates -- without it they
    would snapshot and evict differently and corrupt the shared checkpoint."""
    # (a) exactness against the iterative update
    pool = _pool()
    before = pool.wr.clone()
    n, a = 7, 0.1
    _feed(pool, torch.zeros(n, dtype=torch.long), torch.full((n,), 10.0))
    ref = float(before[0])
    for _ in range(n):
        ref = (1 - a) * ref + a * 1.0
    assert abs(float(pool.wr[0]) - ref) < 1e-6, (float(pool.wr[0]), ref)
    # an opponent that played no games must not move at all
    assert pool.wr[1:].tolist() == before[1:].tolist()

    # (b) order independence over a mixed multiset
    torch.manual_seed(3)
    assign = torch.randint(3, (40,))
    rewards = torch.where(torch.rand(40) < 0.6, 10.0, -10.0)
    p1, p2 = _pool(), _pool()
    eps1, sum1 = _feed(p1, assign, rewards)
    perm = torch.randperm(40)
    eps2, sum2 = _feed(p2, assign[perm], rewards[perm])
    assert (eps1, eps2) == (40, 40) and abs(sum1 - float(rewards.sum())) < 1e-4
    assert abs(sum1 - sum2) < 1e-4
    assert torch.equal(p1.wr, p2.wr), (p1.wr, p2.wr)
    # and it still tracks reality: a 60%-win multiset pulls 0.5 upward
    assert float(p1.wr.mean()) > 0.5


def test_pool_tally_mask():
    pool = _pool()
    tally = pool.new_tally()
    r = torch.tensor([10.0, 10.0])
    done = torch.ones(2, dtype=torch.bool)
    assign = torch.tensor([0, 1])
    tally.update(r, done, assign, mask=torch.tensor([True, False]))
    eps, _ = pool.apply_tally(tally)
    assert eps == 2, "masked games still count as episodes"
    assert float(pool.wr[0]) > 0.5 and float(pool.wr[1]) == 0.5, \
        "a masked game must not move the win rate"


def test_pool_growth_and_eviction():
    pool = _pool(B=16, snapshot_every=2, add_threshold=0.0)   # always grow
    p = DummyPolicy()
    assert pool.size == 3 and pool.cap == 5
    assert pool.maybe_snapshot(2, p) and pool.perm[-1], "periodic = permanent"
    assert pool.cap == 6, "a permanent snapshot must grow the cap"
    for _ in range(6):
        pool.maybe_grow(p)
    assert pool.size <= pool.cap, (pool.size, pool.cap)
    # scripted bots and permanent snapshots survive; ids stay unique
    assert pool.ids[:2] == ["rusher", "turtle"]
    assert sum(pool.perm) >= 1, "a permanent snapshot was evicted"
    assert len(set(pool.ids)) == len(pool.ids), "duplicate opponent ids"
    assert len(pool.nets) == len(pool.perm) == pool.size - 2
    assert pool.wr.numel() == pool.size
    assert int(pool.assign.max()) < pool.size, "assignment points past the pool"

    # eviction remaps in the right order: rows on the evicted index fall back to 0
    pool2 = _pool(B=6, add_threshold=1.1)
    for _ in range(2):
        pool2._append(DummyPolicy(), permanent=False)
    pool2.assign = torch.tensor([0, 1, 2, 3, 4, 4])
    ids_before = list(pool2.ids)
    pool2._evict(0)                     # unified index 2
    assert pool2.ids == ids_before[:2] + ids_before[3:]
    assert pool2.assign.tolist() == [0, 1, 0, 2, 3, 3]


def test_pool_checkpoint_migration():
    """Adding a scripted opponent must not reinterpret the stored indices."""
    old = rlkit.OpponentPool([DummyBot("rusher")], DummyPolicy(), B=6, device="cpu",
                             max_size=5, seed=2)
    old._append(DummyPolicy(), permanent=True)
    old.wr = torch.tensor([0.3, 0.7, 0.9])          # rusher, net0, net1
    old.assign = torch.tensor([0, 1, 2, 2, 1, 0])
    sd = old.state_dict()

    new = rlkit.OpponentPool([DummyBot("rusher"), DummyBot("turtle")], DummyPolicy(),
                             B=6, device="cpu", max_size=5, seed=2)
    new.load_state_dict(sd, DummyPolicy, B=6, rank=0, verbose=False)
    assert new.n_scripted == 2 and new.size == 4
    assert new.ids == ["rusher", "turtle", 0, 1], new.ids
    # rusher keeps its win rate, the new bot starts neutral, nets shift by one
    assert torch.allclose(new.wr, torch.tensor([0.3, 0.5, 0.7, 0.9]))
    assert new.assign.tolist() == [0, 2, 3, 3, 2, 0], new.assign.tolist()
    assert new.perm == [False, True], new.perm

    # same set of bots -> everything restored verbatim
    same = rlkit.OpponentPool([DummyBot("rusher")], DummyPolicy(), B=6,
                              device="cpu", max_size=5, seed=9)
    same.load_state_dict(sd, DummyPolicy, B=6, rank=0, verbose=False)
    assert torch.allclose(same.wr, old.wr)
    assert same.assign.tolist() == old.assign.tolist()
    assert same.next_id == old.next_id

    # a non-zero rank must NOT reuse rank 0's assignment (it would simulate the
    # same games), and must diverge its matchmaking stream
    r1 = rlkit.OpponentPool([DummyBot("rusher")], DummyPolicy(), B=6, device="cpu",
                            max_size=5, seed=9)
    r1.load_state_dict(sd, DummyPolicy, B=6, rank=1, verbose=False)
    r1.reseed(1234, B=6)
    assert r1.assign.tolist() != old.assign.tolist()


def test_pool_sampling_is_winrate_inverse():
    pool = _pool(B=4096, sample_floor=0.05)
    pool.wr = torch.tensor([0.95, 0.5, 0.05])
    draws = pool.sample(20000)
    frac = [float((draws == k).float().mean()) for k in range(3)]
    assert frac[2] > frac[1] > frac[0], f"not win-rate-inverse: {frac}"
    assert frac[0] > 0.005, "the floor must keep a beaten opponent in the mix"


# --------------------------------------------------------------------------- #
# 4. checkpoint round trip
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip():
    cfg = toy.Config(d_model=16, lr=1e-3)
    dev = "cpu"
    pol = toy.DuelPolicy(cfg, dev)
    ppo = rlkit.PPO(pol, cfg, None, dev)
    pool = rlkit.OpponentPool([toy.Rusher(), toy.Turtle()], pol, B=4, device=dev,
                              max_size=5, seed=1)
    pool._append(pol, permanent=True)
    path = os.path.join(TMP, "roundtrip.pt")
    ck = rlkit.Checkpointer(path, None, dev)
    ck.save(7, policy=pol, ppo=ppo, pool=pool, cfg=None)

    pol2 = toy.DuelPolicy(cfg, dev)
    before = rlkit.testing.flat_params(pol2)
    ppo2 = rlkit.PPO(pol2, cfg, None, dev)
    pool2 = rlkit.OpponentPool([toy.Rusher(), toy.Turtle()], pol2, B=4, device=dev,
                               max_size=5, seed=1)
    it = ck.load(policy=pol2, ppo=ppo2, pool=pool2,
                 make_policy=lambda: toy.DuelPolicy(cfg, dev), B=4, verbose=False)
    assert it == 7
    n = rlkit.testing.assert_params_identical(rlkit.testing.flat_params(pol),
                                              rlkit.testing.flat_params(pol2),
                                              "checkpoint")
    assert n > 0 and any(not torch.equal(a, b) for a, b in
                         zip(before, rlkit.testing.flat_params(pol2))), \
        "the load did not actually change anything -- test is vacuous"
    assert pool2.size == pool.size and pool2.perm == pool.perm
    rlkit.testing.assert_params_identical(rlkit.testing.flat_params(pool.nets[-1]),
                                          rlkit.testing.flat_params(pool2.nets[-1]),
                                          "pool snapshot")

    # a guard must turn a meaning-changing shape mismatch into a clear error
    guarded = rlkit.Checkpointer(path, None, dev,
                                 guards={"actor": {"0.weight": 999}})
    try:
        guarded.load(policy=pol2, ppo=ppo2, pool=pool2,
                     make_policy=lambda: toy.DuelPolicy(cfg, dev), B=4,
                     verbose=False)
    except RuntimeError as e:
        assert "incompatible" in str(e)
    else:
        raise AssertionError("guard did not fire")


# --------------------------------------------------------------------------- #
# 5. instance factory (worker processes)
# --------------------------------------------------------------------------- #
def test_instance_factory():
    got = rlkit.make_instances(20, toy.gen_instance, seed=1, workers=2)
    assert len(got) == 20 and all(len(g) == 3 for g in got)
    assert len({g for g in got}) > 1, "parallel generation returned identical maps"

    f = rlkit.InstanceFactory(toy.gen_instance, workers=2, seed=3, depth=32)
    try:
        vals = [f.get() for _ in range(200)]
        assert all(14 <= v[0] <= 26 for v in vals)
        assert f.hits > 0, "the worker queue was never used"
    finally:
        f.close()
    assert not f.procs

    inline = rlkit.InstanceFactory(toy.gen_instance, workers=0, seed=3)
    assert len(inline.get()) == 3 and inline.hits == 0
    inline.close()


# --------------------------------------------------------------------------- #
# 6. end to end: train, then resume
# --------------------------------------------------------------------------- #
def _toy_cfg(**kw):
    cfg = toy.Config(B=32, steps_per_iter=1024, iters=2, minibatch=256, d_model=16,
                     use_wandb=False, phases=None, instance_workers=0,
                     pool_snapshot_every=2, pool_add_threshold=0.0,
                     ckpt_path=os.path.join(TMP, "e2e.pt"), log_every=99)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_end_to_end_and_resume():
    torch.manual_seed(0)
    cfg = _toy_cfg(resume=False)
    policy, pool, task = rlkit.train(cfg, toy.build, device="cpu", verbose=False)
    for name, m in policy.modules().items():
        for k, p in m.state_dict().items():
            assert torch.isfinite(p).all(), f"non-finite parameter {name}.{k}"
    ck = torch.load(cfg.ckpt_path, weights_only=False)
    assert ck["iter"] == 1, ck["iter"]        # saved at the START of each iteration
    size_before = ck["pool"]["wr"].numel()

    # resume: must pick up at the stored iteration and keep the pool
    cfg2 = _toy_cfg(resume=True, iters=3)
    policy2, pool2, _ = rlkit.train(cfg2, toy.build, device="cpu", verbose=False)
    ck2 = torch.load(cfg2.ckpt_path, weights_only=False)
    assert ck2["iter"] == 2, ck2["iter"]
    assert ck2["pool"]["wr"].numel() >= size_before
    assert pool2.size >= size_before
    # ...and a resumed run must NOT be a fresh one
    fresh = toy.DuelPolicy(cfg2, "cpu")
    same = all(torch.equal(a, b) for a, b in
               zip(rlkit.testing.flat_params(fresh),
                   rlkit.testing.flat_params(policy2)))
    assert not same


def test_add_scripted_opponent_midrun():
    """Adding a hand-written bot to the pool DURING a run must just work.

    This is the finals workflow: training starts with one scripted bot, a
    heuristic gets written an hour later, and it should join the pool on the next
    restart without resetting anything or silently reinterpreting the stored
    opponent indices."""
    path = os.path.join(TMP, "midrun.pt")

    def build_one(ctx):
        su = toy.build(ctx)
        su.scripted = [toy.Rusher()]
        return su

    def build_two(ctx):
        su = toy.build(ctx)
        su.scripted = [toy.Rusher(), toy.Turtle()]
        return su

    torch.manual_seed(0)
    cfg = _toy_cfg(resume=False, iters=2, ckpt_path=path)
    _p, pool1, _t = rlkit.train(cfg, build_one, device="cpu", verbose=False)
    assert pool1.ids[0] == "rusher" and "turtle" not in pool1.ids
    n_nets = len(pool1.nets)

    cfg2 = _toy_cfg(resume=True, iters=3, ckpt_path=path)
    policy2, pool2, _t = rlkit.train(cfg2, build_two, device="cpu", verbose=False)
    assert pool2.ids[:2] == ["rusher", "turtle"], pool2.ids
    assert len(pool2.nets) >= n_nets, "the snapshots were lost"
    assert pool2.wr.numel() == pool2.size
    assert int(pool2.assign.max()) < pool2.size
    for m in policy2.modules().values():
        for k, p in m.state_dict().items():
            assert torch.isfinite(p).all(), k


def test_parity_harness():
    """The reference-vs-batched comparator must catch what it is there to catch."""
    cmp = rlkit.parity.compare
    assert cmp({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]}) is None
    assert "b[1]" in cmp({"b": [1, 2]}, {"b": [1, 3]})
    assert "length" in cmp({"b": [1, 2]}, {"b": [1]})
    assert "missing" in cmp({"a": 1}, {"a": 1, "c": 2})
    assert cmp({"x": torch.tensor([1, 2])}, {"x": [1, 2]}) is None   # tensor vs list
    assert cmp({"x": 1.0}, {"x": 1.05}, atol=0.1) is None
    assert cmp({"x": 1.0}, {"x": 1.5}, atol=0.1) is not None

    # a toy "reference" (python ints) vs a "batched" version (a tensor), driven by
    # the same actions -- and then the same thing with a deliberate bug
    class Ref:
        def __init__(self, hp):
            self.hp = hp

    class Batch:
        def __init__(self, hp):
            self.hp = torch.tensor(hp)

    def sample(e, rng):
        acts = [rng.randint(0, 2) for _ in range(4)]
        return torch.tensor(acts), acts

    def check(bug):
        refs = [Ref(10 + i) for i in range(4)]
        env = Batch([10 + i for i in range(4)])
        return rlkit.parity.run(
            refs, env, turns=5, verbose=False, sample_actions=sample,
            step_env=lambda e, a: setattr(e, "hp", e.hp - a),
            step_ref=lambda r, a: setattr(r, "hp", r.hp - a),
            snap_ref=lambda r: {"hp": r.hp},
            # `bug` mimics an off-by-one in the batched rewrite
            snap_env=lambda e, b: {"hp": int(e.hp[b]) - (1 if bug else 0)})

    assert check(bug=False) is True, "the harness reported a difference that isn't there"
    assert check(bug=True) is False, "the harness missed a real difference"


def test_extra_loss_and_metrics():
    """A policy may add its own loss terms and its own logged scalars."""
    seen = {}

    class Extra(toy.DuelPolicy):
        def evaluate(self, mb):
            out = super().evaluate(mb)
            pen = (self.actor[0].weight ** 2).mean()      # a toy regulariser
            seen["actor"] = True
            return rlkit.ActorOut(out.logp, out.entropy, extra_loss=0.01 * pen,
                                  metrics={"my_penalty": pen.detach()})

        def evaluate_value(self, mb):
            out = super().evaluate_value(mb)
            return rlkit.CriticOut(out.value, extra_loss=0.0,
                                   metrics={"my_value_mean": out.value.mean().detach()})

    def build(ctx):
        su = toy.build(ctx)
        su.policy = Extra(ctx.cfg, ctx.device)
        su.make_policy = lambda: Extra(ctx.cfg, ctx.device)
        return su

    cfg = _toy_cfg(resume=False, ckpt_path=os.path.join(TMP, "extra.pt"), iters=1)
    logged = {}
    real_log = rlkit.Logger.log

    def spy(self, it, metrics, line=None):
        logged.update(metrics)

    rlkit.Logger.log = spy
    try:
        rlkit.train(cfg, build, device="cpu", verbose=False)
    finally:
        rlkit.Logger.log = real_log
    assert seen.get("actor"), "the custom evaluate() was never called"
    assert "my_penalty" in logged and "my_value_mean" in logged, sorted(logged)
    assert logged["my_penalty"] > 0


def test_shared_param_group():
    """A single-group policy must train through the shared-trunk code path."""
    class Shared(toy.DuelPolicy):
        def param_groups(self):
            return {"all": list(self.actor.parameters()) + list(self.critic.parameters())}

    def build(ctx):
        su = toy.build(ctx)
        su.policy = Shared(ctx.cfg, ctx.device)
        su.make_policy = lambda: Shared(ctx.cfg, ctx.device)
        return su

    cfg = _toy_cfg(resume=False, ckpt_path=os.path.join(TMP, "shared.pt"))
    policy, _, _ = rlkit.train(cfg, build, device="cpu", verbose=False)
    assert len(rlkit.PPO(policy, cfg, None, "cpu").opts) == 1
    for m in policy.modules().values():
        for k, p in m.state_dict().items():
            assert torch.isfinite(p).all(), k


TESTS = [test_phases, test_gae, test_pool_ema, test_pool_tally_mask,
         test_pool_growth_and_eviction, test_pool_checkpoint_migration,
         test_pool_sampling_is_winrate_inverse, test_checkpoint_roundtrip,
         test_instance_factory, test_end_to_end_and_resume,
         test_add_scripted_opponent_midrun, test_parity_harness,
         test_extra_loss_and_metrics, test_shared_param_group]


def main():
    global TMP
    TMP = tempfile.mkdtemp(prefix="rlkit_test_")
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
