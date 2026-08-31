#!/usr/bin/env python3
"""Verify the numpy submission bot reproduces the torch training pipeline:
encoder features (extract) and the actor net forward, on random states (including
random fog-of-war belief), for both sides. Run after `export_weights.py` (reads
data.bin + checkpoint.pt)."""
import numpy as np
import torch

# --- repo layout: make src/ importable when run directly --------------------
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import fast_env as fe
from fast_env import OWN_LEFT, KIND_HQ, KIND_BASE
from ppo_selfplay import ActorT1, ActorT2, extract, T2_EXTRA
import vanilla_bot as sb


def make_env(seed):
    import random
    rr = random.Random(seed)
    NP = rr.randint(90, 124)
    N = 2 * NP + 1
    import math
    root_n = math.sqrt(N)
    klo = math.ceil(root_n - 1.0); khi = math.floor(root_n + 4.0)
    klo = klo + 1 if klo % 2 == 0 else klo
    khi = khi - 1 if khi % 2 == 0 else khi
    KP = rr.randint((klo - 1) // 2, (khi - 1) // 2)
    m = fe.tt.read_map(fe.tt.generate_map(fe.tt.XoShiro256(seed), NP, KP))
    env = fe.FastEnv([m], device="cpu", max_warriors_per_side=60)
    return env, m


def randomize(env, m, seed):
    r = np.random.default_rng(seed)
    N = env.N
    env.day[0] = int(r.integers(0, 400))
    env.gold[0, 0] = int(r.integers(0, 3000)); env.gold[0, 1] = int(r.integers(0, 3000))
    env.prev_income[0, 0] = int(r.integers(0, 300)); env.prev_income[0, 1] = int(r.integers(0, 300))
    env.b_owner.zero_(); env.b_kind.zero_(); env.b_level.zero_(); env.b_hp.zero_()
    # HQs
    env.b_owner[0, 0] = 1; env.b_kind[0, 0] = KIND_HQ
    env.b_level[0, 0] = int(r.integers(1, 6)); env.b_hp[0, 0] = int(r.integers(1, 30))
    env.b_owner[0, N - 1] = 2; env.b_kind[0, N - 1] = KIND_HQ
    env.b_level[0, N - 1] = int(r.integers(1, 6)); env.b_hp[0, N - 1] = int(r.integers(1, 30))
    # bases on random strongholds
    strong = np.nonzero(env.mb.is_stronghold[0].cpu().numpy())[0]
    for rr in strong:
        if r.random() < 0.5:
            env.b_owner[0, rr] = int(r.integers(1, 3))
            env.b_kind[0, rr] = 2
            env.b_level[0, rr] = int(r.integers(1, 4))
            env.b_hp[0, rr] = int(r.integers(1, 18))
    # warriors
    env.w_hp.zero_(); env.w_region.zero_(); env.w_move.zero_(); env.w_tgt.zero_()
    toks = env.mb.token_ids[0].cpu().numpy()
    toks = toks[toks < N]
    for side in (0, 1):
        base = env.left_base if side == 0 else env.right_base
        nw = int(r.integers(0, 40))
        for j in range(nw):
            slot = base + j
            env.w_hp[0, slot] = int(r.integers(1, 9))
            env.w_region[0, slot] = int(r.integers(0, N))
            if r.random() < 0.5:
                env.w_move[0, slot] = True
                env.w_tgt[0, slot] = int(toks[r.integers(0, len(toks))])
    # ---- random fog-of-war belief (both sides' view of their opponent) ----
    for t in (env.op_seen_kind, env.op_seen_level, env.op_seen_bhp,
             env.op_seen_wcnt, env.op_seen_whp, env.op_seen_age):
        t.zero_()
    for side in (0, 1):
        for rr in strong:
            if r.random() < 0.4:
                kind = int(r.integers(1, 3))
                env.op_seen_kind[0, side, rr] = kind
                env.op_seen_level[0, side, rr] = int(r.integers(1, 4 if kind == KIND_BASE else 6))
                env.op_seen_bhp[0, side, rr] = int(r.integers(0, 30))
            if r.random() < 0.4:
                env.op_seen_wcnt[0, side, rr] = int(r.integers(0, 10))
                env.op_seen_whp[0, side, rr] = int(r.integers(0, 60))
            env.op_seen_age[0, side, rr] = int(r.integers(0, 400))
        opp_hq = (N - 1) if side == 0 else 0
        if r.random() < 0.7:
            env.op_seen_kind[0, side, opp_hq] = KIND_HQ
            env.op_seen_level[0, side, opp_hq] = int(r.integers(1, 6))
            env.op_seen_bhp[0, side, opp_hq] = int(r.integers(1, 30))
        env.op_seen_age[0, side, opp_hq] = int(r.integers(0, 400))

    # ---- enforce the real invariant: a region with my own warrior/building is
    # ALWAYS in my own vision (distance 0), so belief == truth there. The random
    # draws above don't know this; fix it up region-by-region per side so the
    # comparison below is testing a state fog-of-war could actually produce.
    for side in (0, 1):
        me_own = side + 1              # OWN_LEFT=1, OWN_RIGHT=2
        opp_own = 2 if side == 0 else 1
        mine = set(int(x) for x in np.nonzero(
            (env.b_owner[0].cpu().numpy() == me_own))[0])
        for slot in range(env.W):
            is_side = slot < env.Wside if side == 0 else slot >= env.Wside
            if is_side and int(env.w_hp[0, slot]) > 0:
                mine.add(int(env.w_region[0, slot]))
        for rr in mine:
            is_opp_b = int(env.b_owner[0, rr]) == opp_own
            env.op_seen_kind[0, side, rr] = int(env.b_kind[0, rr]) if is_opp_b else 0
            env.op_seen_level[0, side, rr] = int(env.b_level[0, rr]) if is_opp_b else 0
            env.op_seen_bhp[0, side, rr] = int(env.b_hp[0, rr]) if is_opp_b else 0
            opp_base = env.right_base if side == 0 else env.left_base
            opp_lo, opp_hi = (opp_base, opp_base + env.Wside)
            cnt = whp = 0
            for slot in range(opp_lo, opp_hi):
                if int(env.w_region[0, slot]) == rr and int(env.w_hp[0, slot]) > 0:
                    cnt += 1; whp += int(env.w_hp[0, slot])
            env.op_seen_wcnt[0, side, rr] = cnt
            env.op_seen_whp[0, side, rr] = whp
            env.op_seen_age[0, side, rr] = 0


def env_to_bot(env, m, side):
    bot = sb.Bot.__new__(sb.Bot)          # bypass __init__ (no weights needed here)
    bot.stochastic = False
    bot.my_side = 'A' if side == 0 else 'B'
    bot.me_code = OWN_LEFT if side == 0 else 2
    bot.opp = 'B' if side == 0 else 'A'
    N = env.N
    bot.N = N; bot.K = m.K
    bot.x = list(m.x); bot.y = list(m.y); bot.adj = m.adj
    bot.stronghold = env.mb.is_stronghold[0].cpu().numpy().copy()
    bot.is_hq = np.zeros(N, dtype=bool); bot.is_hq[0] = True; bot.is_hq[N - 1] = True
    bot.tok_ids = env.mb.token_ids[0].cpu().numpy()[: m.K + 2].copy()
    bot.tok2idx = {int(rr): i for i, rr in enumerate(bot.tok_ids)}
    bot.tt = sb.compute_travel(N, bot.x, bot.y, bot.adj, bot.tok_ids)
    bot.prev_reach = None
    bot.hq_commit = False

    me_own = OWN_LEFT if side == 0 else 2
    bot.warriors = {}
    Ws = env.Wside
    for slot in range(env.W):
        hp = int(env.w_hp[0, slot])
        if hp <= 0:
            continue
        s = 'A' if slot < Ws else 'B'
        if s != bot.my_side:
            continue
        num = (slot if s == 'A' else slot - Ws) + 1
        w = sb.Warrior(s, num, int(env.w_region[0, slot]), hp)
        w.moving = bool(env.w_move[0, slot]); w.target = int(env.w_tgt[0, slot])
        bot.warriors[(s, num)] = w
    bot.buildings = {}
    for rr in range(N):
        ow = int(env.b_owner[0, rr])
        if ow != me_own:
            continue
        kind = 'HQ' if int(env.b_kind[0, rr]) == KIND_HQ else 'BASE'
        bot.buildings[rr] = sb.Building(rr, bot.my_side, kind, int(env.b_level[0, rr]), int(env.b_hp[0, rr]))
    bot.gold = int(env.gold[0, side])
    bot.income = int(env.prev_income[0, side])
    # belief copied DIRECTLY from the env's own op_seen_* buffers -- this is
    # exactly what extract() reads, so it isolates encode()'s math from vision
    # simulation (which is tested independently, see test_encoder.py).
    bot.op_kind = env.op_seen_kind[0, side, :].cpu().numpy().copy()
    bot.op_level = env.op_seen_level[0, side, :].cpu().numpy().copy()
    bot.op_bhp = env.op_seen_bhp[0, side, :].cpu().numpy().copy()
    bot.op_wcnt = env.op_seen_wcnt[0, side, :].cpu().numpy().copy()
    bot.op_whp = env.op_seen_whp[0, side, :].cpu().numpy().copy()
    bot.op_age = env.op_seen_age[0, side, :].cpu().numpy().copy()
    return bot


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="data.bin")
    ap.add_argument("--ckpt", default="checkpoint.pt")
    args = ap.parse_args()
    npz = np.load(args.weights)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    d = int(ck["cfg"]["d_model"])
    t1net = ActorT1(d=d); t1net.load_state_dict(ck["actor_t1"]); t1net.eval()
    t2net = ActorT2(d + T2_EXTRA, d=d); t2net.load_state_dict(ck["actor_t2"]); t2net.eval()
    net = sb.Net(npz)

    fields = ['t1', 'glob', 'build_cost', 'wc_cur', 'wc_after', 'stat_cnt',
              'extra4', 'tok_dist', 'normx', 'normy']
    bool_fields = ['build_cand', 'is_hq_me', 'can_up_hq', 'owner_me', 'build_new']

    worst = {k: 0.0 for k in fields}
    worst_head = worst_h1 = worst_t2 = 0.0
    bool_ok = True
    tt_ok = True

    for seed in range(1, 31):
        env, m = make_env(seed)
        randomize(env, m, seed)
        for side in (0, 1):
            o_t = extract(env, side)
            bot = env_to_bot(env, m, side)
            # travel cache must match the env's
            if not np.array_equal(bot.tt, env.mb.travel_turns[0].cpu().numpy()[:, :m.K + 2]):
                tt_ok = False
            o_n = bot.encode(int(env.day[0]) + 1)

            # build_new/build_cost/wc_after are only ever CONSUMED where build_cand
            # is true (my_cnt > 0, hence always-visible, hence belief == truth); at
            # other tokens they may legitimately differ (fog) without it mattering --
            # mask the comparison to where it's actually observable/relevant.
            relevant = o_t['build_cand'][0].cpu().numpy().astype(bool)
            for k in fields:
                a = o_t[k][0].detach().cpu().numpy() if o_t[k].dim() >= 1 else o_t[k].item()
                b = np.asarray(o_n[k], dtype=np.float64)
                if k in ('build_cost', 'wc_after'):
                    a = np.where(relevant, a, 0); b = np.where(relevant, b, 0)
                worst[k] = max(worst[k], float(np.max(np.abs(a - b))))
            for k in bool_fields:
                a = o_t[k][0].cpu().numpy().astype(bool)
                b = np.asarray(o_n[k]).astype(bool)
                cmp_a, cmp_b = (a, b) if k != 'build_new' else (a & relevant, b & relevant)
                if not np.array_equal(cmp_a, cmp_b):
                    bool_ok = False
                    print(f"  bool mismatch {k} seed={seed} side={side}")

            # network forward
            T = o_n['t1'].shape[0]
            tmask = torch.ones(1, T, dtype=torch.bool)
            h1_t, head_t = t1net(torch.tensor(o_n['t1'], dtype=torch.float32)[None],
                                 torch.tensor(o_n['glob'], dtype=torch.float32)[None], tmask)
            h1_n, head_n = net.t1(o_n['t1'].astype(np.float32), o_n['glob'].astype(np.float32))
            worst_h1 = max(worst_h1, float(np.max(np.abs(h1_t[0].detach().numpy() - h1_n))))
            worst_head = max(worst_head, float(np.max(np.abs(head_t[0].detach().numpy() - head_n))))

            # t2 on identical inputs (single output channel: the move-target logit --
            # no full-mobilisation head any more). Sources = any 거점 with SURPLUS
            # (stationary beyond work_cap), matching sample_policy's valid_src.
            surplus = np.maximum(o_n['stat_cnt'] - o_n['wc_cur'], 0)
            src = np.nonzero(surplus > 0)[0]
            if src.size:
                X = np.empty((src.size, T, h1_n.shape[1] + T2_EXTRA), dtype=np.float32)
                for j, si in enumerate(src):
                    sf = np.full((T, 1), sb.plog1p(surplus[si]))
                    tv = sb.plog1p(o_n['tok_dist'][si])[:, None]
                    dx = (o_n['normx'] - o_n['normx'][si])[:, None]
                    dy = (o_n['normy'] - o_n['normy'][si])[:, None]
                    X[j] = np.concatenate([h1_n, o_n['extra4'], sf, tv, dx, dy], axis=1)
                lg_n = net.t2(X)[:, :, 0]      # [S,T,1] -> [S,T] (single output channel)
                lg_t = t2net(torch.tensor(X), torch.ones(src.size, T, dtype=torch.bool))
                worst_t2 = max(worst_t2, float(np.max(np.abs(lg_t.detach().numpy() - lg_n))))

    print("travel cache matches env:", tt_ok)
    print("bool fields match:", bool_ok)
    print("max abs diff (encoder features):")
    for k in fields:
        print(f"  {k:10s}: {worst[k]:.2e}")
    print(f"max abs diff h1   : {worst_h1:.2e}")
    print(f"max abs diff head5: {worst_head:.2e}")
    print(f"max abs diff t2   : {worst_t2:.2e}")
    ok = (tt_ok and bool_ok and max(worst.values()) < 1e-4
          and worst_head < 1e-3 and worst_t2 < 1e-3)
    print("VERIFY", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
