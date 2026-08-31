#!/usr/bin/env python3
"""Independent verification of EVERY encoder/observation feature.

We advance the fast env and a reference `GameState` in lock-step (using the
validated bit-exact dynamics), then recompute each observation feature
*independently* from the raw reference state and compare to `env.observe()`.

The tricky features are travel-time based:
  * token feature: enemy warriors that can reach a 거점 within 1..5 turns,
  * token feature: my warriors arriving at a 거점 in exactly 1..5 turns,
  * token feature: turns-distance to every other 거점.
For these we compute the ground-truth "turns to reach a target" by directly
simulating the game's movement rule (dijkstra_from(target) + the same min
edge_weight+dist next-hop, ties -> smaller region id), independent of the env's
precomputed cache.
"""
import random

import torch

# --- repo layout: make src/ importable when run directly --------------------
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import fast_env as fe
import test_fast_env as T

tt = fe.tt
Side = tt.Side
BKind = tt.BKind


def turns_to_target(m, dst):
    """turns_to[r] = #steps for a warrior at r to reach dst, following the exact
    movement rule used by apply_day_movement (unblocked). -1 if unreachable."""
    tt._dijkstra_cache.clear()
    dist = tt.dijkstra_from(m, dst)        # shortest total euclidean-ceil distance
    res = [None] * m.N
    res[dst] = 0

    def steps(cur):
        if res[cur] is not None:
            return res[cur]
        best_v, best_score = -1, None
        for v in sorted(m.adj[cur]):        # ascending -> tie goes to smaller id
            if dist[v] < 0:
                continue
            score = tt.edge_weight(m, cur, v) + dist[v]
            if best_score is None or score < best_score:
                best_score, best_v = score, v
        if best_v < 0:
            res[cur] = -1
            return -1
        r = steps(best_v)
        res[cur] = (r + 1) if r >= 0 else -1
        return res[cur]

    import sys
    sys.setrecursionlimit(10000)
    for r in range(m.N):
        if dist[r] >= 0:
            steps(r)
        elif res[r] is None:
            res[r] = -1
    return res


def warriors_of(st, side):
    return [w for w in st.warriors.values() if w.side is side and w.hp > 0]


# --------------------------------------------------------------------------- #
# Independent python-side belief tracker (fog of war), mirroring
# FastEnv._update_fog exactly but built from the reference GameState + the
# reference's OWN tt._hop_set -- so this is a genuinely independent check of
# the vision/memory logic, not just a restatement of it.
# --------------------------------------------------------------------------- #
def _vis_regions(st, m, side):
    vis = set()
    for w in st.warriors.values():
        if w.side is side and w.hp > 0:
            vis |= tt._hop_set(w.region, tt.HOP_VISION, m.adj)
    for bld in st.buildings.values():
        if bld.side is side:
            vis |= tt._hop_set(bld.region, tt.HOP_VISION, m.adj)
    return vis


def new_belief(N):
    return {r: dict(kind=0, level=0, bhp=0, wcnt=0, whp=0, age=fe.MAX_DAYS) for r in range(N)}


def update_belief(belief, st, m, side):
    """One day's refresh of `side`'s belief about its opponent, in place."""
    opp = Side.RIGHT if side is Side.LEFT else Side.LEFT
    vis = _vis_regions(st, m, side)
    op_w = warriors_of(st, opp)
    wcnt_by_r, whp_by_r = {}, {}
    for w in op_w:
        wcnt_by_r[w.region] = wcnt_by_r.get(w.region, 0) + 1
        whp_by_r[w.region] = whp_by_r.get(w.region, 0) + w.hp
    for r in range(m.N):
        e = belief[r]
        if r in vis:
            bld = st.buildings.get(r)
            if bld is not None and bld.side is opp:
                e['kind'] = 1 if bld.kind is BKind.HQ else 2
                e['level'] = bld.level
                e['bhp'] = bld.hp
            else:
                e['kind'] = 0; e['level'] = 0; e['bhp'] = 0
            e['wcnt'] = wcnt_by_r.get(r, 0)
            e['whp'] = whp_by_r.get(r, 0)
            e['age'] = 0
        else:
            e['age'] += 1


def check_game(env, b, st, m, side, belief, cov):
    me = side
    opp = Side.RIGHT if side is Side.LEFT else Side.LEFT
    tokens, glob, info = env.observe(0 if side is Side.LEFT else 1)
    tok_ids = env.mb.token_ids[b].tolist()
    tmask = env.mb.token_valid[b].tolist()
    T_ = env.mb.T
    HQ_TUR = [0, 1, 2, 2, 3, 3]; HQ_WC = [0, 1, 2, 3, 4, 5]
    BA_TUR = [0, 1, 1, 2]; BA_WC = [0, 1, 2, 3]

    # ground-truth turns-to-target for every token region (cache per token)
    tturns = {}  # token_region -> [turns from each region]
    for ti in range(T_):
        if tmask[ti]:
            r = tok_ids[ti]
            if r not in tturns:
                tturns[r] = turns_to_target(m, r)

    my_w = warriors_of(st, me)

    for ti in range(T_):
        if not tmask[ti]:
            # padded token: must be all zero
            assert float(tokens[b, ti].abs().sum()) == 0.0, f"padded token nonzero b{b} ti{ti}"
            continue
        r = tok_ids[ti]
        f = tokens[b, ti].tolist()

        my_cnt = sum(1 for w in my_w if w.region == r)
        bld = st.buildings.get(r)
        my_base = bld.level if (bld and bld.side is me and bld.kind is BKind.BASE) else 0
        my_hq = bld.level if (bld and bld.side is me and bld.kind is BKind.HQ) else 0
        my_tur = bld.turret() if (bld and bld.side is me) else 0
        my_wc = bld.work_cap() if (bld and bld.side is me) else 0
        my_bhp = bld.hp if (bld and bld.side is me) else 0
        stat = [w for w in my_w if w.region == r and w.moving_target is None]
        surplus = len(stat) - my_wc
        stat_hp = sum(w.hp for w in stat)

        # opponent fields come from this side's BELIEF, not the true state
        e = belief[r]
        op_cnt = e['wcnt']
        op_base = e['level'] if e['kind'] == 2 else 0
        op_hq = e['level'] if e['kind'] == 1 else 0
        op_tur = (HQ_TUR[min(e['level'], 5)] if e['kind'] == 1 else
                  BA_TUR[min(e['level'], 3)] if e['kind'] == 2 else 0)
        op_wc = (HQ_WC[min(e['level'], 5)] if e['kind'] == 1 else
                 BA_WC[min(e['level'], 3)] if e['kind'] == 2 else 0)
        op_bhp = e['bhp']
        age = e['age']

        expect = [my_cnt, op_cnt, my_base, op_base, my_hq, op_hq,
                  my_tur, op_tur, my_wc, op_wc, my_bhp, op_bhp, surplus, stat_hp, age]
        for i, ev in enumerate(expect):
            assert int(f[i]) == ev, \
                f"token feat {i} b{b} side{me} tok r{r}: env={int(f[i])} expect={ev}"

        # arrivals of my movers in exactly k turns (unaffected by fog -- own info)
        for k in range(1, 6):
            ev = sum(1 for w in my_w
                     if w.moving_target == r and tturns[r][w.region] == k)
            assert int(f[15 + (k - 1)]) == ev, \
                f"arrive k{k} b{b} tok r{r}: env={int(f[15 + k - 1])} expect={ev}"
            if ev > 0:
                cov['arrive_pos'] += 1

        # enemy reachable within k turns -- a sum over ALL regions weighted by
        # (that region's BELIEVED opponent count) x (within k turns of r); belief
        # is per-region (not per-warrior), so this mirrors the env exactly.
        for k in range(1, 6):
            ev = sum(belief[src]['wcnt'] for src in range(m.N)
                     if 0 <= tturns[r][src] <= k)
            assert int(f[20 + (k - 1)]) == ev, \
                f"reach k{k} b{b} tok r{r}: env={int(f[20 + k - 1])} expect={ev}"
            cov['reach_max'] = max(cov['reach_max'], ev)

        # distance to every other token (turns)
        for tj in range(T_):
            base = 25 + tj
            if not tmask[tj]:
                assert int(f[base]) == 0
                continue
            rj = tok_ids[tj]
            ev = tturns[rj][r]              # turns from r to token rj
            assert ev >= 0 and int(f[base]) == ev, \
                f"dist b{b} tok r{r}->r{rj}: env={int(f[base])} expect={ev}"
    return True


def main():
    import sys
    maps = T.gen_maps_mixed([(90, 6), (124, 9), (95, 6), (115, 8)], seed0=2024)
    env = fe.FastEnv(maps, device='cpu')
    refs = [tt.init_state(m) for m in maps]
    rng = random.Random(7)
    B = len(maps)

    # independent fog-of-war belief tracker: belief[b][side] = this side's
    # persisted memory of the opponent, refreshed via the reference's own
    # tt._hop_set (not env's hop2_reach) -- see update_belief().
    belief = [{Side.LEFT: new_belief(maps[b].N), Side.RIGHT: new_belief(maps[b].N)}
             for b in range(B)]

    # play random turns and validate observe() at several snapshots, so the
    # travel-based features are exercised across many different states.
    done = [False] * B
    cov = {'arrive_pos': 0, 'reach_max': 0}
    checked = 0
    transit = 0

    def validate_now():
        nonlocal checked, transit
        for b in range(B):
            if done[b]:
                continue
            toks = set(int(x) for x, v in zip(env.mb.token_ids[b].tolist(),
                                              env.mb.token_valid[b].tolist()) if v)
            for w in refs[b].warriors.values():
                if w.hp > 0 and w.region not in toks:
                    transit += 1
            for side in (Side.LEFT, Side.RIGHT):
                check_game(env, b, refs[b], maps[b], side, belief[b][side], cov)
                checked += 1

    for t in range(60):
        per_game = []
        for b in range(B):
            if done[b]:
                per_game.append((([], [], 0), ([], [], 0)))
                continue
            tt._dijkstra_cache.clear()
            per_game.append(T.play_ref_turn(refs[b], maps[b], rng))
        env.step(T.assemble_actions(env, per_game))
        for b in range(B):
            if done[b]:
                continue
            # belief refreshes off the SAME post-day state env._update_fog uses
            for side in (Side.LEFT, Side.RIGHT):
                update_belief(belief[b][side], refs[b], maps[b], side)
        for b in range(B):
            if tt.hq_of(refs[b], Side.LEFT) is None or tt.hq_of(refs[b], Side.RIGHT) is None:
                done[b] = True
        if t >= 5 and t % 7 == 0:
            validate_now()

    # also directly validate the travel-time cache itself
    for b in range(B):
        m = maps[b]
        for ti in range(env.mb.T):
            if not env.mb.token_valid[b, ti]:
                continue
            r = int(env.mb.token_ids[b, ti])
            gt = turns_to_target(m, r)
            for src in range(m.N):
                if gt[src] is not None and gt[src] >= 0:
                    cv = int(env.mb.travel_turns[b, src, ti])
                    assert cv == gt[src], \
                        f"travel_turns b{b} src{src}->tok r{r}: cache={cv} truth={gt[src]}"
    print("travel_turns cache matches independent movement simulation: OK")

    validate_now()  # final snapshot too
    assert cov['arrive_pos'] > 0, "no nonzero arrival features exercised"
    assert cov['reach_max'] > 0, "no nonzero reach features exercised"
    assert transit > 0, "no warriors in transit (non-token) regions to test reach"
    print(f"coverage: nonzero-arrive cells={cov['arrive_pos']}, "
          f"max enemy-reach value={cov['reach_max']}, transit warriors={transit}")
    # global feature spot-checks (opponent aggregates are belief-based, i.e.
    # summed over this side's per-region memory -- see fast_env.observe)
    for side in (0, 1):
        tokens, glob, info = env.observe(side)
        me = Side.LEFT if side == 0 else Side.RIGHT
        opp = Side.RIGHT if side == 0 else Side.LEFT
        for b in range(B):
            if done[b]:
                continue
            st = refs[b]
            bel = belief[b][me]
            assert int(glob[b, 1]) == len(warriors_of(st, me))
            assert int(glob[b, 2]) == sum(e['wcnt'] for e in bel.values())
            assert int(glob[b, 5]) == st.gold[me.value]
            assert int(glob[b, 6]) == st.gold[opp.value]     # opp gold is NOT fogged in observe()
            lvl_my = sum(bb.level for bb in st.buildings.values() if bb.side is me)
            lvl_op = sum(e['level'] for e in bel.values())
            assert int(glob[b, 9]) == lvl_my and int(glob[b, 10]) == lvl_op
            # opponent income (glob[8]) is belief-based: 15 x min(believed count,
            # believed work_cap) per region, summed -- NOT the true st income.
            HQ_WC = [0, 1, 2, 3, 4, 5]; BA_WC = [0, 1, 2, 3]
            inc_op = 0
            for e in bel.values():
                if e['kind'] == 0:
                    continue
                wc = HQ_WC[min(e['level'], 5)] if e['kind'] == 1 else BA_WC[min(e['level'], 3)]
                inc_op += 15 * min(e['wcnt'], wc)
            assert int(glob[b, 8]) == inc_op, \
                f"income_op b{b} side{me}: env={int(glob[b, 8])} expect={inc_op}"

    print(f"all token features (counts, buildings, surplus, stat_hp, "
          f"arrive 1-5, enemy-reach 1-5, token distances) verified for "
          f"{checked} (game,side) views")
    print("global features (totals, gold, level sums) verified")
    print("\nRESULT: ENCODER OK")


if __name__ == "__main__":
    main()
