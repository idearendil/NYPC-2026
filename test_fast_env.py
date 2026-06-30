#!/usr/bin/env python3
"""Bit-exact parity test: FastEnv vs the original testing-tool.py dynamics.

For B games we drive a reference ``tt.GameState`` with the *original* phase
functions (the same calls ``run_game`` makes), feeding it the per-warrior MOVE
commands that the RL action (src->target) expands to. The identical RL-level
actions are fed to ``FastEnv``. After every turn we compare, per game:
  * gold (both sides)
  * every building (owner, kind, level, hp)
  * every warrior by suffix (region, hp, moving) -- the strongest possible check
"""
from __future__ import annotations

import random
import sys

import torch

import fast_env as fe

tt = fe.tt
Side = tt.Side
BKind = tt.BKind


# --------------------------------------------------------------------------- #
# Map generation (reuse the original generator)
# --------------------------------------------------------------------------- #
def gen_maps(B, NP, KP, seed0):
    maps = []
    for b in range(B):
        lines = tt.generate_map(tt.XoShiro256(seed0 + 7919 * b + 1), NP, KP)
        maps.append(tt.read_map(lines))
    return maps


def gen_maps_mixed(specs, seed0):
    """specs = list of (NP, KP); returns one map per spec (varying N,K)."""
    maps = []
    for b, (NP, KP) in enumerate(specs):
        lines = tt.generate_map(tt.XoShiro256(seed0 + 7919 * b + 1), NP, KP)
        maps.append(tt.read_map(lines))
    return maps


# --------------------------------------------------------------------------- #
# Reference turn driver that also records the RL-level action it used.
# --------------------------------------------------------------------------- #
def _token_regions(m):
    return sorted(set(m.strongholds) | {0, m.N - 1})


def _build_cost(st, m, side, r):
    b = st.buildings.get(r)
    if b is None:
        if r == 0 or r == m.N - 1:
            return None
        if r not in m.strongholds:
            return None
        return tt.BASE_LEVELS[1].cost
    if b.side is not side:
        return None
    if b.level >= b.max_level():
        return b.heal_cost()
    return b.next_level_cost()


def _translate_moves(st, side, rl_moves):
    """Expand RL (src,tgt) pairs into per-warrior (suffix,tgt) MOVE commands
    using the *post-build* state -- exactly what FastEnv does internally."""
    out = []
    for (src, tgt) in rl_moves:
        ws = [w for w in st.warriors.values()
              if w.side is side and w.region == src and w.hp > 0
              and w.moving_target is None]
        b = st.buildings.get(src)
        if b is not None and b.side is side:
            ws.sort(key=lambda w: (w.hp, w.suffix))
            movers = ws[b.work_cap():]
        else:
            movers = ws
        for w in movers:
            out.append((w.suffix, tgt))
    return out


def _present(st, r, side, friendly):
    return any(w.region == r and w.hp > 0 and ((w.side is side) == friendly)
               for w in st.warriors.values())


def play_ref_turn(st, m, rng):
    """Run one full reference turn; return ((build_l, moves_l, train_l),
    (build_r, moves_r, train_r)) -- the RL-level actions used."""
    rb_l = tt.ResultBlock()
    rb_r = tt.ResultBlock()
    tokens = _token_regions(m)
    recorded = {}

    def phase1_side(side, rb):
        u = side.value
        # reserve gold for training first so the train/upgrade/heal gold paths
        # actually get exercised (a pure-move policy keeps gold near zero).
        hq0 = tt.hq_of(st, side)
        cap0 = hq0.train_cap() if hq0 is not None else 0
        want_train = min(cap0, st.gold[u] // fe.TRAIN_COST) if rng.random() < 0.7 else 0
        reserve = want_train * fe.TRAIN_COST

        # ---- build / upgrade / heal ----
        cand = [r for r in tokens if _present(st, r, side, True)
                and not _present(st, r, side, False)]
        rng.shuffle(cand)
        builds = []
        spent_b = 0
        for r in cand:
            if rng.random() < 0.5:
                c = _build_cost(st, m, side, r)
                if c is None:
                    continue
                if st.gold[u] - reserve - spent_b - c >= 0:
                    builds.append(r)
                    spent_b += c
        sub = tt.Submission()
        sub.upgrades = builds
        tt.apply_upgrades(st, m, side, sub, rb)

        # ---- moves (chosen pre-build sources, expanded post-build) ----
        src_pool = [r for r in tokens
                    if any(w.region == r and w.side is side and w.hp > 0
                           and w.moving_target is None for w in st.warriors.values())]
        rng.shuffle(src_pool)
        rl_moves = []
        used_src = set()
        for src in src_pool:
            if src in used_src:
                continue
            if rng.random() < 0.3:
                tgt = rng.choice(tokens)
                rl_moves.append((src, tgt))
                used_src.add(src)
        # expand + budget-check against live gold
        movelist = _translate_moves(st, side, rl_moves)
        # compute cost and trim to affordable (keep whole src groups for fidelity)
        affordable_moves = []
        spent = 0
        # re-expand per rl pair so we can drop whole pairs if unaffordable
        final_rl = []
        for (src, tgt) in rl_moves:
            grp = _translate_moves(st, side, [(src, tgt)])
            b = st.buildings.get(tgt)
            free = (b is not None and b.side is side)
            cost = 0 if free else fe.MOVE_COST * len(grp)
            if st.gold[u] - spent - cost >= reserve:   # keep the train reserve
                final_rl.append((src, tgt))
                affordable_moves.extend(grp)
                spent += cost
        sub2 = tt.Submission()
        sub2.moves = affordable_moves
        tt.apply_moves(st, m, side, sub2)

        # ---- train (the amount we reserved gold for) ----
        hq = tt.hq_of(st, side)
        cap = hq.train_cap() if hq is not None else 0
        n = min(want_train, cap, st.gold[u] // fe.TRAIN_COST)
        sub3 = tt.Submission()
        sub3.has_train = True
        sub3.train_n = n
        n = tt.apply_train_charge(st, side, sub3)

        recorded[side] = (builds, final_rl, n)
        return n

    n_l = phase1_side(Side.LEFT, rb_l)
    n_r = phase1_side(Side.RIGHT, rb_r)
    tt.apply_day_movement(st, m, rb_l, rb_r)
    tt.spawn_trained(st, Side.LEFT, n_l, rb_l)
    tt.spawn_trained(st, Side.RIGHT, n_r, rb_r)
    siege = {}
    tt.apply_day_combat(st, rb_l, rb_r, siege)
    tt.apply_day_siege(st, rb_l, rb_r, siege)
    tt.apply_evening_work(st)
    tt.apply_evening_upkeep(st, rb_l, rb_r)

    return recorded[Side.LEFT], recorded[Side.RIGHT]


# --------------------------------------------------------------------------- #
# State extraction for comparison
# --------------------------------------------------------------------------- #
def ref_snapshot(st, m):
    bld = {}
    for r, b in st.buildings.items():
        bld[r] = (b.side.value + 1, 1 if b.kind is BKind.HQ else 2, b.level, b.hp)
    war = {}
    for w in st.warriors.values():
        if w.hp > 0:
            war[(w.side.value, w.suffix)] = (w.region, w.hp, w.moving_target is not None)
    return list(st.gold), bld, war


def fast_snapshot(env, b):
    N = env.N
    gold = [int(env.gold[b, 0]), int(env.gold[b, 1])]
    bld = {}
    owner = env.b_owner[b].tolist()
    kind = env.b_kind[b].tolist()
    level = env.b_level[b].tolist()
    hp = env.b_hp[b].tolist()
    for r in range(N):
        if owner[r] != 0:
            bld[r] = (owner[r], kind[r], level[r], hp[r])
    war = {}
    for side in (0, 1):
        base = env.left_base if side == 0 else env.right_base
        whp = env.w_hp[b, base:base + env.Wside].tolist()
        wreg = env.w_region[b, base:base + env.Wside].tolist()
        wmov = env.w_move[b, base:base + env.Wside].tolist()
        for j in range(env.Wside):
            if whp[j] > 0:
                war[(side, j + 1)] = (wreg[j], whp[j], bool(wmov[j]))
    return gold, bld, war


def compare(turn, b, ref, fast):
    rg, rb, rw = ref
    fg, fb, fw = fast
    if rg != fg:
        return f"gold mismatch: ref={rg} fast={fg}"
    if rb != fb:
        # find first differing region
        for r in sorted(set(rb) | set(fb)):
            if rb.get(r) != fb.get(r):
                return f"building@{r} ref={rb.get(r)} fast={fb.get(r)}"
    if rw != fw:
        for k in sorted(set(rw) | set(fw)):
            if rw.get(k) != fw.get(k):
                return f"warrior(side,suffix)={k} ref={rw.get(k)} fast={fw.get(k)}"
    return None


# --------------------------------------------------------------------------- #
# Build batched action tensors from per-game recorded RL actions
# --------------------------------------------------------------------------- #
def assemble_actions(env, per_game):
    B, N = env.B, env.N
    dev = env.device
    act = {}
    for side, key in ((0, 'left'), (1, 'right')):
        build = torch.zeros((B, N), dtype=torch.bool, device=dev)
        move = torch.full((B, N), -1, dtype=torch.int64, device=dev)
        train = torch.zeros(B, dtype=torch.int64, device=dev)
        for b in range(B):
            builds, rl_moves, n = per_game[b][side]
            for r in builds:
                build[b, r] = True
            for (src, tgt) in rl_moves:
                move[b, src] = tgt
            train[b] = n
        act[key] = {'build': build, 'move': move, 'train': train}
    return act


# --------------------------------------------------------------------------- #
def run_parity(B, NP, KP, turns, seed, device='cpu', verbose=True):
    maps = gen_maps(B, NP, KP, seed)
    return run_parity_maps(maps, turns, seed, device, verbose,
                           label=f"N={2*NP+1} K={2*KP+1}")


def run_parity_maps(maps, turns, seed, device='cpu', verbose=True, label=""):
    B = len(maps)
    rng = random.Random(seed)
    env = fe.FastEnv(maps, device=device)
    refs = [tt.init_state(m) for m in maps]

    # sanity: identical initial state
    for b in range(B):
        err = compare(0, b, ref_snapshot(refs[b], maps[b]), fast_snapshot(env, b))
        assert err is None, f"INIT mismatch game {b}: {err}"

    done = [False] * B
    for t in range(1, turns + 1):
        per_game = []
        for b in range(B):
            if done[b]:
                per_game.append((([], [], 0), ([], [], 0)))      # finished -> noop
                continue
            # the reference dijkstra cache is keyed only by target region, so it
            # must be cleared when switching between different maps.
            tt._dijkstra_cache.clear()
            al, ar = play_ref_turn(refs[b], maps[b], rng)
            per_game.append((al, ar))
        pre = [fast_snapshot(env, b) for b in range(B)]
        act = assemble_actions(env, per_game)
        env.step(act, apply_agent_rules=False)   # parity = pure game rules only

        for b in range(B):
            if done[b]:
                continue
            err = compare(t, b, ref_snapshot(refs[b], maps[b]), fast_snapshot(env, b))
            if err is not None:
                print(f"[FAIL] turn {t} game {b} (N={maps[b].N}): {err}")
                pg = per_game[b]
                print("  pre gold:", pre[b][0])
                print("  actions L:", pg[0])
                print("  actions R:", pg[1])
                rs = ref_snapshot(refs[b], maps[b])
                fs = fast_snapshot(env, b)
                print("  ref buildings:", rs[1])
                print("  fast buildings:", fs[1])
                # warriors per region per side count
                def wcount(war):
                    d = {}
                    for (s, sfx), (reg, hp, mv) in war.items():
                        d[(s, reg)] = d.get((s, reg), 0) + 1
                    return d
                print("  ref wcount:", sorted(wcount(rs[2]).items()))
                print("  fast wcount:", sorted(wcount(fs[2]).items()))
                return False
            if (tt.hq_of(refs[b], Side.LEFT) is None
                    or tt.hq_of(refs[b], Side.RIGHT) is None):
                done[b] = True   # validated through the deciding turn; stop here
    if verbose:
        print(f"[OK] {label} B={B} turns={turns} device={device}: "
              f"identical for all {B} games every turn")
    return True


def ref_apply_turn(st, m, actL, actR):
    """Apply one reference turn from explicit RL actions (builds, rl_moves, n)."""
    rb_l, rb_r = tt.ResultBlock(), tt.ResultBlock()

    def side_apply(side, act, rb):
        sub = tt.Submission(); sub.upgrades = list(act[0])
        tt.apply_upgrades(st, m, side, sub, rb)
        sub2 = tt.Submission(); sub2.moves = _translate_moves(st, side, act[1])
        tt.apply_moves(st, m, side, sub2)
        sub3 = tt.Submission(); sub3.has_train = True; sub3.train_n = act[2]
        return tt.apply_train_charge(st, side, sub3)

    n_l = side_apply(Side.LEFT, actL, rb_l)
    n_r = side_apply(Side.RIGHT, actR, rb_r)
    tt.apply_day_movement(st, m, rb_l, rb_r)
    tt.spawn_trained(st, Side.LEFT, n_l, rb_l)
    tt.spawn_trained(st, Side.RIGHT, n_r, rb_r)
    siege = {}
    tt.apply_day_combat(st, rb_l, rb_r, siege)
    tt.apply_day_siege(st, rb_l, rb_r, siege)
    tt.apply_evening_work(st)
    tt.apply_evening_upkeep(st, rb_l, rb_r)


def econ_action(st, m, side, cov):
    """Greedy economic policy that exercises build/upgrade/heal/move/train,
    budgeted in the reference deduction order (build, move, train)."""
    u = side.value
    gold = st.gold[u]
    spent = 0
    builds = []
    hqr = 0 if side is Side.LEFT else m.N - 1
    hq = st.buildings.get(hqr)

    # build new bases where we stand on an empty stronghold
    for r in m.strongholds:
        b = st.buildings.get(r)
        if b is None and _present(st, r, side, True) and not _present(st, r, side, False):
            if gold - spent - tt.BASE_LEVELS[1].cost >= 0:
                builds.append(r); spent += tt.BASE_LEVELS[1].cost; cov['build'] += 1
    # upgrade or heal HQ
    if hq is not None and _present(st, hqr, side, True):
        if hq.level < hq.max_level():
            c = hq.next_level_cost()
            if gold - spent - c >= 0:
                builds.append(hqr); spent += c; cov['upgrade'] += 1
        else:
            c = hq.heal_cost()
            if gold - spent - c >= 0:
                builds.append(hqr); spent += c; cov['heal'] += 1
    # upgrade or heal our bases
    for r in m.strongholds:
        b = st.buildings.get(r)
        if b is not None and b.side is side and _present(st, r, side, True) \
                and not _present(st, r, side, False) and r not in builds:
            if b.level < b.max_level():
                c = b.next_level_cost(); kind = 'upgrade'
            else:
                c = b.heal_cost(); kind = 'heal'
            if gold - spent - c >= 0:
                builds.append(r); spent += c; cov[kind] += 1

    # move: send surplus HQ warriors to nearest unclaimed stronghold
    rl_moves = []
    stat_hq = [w for w in st.warriors.values()
               if w.side is side and w.region == hqr and w.hp > 0
               and w.moving_target is None]
    empties = [r for r in m.strongholds if st.buildings.get(r) is None]
    if len(stat_hq) > (hq.work_cap() if hq else 1) and empties:
        tgt = min(empties, key=lambda r: abs(r - hqr))
        grp = _translate_moves(st, side, [(hqr, tgt)])
        cost = fe.MOVE_COST * len(grp)
        if grp and gold - spent - cost >= 0:
            rl_moves.append((hqr, tgt)); spent += cost

    # hoard gold for buildings (training is already covered by the random test)
    train_n = 0
    return builds, rl_moves, train_n


def run_scripted_coverage(NP, KP, turns, seed):
    m = tt.read_map(tt.generate_map(tt.XoShiro256(seed), NP, KP))
    env = fe.FastEnv([m], device='cpu')
    st = tt.init_state(m)
    cov = dict(build=0, upgrade=0, heal=0)
    err = compare(0, 0, ref_snapshot(st, m), fast_snapshot(env, 0))
    assert err is None, f"init: {err}"
    for t in range(1, turns + 1):
        if tt.hq_of(st, Side.LEFT) is None or tt.hq_of(st, Side.RIGHT) is None:
            break
        tt._dijkstra_cache.clear()
        actL = econ_action(st, m, Side.LEFT, cov)
        actR = econ_action(st, m, Side.RIGHT, cov)
        ref_apply_turn(st, m, actL, actR)
        act = assemble_actions(env, [(actL, actR)])
        env.step(act, apply_agent_rules=False)   # parity = pure game rules only
        err = compare(t, 0, ref_snapshot(st, m), fast_snapshot(env, 0))
        if err is not None:
            print(f"[FAIL scripted] turn {t}: {err}")
            return False
    print(f"[OK] scripted economy N={2*NP+1} turns={turns}: identical "
          f"(coverage build={cov['build']} upgrade={cov['upgrade']} heal={cov['heal']})")
    return cov['upgrade'] > 0 and cov['heal'] > 0 and cov['build'] > 0


def main():
    configs = [
        # (NP, KP) -- KP must be in the legal odd range for N=2NP+1
        (25, 4),    # N=51,  K=9   (smallest)
        (30, 5),    # N=61,  K=11
        (40, 6),    # N=81,  K=13
        (54, 10),   # N=109, K=21  (largest)
    ]
    ok = True
    for (NP, KP) in configs:
        ok &= run_parity(B=16, NP=NP, KP=KP, turns=200, seed=1234 + NP, device='cpu')

    # deterministic economy: guarantees build/upgrade/heal branches are hit
    for (NP, KP) in [(25, 4), (40, 6)]:
        ok &= run_scripted_coverage(NP, KP, turns=200, seed=77 + NP)

    # MIXED SIZES in one padded batch (different N and K per game)
    mixed_specs = [(25, 4), (54, 10), (31, 5), (40, 6), (25, 4), (47, 8),
                   (54, 9), (33, 6)]
    for dev in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        mmaps = gen_maps_mixed(mixed_specs, seed0=555)
        ns = sorted({mm.N for mm in mmaps})
        ok &= run_parity_maps(mmaps, turns=200, seed=555, device=dev,
                              label=f"MIXED N={ns}")

    # GPU == CPU determinism check on one config (compare to its own reference)
    if torch.cuda.is_available():
        ok &= run_parity(B=16, NP=40, KP=6, turns=200, seed=999, device='cuda')
    else:
        print("[skip] CUDA not available for GPU parity")

    print("\nRESULT:", "ALL PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
