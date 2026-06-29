#!/usr/bin/env python3
"""Per-PHASE parity test.

Instead of comparing only at end-of-turn, this drives the reference and the fast
env one stage at a time and compares the full state after EACH stage:
    build -> move -> train -> movement -> spawn -> combat+siege -> work -> upkeep
so any single phase that diverges is pinpointed.

Actions are decided once per turn on a throwaway deepcopy (via the validated
`play_ref_turn`), then replayed stage-by-stage onto the real reference while the
matching `FastEnv._phase_*` method is called on the batch.
"""
import copy
import random

import torch

import fast_env as fe
import test_fast_env as T

tt = fe.tt
Side = tt.Side
Sub = tt.Submission
RB = tt.ResultBlock


def _cmp_all(phase, t, refs, env, done):
    for b in range(len(refs)):
        if done[b]:
            continue
        err = T.compare(t, b, T.ref_snapshot(refs[b], None), T.fast_snapshot(env, b))
        if err is not None:
            print(f"[FAIL] phase={phase} turn={t} game={b} "
                  f"(N={env.mb.n_regions[b].item()}): {err}")
            return False
    return True


def run(maps, turns, seed, device, label):
    B = len(maps)
    rng = random.Random(seed)
    env = fe.FastEnv(maps, device=device)
    refs = [tt.init_state(m) for m in maps]
    done = [False] * B

    for b in range(B):
        err = T.compare(0, b, T.ref_snapshot(refs[b], None), T.fast_snapshot(env, b))
        assert err is None, f"init game {b}: {err}"

    for t in range(1, turns + 1):
        # ---- decide actions once (on a deepcopy so the real ref is untouched) ----
        per_game = []
        ntrain = []
        for b in range(B):
            if done[b]:
                per_game.append((([], [], 0), ([], [], 0)))
                ntrain.append((0, 0))
                continue
            cp = copy.deepcopy(refs[b])
            tt._dijkstra_cache.clear()
            al, ar = T.play_ref_turn(cp, maps[b], rng)
            per_game.append((al, ar))
            ntrain.append((al[2], ar[2]))
        act = T.assemble_actions(env, per_game)

        def ref_each(fn):
            for b in range(B):
                if not done[b]:
                    fn(b)

        # ---- BUILD ----
        def do_build(b):
            m = maps[b]; rb = RB()
            for side, a in ((Side.LEFT, per_game[b][0]), (Side.RIGHT, per_game[b][1])):
                s = Sub(); s.upgrades = list(a[0])
                tt.apply_upgrades(refs[b], m, side, s, rb)
        ref_each(do_build)
        env._phase_build(act)
        if not _cmp_all("build", t, refs, env, done):
            return False

        # ---- MOVE (register) ----
        def do_move(b):
            m = maps[b]
            for side, a in ((Side.LEFT, per_game[b][0]), (Side.RIGHT, per_game[b][1])):
                s = Sub(); s.moves = T._translate_moves(refs[b], side, a[1])
                tt.apply_moves(refs[b], m, side, s)
        ref_each(do_move)
        env._phase_register_moves(act)
        if not _cmp_all("move", t, refs, env, done):
            return False

        # ---- TRAIN charge ----
        def do_train(b):
            for side, a in ((Side.LEFT, per_game[b][0]), (Side.RIGHT, per_game[b][1])):
                s = Sub(); s.has_train = True; s.train_n = a[2]
                tt.apply_train_charge(refs[b], side, s)
        ref_each(do_train)
        env._phase_train_charge(act)
        if not _cmp_all("train", t, refs, env, done):
            return False

        # ---- MOVEMENT ----
        def do_movement(b):
            tt._dijkstra_cache.clear()   # cache is keyed only by target region
            tt.apply_day_movement(refs[b], maps[b], RB(), RB())
        ref_each(do_movement)
        env._phase_move()
        if not _cmp_all("movement", t, refs, env, done):
            return False

        # ---- SPAWN ----
        def do_spawn(b):
            tt.spawn_trained(refs[b], Side.LEFT, ntrain[b][0], RB())
            tt.spawn_trained(refs[b], Side.RIGHT, ntrain[b][1], RB())
        ref_each(do_spawn)
        env._phase_spawn()
        if not _cmp_all("spawn", t, refs, env, done):
            return False

        # ---- COMBAT + SIEGE ----
        def do_combat(b):
            rb_l, rb_r = RB(), RB(); siege = {}
            tt.apply_day_combat(refs[b], rb_l, rb_r, siege)
            tt.apply_day_siege(refs[b], rb_l, rb_r, siege)
        ref_each(do_combat)
        env._phase_combat()
        if not _cmp_all("combat", t, refs, env, done):
            return False

        # ---- WORK ----
        ref_each(lambda b: tt.apply_evening_work(refs[b]))
        env._phase_work()
        if not _cmp_all("work", t, refs, env, done):
            return False

        # ---- UPKEEP ----
        ref_each(lambda b: tt.apply_evening_upkeep(refs[b], RB(), RB()))
        env._phase_upkeep()
        if not _cmp_all("upkeep", t, refs, env, done):
            return False

        env.day += 1
        for b in range(B):
            if not done[b] and (tt.hq_of(refs[b], Side.LEFT) is None
                                or tt.hq_of(refs[b], Side.RIGHT) is None):
                done[b] = True

    print(f"[OK] per-phase {label} B={B} turns={turns} dev={device}: "
          f"every phase identical every turn")
    return True


def main():
    import sys
    ok = True
    # uniform sizes
    for (NP, KP) in [(25, 4), (40, 6), (54, 10)]:
        maps = T.gen_maps(12, NP, KP, 4321 + NP)
        ok &= run(maps, turns=200, seed=4321 + NP, device='cpu',
                  label=f"N={2*NP+1}")
    # mixed sizes (padded batch)
    mixed = T.gen_maps_mixed([(25, 4), (54, 10), (31, 5), (40, 6),
                              (47, 8), (33, 6)], seed0=999)
    ok &= run(mixed, turns=200, seed=999, device='cpu', label="MIXED")
    if torch.cuda.is_available():
        ok &= run(mixed, turns=200, seed=999, device='cuda', label="MIXED")
    print("\nRESULT:", "ALL PHASES IDENTICAL" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
