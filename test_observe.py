#!/usr/bin/env python3
"""Sanity checks for FastEnv.observe(): shapes match the encoder spec and a few
token/global features are validated against a direct recompute from raw state."""
import random

import torch

import fast_env as fe
import test_fast_env as T

tt = fe.tt
Side = tt.Side


def main():
    B, NP, KP = 8, 35, 6
    maps = T.gen_maps(B, NP, KP, 2024)
    env = fe.FastEnv(maps, device='cpu')
    rng = random.Random(0)

    # play a handful of random turns so the state is non-trivial
    for _ in range(30):
        per_game = []
        for b in range(B):
            tt._dijkstra_cache.clear()
            # reuse the reference driver only to advance fast via same actions
            # (we don't need the reference state here; just drive fast)
            per_game.append((([], [], rng.randint(0, 1)), ([], [], rng.randint(0, 1))))
        act = T.assemble_actions(env, per_game)
        env.step(act)

    T_tok = env.mb.T
    N = env.N
    # token feature count per spec:
    #  14 scalar features + arrive(5) + reach(5) + dist-to-all-tokens(T)
    exp_F = 14 + 5 + 5 + T_tok
    for side in (0, 1):
        tokens, glob, info = env.observe(side)
        assert tokens.shape == (B, T_tok, exp_F), \
            f"tokens shape {tokens.shape} != {(B, T_tok, exp_F)}"
        assert glob.shape == (B, 11), f"global shape {glob.shape}"
        assert info['build_candidates'].shape == (B, T_tok)
        assert info['move_sources'].shape == (B, T_tok)
        assert info['gold'].shape == (B,)
        assert torch.isfinite(tokens).all() and torch.isfinite(glob).all()
        print(f"side {side}: tokens {tuple(tokens.shape)}, global {tuple(glob.shape)} OK")

    # ---- validate a few features against a direct recompute (left side) ----
    side = 0
    tokens, glob, info = env.observe(side)
    tok_ids = env.mb.token_ids
    # feature 0 = my warrior count on each token region
    my_alive = (env.w_hp[:, :env.Wside] > 0)
    for b in range(B):
        for ti in range(T_tok):
            r = int(tok_ids[b, ti])
            my_cnt = int(((env.w_region[b, :env.Wside] == r) & my_alive[b]).sum())
            feat = int(tokens[b, ti, 0])
            assert feat == my_cnt, f"my_cnt mismatch b{b} tok{ti}: {feat} vs {my_cnt}"
    # global feature 1/2 = total warriors per side
    for b in range(B):
        mt = int((env.w_hp[b, :env.Wside] > 0).sum())
        ot = int((env.w_hp[b, env.Wside:] > 0).sum())
        assert int(glob[b, 1]) == mt and int(glob[b, 2]) == ot
    # distance block is symmetric in turns: dist(tok i -> HQ) sanity (>=0)
    dist_block = tokens[:, :, 14 + 10:]
    assert (dist_block >= 0).all()
    print("feature recompute checks (my_cnt, totals, dist>=0): OK")

    # ---- mixed-size batch: token_mask must mark exactly K_b+2 valid tokens ----
    mmaps = T.gen_maps_mixed([(25, 4), (54, 10), (31, 5), (40, 6)], 321)
    menv = fe.FastEnv(mmaps, device='cpu')
    for _ in range(10):
        menv.step(T.assemble_actions(menv, [(([], [], 1), ([], [], 1))] * len(mmaps)))
    tk, gl, info = menv.observe(0)
    Tmax = menv.mb.T
    assert tk.shape == (len(mmaps), Tmax, 14 + 5 + 5 + Tmax)
    for b, mm in enumerate(mmaps):
        nvalid = int(info['token_mask'][b].sum())
        assert nvalid == mm.K + 2, f"game {b}: {nvalid} valid tokens != {mm.K + 2}"
        # padded tokens must be all-zero feature rows
        pad = ~info['token_mask'][b]
        assert float(tk[b, pad].abs().sum()) == 0.0
    print(f"mixed-size observe: token_mask correct (Tmax={Tmax}), padded tokens zeroed: OK")
    print("\nRESULT: observe() OK")


if __name__ == "__main__":
    main()
