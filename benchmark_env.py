#!/usr/bin/env python3
"""Throughput benchmark: FastEnv (batched GPU) vs the original CPU simulator.

Reports environment-steps per second (one env-step = advancing one game one day).
"""
import time

import torch

import fast_env as fe
import test_fast_env as T

tt = fe.tt
Side = tt.Side


def rand_actions(env, gen):
    B, N = env.B, env.N
    dev = env.device
    tok_mask = torch.zeros((B, N), dtype=torch.bool, device=dev)
    tok_mask.scatter_(1, env.mb.token_ids, True)
    act = {}
    for key in ('left', 'right'):
        build = tok_mask & (torch.rand((B, N), device=dev, generator=gen) < 0.1)
        move = torch.full((B, N), -1, dtype=torch.int64, device=dev)
        pick = tok_mask & (torch.rand((B, N), device=dev, generator=gen) < 0.15)
        rnd_tgt = torch.randint(0, N, (B, N), device=dev, generator=gen)
        move = torch.where(pick, rnd_tgt, move)
        train = torch.randint(0, 4, (B,), device=dev, generator=gen)
        act[key] = {'build': build, 'move': move, 'train': train}
    return act


def bench_fast(B, NP, KP, steps, device, seed=0, max_w=None):
    maps = T.gen_maps(min(B, 8), NP, KP, seed)
    # replicate the few generated maps to fill the batch cheaply
    maps = (maps * ((B // len(maps)) + 1))[:B]
    t0 = time.time()
    env = fe.FastEnv(maps, device=device, max_warriors_per_side=max_w)
    reset_t = time.time() - t0
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    # warmup
    for _ in range(3):
        env.step(rand_actions(env, gen))
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        env.step(rand_actions(env, gen))
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    eps = B * steps / dt
    mem = (torch.cuda.max_memory_allocated() / 1e9) if device == 'cuda' else 0.0
    print(f"  FastEnv  B={B:5d} N={2*NP+1:3d} Wside={env.Wside:4d} dev={device:4s}: "
          f"{eps:12,.0f} env-steps/s  ({steps} steps in {dt:5.2f}s, "
          f"setup {reset_t:4.2f}s, peakmem {mem:.2f}GB)")
    return eps


def bench_reference(B, NP, KP, steps, seed=0):
    import random
    rng = random.Random(seed)
    maps = T.gen_maps(B, NP, KP, seed)
    sts = [tt.init_state(m) for m in maps]
    t0 = time.time()
    done = [False] * B
    for _ in range(steps):
        for b in range(B):
            if done[b]:
                continue
            if tt.hq_of(sts[b], Side.LEFT) is None or tt.hq_of(sts[b], Side.RIGHT) is None:
                done[b] = True
                continue
            tt._dijkstra_cache.clear()
            T.play_ref_turn(sts[b], maps[b], rng)
    dt = time.time() - t0
    eps = B * steps / dt
    print(f"  Reference B={B:5d} N={2*NP+1:3d} dev=cpu : "
          f"{eps:12,.0f} env-steps/s  ({steps} steps in {dt:5.2f}s)")
    return eps


def main():
    print("== original CPU simulator (single-thread reference dynamics) ==")
    ref = bench_reference(B=16, NP=40, KP=6, steps=50)

    print("\n== FastEnv (full pool, bit-exact) ==")
    if torch.cuda.is_available():
        bench_fast(B=1024, NP=40, KP=6, steps=200, device='cuda')
        fast = bench_fast(B=4096, NP=40, KP=6, steps=200, device='cuda')
        bench_fast(B=2048, NP=54, KP=10, steps=200, device='cuda')  # largest map
        print(f"\nSpeedup (B=4096 cuda, full pool, vs reference): {fast/ref:,.0f}x")

        print("\n== FastEnv (realistic pool Wside=128 for throughput) ==")
        f2 = bench_fast(B=4096, NP=40, KP=6, steps=200, device='cuda', max_w=128)
        bench_fast(B=8192, NP=40, KP=6, steps=200, device='cuda', max_w=128)
        print(f"\nSpeedup (B=4096 cuda, Wside=128, vs reference): {f2/ref:,.0f}x")


if __name__ == "__main__":
    main()
