#!/usr/bin/env python3
"""Measure throughput vs batch size and pick B for THIS machine and THIS game.

Why this exists: a batched GPU env is normally kernel-LAUNCH bound rather than
FLOP bound, so env-steps/s keeps climbing with B long after the GPU looks busy --
until it doesn't. Where the curve flattens depends on the game's env and on the
card, so it is worth 5 minutes of measurement rather than a guess.

The one thing to keep in mind while reading the table: raising B at a FIXED
steps_per_iter shortens the GAE horizon (= steps_per_iter / B). The advantage
estimate only reaches back about ``1/(1 - gamma*lam)`` steps (~20 with the
defaults), so a horizon below that starts throwing away signal no matter how good
the throughput looks. This script refuses to recommend such a B.

    python tune_batch.py examples.toy_duel
    python tune_batch.py mygame --B 512 1024 2048 4096 --steps 40000
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import tempfile

import torch

import rlkit


def measure(mod, B, steps_per_iter, minibatch, iters, device, tmp, extra=None):
    """One config -> (steps_per_s, iter_seconds, peak VRAM bytes) or None on OOM.

    The LAST iteration is the one reported: the first pays for cudnn autotuning,
    lazy CUDA context setup and a cold instance queue.
    """
    cfg = mod.Config(B=B, steps_per_iter=steps_per_iter, minibatch=minibatch,
                     iters=iters, use_wandb=False, resume=False, phases=None,
                     log_every=99,
                     ckpt_path=os.path.join(tmp, f"tune_{B}.pt"))
    for k, v in (extra or {}).items():
        setattr(cfg, k, v)
    got = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        rlkit.train(cfg, mod.build, device=device, verbose=False,
                    on_metrics=lambda it, m: got.append(m))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    last = got[-1]
    return last["steps_per_s"], last["iter_seconds"], peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="module with Config + build (e.g. examples.toy_duel)")
    ap.add_argument("--B", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=None,
                    help="steps_per_iter, held FIXED across B (default: 50 * max B)")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=2, help="iterations per B (>=2)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-horizon", type=int, default=20,
                    help="reject a B whose steps_per_iter/B falls below this")
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    for need in ("Config", "build"):
        if not hasattr(mod, need):
            sys.exit(f"{args.module} has no {need}() -- see examples/toy_duel.py")
    steps = args.steps or 50 * max(args.B)
    tmp = tempfile.mkdtemp(prefix="rlkit_tune_")
    rows = []
    try:
        for B in sorted(args.B):
            if B > steps:
                continue
            mb = min(args.minibatch, steps)
            print(f"  B={B:6d} ...", end="", flush=True)
            out = measure(mod, B, steps, mb, max(2, args.iters), args.device, tmp)
            if out is None:
                print("  OOM")
                rows.append((B, steps // B, None, None, None))
                break               # everything larger will OOM too
            sps, dt, peak = out
            print(f"  {sps:>9,.0f} steps/s   {dt:5.1f}s/iter   "
                  f"{peak / 2 ** 30:5.2f} GiB")
            rows.append((B, steps // B, sps, dt, peak))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nsteps_per_iter = {steps:,} (fixed), minibatch = {args.minibatch}")
    print(f"{'B':>7} {'horizon':>8} {'steps/s':>11} {'s/iter':>8} {'peak VRAM':>10}")
    for B, hz, sps, dt, peak in rows:
        if sps is None:
            print(f"{B:>7} {hz:>8} {'OOM':>11}")
            continue
        flag = "" if hz >= args.min_horizon else "  <- horizon too short"
        print(f"{B:>7} {hz:>8} {sps:>11,.0f} {dt:>8.1f} "
              f"{peak / 2 ** 30:>9.2f}G{flag}")

    ok = [r for r in rows if r[2] is not None and r[1] >= args.min_horizon]
    if not ok:
        print("\nno usable B: every candidate either OOMs or has a horizon below "
              f"{args.min_horizon}. Raise --steps and re-run.")
        return
    best = max(ok, key=lambda r: r[2])
    print(f"\n=> B: {best[0]}   (horizon {best[1]}, {best[2]:,.0f} steps/s, "
          f"peak {best[4] / 2 ** 30:.2f} GiB)")
    print(f"   steps_per_iter: {steps}")
    if best[0] == max(r[0] for r in ok) and best[1] > 2 * args.min_horizon:
        print("   the curve had not flattened yet -- try a larger --B and a "
              "proportionally larger --steps")
    print("   remember B and minibatch are TOTALS: --gpus N splits them.")


if __name__ == "__main__":
    main()
