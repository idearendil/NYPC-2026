#!/usr/bin/env python3
"""Offline: convert a PPO checkpoint's actor nets into a torch-free .npz so the
submission bot can run with numpy only (no ~2.3s torch import at handshake).

Usage:
    python export_weights.py --ckpt checkpoint.pt --out data.bin
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint.pt")
    ap.add_argument("--out", default="data.bin")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    d_model = int(ck.get("cfg", {}).get("d_model", 64))

    arrays = {}
    for k, v in ck["actor_t1"].items():
        arrays["t1." + k] = v.detach().cpu().numpy().astype(np.float32)
    for k, v in ck["actor_t2"].items():
        arrays["t2." + k] = v.detach().cpu().numpy().astype(np.float32)
    # critic net: needed by the submission's lookahead search to value leaf states
    # (the numpy Net.value() mirrors Critic.value: encoder -> per-token head -> mean).
    for k, v in ck["critic"].items():
        arrays["critic." + k] = v.detach().cpu().numpy().astype(np.float32)
    # metadata (numpy scalars)
    arrays["meta.d_model"] = np.int64(d_model)
    arrays["meta.heads"] = np.int64(4)        # ActorT1/ActorT2 default

    # Pass a file handle (not a path) so numpy writes EXACTLY args.out and does
    # not append a ".npz" extension (the submission binary must be named data.bin).
    with open(args.out, "wb") as f:
        np.savez(f, **arrays)
    print(f"wrote {args.out}: {len(arrays)} arrays, d_model={d_model}")


if __name__ == "__main__":
    main()
