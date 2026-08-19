#!/usr/bin/env python3
"""Data-parallel lockstep test -- no NCCL/gloo required.

The one property that MUST hold in a multi-rank run: at the end of every iteration
every rank holds bit-identical weights AND an identical opponent pool. Each rank
decides on its own when to snapshot, grow and evict, so if the pool's win rates
drift apart the ranks silently become two different trainers writing one
checkpoint. That failure does not crash and does not show up in the loss curves.

This runs the ranks as THREADS of one process with the collectives implemented over
a barrier (``rlkit.testing``), so every distributed code path is exercised -- which
quantities get reduced, where, and whether the ranks stay in lockstep -- and only
the wire protocol is stubbed. Threads share the global torch RNG, so the ranks here
do not simulate genuinely independent games; that is fine, because what is under
test is agreement, and agreement is produced by the reductions rather than by the
sampling.

Still worth doing once on the real box before a long run:
    python -m examples.toy_duel --gpus 2 --iters 3 --no-wandb --no-resume

    python test_rlkit_dist.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

import rlkit
from examples import toy_duel as toy

WORLD = 2
ITERS = 4


def make_cfg(tmp, rank):
    return toy.Config(
        B=32, steps_per_iter=1024, iters=ITERS, minibatch=256, d_model=16,
        use_wandb=False, resume=False, phases=None, instance_workers=0,
        store_device="cpu", log_every=99,
        # force the pool to churn: a permanent snapshot every 2 iterations and an
        # add every iteration, so growth AND eviction both happen during the test
        pool_snapshot_every=2, pool_add_threshold=0.0, pool_max_size=4,
        ckpt_path=os.path.join(tmp, f"dp{rank}.pt"))


def main():
    tmp = tempfile.mkdtemp(prefix="rlkit_dist_")
    torch.manual_seed(0)

    def entry(rank):
        policy, pool, _task = rlkit.train(make_cfg(tmp, rank), toy.build,
                                         device="cpu", verbose=False)
        return dict(w=rlkit.testing.flat_params(policy), wr=pool.wr.clone(),
                    ids=list(pool.ids), perm=list(pool.perm), size=pool.size)

    try:
        out = rlkit.testing.run_threaded_ranks(entry, world=WORLD)
        a, b = out[0], out[1]
        n = rlkit.testing.assert_params_identical(a["w"], b["w"], "ranks")
        assert torch.equal(a["wr"], b["wr"]), \
            f"pool win rates diverged:\n  {a['wr']}\n  {b['wr']}"
        assert a["ids"] == b["ids"], f"pool membership diverged: {a['ids']} vs {b['ids']}"
        assert a["perm"] == b["perm"], "permanent-snapshot flags diverged"
        assert a["size"] > 3, f"the pool never grew, so nothing was tested ({a['size']})"
    except Exception as e:                    # noqa: BLE001 - reported below
        import traceback
        traceback.print_exc()
        print(f"\nRESULT: FAILED ({e})")
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{n} parameter tensors compared across {WORLD} ranks over {ITERS} iters "
          f"(pool grew to {a['size']}: ids {a['ids']})")
    print("RESULT: RANKS BIT-IDENTICAL -- data-parallel logic OK")


if __name__ == "__main__":
    main()
