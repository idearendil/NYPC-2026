#!/usr/bin/env python3
"""Instrumented LEFT player for benchmarking submit_bot.py inside a REAL judge game.
Launched by bench_submit.py as the judge's exec1. It monkeypatches timing wrappers
onto Bot/Net, runs the normal protocol loop, and on FINISH dumps a per-component
timing breakdown (JSON) to the path in $BENCH_OUT.

Buckets are mutually exclusive leaves (they don't call each other):
    sim_step   -> forward simulator (_sim_step)
    encode     -> feature extraction (encode)
    actor      -> policy net matmuls (Net.t1 + Net.t2)
    critic     -> value net matmuls (Net.value)
    ensure     -> one-time precompute (travel turns + all-pairs next-hop), turn 1 only
Everything else in a decide() (candidate assembly, greedy alloc, softmax, cloning,
_to_commands, python glue) shows up as the "glue" residual.
"""
import functools
import json
import os
import sys
import time

import submit_bot as sb

STATS = {}          # key -> [total_seconds, calls]
DECIDE_MS = []      # per-turn decide wall time (ms)


def timed(fn, key):
    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            r = STATS.setdefault(key, [0.0, 0])
            r[0] += time.perf_counter() - t0
            r[1] += 1
    return w


def _decide_timed(fn):
    @functools.wraps(fn)
    def w(self, turn):
        t0 = time.perf_counter()
        try:
            return fn(self, turn)
        finally:
            DECIDE_MS.append((time.perf_counter() - t0) * 1000.0)
    return w


def install():
    sb.Bot.decide = _decide_timed(sb.Bot.decide)
    sb.Bot._ensure_ready = timed(sb.Bot._ensure_ready, "ensure")
    sb.Bot.encode = timed(sb.Bot.encode, "encode")
    sb.Bot._sim_step = timed(sb.Bot._sim_step, "sim_step")
    sb.Bot._select_action = timed(sb.Bot._select_action, "select_action")  # inclusive (ctx)
    sb.Net.t1 = timed(sb.Net.t1, "actor_t1")
    sb.Net.t2 = timed(sb.Net.t2, "actor_t2")
    sb.Net.value = timed(sb.Net.value, "critic")
    if hasattr(sb.Net, "t1_batch"):
        sb.Net.t1_batch = timed(sb.Net.t1_batch, "actor_t1b")
    if hasattr(sb.Net, "value_batch"):
        sb.Net.value_batch = timed(sb.Net.value_batch, "critic_b")


def dump():
    out = os.environ.get("BENCH_OUT")
    if not out:
        return
    payload = {"stats": STATS, "decide_ms": DECIDE_MS}
    with open(out, "w") as f:
        json.dump(payload, f)


def main():
    # same minimal argv contract as submit_bot.main()
    weights, stochastic = "data.bin", True
    search, n_cand, depth = True, 5, 2
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--stochastic":
            stochastic = True
        elif a == "--no-search":
            search = False
        elif a == "--cand":
            i += 1; n_cand = int(args[i])
        elif a == "--depth":
            i += 1; depth = int(args[i])
        elif a == "--weights":
            i += 1; weights = args[i]
        elif a.startswith("--weights="):
            weights = a.split("=", 1)[1]
        i += 1

    install()
    try:
        sb.Bot(weights, stochastic=stochastic, search=search,
               n_cand=n_cand, depth=depth).run()
    finally:
        dump()


if __name__ == "__main__":
    main()
