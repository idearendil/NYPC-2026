#!/usr/bin/env python3
"""Benchmark submit_bot.py's decide() by playing REAL judge games and profiling the
LEFT player (an instrumented submit_bot via bench_runner.py) against old_submit_bot.

Reports where the time goes -- forward simulator vs encoder vs actor vs critic vs
python glue -- aggregated over all turns of all games, plus per-turn latency stats
(mean / p50 / p95 / max and the one-time turn-1 precompute cost).

Usage:
    python bench_submit.py --games 3
    python bench_submit.py --games 5 --weights data.bin --depth 2 --cand 5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PY = r"D:\other_programs\anaconda3\envs\orbit\python.exe"  # has numpy

# leaf buckets (mutually exclusive) in the order we display them
LEAVES = [
    ("sim_step", "game step forward (_sim_step)"),
    ("encode",   "encoder / feature extraction"),
    ("actor",    "actor net (t1 + t2 matmuls)"),
    ("critic",   "critic net (value matmuls)"),
    ("ensure",   "one-time precompute (turn 1)"),
]


def run_game(py, seed, weights, depth, cand, timeout):
    """Play one game (bench_runner LEFT vs old_submit RIGHT); return timing payload."""
    fd, out_json = tempfile.mkstemp(suffix=".json"); os.close(fd)
    log = out_json + ".log"
    left = (f'"{py}" bench_runner.py --weights "{weights}" '
            f'--depth {depth} --cand {cand}')
    right = f'"{py}" old_submit_bot.py --weights "{weights}"'
    env = dict(os.environ, BENCH_OUT=out_json)
    cmd = [py, "testing-tool.py", "--seed", str(seed),
           "--exec1", left, "--exec2", right, "--log", log]
    try:
        subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"  seed {seed}: JUDGE_TIMEOUT", file=sys.stderr)
        return None
    if not os.path.exists(out_json) or os.path.getsize(out_json) == 0:
        print(f"  seed {seed}: no timing output (bot crashed? see {log})", file=sys.stderr)
        return None
    with open(out_json) as f:
        payload = json.load(f)
    os.remove(out_json)
    return payload


def merge(dst, payload):
    for k, (tot, cnt) in payload["stats"].items():
        r = dst["stats"].setdefault(k, [0.0, 0])
        r[0] += tot; r[1] += cnt
    dst["decide_ms"].extend(payload["decide_ms"])


def bucket_totals(stats):
    """Collapse raw keys into display buckets. actor = t1+t2."""
    g = lambda k: stats.get(k, [0.0, 0])[0]
    return {
        "sim_step": g("sim_step"),
        "encode":   g("encode"),
        "actor":    g("actor_t1") + g("actor_t2") + g("actor_t1b"),
        "critic":   g("critic") + g("critic_b"),
        "ensure":   g("ensure"),
    }


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def report(agg, n_games):
    stats = agg["stats"]
    dm = agg["decide_ms"]
    turns = len(dm)
    b = bucket_totals(stats)
    total_decide_s = sum(dm) / 1000.0
    leaf_sum = sum(b.values())
    glue = max(total_decide_s - leaf_sum, 0.0)

    print("\n" + "=" * 66)
    print(f"games: {n_games}   turns (LEFT decides): {turns}   "
          f"total decide: {total_decide_s*1000:.0f} ms")
    print("=" * 66)
    print(f"{'component':<34}{'total ms':>10}{'/turn ms':>10}{'% ':>8}")
    print("-" * 66)
    per = max(turns, 1)
    for key, label in LEAVES:
        t = b[key] * 1000.0
        print(f"{label:<34}{t:>10.1f}{t/per:>10.3f}{100*b[key]/max(total_decide_s,1e-9):>7.1f}%")
    gt = glue * 1000.0
    print(f"{'python glue (assembly/clone/etc)':<34}{gt:>10.1f}"
          f"{gt/per:>10.3f}{100*glue/max(total_decide_s,1e-9):>7.1f}%")
    print("-" * 66)
    tt = total_decide_s * 1000.0
    print(f"{'TOTAL decide':<34}{tt:>10.1f}{tt/per:>10.3f}{100.0:>7.1f}%")

    # call counts + per-call cost for the net (helps decide batching vs port)
    print("\ncall counts / per-call (per LEFT turn):")
    for k in ("encode", "sim_step", "actor_t1", "actor_t1b", "actor_t2",
              "critic", "critic_b", "select_action"):
        tot, cnt = stats.get(k, [0.0, 0])
        if cnt:
            print(f"  {k:<14} calls={cnt:>6} ({cnt/per:>5.1f}/turn)  "
                  f"{tot/cnt*1e3:.4f} ms/call")

    print("\nper-turn decide latency (ms):")
    print(f"  turn-1 (incl. precompute): {dm[0] if dm else 0:.1f}")
    steady = dm[1:] if len(dm) > 1 else dm
    print(f"  mean {sum(steady)/max(len(steady),1):.2f}   p50 {pct(steady,.5):.2f}   "
          f"p95 {pct(steady,.95):.2f}   max {max(steady) if steady else 0:.2f}")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=2000)
    ap.add_argument("--weights", default="data.bin")
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--depth", type=int, default=6)   # hard ceiling; horizon is time-adaptive
    ap.add_argument("--cand", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    weights = args.weights if os.path.isabs(args.weights) else os.path.join(HERE, args.weights)
    if not os.path.exists(weights):
        print(f"weights not found: {weights}", file=sys.stderr); sys.exit(1)
    try:
        import numpy as np
        with np.load(weights) as z:
            if not any(k.startswith("critic.") for k in z.files):
                print("WARNING: data.bin has NO critic -> search inactive; "
                      "you'll be profiling the single-action path only.\n", file=sys.stderr)
    except Exception:
        pass

    print(f"python : {args.python}")
    print(f"weights: {weights}")
    print(f"config : {args.games} games  depth={args.depth} cand={args.cand}")

    agg = {"stats": {}, "decide_ms": []}
    n_ok = 0
    for i in range(args.games):
        seed = args.base_seed + i
        print(f"running seed {seed} ...", flush=True)
        p = run_game(args.python, seed, weights, args.depth, args.cand, args.timeout)
        if p is not None:
            merge(agg, p); n_ok += 1

    if n_ok == 0:
        print("no successful games", file=sys.stderr); sys.exit(1)
    report(agg, n_ok)


if __name__ == "__main__":
    main()
