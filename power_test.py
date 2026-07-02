#!/usr/bin/env python3
"""Head-to-head win-rate test: current submit_bot.py (lookahead search) vs
old_submit_bot.py (single-action baseline), over random map seeds.

Each seed is played TWICE with sides swapped (new plays LEFT then RIGHT) so any
LEFT/RIGHT asymmetry cancels out. Wins/losses/draws are tallied from the judge's
`RESULT <outcome> <reason>` line. Both bots read the SAME data.bin -- the new bot
additionally uses the critic in it for its search; make sure data.bin was exported
with a critic (export_weights.py), otherwise the new bot silently falls back to the
single-action path and the two bots play identically (~50%).

Each side's engine is selectable via --new-engine / --old-engine:
  * baseline -> old_submit_bot.py (single-action python baseline)
  * py       -> submit_bot.py     (python lookahead-search bot)

Usage:
    python power_test.py --games 20                       # py-search vs baseline
    python power_test.py --games 20 --old-engine py       # py-search vs py-search
    python power_test.py --games 40 --weights data.bin --jobs 4 --cand 5
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PY = r"D:\other_programs\anaconda3\envs\orbit\python.exe"  # has numpy
RESULT_RE = re.compile(r"RESULT\s+(LEFT_WIN|RIGHT_WIN|DRAW)\s+(\S+)")


def build_cmd(engine, py, weights, depth, cand, budget, stochastic):
    """Command line for one contender. engine:
      * "baseline" -> old_submit_bot.py (single-action python baseline)
      * "py"       -> submit_bot.py     (python lookahead-search bot)"""
    if engine == "baseline":
        s = " --stochastic" if stochastic else ""
        return f'"{py}" old_submit_bot.py --weights "{weights}"{s}'
    dep = f" --depth {depth}" if depth and depth > 0 else ""   # else use submit_bot's default (unlimited)
    s = " --stochastic" if stochastic else ""
    return f'"{py}" submit_bot.py --weights "{weights}"{dep} --cand {cand} --budget {budget}{s}'


def run_one(py, seed, left_cmd, right_cmd, logdir, timeout):
    """Run one judge game; return (outcome, reason) or (None, err)."""
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, f"seed{seed}.log")
    cmd = [py, "testing-tool.py", "--seed", str(seed),
           "--exec1", left_cmd, "--exec2", right_cmd, "--log", log]
    try:
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "JUDGE_TIMEOUT"
    m = RESULT_RE.search(p.stdout) or RESULT_RE.search(p.stderr)
    if not m:
        return None, "NO_RESULT"
    return m.group(1), m.group(2)


def play_seed(py, seed, weights, depth, cand, budget, stochastic, logdir, timeout,
              new_engine="py", old_engine="baseline"):
    """Two games for one seed: new=LEFT then new=RIGHT. Returns a list of dicts."""
    ncmd = build_cmd(new_engine, py, weights, depth, cand, budget, stochastic)
    ocmd = build_cmd(old_engine, py, weights, depth, cand, budget, stochastic)
    out = []
    # game 1: new is LEFT (exec1), old is RIGHT (exec2)
    oc, rs = run_one(py, seed, ncmd, ocmd, logdir + "_A", timeout)
    out.append(_score(seed, "new=LEFT", oc, rs, new_side="LEFT"))
    # game 2: new is RIGHT (exec2), old is LEFT (exec1)
    oc, rs = run_one(py, seed, ocmd, ncmd, logdir + "_B", timeout)
    out.append(_score(seed, "new=RIGHT", oc, rs, new_side="RIGHT"))
    return out


def _score(seed, tag, outcome, reason, new_side):
    if outcome is None:
        return {"seed": seed, "tag": tag, "winner": "error", "reason": reason}
    if outcome == "DRAW":
        winner = "draw"
    else:
        won_side = "LEFT" if outcome == "LEFT_WIN" else "RIGHT"
        winner = "new" if won_side == new_side else "old"
    return {"seed": seed, "tag": tag, "winner": winner, "reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="number of seeds (2 games each)")
    ap.add_argument("--base-seed", type=int, default=3000)
    ap.add_argument("--weights", default="data.bin")
    ap.add_argument("--python", default=DEFAULT_PY, help="interpreter for the bots + judge")
    ap.add_argument("--depth", type=int, default=0,
                    help="lookahead ceiling; 0 = omit (submit_bot uses its default: unlimited/time-only)")
    ap.add_argument("--cand", type=int, default=5, help="number of action candidates per turn")
    ap.add_argument("--budget", type=int, default=70,
                    help="new bot's per-turn INTERNAL search budget in ms (kept <100ms judge cap "
                         "incl. IPC overhead; 90 spikes to ~110ms judge-side -> token drain -> WA)")
    ap.add_argument("--stochastic", action="store_true", default=True)
    ap.add_argument("--greedy", dest="stochastic", action="store_false",
                    help="run both bots in argmax mode")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel games (use 1 for accurate timing: the new bot's "
                         "adaptive depth measures wall-clock, so CPU contention shrinks it)")
    ap.add_argument("--timeout", type=int, default=600, help="per-game seconds")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--new-engine", choices=("py", "baseline"), default="py",
                    help="engine for the 'new' side: py=submit_bot.py, baseline=old_submit_bot.py")
    ap.add_argument("--old-engine", choices=("py", "baseline"), default="baseline",
                    help="engine for the 'old' side (default: baseline=old_submit_bot.py)")
    args = ap.parse_args()

    weights = args.weights if os.path.isabs(args.weights) else os.path.join(HERE, args.weights)
    if not os.path.exists(weights):
        print(f"weights not found: {weights}", file=sys.stderr); sys.exit(1)
    _warn_if_no_critic(weights)

    logdir = args.logdir or tempfile.mkdtemp(prefix="power_")
    os.makedirs(logdir, exist_ok=True)
    def _desc(engine):
        return {"py": "py (submit_bot.py)",
                "baseline": "baseline (old_submit_bot.py)"}[engine]
    print(f"python : {args.python}")
    print(f"weights: {weights}")
    print(f"new bot: {_desc(args.new_engine)}   vs   old bot: {_desc(args.old_engine)}")
    depth_desc = "unlimited(time-only)" if args.depth <= 0 else f"<={args.depth}"
    print(f"games  : {args.games} seeds x2 (sides swapped)  depth={depth_desc} "
          f"cand={args.cand} budget={args.budget}ms stochastic={args.stochastic}")
    print(f"logs   : {logdir}\n")

    seeds = [args.base_seed + i for i in range(args.games)]
    results = []

    def work(sd):
        return play_seed(args.python, sd, weights, args.depth, args.cand, args.budget,
                         args.stochastic, os.path.join(logdir, f"s{sd}"), args.timeout,
                         args.new_engine, args.old_engine)

    if args.jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for games in ex.map(work, seeds):
                results.extend(games); _report_partial(games)
    else:
        for sd in seeds:
            games = work(sd); results.extend(games); _report_partial(games)

    _summary(results)


def _warn_if_no_critic(weights):
    try:
        import numpy as np
        with np.load(weights) as z:
            if not any(k.startswith("critic.") for k in z.files):
                print("WARNING: data.bin has NO critic net -> new bot falls back to "
                      "single-action; expect ~50% (identical play).\n", file=sys.stderr)
    except Exception as e:
        print(f"(could not inspect weights: {e})", file=sys.stderr)


def _report_partial(games):
    for g in games:
        extra = f" [{g['reason']}]" if g["winner"] in ("error",) or g["reason"] not in (
            "HQ_DESTROYED", "TURN_LIMIT") else ""
        print(f"  seed {g['seed']:>6} {g['tag']:<9} -> {g['winner']}{extra}")


def _summary(results):
    n = len(results)
    new = sum(1 for r in results if r["winner"] == "new")
    old = sum(1 for r in results if r["winner"] == "old")
    draw = sum(1 for r in results if r["winner"] == "draw")
    err = sum(1 for r in results if r["winner"] == "error")
    decisive = new + old
    print("\n" + "=" * 48)
    print(f"games played : {n}")
    print(f"new wins     : {new}")
    print(f"old wins     : {old}")
    print(f"draws        : {draw}")
    if err:
        print(f"errors       : {err}  (see logs / reasons above)")
    if decisive:
        print(f"\nnew win rate (excl. draws/errors): {new/decisive:6.1%}")
    if n:
        print(f"new score    (win=1 draw=.5)     : {(new + 0.5*draw)/max(n-err,1):6.1%}")
    print("=" * 48)


if __name__ == "__main__":
    main()
