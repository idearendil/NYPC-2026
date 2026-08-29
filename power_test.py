#!/usr/bin/env python3
"""Head-to-head win-rate test between two vanilla_bot weight files, over random map
seeds.

Each seed is played TWICE with sides swapped (new plays LEFT then RIGHT) so any
LEFT/RIGHT asymmetry cancels out. Wins/losses/draws are tallied from the judge's
`RESULT <outcome> <reason>` line. Both sides run the SAME bot (`vanilla_bot.py`);
what differs is the weights each one loads -- so this measures "is the new
checkpoint stronger than the old one?".

Usage:
    python power_test.py --games 20                              # data.bin vs data.bin (sanity: ~50%)
    python power_test.py --games 40 --old-weights old.bin        # new checkpoint vs an archived one
    python power_test.py --games 40 --jobs 4 --greedy
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PY = r"D:\other_programs\anaconda3\envs\nypc\python.exe"  # has numpy
RESULT_RE = re.compile(r"RESULT\s+(LEFT_WIN|RIGHT_WIN|DRAW)\s+(\S+)")


def build_cmd(py, weights, stochastic):
    """Command line for one contender: vanilla_bot with the given weight file."""
    s = " --stochastic" if stochastic else " --greedy"
    return f'"{py}" vanilla_bot.py --weights "{weights}"{s}'


def run_one(py, seed, left_cmd, right_cmd, logdir, timeout):
    """Run one judge game; return (outcome, reason) or (None, err)."""
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, f"seed{seed}.log")
    cmd = [py, "testing-tool2.py", "--seed", str(seed),
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


def play_seed(py, seed, new_weights, old_weights, stochastic, logdir, timeout):
    """Two games for one seed: new=LEFT then new=RIGHT. Returns a list of dicts."""
    ncmd = build_cmd(py, new_weights, stochastic)
    ocmd = build_cmd(py, old_weights, stochastic)
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
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--weights", default="data.bin",
                    help="weights for the 'new' side (default: data.bin)")
    ap.add_argument("--old-weights", default=None,
                    help="weights for the 'old' side (default: same as --weights)")
    ap.add_argument("--python", default=DEFAULT_PY, help="interpreter for the bots + judge")
    ap.add_argument("--stochastic", action="store_true", default=True)
    ap.add_argument("--greedy", dest="stochastic", action="store_false",
                    help="run both bots in argmax mode")
    ap.add_argument("--jobs", type=int, default=1, help="parallel games")
    ap.add_argument("--timeout", type=int, default=600, help="per-game seconds")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    def _resolve(w):
        return w if os.path.isabs(w) else os.path.join(HERE, w)
    new_w = _resolve(args.weights)
    old_w = _resolve(args.old_weights or args.weights)
    for w in (new_w, old_w):
        if not os.path.exists(w):
            print(f"weights not found: {w}", file=sys.stderr); sys.exit(1)

    logdir = args.logdir or tempfile.mkdtemp(prefix="power_")
    os.makedirs(logdir, exist_ok=True)
    print(f"python : {args.python}")
    print(f"new bot: vanilla_bot.py + {new_w}")
    print(f"old bot: vanilla_bot.py + {old_w}")
    if new_w == old_w:
        print("         (identical weights -> this is a ~50% sanity run)")
    print(f"games  : {args.games} seeds x2 (sides swapped)  stochastic={args.stochastic}")
    print(f"logs   : {logdir}\n")

    seeds = [args.base_seed + i for i in range(args.games)]
    results = []

    def work(sd):
        return play_seed(args.python, sd, new_w, old_w, args.stochastic,
                         os.path.join(logdir, f"s{sd}"), args.timeout)

    if args.jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for games in ex.map(work, seeds):
                results.extend(games); _report_partial(games)
    else:
        for sd in seeds:
            games = work(sd); results.extend(games); _report_partial(games)

    _summary(results)


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
