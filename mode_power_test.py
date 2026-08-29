#!/usr/bin/env python3
"""Head-to-head test: is a SINGLE checkpoint stronger playing greedy (argmax) or
stochastic (sampled) action selection? Both sides run `vanilla_bot.py` with the
SAME `--weights` file; only the `--greedy`/`--stochastic` flag differs.

Each seed is played TWICE with sides swapped (greedy=LEFT then greedy=RIGHT) so
any LEFT/RIGHT map asymmetry cancels out. Wins/losses/draws come from the
judge's `RESULT <outcome> <reason>` line.

Parallelism: each game is a real judge subprocess (testing-tool2.py) plus two
real bot subprocesses -- the actual work happens in those OS processes, not in
this script's Python. A ThreadPoolExecutor is enough to get true parallel
execution (subprocess.run releases the GIL while waiting), so this avoids the
extra process-spawn overhead a ProcessPoolExecutor would add on top for no
benefit.

--jobs defaults to cpu_count()//5, NOT cpu_count(): the judge enforces a
token-bucket turn budget (starts at 5 tokens, ~(tokens+1)*100ms per turn) that
a slow turn eats into, and each concurrent game costs 3 OS processes (1 judge +
2 bots). Oversubscribing the CPU makes an otherwise-cheap turn (normally
~0-20ms; the one expensive turn is turn 1's weight-load + travel-table
precompute, normally ~200-300ms) take long enough under contention to blow the
handshake/turn budget -- that shows up as a `WA` result with NO turns played at
all, which is a local-contention artifact, not a real skill difference. This
is made WORSE by a structural asymmetry in the judge itself: it checks LEFT's
handshake response first with a hard 1s wait, then RIGHT -- but RIGHT's reader
thread has been buffering output the whole time LEFT was being waited on, so
RIGHT effectively gets a longer window. Under contention LEFT fails first and
far more often (~90%+ of the WAs we measured were LEFT). Not fixable here --
it's the reference judge's own handshake loop, not vanilla_bot.py or this
script. Pinning BLAS threads alone does NOT fix it (tried; no effect) --
only running fewer concurrent games does. Measured on this machine (16 cores):
jobs=3 (9 procs) -> ~5% WA, clean enough to trust; jobs=5 (15 procs) -> ~25%;
jobs=8+ -> most games. If you raise --jobs, watch the WA rate in the output --
if it's high, the win-rate numbers below are meaningless and you need to lower
--jobs and rerun. (This same asymmetry is also a real submission-bot risk, not
just a test-harness quirk: it means turn-1 init speed matters more than a
"1000ms handshake budget" number alone suggests, since LEFT gets no grace
period under real contention either.)

Usage:
    python mode_power_test.py --games 30                 # data.bin, jobs=cpu_count
    python mode_power_test.py --games 50 --weights ckpt_export.bin --jobs 12
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PY = r"D:\applications\anaconda3\envs\nypc\python.exe"
RESULT_RE = re.compile(r"RESULT\s+(LEFT_WIN|RIGHT_WIN|DRAW)\s+(\S+)")
# numpy/BLAS spins up one thread PER CORE by default on import. vanilla_bot's
# matmuls are tiny (single-token forward pass), so that's pure overhead even
# in a single game -- and with N games running at once it's N processes each
# trying to grab every core, which is what actually blows the judge's turn
# budget (far more than the process count alone would). Pin every child to 1
# BLAS thread; each process still gets its own real core from the OS scheduler.
_SUBPROC_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def build_cmd(py, weights, stochastic):
    """Command line for one contender: vanilla_bot in greedy or stochastic mode."""
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
                           timeout=timeout, env=_SUBPROC_ENV)
    except subprocess.TimeoutExpired:
        return None, "JUDGE_TIMEOUT"
    m = RESULT_RE.search(p.stdout) or RESULT_RE.search(p.stderr)
    if not m:
        return None, "NO_RESULT"
    return m.group(1), m.group(2)


def play_seed(py, seed, weights, logdir, timeout):
    """Two games for one seed: greedy=LEFT then greedy=RIGHT. Returns a list of dicts."""
    gcmd = build_cmd(py, weights, stochastic=False)
    scmd = build_cmd(py, weights, stochastic=True)
    out = []
    # game 1: greedy is LEFT (exec1), stochastic is RIGHT (exec2)
    oc, rs = run_one(py, seed, gcmd, scmd, logdir + "_A", timeout)
    out.append(_score(seed, "greedy=LEFT", oc, rs, greedy_side="LEFT"))
    # game 2: greedy is RIGHT (exec2), stochastic is LEFT (exec1)
    oc, rs = run_one(py, seed, scmd, gcmd, logdir + "_B", timeout)
    out.append(_score(seed, "greedy=RIGHT", oc, rs, greedy_side="RIGHT"))
    return out


def _score(seed, tag, outcome, reason, greedy_side):
    if outcome is None:
        return {"seed": seed, "tag": tag, "winner": "error", "reason": reason}
    if outcome == "DRAW":
        winner = "draw"
    else:
        won_side = "LEFT" if outcome == "LEFT_WIN" else "RIGHT"
        winner = "greedy" if won_side == greedy_side else "stochastic"
    return {"seed": seed, "tag": tag, "winner": winner, "reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="number of seeds (2 games each)")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--weights", default="data.bin", help="weights both sides load")
    ap.add_argument("--python", default=DEFAULT_PY, help="interpreter for the bots + judge")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel games (default: os.cpu_count()//5 -- see "
                         "module docstring; each game costs 3 OS processes and "
                         "there's a LEFT-vs-RIGHT handshake asymmetry that "
                         "makes WA worse than raw process count alone predicts)")
    ap.add_argument("--timeout", type=int, default=600, help="per-game seconds")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) // 5)

    weights = args.weights if os.path.isabs(args.weights) else os.path.join(HERE, args.weights)
    if not os.path.exists(weights):
        print(f"weights not found: {weights}", file=sys.stderr); sys.exit(1)

    logdir = args.logdir or tempfile.mkdtemp(prefix="modepower_")
    os.makedirs(logdir, exist_ok=True)
    print(f"python  : {args.python}")
    print(f"weights : {weights}  (same file both sides)")
    print(f"games   : {args.games} seeds x2 (sides swapped)  jobs={jobs}")
    print(f"logs    : {logdir}\n")

    seeds = [args.base_seed + i for i in range(args.games)]
    results = []

    def work(sd):
        return play_seed(args.python, sd, weights, os.path.join(logdir, f"s{sd}"), args.timeout)

    if jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
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
        print(f"  seed {g['seed']:>6} {g['tag']:<12} -> {g['winner']}{extra}")


def _summary(results):
    n = len(results)
    greedy = sum(1 for r in results if r["winner"] == "greedy")
    stoch = sum(1 for r in results if r["winner"] == "stochastic")
    draw = sum(1 for r in results if r["winner"] == "draw")
    err = sum(1 for r in results if r["winner"] == "error")
    wa = sum(1 for r in results if r.get("reason") == "WA")
    decisive = greedy + stoch
    print("\n" + "=" * 52)
    print(f"games played      : {n}")
    print(f"greedy wins       : {greedy}")
    print(f"stochastic wins   : {stoch}")
    print(f"draws             : {draw}")
    if err:
        print(f"errors            : {err}  (see logs / reasons above)")
    if wa and n and wa / n > 0.15:
        print(f"\n*** {wa}/{n} games ended in WA (disqualification) -- that's high. ***\n"
              f"*** Likely CPU contention from --jobs, not a real bug -- see the  ***\n"
              f"*** module docstring. Rerun with a lower --jobs before trusting   ***\n"
              f"*** the win rate below.                                          ***")
    if decisive:
        wr = greedy / decisive
        # normal-approx 95% CI on the decisive-game win rate -- just enough to
        # judge whether the gap could plausibly be noise at this sample size.
        se = (wr * (1 - wr) / decisive) ** 0.5
        lo, hi = max(0.0, wr - 1.96 * se), min(1.0, wr + 1.96 * se)
        print(f"\ngreedy win rate (excl. draws/errors): {wr:6.1%}  "
              f"95% CI [{lo:.1%}, {hi:.1%}]  (n={decisive})")
        verdict = ("greedy stronger" if lo > 0.5 else
                   "stochastic stronger" if hi < 0.5 else
                   "no significant difference yet -- run more games")
        print(f"verdict: {verdict}")
    if n:
        print(f"greedy score (win=1 draw=.5)        : {(greedy + 0.5*draw)/max(n-err,1):6.1%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
