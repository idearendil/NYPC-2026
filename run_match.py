#!/usr/bin/env python3
"""Run a vanilla_bot.py vs vanilla_bot.py match and write a replay log.

Thin wrapper around testing-tool.py (the judge) so you don't have to type the
nested exec commands by hand. Uses the *current* Python interpreter for both the
judge and the two bots, so just run it with the interpreter that has numpy:

    python run_match.py                      # LEFT=argmax, RIGHT=stochastic, random board
    python run_match.py --seed 42            # fixed board
    python run_match.py --left stochastic    # make LEFT stochastic too
    python run_match.py --right argmax       # make RIGHT argmax
    python run_match.py --np 40 --kp 6       # fix board size
    python run_match.py --log mygame.log     # choose output file

Defaults: LEFT argmax, RIGHT stochastic, random seed, log -> replay.log
"""
import argparse
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def bot_cmd(py, weights, mode):
    cmd = f'"{py}" "{os.path.join(HERE, "vanilla_bot.py")}" --weights "{weights}"'
    # vanilla_bot CLI: --greedy = argmax, --stochastic = sample (default stochastic)
    cmd += " --greedy" if mode == "argmax" else " --stochastic"
    return cmd


def main():
    ap = argparse.ArgumentParser(description="vanilla_bot vs vanilla_bot -> replay log")
    ap.add_argument("--left", choices=["argmax", "stochastic"], default="argmax",
                    help="LEFT player policy (default: argmax)")
    ap.add_argument("--right", choices=["argmax", "stochastic"], default="stochastic",
                    help="RIGHT player policy (default: stochastic)")
    ap.add_argument("--weights", default=os.path.join(HERE, "data.bin"),
                    help="weights file for both bots (default: data.bin)")
    ap.add_argument("--log", default=os.path.join(HERE, "replay.log"),
                    help="output replay log (default: replay.log)")
    ap.add_argument("--seed", type=int, default=None,
                    help="map seed (default: random)")
    ap.add_argument("--np", type=int, default=None, help="zones per half (NP)")
    ap.add_argument("--kp", type=int, default=None, help="neutral strongholds per half (KP)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    py = sys.executable

    cmd = [
        py, os.path.join(HERE, "testing-tool2.py"),
        "--seed", str(seed),
        "-l", args.log,
        "-a", bot_cmd(py, args.weights, args.left),
        "-b", bot_cmd(py, args.weights, args.right),
    ]
    if args.np is not None:
        cmd += ["--NP", str(args.np)]
    if args.kp is not None:
        cmd += ["--KP", str(args.kp)]

    print(f"LEFT={args.left}  RIGHT={args.right}  seed={seed}")
    print(f"log -> {args.log}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
