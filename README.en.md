<p align="right">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/README-English-1f6feb?style=for-the-badge"></a>
  <a href="README.md"><img alt="한국어" src="https://img.shields.io/badge/README-%ED%95%9C%EA%B5%AD%EC%96%B4-6e7681?style=for-the-badge"></a>
</p>

# NYPC 2026 — self-play RL agent (finals)

<img alt="1st place" src="https://img.shields.io/badge/NYPC%202026%20Master%20Track%20Finals-1st%20place-f5b400?style=for-the-badge">

**The 1st-place solution of the NYPC 2026 Master Track final round.**

A reinforcement-learning agent for the NYPC 2026 strategy game, trained end-to-end
with self-play PPO on a custom batched GPU re-implementation of the judge, and
submitted as a **numpy-only** bot with no torch dependency.

This is the `main` branch: the **finals** ruleset (400 days, fog of war, larger
maps). The qualification-round code lives on the
[`qualification_round`](../../tree/qualification_round) branch — same architecture,
different rules and a separately trained network.

Contest write-ups: [qualification](docs/Qualification%20Round%20Replay.pdf) ·
[finals](docs/Final%20Round%20Replay.pdf)

| Piece | File | What it is |
|---|---|---|
| Environment | `src/fast_env.py` | The judge's dynamics re-written as batched tensor ops. `B` games step in parallel on the GPU, bit-exact against `judge/testing-tool2.py`. |
| Trainer | `src/ppo_selfplay.py` | Self-play PPO over that env: two-stage factored actor, privileged critic, opponent pool, multi-GPU. |
| Submission | `src/vanilla_bot.py` | The trained actor, re-implemented in pure numpy. One inference per turn, no search. Loads `data.bin`. |

## The game

Two players fight over a randomly generated graph of regions (`N` in 181–249,
`K` strongholds ≈ √N). Each side starts with a headquarters, 3 warriors and
750 gold, and has **400 days**.

- **Buildings.** Bases can be built on strongholds (500 gold, up to level 3).
  The HQ upgrades to level 5 (600 / 1000 / 2000 / 3000 gold). Levels raise HP,
  turret damage, the training cap and the *work cap* (how many warriors can earn
  gold there).
- **Economy.** Each working warrior earns 15 gold/day; every warrior costs 2
  gold/day in upkeep. Training a warrior costs 120, moving one costs 10.
- **Fog of war.** You see enemy units and buildings only within **2 hops** of one
  of your own warriors or buildings. Everything else must be remembered and
  guessed.
- **Winning.** Destroy the enemy HQ, or be ahead on building HP when day 400 hits.

`judge/testing-tool2.py` is the finals judge; see
[docs/testing-tool.md](docs/testing-tool.md) for its CLI, map format and log format.

## Quickstart

Run everything from the **repo root**. Install torch **first**, matching your
CUDA build, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick your CUDA
pip install -r requirements.txt
```

```bash
# 1. sanity-check the whole pipeline end to end (tiny nets, a couple of iterations)
python src/ppo_selfplay.py --smoke --ckpt smoke.pt

# 2. train for real (edit config.yaml first; --gpus N for data parallel)
python src/ppo_selfplay.py --config config.yaml --gpus 1

# 3. convert the checkpoint into the torch-free weights the bot loads
python src/export_weights.py --ckpt checkpoint.pt --out data.bin

# 4. verify the numpy bot reproduces the torch pipeline (features + forward pass)
python tests/verify_np_bot.py

# 5. watch it play, and measure it
python tools/run_match.py --seed 42                     # writes replay.log
python tools/power_test.py --games 40 --old-weights old.bin
```

Training logs to Weights & Biases when `use_wandb: true` in `config.yaml`
(`--no-wandb` to disable). Credentials come from `WANDB_API_KEY` in the
environment or from `wandb login` — nothing is stored in this repo.

## How it works

**Batched environment.** RL needs far more games than a subprocess-per-game judge
can produce, so `fast_env.py` re-implements the rules as tensor ops over `B`
simultaneous games. It is verified **bit-exact** against the judge — gold, every
building's owner/kind/level/HP, and every warrior — both at end-of-turn
(`test_fast_env.py`) and after every individual phase (`test_phases.py`). The
rollout is launch-bound, so throughput scales almost linearly with `B`
(`B: 12288` in the shipped config). Details in [docs/fast_env.md](docs/fast_env.md).

**Factored action space.** A turn is a *set* of commands, not one choice, so the
actor is split in two:

- **T1** — a 3-block transformer over one token per stronghold (32 token features
  + 14 global features, all log1p / normalised). Its 5-dim head gives a per-token
  build Bernoulli and, mask-averaged over tokens, a 4-way "what to train / upgrade"
  softmax.
- **T2** — a 2-block transformer re-run per move source, taking T1's token output
  plus 8 extra features, producing a softmax over tokens: where to send this
  group.

Gold is enforced twice: an affordability mask *before* sampling, then a greedy
allocation of the remaining gold afterwards (build → move → train). Commands
dropped by the greedy step still count as taken for PPO; commands masked to
probability 0 never do.

**Critic and auxiliary tasks.** A separate encoder of the same shape predicts the
value. Both actor and critic carry a 7-dim auxiliary head — per stronghold, the
enemy garrison reachable within 1–5 turns, plus a global estimate of the
opponent's hidden gold. The auxiliary targets are training-only supervision that
shapes the encoders; the submission never runs them.

**Self-play.** The agent (LEFT) plays a pool of opponents (RIGHT) sampled toward
whichever is currently hardest, by EMA win rate. The first three pool slots are
fixed scripted bots — batched, vectorised ports of `bots/final_rush_bot.py`,
`bots/final_rush_bot2.py` and `bots/final_defence_bot.py` — which never get
evicted, so the pool cannot collapse into self-play homogeneity. When the agent
beats every opponent above `pool_add_threshold`, it snapshots itself into the
pool; every `perm_snapshot_every` iterations that snapshot is made permanent.

**Submission.** torch's import alone (~2.3 s) does not fit the judge's 1 s
handshake, so `export_weights.py` flattens the actor into a numpy `.npz`
(`data.bin`) and `vanilla_bot.py` re-implements the forward pass — layernorm,
multi-head attention, GELU — in numpy. It also reconstructs the hidden state the
protocol never sends: a per-region belief about the opponent (kind, level, HP,
garrison, and the age of that observation) refreshed inside the locally computed
vision set and aged outside it, mirroring `fast_env`'s fog exactly so the bot sees
the same features it was trained on.

## Repo map

```
config.yaml              all training hyperparameters
requirements.txt

src/                     the project itself (this dir is the import path)
  fast_env.py            batched GPU environment (+ observation encoder)
  map_gen.py             background random-map generation for training
  ppo_selfplay.py        self-play PPO trainer (nets, rollout, pool, update loop)
  export_weights.py      checkpoint.pt -> data.bin (torch-free)
  vanilla_bot.py         THE submission bot (numpy only)

judge/                   the organisers' simulator, as shipped
  testing-tool2.py       the finals judge
  config.ini             judge config example
  sample-code.py         the organisers' protocol sample bot

bots/                    rule-based bots
  final_rush_bot.py      scripted opponents; ppo_selfplay contains batched ports
  final_rush_bot2.py       of these three as the fixed part of the opponent pool
  final_defence_bot.py
  basic_bot.py           qualification-era baselines, still runnable as sparring
  rush_bot.py              partners against the judge
  japper_bot.py

tools/                   run / evaluate
  run_match.py           one bot-vs-bot match -> replay log
  power_test.py          win rate between two weight files (sides swapped)
  mode_power_test.py     win rate of greedy vs stochastic action selection
  benchmark_env.py       env throughput benchmark

tests/                   verification (all run directly as scripts)
  test_fast_env.py       bit-exact end-of-turn parity vs the judge
  test_phases.py         bit-exact parity after every phase
  test_observe.py        observation shapes, token masks
  test_encoder.py        independent recompute of every encoder feature
  verify_np_bot.py       numpy bot == torch pipeline

docs/                    judge docs, the fast_env deep-dive, contest write-ups
```

Scripts in `tests/` and `tools/` put `src/` on `sys.path` themselves, so there is
nothing to install — just run them from the repo root.

## Tests

```bash
python tests/test_fast_env.py    # bit-exact end-of-turn parity vs the judge (slow)
python tests/test_phases.py      # bit-exact parity after every phase, cpu + cuda
python tests/test_observe.py     # observation shapes, mixed-size token masks
python tests/test_encoder.py     # independent recompute of every encoder feature
python tests/verify_np_bot.py    # numpy bot == torch pipeline (needs checkpoint.pt + data.bin)
```

If you change the network, `verify_np_bot.py` alone is **not** enough — it checks
the forward pass, not action selection. Also compare `ppo_selfplay.sample_policy`
against `vanilla_bot._select_action` on the same states with both forced
deterministic: a divergence in a mask, or in a mask's *polarity*, shows up nowhere
else. (Give the states plenty of gold when you do — the greedy allocator funds a
randomly permuted subset when gold binds, so a budget-bound state disagrees for
legitimate reasons.)

## Notes

- No weights are committed. Train your own, or the bot has nothing to load.
- Everything runs on one GPU; `--gpus N` is plain data parallelism (the same
  data per iteration, split across ranks). `minibatch` must be divisible by the
  GPU count.
- `python src/ppo_selfplay.py --smoke` is the fast end-to-end check: tiny nets, a
  handful of games, two iterations, no wandb.
