# NYPC 2026 — self-play PPO agent

A reinforcement-learning bot for the NYPC 2026 strategy game, trained end to end by
**self-play PPO** on a **batched GPU re-implementation of the official judge**, and
shipped as a **numpy-only** submission that runs one network inference per turn.

Three pieces:

| piece | file | what it is |
|---|---|---|
| environment | `fast_env.py` | the game rules re-written as batched tensor ops, bit-exact vs the judge |
| trainer | `ppo_selfplay.py` | self-play PPO with an opponent pool, exploiters, multi-GPU |
| submission | `vanilla_bot.py` | the trained actor, ported to numpy, speaking the judge's stdio protocol |

No weights are committed — train your own (see below), or point `export_weights.py`
at any checkpoint.

## The game, briefly

Two players (LEFT / RIGHT) fight over a graph of `N` regions (51–109), `K` of which
are **strongholds** (거점) — the only places a building can stand. Each side starts
with an HQ at its end of the map. Every day you may, subject to gold:

- **UPGRADE** a stronghold (build a base for 300, or level it up; the HQ goes to level 5),
- **TRAIN** warriors at your buildings (120 each, capped by building level),
- **MOVE** warriors between regions (10 each; travel takes several days).

Warriors garrisoned at a building up to its `work_cap` become labourers and earn
15 gold/day; the rest fight. Warriors take turret damage, fight enemies in the same
region, siege buildings, and starve if upkeep is not paid. The game ends when an HQ
falls or on day 200 (HP tiebreak).

The judge (`testing-tool.py`) and the protocol reference (`sample-code.py`) are the
organisers' files, included so matches and the parity tests can be run locally; the
protocol is documented in [`docs/testing-tool.md`](docs/testing-tool.md).

## Quickstart

```bash
# 1. install torch for YOUR cuda version first, then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. sanity-check the whole loop (a couple of minutes, CPU is fine)
python ppo_selfplay.py --smoke

# 3. train -> checkpoint.pt (edit config.yaml; --gpus N for data parallel)
python ppo_selfplay.py --config config.yaml

# 4. convert the actor to a torch-free weight file
python export_weights.py --ckpt checkpoint.pt --out data.bin

# 5. check the numpy bot reproduces the torch pipeline
python verify_np_bot.py --weights data.bin --ckpt checkpoint.pt

# 6. play it
python run_match.py --seed 42                                  # bot vs bot -> replay.log
python power_test.py --games 40 --old-weights data_prev.bin    # A/B two weight files
```

Run a single game against the judge directly:

```bash
python testing-tool.py --seed 1 --NP 40 --KP 6 \
  -a "python vanilla_bot.py --weights data.bin --stochastic" \
  -b "python sample-code.py P2"
```

## How it works

**Environment.** `fast_env.FastEnv` steps `B` games in parallel as tensor ops on the
GPU — the rollout is launch-bound, so throughput scales almost linearly with `B`
(256 → 2048 is ~5× the env-steps/s). Games of different `N`/`K` share a batch via
padding. Every phase of a day is **bit-exact** against `testing-tool.py`
(`test_fast_env.py`, `test_phases.py`). Maps come from background worker processes
(`map_gen.py`) so episode regeneration never stalls the rollout. Design notes:
[`docs/fast_env.md`](docs/fast_env.md).

**Policy.** The action is factored into two transformer heads over 거점-tokens:

- **T1** (3 blocks) sees per-token features (31) ++ global features (16) and emits
  5 numbers per token: a per-token *build* Bernoulli logit, and a 4-way *train*
  contribution that is mask-averaged across tokens.
- **T2** (2 blocks) runs once per move-source over T1's token embeddings ++ 8 extra
  source-relative features, emitting for each candidate target a *move-target* logit
  (softmax over tokens) and a *full-mobilisation* logit (send the source's labourers
  too, not just the surplus beyond `work_cap`).
- A **critic** shaped like T1 produces the value. Actor and critic both carry an
  auxiliary head predicting next-turn enemy pressure per 거점 and the opponent's
  hidden gold; the actor's own gold prediction is fed back to it as a feature.

Gold is enforced twice: affordability masks before sampling, then greedy allocation
(build → move → train) of whatever is left. Reward is ±10 at the end of the game only.

**Self-play.** Opponents are drawn from a pool, weighted toward the ones the agent
beats least. The pool always holds two scripted bots (`rush_bot.py`, `japper_bot.py`,
ported to batched form inside the trainer) and grows with snapshots of the agent as
it improves. Every `perm_snapshot_every` iterations the current agent is frozen in
permanently; every `exploiter_every` iterations training pauses to raise a dedicated
**exploiter** against a frozen copy of the main agent, and whatever hole it finds
joins the pool for good. Hyperparameters follow a **phase schedule** (lr / epochs /
entropy / rollout size step down as training progresses), resolved from the iteration
number so a resumed run lands in the right phase.

**Submission.** `vanilla_bot.py` imports only `math`, `sys` and `numpy` — torch's
~2.3 s import does not fit the 1 s handshake, numpy's ~0.15 s does. The encoder and
the transformer forward are hand-ported to numpy and verified numerically against the
torch path (`verify_np_bot.py`: features to ~1e-6, net outputs to ~1e-5, every action
mask exact). The protocol never reveals the opponent's gold or its warriors'
destinations, so the bot reconstructs both from visible play — and the trainer feeds
the *same* reconstruction to the net, so there is no train/inference gap.

## Files

```
fast_env.py         batched GPU env (FastEnv, MapBatch, observe)
map_gen.py          background random-map generation
ppo_selfplay.py     self-play PPO trainer (pool, exploiters, phases, multi-GPU)
config.yaml         all training hyperparameters, commented
export_weights.py   checkpoint.pt -> data.bin (torch-free npz)
vanilla_bot.py      the submission bot (numpy only)

run_match.py        bot vs bot -> replay log
power_test.py       head-to-head win rate between two weight files
benchmark_env.py    env throughput benchmark
rush_bot.py         scripted opponent: rush
japper_bot.py       scripted opponent: expand-and-wave
basic_bot.py        simple heuristic baseline

testing-tool.py     official judge (organisers')
sample-code.py      official protocol reference (organisers')
config.ini          judge config file

test_fast_env.py    bit-exact end-of-day parity, env vs judge
test_phases.py      bit-exact per-phase parity
test_observe.py     observation shapes + padding masks
test_encoder.py     independent recompute of every encoder feature
verify_np_bot.py    numpy bot vs torch pipeline
```

## Tests

```bash
python test_fast_env.py     # bit-exact vs the judge (slow)
python test_phases.py
python test_observe.py
python test_encoder.py
python verify_np_bot.py     # needs a matching checkpoint.pt + data.bin pair
```

If you change the network, `verify_np_bot.py` alone is **not** enough — it checks the
forward pass, not action selection. Also compare `ppo_selfplay.sample_policy` against
`vanilla_bot._select_action` on the same states with both forced deterministic: a
divergence in a mask, or in a mask's polarity, shows up nowhere else.

## Notes

- Attention is a hand-written multi-head implementation, not `nn.MultiheadAttention`:
  the fused CUDA path emits all-NaN rows when a batch mixes padded and unpadded token
  sequences, which is exactly the mixed-map-size case here.
- `wandb` logging is on by default; pass `--no-wandb` to disable it.
- Training was run on Linux across several GPUs. `--gpus N` splits `B` and `minibatch`,
  so an N-GPU run collects data identical to a 1-GPU run with the same config.
