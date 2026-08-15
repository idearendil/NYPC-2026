# FastEnv — GPU-accelerated batched env for RL

A batched, GPU (CUDA) re-implementation of `testing-tool.py`'s game dynamics for
fast RL data collection. `B` games are stepped in parallel as tensor ops.

## Files
- `fast_env.py` — the env (`FastEnv`, `MapBatch`, `observe`).
- `map_gen.py` — background (multi-process) random-map generation for training.
- `test_fast_env.py` — **bit-exact end-of-turn parity** (uniform + mixed sizes).
- `test_phases.py` — **bit-exact per-phase parity** (compares after every stage).
- `test_observe.py` — observation shapes + mixed-size token-mask checks.
- `test_encoder.py` — **independent recompute of every encoder feature**.
- `benchmark_env.py` — throughput benchmark.

Run with the env that has torch+CUDA:
```
D:/other_programs/anaconda3/envs/nypc/python.exe test_fast_env.py
D:/other_programs/anaconda3/envs/nypc/python.exe benchmark_env.py
```

## Correctness
The dynamics are a faithful copy of the original, verified to be **bit-exact**
(gold, every building's owner/kind/level/hp, and **every warrior by suffix**
incl. region/hp/moving):
- **End-of-turn** parity (`test_fast_env.py`): random play, 200 turns,
  N∈{51,61,81,109} and **mixed-size batches**, on **CPU and CUDA**; plus a
  deterministic economy run covering the **build / upgrade / heal** branches.
- **Per-phase** parity (`test_phases.py`): compares the full state after **each
  stage** — build, move, train, movement, spawn, combat+siege, work, upkeep —
  every turn, on uniform and mixed batches, CPU and CUDA. This pins any
  divergence to a single phase.

Map generation reuses the original `generate_map` verbatim, so boards come from
the exact same distribution and generation rules.

### Encoder verification (`test_encoder.py`)
Every observation feature is recomputed **independently** from the raw reference
state and compared to `observe()`, across many states and both sides:
- per-region scalars (warrior counts, base/HQ levels, turret, work-cap, building
  HP), `surplus`, `stat_hp`;
- **my arrivals at a 거점 in exactly 1..5 turns**;
- **enemy warriors able to reach a 거점 within 1..5 turns** (incl. warriors in
  transit / non-거점 regions);
- **turns-distance to every other 거점**;
- global features (totals, gold, HQ levels, level sums).

The travel-time-in-turns cache is itself checked against a direct simulation of
the game's movement rule (`dijkstra_from(target)` + min `edge_weight+dist`
next-hop, ties → smaller id). Coverage asserts the travel-based features actually
take nonzero values (not vacuously passing).

## Throughput (RTX 3070 Ti)
Single-thread original simulator ≈ **2k env-steps/s**.

| config | env-steps/s | speedup |
|---|---|---|
| B=4096, full pool (bit-exact) | ~79k | 40× |
| B=2048, N=109, full pool | ~66k | 33× |
| B=4096, `max_warriors_per_side=128` | ~197k | 100× |
| B=8192, `max_warriors_per_side=128` | ~263k | 130× |

`setup` (per `FastEnv` construction) runs the all-pairs path/precompute and takes
~10–37s for large batches; it is a **one-time** cost — reuse one `FastEnv` and
call `reset()` between episodes (cheap).

## Usage
```python
import fast_env as fe
tt = fe.tt
maps = [tt.read_map(tt.generate_map(tt.XoShiro256(s), NP=40, KP=6)) for s in range(B)]
env = fe.FastEnv(maps, device="cuda")          # full pool = bit-exact
# env = fe.FastEnv(maps, device="cuda", max_warriors_per_side=128)  # faster

act = {
  'left':  {'build': bool[B,N], 'move': long[B,N] (-1=none), 'train': long[B]},
  'right': {...},
}
env.step(act)
tokens, glob, info = env.observe(side=0)         # side 0=left, 1=right
done = env.hq_alive()                             # [B,2] bool
```

### Action space (matches the spec you gave)
- `build` `[B,N]` bool — regions to build/upgrade/heal (build candidates).
- `move`  `[B,N]` long — `move[b,src]=tgt` (or `-1`). Encodes "one source → one
  target". For a source with your building, the env keeps the `work_cap`
  lowest-HP warriors (built/upgraded **before** moves, so the increased work_cap
  applies) and moves the rest; otherwise it moves all stationary warriors there.
- `mobilize` `[B,N]` bool *(optional)* — **full mobilisation**: for these regions
  the keep-cap drops to 0, so the commanded move takes *every* stationary warrior,
  labourers included. Omit the key (or pass `None`) for the classic behaviour.
- `train` `[B]` long — 0..3 (clamped to HQ train cap & affordable gold).

`info` returns: `gold`, `hq_level`, `token_ids`, `build_candidates [B,T]`,
`move_sources [B,T]`.

### Encoder output
`observe()` returns transformer-ready tensors with one **token per 거점**
(strongholds + both HQs, `T=K+2`):
- `tokens [B, T, 39]`: the 14 per-region scalar features, then arrivals of your
  movers in 1..5 turns (5), enemy reach in 1..5 turns (5), and turns-distance to
  every token (T). Travel times are precomputed at reset.
- `glob [B, 11]`: day, both warrior totals, both HQ levels, both golds, both
  previous-turn incomes, both building-level sums.

## Self-play PPO (`ppo_selfplay.py`)
Trains an agent (LEFT) against an **opponent pool** (RIGHT). Reward +10 win /
−10 loss / 0 draw (HP tiebreak at day 200).

**Opponent pool:** starts with one frozen copy of the initial policy. Each game
samples an opponent from the pool, weighted toward those with a **lower EMA win
rate** (= harder; weight ∝ `(1 − winrate).clamp(min=0.05)`). The EMA win rate
(the agent's win rate vs that opponent; draw = 0.5) is updated from the
data-collection results as games finish, and the finished slot is re-assigned a
fresh opponent + a fresh map. When the **minimum** EMA win rate across the pool
exceeds `pool_add_threshold` (0.6), the current agent is snapshotted into the
pool (new entry seeded at 0.5). Opponent actions are sampled **in proportion to
the policy's probabilities** (same `sample_policy` as the agent), not argmax.
Slots are grouped by assigned opponent so each pooled net runs once per step.

**wandb logging** (project `nypc2026-selfplay`, disable with `--no-wandb`):
`avg_ep_R`, `ploss`/`vloss`/`entropy`, value-net **explained variance**
(`value_ev`), `pool_size`, per-opponent EMA win rate keyed by a **stable id**
(`opp_winrate/<id>`, so a curve tracks the same opponent across FIFO evictions),
`opp_winrate_min`/`_mean`, `pool_added`, `steps_per_s`, `episodes`.

**Hyperparameters** live in `config.yaml` (one field per `Config` field, with
comments). Edit it freely; `load_config` errors on unknown keys. CLI flags
(`--B`, `--steps`, `--iters`, `--ckpt`, `--gpus`, `--map-workers`, `--no-wandb`,
`--no-resume`, `--no-phases`) override the file.

### Throughput / multi-GPU
The rollout is **launch-bound** — thousands of tiny kernels per step — so
env-steps/s scales almost linearly with `B` until the GPU saturates (RTX 3070 Ti,
full pool: 1.8k → 3.2k → 5.8k → 9.8k → 12.8k as B goes 256 → 512 → 1024 → 2048 →
4096). The default `B` is therefore **2048**, and the following are on:

- **`--gpus N` (data parallel).** `B` and `minibatch` are TOTALS: rank *r*
  simulates `B/N` games in its own `FastEnv` against its own opponent draws, and
  the PPO gradients are averaged across ranks, so an N-GPU run consumes exactly
  the same data per iteration as a 1-GPU run with the same config — same horizon,
  same number of optimizer steps, same lr. `python ppo_selfplay.py --gpus 2`
  spawns the workers itself; `torchrun --nproc_per_node=2 ppo_selfplay.py` also
  works (the rank is read from the environment). Default is *all* visible GPUs.
  The nets are **not** wrapped in `DistributedDataParallel` — the code calls
  submodules directly (`t1net.aux(h1)`, T2 on a flattened source subset), which
  DDP's forward-hook reducer can't see; an explicit flat all-reduce of ~300k
  parameters costs microseconds instead.
  Everything the ranks must agree on is kept in lockstep explicitly: weights are
  broadcast at startup and stay identical because every rank applies the same
  averaged gradient; **advantage whitening uses global statistics**; and the
  opponent pool's EMA win rates are all-reduced once per iteration so both ranks
  make the same snapshot/eviction decision. The per-game EMA is applied in its
  order-independent closed form (`wr ← (1−α)ⁿ·wr + (1−(1−α)ⁿ)·r̄` for the *n*
  results an opponent collected, mean `r̄`) — one iteration's worth of games is
  drawn against the win rates from the *start* of that iteration, which at α=0.02
  and hundreds of games per iteration is indistinguishable from the old
  update-as-you-go behaviour.
- **Background map generation** (`map_gen.py`, `map_workers: 6` **per rank**).
  One random map costs ~25 ms of single-core Python and every finished episode
  needs one; at B=2048 that is thousands of maps per iteration, serially, in the
  middle of the rollout. A pool of worker processes keeps a queue of ready maps
  topped up and `FastEnv.regen` just pops one. `map_workers: 0` restores
  deterministic inline generation. (The inline `make_maps` also gained the retry the generator needs —
  it raises "could only place k/K strongholds" every few hundred maps, which used
  to crash startup at large B.)
- **Rollout buffer location** (`store_device`: `cpu` / `cuda` / `auto`).
  Keeping the iteration's transitions in VRAM removes a device→host copy of ~25
  tensors on every step and makes the per-minibatch gather device-side, but the
  A/B says it **doesn't matter**: at B=2048, 100k steps/iter took 33.8/33.2 s
  resident and 32.9/34.3 s on the host — the gather and H2D overlap with GPU
  compute. So the default stays `cpu` and the ~2.7 GiB (≈1.4 GiB/rank on two
  GPUs, at ~9.5 KiB/transition) is left free for a larger `B`/`minibatch`, which
  do matter. `auto` picks VRAM when it fits in `store_vram_frac` (0.45) of free
  memory and says which it chose.
  Building this exposed a latent aliasing bug worth knowing about: on a resident
  buffer, storing is an *alias*, and `observe()` hands back `mb.token_valid`
  itself while `regen()` overwrites it in place — so every stored `tmask` would
  have become the newest map's. `extract` now clones the token mask.
- **TF32 matmuls + fused Adam + `set_to_none` grads**, and per-minibatch metrics
  accumulated on-device (`.item()` per minibatch used to sync the stream ~250
  times per iteration). bf16/fp16 autocast is deliberately *not* used: PPO's
  ratio compares a stored `old_logp` against a recomputed one, and bf16's ~3
  decimal digits over a sum of ~20 log-probs would inject percent-level noise
  into the ratio. TF32's ~1e-3 relative error does not.
- **`_greedy`** (the gold allocator, the hottest launch site in the rollout: T
  iterations × 2 calls × every distinct opponent) hoists its gathers out of the
  loop and writes back with one scatter — ~3× faster, bit-identical output.

Once `B` is large the **PPO update**, not the rollout, is the bottleneck; the
remaining lever there is a bigger `minibatch` (which changes the optimization, so
it is left alone by default).

**Phase schedule.** Iterations are grouped into blocks of `phase_iters` (250); each
block swaps in its own `lr / epochs / ent_coef / steps_per_iter`, and the **last
entry is held for every later phase**:

| phase | iters | lr | epochs | ent_coef | steps/iter |
|---|---|---|---|---|---|
| 1 | 1–250 | 5e-4 | 5 | 5e-3 | 200k |
| 2 | 251–500 | 2e-4 | 4 | 2e-3 | 250k |
| 3 | 501–750 | 1e-4 | 3 | 1e-3 | 300k |
| 4+ | 751– | 5e-5 | 3 | 5e-4 | 300k |

The phase is derived from the iteration number, so a resumed run re-enters the
right phase (and re-applies its lr to both optimizers) automatically. Each entry
is printed when it takes effect and logged to wandb as `phase`/`lr`/`ent_coef`/
`steps_per_iter`; the console line shows `p<N>`. `phases: null`, `--no-phases`, or
an explicit `--steps` disables the schedule and uses the flat config values.

**Checkpoint / resume:** at the start of every iter, `checkpoint.pt` is written
atomically with the agent (actor T1/T2 + critic), both optimizers, the whole
opponent pool (actor nets + EMA win rates + stable ids), the current opponent
assignments, and RNG state. On restart (`resume: true`, default) training loads it
and continues from the saved iter — so an interrupted run picks up where it left
off. Pass `--no-resume` (or delete the file) to start fresh. In a multi-GPU run
only **rank 0** writes it (every rank holds identical weights and pool); each rank
offsets its RNG on resume so the ranks keep simulating different games, and the
opponent assignment is simply redrawn (the env is rebuilt from scratch anyway).

Networks per the spec:
- **T1** (3 transformer blocks): 거점-token transformer; token feats (24, log1p) +
  global feats (11, per-spec transforms) → token MLP → 5 dims: `[0]` build logit
  (Bernoulli per candidate token), `[1:5]` averaged → 4-way train softmax.
- **T2** (2 blocks): per move-source, T1 token outputs ++ 6 extra feats → token
  MLP → **2 values per target**: `[0]` softmax over targets = target distribution
  (self = no move); `[1]` sigmoid = **full mobilisation** (send the source's
  labourers too). Only the chosen target's `[1]` is sampled (Bernoulli in training,
  `>0.5` at submission) and only it gets a log-prob — and only when the source is
  really dispatched (target ≠ self) and is actually keeping labourers home. A 거점
  whose warriors are *all* labouring (surplus 0) is a legal move source too: it can
  only leave via full mobilisation, otherwise the move resolves to a no-op.
- **Critic**: same structure as T1, independent; token MLP → mean = value.

Action gold-gating is exactly as specified: per-command affordability masks
before sampling, then greedy gold allocation (build → moves → train); commands
dropped only by the greedy step still count for PPO, masked-to-0 ones never do.

Each finished episode gets a **brand-new random map** (`FastEnv.regen`); the env
reserves capacity for the largest possible map (N=109, T=23). T1 token features
also include **normalized 거점 coordinates** (per-map x/y range → [−10,10]); T2
adds the **source→token normalized coordinate difference**.

Run:
```
python ppo_selfplay.py                       # config.yaml: B=2048, phase schedule, all GPUs
python ppo_selfplay.py --gpus 2              # explicit 2-GPU data-parallel run
python ppo_selfplay.py --gpus 1              # single GPU
python ppo_selfplay.py --smoke               # tiny end-to-end sanity run
python ppo_selfplay.py --B 256 --steps 1000000 --iters 50   # the original setting
```
Verified learning: avg episode reward rises from ≈ −6 toward +6 within ~10 short
iters (agent beating the frozen initial opponent), now against fresh maps.

**On `steps_per_iter`:** 1e6 is workable but coarse — it's only ~50 policy
refreshes, with a large, increasingly stale on-policy batch (3 epochs over 1e6 ≈
730 minibatch updates on the *same* data) and ~3–5 GB of CPU buffer. Prefer a
smaller rollout (≈ 100k–250k = B × horizon ~400–1000) with more iterations
(≈ 200–500) for the same/greater total budget: more frequent policy updates,
less staleness, less memory. Defaults are now 200k × 250 (= 50M total).

> Note: attention uses a hand-written multi-head implementation, not
> `nn.MultiheadAttention` — the fused CUDA fast-path emits all-NaN rows when a
> batch mixes padded and unpadded token sequences (our mixed-size case).

## Submission bot (`vanilla_bot.py`)
A ready-to-submit player that speaks `sample-code.py`'s stdin/stdout protocol but
picks actions by running the trained actor (one inference per turn, no search). It
is **numpy-only at runtime** — the 1000ms handshake budget can't fit torch's ~2.3s
import, while numpy imports in ~0.15s. Workflow:

```
python ppo_selfplay.py ...                    # trains -> checkpoint.pt
python export_weights.py --ckpt checkpoint.pt --out data.bin      # offline, uses torch
python verify_np_bot.py                       # checks numpy == torch pipeline (needs both files)
# then submit / play:
testing-tool.py -a "python vanilla_bot.py --weights data.bin" -b "..." --seed 1 --NP 40 --KP 6
python run_match.py --seed 42                 # vanilla vs vanilla -> replay.log
python power_test.py --games 20 --old-weights data_prev.bin   # A/B two weight files
```

- The encoder (`extract`/`observe`) and the transformer forward are ported to
  numpy and verified numerically against the torch path (`verify_np_bot.py`:
  encoder features match to ~1e-6, net outputs to ~1e-6, travel cache and all
  action masks match exactly).
- Region→거점 travel time (turns) is precomputed once at map load, inside the 1s
  handshake. Per-turn `decide` is ~1–4ms on one CPU thread (budget 100ms).
- `--stochastic` samples actions ∝ policy probs (and samples the mobilisation
  Bernoulli); `--greedy` is argmax + the `p > 0.5` mobilisation threshold.
- Move interpretation matches the env: a chosen (source, target) keeps the
  `work_cap` (post-build) lowest-HP stationary warriors at the source (ties →
  smaller suffix) and moves the rest, emitting one `MOVE` per moved warrior — or
  keeps **nobody** home when T2's mobilisation output fires for that target.
- The agent remembers its movers' destinations (the protocol never reveals them).
  The opponent's gold/income (also never sent) are reconstructed from the visible
  economy — exact except for rare opponent move-cost edge cases.
- `fast_env.py`'s import of `testing-tool.py` is now lazy, so importing the env
  package doesn't require the reference simulator to be present.

## Modeling notes / assumptions
- **Mixed sizes are supported via padding.** A batch may contain games with
  different `N`/`K`; everything is padded to `Nmax`/`Tmax`. Padded regions are
  isolated (inert) and the right HQ sits at each game's own `n_b-1`. `observe()`
  returns `info['token_mask'] [B,T]` marking the valid tokens (= `K_b+2`) per
  game; feed it as the transformer's padding mask. Padded token rows are zeroed.
- **`max_warriors_per_side`** defaults to the theoretical max (3 + 3·200 = 603)
  so the pool can never overflow → bit-exact. Lowering it narrows the per-row
  sort (big speedup); real games are gold-limited to far fewer warriors, but a
  too-small pool silently drops training past capacity, so keep the default when
  you need exactness.
- Combat / "keep lowest-HP workers" / hunger-by-suffix all go through one
  segmented order-statistics primitive (`_seg_stats`, a batched per-row sort).
