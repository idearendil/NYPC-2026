# FastEnv — GPU-accelerated batched env for RL

A batched, GPU (CUDA) re-implementation of `testing-tool.py`'s game dynamics for
fast RL data collection. `B` games are stepped in parallel as tensor ops.

## Files
- `fast_env.py` — the env (`FastEnv`, `MapBatch`, `observe`).
- `test_fast_env.py` — **bit-exact end-of-turn parity** (uniform + mixed sizes).
- `test_phases.py` — **bit-exact per-phase parity** (compares after every stage).
- `test_observe.py` — observation shapes + mixed-size token-mask checks.
- `test_encoder.py` — **independent recompute of every encoder feature**.
- `benchmark_env.py` — throughput benchmark.

Run with the env that has torch+CUDA:
```
D:/other_programs/anaconda3/envs/orbit/python.exe test_fast_env.py
D:/other_programs/anaconda3/envs/orbit/python.exe benchmark_env.py
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
(`value_ev`), `pool_size`, per-opponent EMA win rate (`opp_winrate/<i>`),
`opp_winrate_min`/`_mean`, `pool_added`, `steps_per_s`, `episodes`.

Networks per the spec:
- **T1** (3 transformer blocks): 거점-token transformer; token feats (24, log1p) +
  global feats (11, per-spec transforms) → token MLP → 5 dims: `[0]` build logit
  (Bernoulli per candidate token), `[1:5]` averaged → 4-way train softmax.
- **T2** (2 blocks): per move-source, T1 token outputs ++ 6 extra feats → token
  MLP → scalar → softmax = target distribution (self = no move).
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
python ppo_selfplay.py                       # defaults: B=256, 200k steps/iter, 250 iters
python ppo_selfplay.py --smoke               # tiny end-to-end sanity run
python ppo_selfplay.py --B 256 --steps 1000000 --iters 50   # the original setting
```
Throughput ≈ **1,300–1,900 agent-steps/s** at B=256 on the 3070 Ti (down from
~3,000 because per-episode map regeneration runs the CPU Voronoi/Delaunay
generator whenever games end). Rollout buffer is on CPU.
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
