# rlkit — the game-independent half of a self-play PPO trainer

Everything here works for **any** two-player game. It contains no knowledge of a
board, an action space, an observation layout or a reward rule. To train a new
game you write three things — a `Task`, a `Policy`, and optionally some
`ScriptedOpponent`s — and get the rest for free.

```
                    YOU WRITE                         RLKIT PROVIDES
        ┌──────────────────────────────┐    ┌────────────────────────────────┐
        │ Task    board, turn, reward  │    │ rollout driver, opponent pool, │
        │ Policy  networks, sampling   │◄──►│ GAE, PPO, phases, checkpoints, │
        │ Scripted hand-written bots   │    │ multi-GPU, logging, instances  │
        └──────────────────────────────┘    └────────────────────────────────┘
```

**Start by reading `examples/toy_duel.py`** (≈250 lines, runs in seconds). It is a
complete integration of a tiny game and exercises every feature you will want.
`examples/nypc2026.py` is the same thing for a large real game.

```bash
python -m examples.toy_duel --smoke              # 20 seconds, proves the wiring
python -m examples.toy_duel --config examples/toy_duel.yaml
python test_rlkit.py && python test_rlkit_dist.py
python tune_batch.py examples.toy_duel           # measure the batch size to use
```

Nothing in this kit hardcodes an interpreter path, a GPU count or a thread count --
the machine is described to the session (see `CLAUDE.md`), and `tune_batch.py`
measures the batch size this machine and this game actually want.

---

## The three interfaces

### `Policy` — the networks

```python
modules()        -> {name: nn.Module}      # checkpoint keys + startup broadcast
param_groups()   -> {'actor': [...], 'critic': [...]}   or   {'all': [...]}
act(obs, **kw)   -> (action, store, extra) # sampling, under no_grad
value(obs)       -> [B]                    # critic
evaluate(mb)     -> ActorOut(logp, entropy, extra_loss, metrics)
evaluate_value(mb) -> CriticOut(value, extra_loss, metrics)
```

* `store` is whatever `evaluate` will need later — observations, masks, the
  sampled outcomes, and **`old_logp`**. There is no schema; the trainer treats it
  as an opaque dict of `[B, ...]` tensors and hands minibatches of it back.
* `extra` is for the **task**, not the trainer: a multi-turn commitment, a
  prediction to feed back as a feature next turn. May be `{}`.
* Two parameter groups → one Adam and one backward pass each (right when actor
  and critic are separate nets); one group → `policy + vf_coef*value + extra` in
  a single step (right for a shared trunk). Two groups must be **disjoint**.

### `Task` — the game

```python
rollout_turn(ctx) -> store          # advance one turn, return what to store
reward_done()     -> (r [B], done [B] bool)
reset_finished(done, rows)          # regenerate instances, reset carried state
```

plus optional `tally_mask()`, `metrics()`, `log_extra()`, `state_dict()`,
`close()`, `bootstrap_value(policy)`.

Subclass **`TwoPlayerTask`** and you only implement `observe(side)`,
`env_step(agent_action, opp_action)` and `empty_opponent_out(B)`; it writes
`rollout_turn` for the ordinary "both sides act, then the env resolves" structure.
Override `rollout_turn` when a turn needs something stranger (see the opening
split in `examples/nypc2026.py`).

Inside `rollout_turn`, `ctx` gives you exactly three things:

```python
ctx.agent_act(obs, **kw)     -> (action, store, extra)
ctx.agent_value(obs)         -> [B]
ctx.opponent_act(obs, **kw)  -> (action, extra)   # dispatched over the pool
```

### `ScriptedOpponent` — hand-written bots

```python
name = 'rusher'; full_batch = True
act(task, obs, rows, **kw) -> (action, extra)
reset_rows(rows)
```

They occupy fixed pool slots and are never evicted. **Write at least one.** A pool
of nothing but your own snapshots is homogeneous, and a policy can climb by
learning to beat its own habits while losing to a committed strategy; a scripted
bot is an absolute yardstick in the win-rate curves, and `wr` vs the rusher is
usually the first number that tells you training is real.

---

## What the trainer does with all that

```
per iteration:
  phase = schedule.at(it)               lr / epochs / entropy / rollout size
  checkpoint                            BEFORE any work: a crash loses ≤ 1 iter
  for step in range(steps_per_iter // B):
      store = task.rollout_turn(ctx)    ← your game
      r, done = task.reward_done()
      buffer.add(store + reward + done)
      tally.update(...)                 on-device, read once per iteration
      if done.any(): pool.reassign(rows); task.reset_finished(...)
  pool.apply_tally(tally, dist)         closed-form EMA, all-reduced
  GAE → flatten → whiten (global stats)
  for epoch, for minibatch:             PPO clip + entropy + your extra losses
  pool.maybe_snapshot(it) / maybe_grow()
  log
```

---

## Things that are easy to get wrong

1. **Never store an alias of mutable env state.** Storing a tensor is a `.to()`,
   which is a no-op alias when the buffer already lives on that device. If your
   env hands back one of its own buffers (a validity mask, an id table) and later
   overwrites it in place, every stored transition silently becomes the newest
   game's. `.clone()` it where you build the observation. This bug is invisible
   until you move the buffer to VRAM, and then it is a silent wrong-data bug.
2. **The log-prob must be summed identically at sampling and at re-evaluation** —
   same factors, same masks. `exp(logp - old_logp)` means nothing otherwise, and
   PPO will look like it is training while optimizing noise.
3. **A masked-off factor contributes no log-prob and no entropy.** A decision the
   policy never made must not be reinforced.
4. **Canonicalise the perspective in `observe`.** Every feature is "mine" vs
   "theirs", never "player 0" vs "player 1". That is what lets one network play
   both sides — i.e. what makes self-play possible at all.
5. **`empty_opponent_out` must use the real no-op value**, not zeros. `-1` for
   "no move"; zero would order every idle unit to region 0.
6. **`Task.labels()` runs after the step.** Anything you store from there sees
   the resolved turn, not the state the action was chosen in.

---

## Configuration

Subclass `BaseConfig` for your game's fields; they can be phase-scheduled too:

```python
@dataclass
class Config(rlkit.BaseConfig):
    d_model: int = 64
    n_layers: int = 3

cfg = rlkit.load_config("config.yaml", Config)   # unknown keys are a hard error
```

`phases` is a list of dicts applied in blocks of `phase_iters`, with the last entry
held forever, resolved from the iteration number so a **resumed run lands in the
right phase**. Anything structural (`B`, `iters`, buffer/worker settings) is
rejected as a phase key.

---

## Multi-GPU

```bash
python -m examples.toy_duel --gpus 2
```

`B` and `minibatch` are **totals**: each rank simulates `B/N` games and takes
`minibatch/N` rows, so an N-GPU run collects exactly the same data per iteration
as a 1-GPU run with the same config. Gradients are averaged with one flat
all-reduce (not `DistributedDataParallel`, which cannot see a policy that calls its
submodules directly).

Everything the ranks must agree on is kept in lockstep explicitly: weights are
broadcast at startup and stay identical because every rank applies the same
averaged gradient, and the pool's win rates are all-reduced once per iteration
using an **order-independent** closed-form EMA. This matters more than it sounds:
each rank decides on its own when to snapshot and evict, so drifting win rates
would silently turn one run into two trainers sharing a checkpoint. `test_rlkit_dist.py`
asserts bit-identity across ranks without needing NCCL.

---

## Checkpoints

Written at the **start** of every iteration by rank 0 only. Resume is deliberately
lenient — parameters load by name where the shape still matches, optimizer moments
are rebuilt if the parameter set changed, and the pool re-aligns its scripted slots
by name — because during a long run you *will* add a feature, a head, or a bot. For
changes that silently reinterpret learned weights (a head whose meaning changed),
declare a `guard` in `Setup` and get a loud error instead.

---

## Throughput, in order of impact

| Lever | Notes |
|---|---|
| **`B`** | A batched GPU env is launch-bound, not FLOP-bound: throughput scales nearly linearly with `B` until the GPU saturates. Raise it until VRAM or the GAE horizon (`steps_per_iter / B`, which wants to stay well above `1/(1-γλ)` ≈ 20) stops you. |
| **`--gpus`** | Parallelises rollout *and* update. |
| **`instance_workers`** | Per-episode map/scenario generation is pure Python; without workers it is a serial stall in the middle of the rollout with the GPU idle. |
| `minibatch` | Once `B` is large the PPO update dominates. Raising it means fewer, larger optimizer steps — a real change to the optimization, not a free win. |
| TF32 + fused Adam | On by default (`tune_backend`). |
| `store_device` | Measured: host RAM and VRAM are within noise, because the per-minibatch gather and H2D overlap with compute. Leave it on the host and spend the VRAM on `B`. Do **not** pin the host buffer. |
| bf16/fp16 autocast | **Rejected.** PPO compares a stored `old_logp` against a recomputed one; bf16 over a sum of ~20 log-probs injects percent-level noise straight into the ratio. TF32's ~1e-3 does not. |

---

## Files

| file | what |
|---|---|
| `interfaces.py` | `Task`, `TwoPlayerTask`, `Policy`, `ScriptedOpponent`, `ActorOut`, `CriticOut` — **the contract** |
| `trainer.py` | `train(cfg, setup)`, `SetupCtx`/`Setup`, the standard CLI flags |
| `rollout.py` | the collection loop and `RolloutContext` |
| `ppo.py` | the clipped-surrogate update, KL early stop, metrics |
| `buffer.py` | storage, GAE, flatten, whitening, minibatching, explained variance |
| `pool.py` | the opponent pool and its win-rate maths |
| `config.py` | `BaseConfig`, YAML loading, `PhaseSchedule` |
| `dist.py` | `Dist`, `tune_backend`, `launch` |
| `checkpoint.py` | atomic save, lenient resume, guards |
| `factory.py` | background per-episode instance generation |
| `logger.py` | console line + wandb |
| `parity.py` | turn-by-turn comparison of a batched env against a reference simulator |
| `testing.py` | in-process fake of the collectives, for the multi-rank test |
