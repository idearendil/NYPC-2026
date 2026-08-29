#!/usr/bin/env python3
"""Self-play PPO for the board game, on top of the batched GPU env (fast_env).

Opponent pool: starts with a single frozen copy of the initial policy. For each
game we sample an opponent from the pool, weighted toward opponents with a lower
EMA win rate (harder opponents). Win rates (the agent's win rate vs that
opponent) are updated from the data-collection game results; when the minimum EMA
win rate across the pool exceeds a threshold, the current agent is snapshotted
into the pool. Opponent actions are sampled in proportion to their probabilities
(not argmax), like the agent's. Agent plays LEFT, opponents play RIGHT.

Networks (per the spec):
  * T1  : transformer over 거점-tokens (3 blocks). Token feats (24, all log1p) +
          global feats (11, per-spec transforms) concatenated per token. Token
          MLP -> 5 dims: [0]=build logit (Bernoulli per candidate token),
          [1:5]=train contribution (mean over tokens -> 4-way softmax).
  * T2  : transformer over tokens (2 blocks), run once per move-source. Input =
          T1 token output ++ 6 extra feats. Token MLP -> scalar -> softmax over
          tokens = target distribution for that source.
  * Critic : same structure as T1 but independent; token MLP -> mean = value.

Action gold-gating exactly as specified: per-command affordability masks before
sampling, then greedy allocation (build first, then moves, then train) of the
remaining gold; commands dropped only by the greedy step still count as taken
for PPO, commands masked to prob 0 never count.

Reward: +10 win, -10 loss, 0 draw (HP tiebreak at the 200-day limit).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import fast_env as fe
import map_gen
from fast_env import (OWN_LEFT, OWN_RIGHT, KIND_HQ, KIND_BASE, MOVE_COST,
                      TRAIN_COST, HQ_MAXLEVEL, BASE_MAXLEVEL, HQ_HEAL, BASE_HEAL,
                      MAX_DAYS)

TOK_FEAT = 32          # 15 scalars (incl. fog-of-war "turns since last seen") + 5 arrive
                       # + 5 reach (all log1p) + 2 norm coords + 5 reach-delta vs prev turn
GLOB_FEAT = 14         # 10 + HQ-turns + 거점-count + x/y map-span (no opponent-gold
                       # feature -- fog-of-war makes it unreconstructible; the
                       # opponent's true gold is an aux LOSS target only, never fed in)
T2_EXTRA = 8           # 4 logged + surplus + travel + 2 norm coord diffs
COST_INF = 1_000_000_000


# --------------------------------------------------------------------------- #
# throughput: backend tuning and multi-GPU data parallelism
# --------------------------------------------------------------------------- #
def tune_backend(threads=None):
    """Global knobs that only affect speed, applied once per process.

    TF32 matmuls (Ampere/Ada tensor cores) are a free ~1.5x on the transformer
    GEMMs -- their 10-bit mantissa is far more precision than log1p'd game
    features carry. cudnn.benchmark helps the fixed-shape encoder kernels. The
    CPU thread count matters because the rollout buffer (when it lives in host
    RAM) is indexed and copied on the CPU every minibatch; more than ~8 threads
    on those small copies just adds sync overhead, so it is capped."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if threads is None:
        threads = min(8, max(1, (os.cpu_count() or 8) // 2))
    torch.set_num_threads(max(1, int(threads)))


class Dist:
    """Data-parallel state for one process.

    The split is over GAMES: rank r simulates ``cfg.B // world`` of the batch in
    its own FastEnv and against its own opponent draws, and the PPO gradients are
    averaged across ranks -- so both halves of the iteration (rollout and update)
    are parallelised, and the data collected per iteration is identical to a
    single-GPU run with the same config. Everything the ranks must agree
    on -- network weights, the opponent pool's membership and win rates -- is
    kept in lockstep explicitly (weights are broadcast once at startup and stay
    identical because every rank applies the same averaged gradient; pool win
    rates are all-reduced once per iteration, see the batched EMA in train()).

    The nets are NOT wrapped in nn.parallel.DistributedDataParallel: the code
    calls submodules directly (``t1net.aux(h1)`` on an already-computed hidden
    state, T2 on a flattened subset of sources), which DDP's forward-hook
    reducer does not support. With ~300k parameters total, an explicit flat
    all-reduce per optimizer step costs microseconds anyway.

    ``world == 1`` makes every method a no-op, so the single-GPU path is
    unchanged."""

    def __init__(self):
        self.rank, self.world, self.local_rank = 0, 1, 0
        self.enabled = False

    @classmethod
    def init(cls):
        d = cls()
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world <= 1:
            return d
        import torch.distributed as dist
        d.rank = int(os.environ["RANK"])
        d.local_rank = int(os.environ.get("LOCAL_RANK", d.rank))
        d.world = world
        n_gpu = torch.cuda.device_count()
        # nccl needs one distinct GPU per rank; anything else (CPU debugging, more
        # ranks than GPUs) falls back to gloo rather than deadlocking.
        use_nccl = n_gpu >= world
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if use_nccl else "gloo")
        if n_gpu:
            torch.cuda.set_device(min(d.local_rank, n_gpu - 1))
        d.enabled = True
        return d

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()

    def broadcast_params(self, modules):
        """Make every rank's weights bit-identical to rank 0's."""
        if not self.enabled:
            return
        import torch.distributed as dist
        for m in modules:
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t.data, src=0)

    def all_reduce_grads(self, params):
        """Average gradients in ONE flat all-reduce (many tiny tensors here)."""
        if not self.enabled:
            return
        import torch.distributed as dist
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= self.world
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)

    def all_reduce_(self, t, op="sum"):
        """In-place sum/mean all-reduce of a tensor (returns it)."""
        if not self.enabled:
            return t
        import torch.distributed as dist
        dist.all_reduce(t)
        if op == "mean":
            t /= self.world
        return t

    def sum_scalars(self, values, device):
        """All-reduce a list of python floats; returns a list of floats."""
        t = torch.tensor([float(v) for v in values], device=device,
                         dtype=torch.float64)
        self.all_reduce_(t)
        return t.tolist()

    def shutdown(self):
        if self.enabled:
            import torch.distributed as dist
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# feature transforms
# --------------------------------------------------------------------------- #
def slog1p(x):                      # signed log1p (handles negative surplus)
    return torch.sign(x) * torch.log1p(x.abs())


def plog1p(x):                      # log1p for non-negative quantities
    return torch.log1p(x.clamp(min=0).float())


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
class MHA(nn.Module):
    """Explicit multi-head self-attention. We avoid nn.MultiheadAttention because
    its fused CUDA fast-path can emit all-NaN rows when a batch mixes padded and
    unpadded sequences (key_padding_mask). This version is numerically safe."""
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, kpm):                        # x [B,T,d], kpm [B,T] True=ignore
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=2)
        def split(t):
            return t.view(B, T, self.h, self.dk).transpose(1, 2)  # [B,h,T,dk]
        q, k, v = split(q), split(k), split(v)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)        # [B,h,T,T]
        if kpm is not None:
            att = att.masked_fill(kpm[:, None, None, :], float('-inf'))
        att = att.softmax(dim=-1)
        att = torch.nan_to_num(att)                  # safe if a row is fully masked
        out = (att @ v).transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, kpm):
        x = x + self.attn(self.ln1(x), kpm)
        x = x + self.ff(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, d, heads, ff, layers):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])

    def forward(self, x, kpm):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, kpm)
        return h


class ActorT1(nn.Module):
    def __init__(self, d=64, heads=4, ff=128, layers=3):
        super().__init__()
        self.enc = Encoder(TOK_FEAT + GLOB_FEAT, d, heads, ff, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 5))
        # auxiliary head (per token, 7 dims; shapes the shared encoder, unused at
        # inference). 0..5 = ln(1 + enemy garrison / reachable-within-1..5-turns) at
        # this 거점 NEXT turn; 6 = per-token contribution to the GLOBAL opp-gold
        # target, masked-averaged across tokens (see aux_losses).
        self.aux = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))
        self.d = d

    def forward(self, t1, glob, tmask):
        B, T, _ = t1.shape
        x = torch.cat([t1, glob[:, None, :].expand(B, T, GLOB_FEAT)], dim=2)
        kpm = ~tmask
        h = self.enc(x, kpm)
        return h, self.head(h)


class ActorT2(nn.Module):
    """Per-source target head: the move-target logit (softmax over tokens ->
    which 거점 to send to). No full-mobilisation head -- a move source only ever
    sends its surplus (stationary beyond work_cap); labourers can't be ordered
    out."""

    def __init__(self, d_in, d=64, heads=4, ff=128, layers=2):
        super().__init__()
        self.enc = Encoder(d_in, d, heads, ff, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, tmask):
        h = self.enc(x, ~tmask)
        return self.head(h).squeeze(-1)         # [Q,T]


class Critic(nn.Module):
    def __init__(self, d=64, heads=4, ff=128, layers=3):
        super().__init__()
        self.enc = Encoder(TOK_FEAT + GLOB_FEAT, d, heads, ff, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # auxiliary head (per token, 7 dims; shapes the critic's encoder). Same
        # targets as ActorT1.aux: 0..5 = ln(1 + enemy garrison / reachable-within-
        # 1..5-turns) NEXT turn, 6 = per-token piece of the global opp-gold target.
        self.aux = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))

    def _encode(self, t1, glob, tmask):
        B, T, _ = t1.shape
        x = torch.cat([t1, glob[:, None, :].expand(B, T, GLOB_FEAT)], dim=2)
        return self.enc(x, ~tmask)

    def _pool(self, h, tmask):
        v = self.head(h).squeeze(-1)           # [B,T]
        m = tmask.float()
        return (v * m).sum(1) / m.sum(1).clamp(min=1)

    def value(self, t1, glob, tmask):
        return self._pool(self._encode(t1, glob, tmask), tmask)

    def value_aux(self, t1, glob, tmask):
        """Value plus the per-token gold-production aux prediction (one encode)."""
        h = self._encode(t1, glob, tmask)
        return self._pool(h, tmask), self.aux(h)


# --------------------------------------------------------------------------- #
# feature extraction from the env (one side's perspective)
# --------------------------------------------------------------------------- #
def extract(env, side, prev_reach=None):
    B, N, T = env.B, env.N, env.mb.T
    me = OWN_LEFT if side == 0 else OWN_RIGHT          # owner code (1/2)
    opp_idx = 1 - side                                  # side index (0/1)
    tokens, glob, info = env.observe(side)
    raw25 = slog1p(tokens[:, :, :25].float())          # the 25 log1p features
    # raw (un-log'd) enemy-reachable-within-1..5-turns counts, packed at [20:25];
    # returned so the caller can feed it back next turn as prev_reach for the delta.
    reach_raw = tokens[:, :, 20:25].float()            # [B,T,5]
    # the fog-of-war "turns since this token was last visible" feature (index 14 of
    # the 15 scalar block) gets its OWN transform, ln(x/10 + 1), instead of the
    # generic slog1p everything else in raw25 got -- overwrite it in place.
    raw25[:, :, 14] = torch.log(tokens[:, :, 14].float() / 10 + 1)

    g = glob.float()
    # The opponent's gold is HIDDEN in the real protocol and NOT reconstructible
    # from visible events any more (the fog-of-war protocol only ever sends this
    # side's OWN build/train/move events -- see FastEnv.step), so no gold estimate
    # or prediction is fed to the network as an input feature at all. Our OWN gold
    # (g[:,5]) stays exact; the opponent's true gold remains an AUXILIARY LOSS
    # target only (see aux_label / gold_glob below), never an input.
    glob_t = torch.stack([
        g[:, 0] / 10 - 10,
        plog1p(g[:, 1] / 10), plog1p(g[:, 2] / 10),
        plog1p(g[:, 3]),      plog1p(g[:, 4]),
        plog1p(g[:, 5] / 100),
        plog1p(g[:, 7] / 10), plog1p(g[:, 8] / 10),
        plog1p(g[:, 9] / 5),  plog1p(g[:, 10] / 5),
    ], dim=1)

    # HQ-upgrade "turns to afford" feature: max(0, cost-gold) / net_income, where
    # net_income = last turn's building income - upkeep (2 per warrior). Denominator
    # floored at 1 so net<=0 reads as "very far" (large) rather than dividing badly.
    hq_lvl = info['hq_level'].long()
    hq_can_up = hq_lvl < HQ_MAXLEVEL
    hq_cost = env.HQ_UPCOST[(hq_lvl + 1).clamp(max=HQ_MAXLEVEL)]
    my_total = ((env.w_hp > 0) & (env.slot_side[None, :].expand(B, env.W) == side)).sum(1)
    net_income = env.prev_income[:, side] - 2 * my_total
    need = (hq_cost - info['gold'].long()).clamp(min=0).float()
    turns = torch.where(hq_can_up, need / net_income.clamp(min=1).float(),
                        torch.zeros(B, device=glob_t.device))
    glob_t = torch.cat([glob_t, plog1p(turns)[:, None]], dim=1)

    # static map-geometry globals: 거점 count (K+2 tokens) and physical map size.
    # Spans are over ALL regions; coords are point-symmetric about the origin, so the
    # zero-padded nonexistent-region slots stay interior and don't skew min/max. The
    # span is ~2*GEN_L (~21000) and nearly constant across maps; /10000 (the disk
    # radius) keeps it O(1) like the other globals instead of saturating log1p.
    n_tok = info['token_mask'].sum(1).float()
    cx, cy = env.mb.cx.float(), env.mb.cy.float()
    x_span = cx.max(1).values - cx.min(1).values
    y_span = cy.max(1).values - cy.min(1).values
    glob_t = torch.cat([glob_t,
                        plog1p(n_tok / 7.0)[:, None],
                        plog1p(x_span / 10000.0)[:, None],
                        plog1p(y_span / 10000.0)[:, None]], dim=1)

    # observe() hands back mb.token_valid ITSELF, and regen() overwrites that tensor
    # in place when an episode ends. tmask is stored in the rollout buffer, so on a
    # GPU-resident buffer (where storing is an alias, not a copy) every stored mask
    # would silently change to the newest map's. Clone it once here -- [B,T] bools,
    # nothing measurable -- so no caller can be caught by the aliasing.
    tmask = info['token_mask'].clone()
    tok_ids = info['token_ids']
    tok_g = tok_ids.clamp(max=N - 1)
    gold = info['gold'].long()
    hq_level = info['hq_level'].long()
    build_cand = info['build_candidates']

    # normalized 거점 coordinates: per game map x-range -> [-10,10], y -> [-10,10]
    tx = env.mb.cx.gather(1, tok_g).float()
    ty = env.mb.cy.gather(1, tok_g).float()
    BIG = 1e18
    xmn = torch.where(tmask, tx, torch.full_like(tx, BIG)).min(1, keepdim=True).values
    xmx = torch.where(tmask, tx, torch.full_like(tx, -BIG)).max(1, keepdim=True).values
    ymn = torch.where(tmask, ty, torch.full_like(ty, BIG)).min(1, keepdim=True).values
    ymx = torch.where(tmask, ty, torch.full_like(ty, -BIG)).max(1, keepdim=True).values
    normx = ((tx - xmn) / (xmx - xmn).clamp(min=1) * 20 - 10) * tmask.float()
    normy = ((ty - ymn) / (ymx - ymn).clamp(min=1) * 20 - 10) * tmask.float()
    # The board is point-symmetric about its centre, so RIGHT's view is LEFT's
    # view point-reflected -- which, for normalized coords, is exactly a negation
    # (token features are already perspective-swapped via me/opp, and tok_dist is
    # reflection-invariant). Feeding both sides a canonical (LEFT) orientation lets
    # one shared net play either side correctly.
    if side == 1:
        normx = -normx
        normy = -normy
    # per-turn CHANGE in enemy reachability (this turn's reach minus last turn's,
    # per 거점 per horizon k). An enemy force marching toward a token shows up as a
    # positive delta on that token's near-horizon dims -> the net can pre-empt it.
    if prev_reach is None:
        prev_reach = torch.zeros_like(reach_raw)
    reach_delta = slog1p(reach_raw - prev_reach) * tmask[:, :, None].float()  # [B,T,5]
    t1 = torch.cat([raw25, normx[:, :, None], normy[:, :, None], reach_delta],
                   dim=2)  # [B,T,32]

    # ---- CRITIC-ONLY view -------------------------------------------------- #
    # The submission never runs the critic, so during training it may condition
    # on privileged (hidden) information the actor cannot see: the per-거점 "enemy
    # reachable within k turns" block (raw25[:,:,20:25]) is replaced by the enemy's
    # PLANNED arrivals within k turns, computed from the (hidden) opponent move
    # orders EXACTLY like observe()'s own-arrival block -- so the value net sees who
    # is actually coming, not who merely could. reach_delta (t1's last 5 dims) is
    # left as the actor's. NOTE: the other 20 scalars of raw25_crit are still the
    # FOGGED (belief) view -- raw25_crit is a clone of the actor's raw25, not a
    # fresh true-state recompute; only the arrival slice is genuinely privileged. A
    # fully unfogged critic is possible future work. glob has no privileged swap
    # (opponent gold isn't fed as a feature to either net -- see extract()'s top).
    sd_all = env.slot_side[None, :].expand(B, env.W)
    op_moving = (env.w_hp > 0) & (sd_all == opp_idx) & env.w_move
    tt_full = env.mb.travel_turns                      # [B,N,T]
    tok_sorted = env.mb.token_ids                      # [B,T] ascending
    ti_of = torch.searchsorted(tok_sorted, env.w_tgt.clamp(min=0)).clamp(max=T - 1)
    is_tok = tok_sorted.gather(1, ti_of) == env.w_tgt
    valid_mv = op_moving & is_tok                      # opp movers heading to a token
    mcount = torch.zeros((B, N, T), dtype=torch.int64, device=t1.device)
    if bool(valid_mv.any()):
        flat = ((env.b_ar[:, None] * N + env.w_region) * T + ti_of)[valid_mv]
        mcount.view(-1).scatter_add_(0, flat, torch.ones_like(flat))
    op_arrive = torch.zeros((B, T, 5), dtype=torch.float32, device=t1.device)
    for k in range(1, 6):                              # arrivals in EXACTLY k turns
        op_arrive[:, :, k - 1] = (mcount * (tt_full == k)).sum(1).float()
    raw25_crit = raw25.clone()
    raw25_crit[:, :, 20:25] = slog1p(op_arrive) * tmask[:, :, None].float()
    t1_crit = torch.cat([raw25_crit, normx[:, :, None], normy[:, :, None],
                         reach_delta], dim=2)          # [B,T,32]
    glob_crit = glob_t.clone()

    cnt, sumhp = env._side_counts()
    turret, workcap = env._turret(), env._workcap()

    def gtok(field):
        return field.gather(1, tok_g)

    # NOTE: op_cnt here is the EXACT enemy count -- kept true because it only ever
    # feeds the build/heal legality checks below (can_up/can_heal), which only
    # apply where my_cnt > 0 (a region I already occupy, hence always visible to
    # me -- true == belief there). e1/e2 below use the FOGGED belief versions
    # instead, since those are real tactical features fed to the T2 (move-target)
    # actor head and must not leak hidden enemy strength.
    my_cnt = gtok(cnt[:, :, side]); op_cnt = gtok(cnt[:, :, opp_idx])
    my_hps = gtok(sumhp[:, :, side])
    own_t = gtok(env.b_owner); kind_t = gtok(env.b_kind)
    lvl_t = gtok(env.b_level); bhp_t = gtok(env.b_hp)
    tur_t = gtok(turret); wc_t = gtok(workcap)
    me_b = own_t == me; opp_b = (own_t != 0) & (own_t != me)

    my_tur = torch.where(me_b, tur_t, torch.zeros_like(tur_t))
    my_bhp = torch.where(me_b, bhp_t, torch.zeros_like(bhp_t))
    my_wc = torch.where(me_b, wc_t, torch.zeros_like(wc_t))

    bel_kind = gtok(env.op_seen_kind[:, side, :]); bel_level = gtok(env.op_seen_level[:, side, :])
    bel_op_cnt = gtok(env.op_seen_wcnt[:, side, :])
    bel_op_hps = gtok(env.op_seen_whp[:, side, :])
    bel_op_bhp = gtok(env.op_seen_bhp[:, side, :])
    bel_op_tur = env._lvl_derived(bel_kind, bel_level, env.HQ_TURRET, env.BASE_TURRET)

    sd = env.slot_side[None, :].expand(B, env.W)
    my_stat = (env.w_hp > 0) & (sd == side) & (~env.w_move)
    stat_cnt = env._scatter_region(my_stat).gather(1, tok_g)

    # build cost / workcap-after per token
    maxlev = torch.where(kind_t == KIND_HQ, HQ_MAXLEVEL, BASE_MAXLEVEL)
    up_cost = torch.where(kind_t == KIND_HQ,
                          env.HQ_UPCOST[(lvl_t + 1).clamp(max=HQ_MAXLEVEL)],
                          env.BASE_COST[(lvl_t + 1).clamp(max=BASE_MAXLEVEL)])
    heal_cost = torch.where(kind_t == KIND_HQ, torch.full_like(lvl_t, HQ_HEAL),
                            torch.full_like(lvl_t, BASE_HEAL))
    is_strong = env.mb.is_stronghold.gather(1, tok_g)
    is_hq = env.is_hq_mask.gather(1, tok_g)
    # Upgrade legality mirrors the judge (testing-tool.apply_upgrades): a region is
    # only upgradeable when a friendly warrior is present AND no enemy warrior is on
    # it. up_room = "has room to upgrade" (ignores transient occupancy) is used by
    # the HQ-saving commitment so it can keep saving while an enemy sits on the HQ.
    up_room = me_b & (lvl_t < maxlev)
    can_up = up_room & (my_cnt > 0) & (op_cnt == 0)
    can_heal = me_b & (lvl_t >= maxlev) & (my_cnt > 0) & (op_cnt == 0)
    build_new = (own_t == 0) & is_strong & (~is_hq)
    cost = torch.full_like(lvl_t, COST_INF)
    cost = torch.where(build_new, torch.full_like(cost, env.BASE_COST[1]), cost)
    cost = torch.where(can_up, up_cost, cost)
    cost = torch.where(can_heal, heal_cost, cost)

    wc_up = torch.where(kind_t == KIND_HQ,
                        env.HQ_WCAP[(lvl_t + 1).clamp(max=HQ_MAXLEVEL)],
                        env.BASE_WCAP[(lvl_t + 1).clamp(max=BASE_MAXLEVEL)])
    wc_after = my_wc.clone()
    wc_after = torch.where(can_up, wc_up, wc_after)
    wc_after = torch.where(build_new, torch.full_like(wc_after, env.BASE_WCAP[1]), wc_after)

    # T2 token-specific extra features (logged) -- FOGGED (belief) enemy strength,
    # since these feed the actor's move-target head.
    e1 = plog1p((bel_op_cnt + bel_op_tur).float())
    e2 = plog1p((bel_op_hps + bel_op_bhp).float() / 5)
    e3 = plog1p((my_cnt + my_tur).float())
    e4 = plog1p((my_hps + my_bhp).float() / 5)
    extra4 = torch.stack([e1, e2, e3, e4], dim=2)

    tok_dist = tokens[:, :, 25:25 + T].float()       # observe packs dist after the 25 feats

    return dict(
        t1=t1, glob=glob_t, t1_crit=t1_crit, glob_crit=glob_crit,
        tmask=tmask, tok_ids=tok_ids,
        gold=gold, hq_level=hq_level,
        build_cand=build_cand, build_cost=cost,
        wc_cur=my_wc, wc_after=wc_after, stat_cnt=stat_cnt,
        is_hq_me=(is_hq & me_b), can_up_hq=(is_hq & can_up),
        can_up_hq_room=(is_hq & up_room),
        hq_upcost=env.HQ_UPCOST[(hq_level + 1).clamp(max=HQ_MAXLEVEL)],
        owner_me=me_b, build_new=build_new,
        extra4=extra4, tok_dist=tok_dist, normx=normx, normy=normy,
        reach_raw=reach_raw,
        # free workers = non-moving friendly not currently labouring (surplus
        # beyond each region's work_cap); used to gate builds.
        free_total=(env._scatter_region(my_stat)
                    - torch.where(env.b_owner == me, workcap,
                                  torch.zeros_like(workcap))).clamp(min=0).sum(1),
        hq_commit=env.hq_commit[:, side],
    )


# --------------------------------------------------------------------------- #
# helpers shared by sampling and evaluation
# --------------------------------------------------------------------------- #
def t2_logits_sources(t2, h1, extra4, surplus_pb, tok_dist, normx, normy, tmask, src_mask):
    """Run T2 only for the valid move-sources (flattened), returns [B,T_src,T_tok]
    target logits (rows for invalid sources are filled with a flat -1e9)."""
    B, T, d = h1.shape
    out = torch.full((B, T, T), -1e9, device=h1.device)
    qb, qs = src_mask.nonzero(as_tuple=True)
    if qb.numel() == 0:
        return out
    h1q = h1[qb]                                                 # [Q,T,d]
    exq = extra4[qb]                                             # [Q,T,4]
    Q = qb.numel()
    sf = plog1p(surplus_pb[qb, qs].float())[:, None, None].expand(Q, T, 1)
    tv = plog1p(tok_dist[qb, qs, :])[:, :, None]                 # [Q,T,1]
    dx = (normx[qb] - normx[qb, qs][:, None])[:, :, None]        # source->token coord diff
    dy = (normy[qb] - normy[qb, qs][:, None])[:, :, None]
    x = torch.cat([h1q, exq, sf, tv, dx, dy], dim=2)
    kpm = ~tmask[qb]
    lg = t2(x, kpm).masked_fill(kpm, -1e9)                       # [Q,T]
    out[qb, qs] = lg
    return out


def bern_logp(p, x):
    p = p.clamp(1e-6, 1 - 1e-6)
    return x * torch.log(p) + (1 - x) * torch.log(1 - p)


def train_logits_from_head(head5, tmask):
    m = tmask.float()[:, :, None]
    return (head5[:, :, 1:5] * m).sum(1) / m.sum(1).clamp(min=1)   # [B,4]


# --------------------------------------------------------------------------- #
# action sampling (collection)
# --------------------------------------------------------------------------- #
def sample_policy(t1net, t2net, o, N, relax_reg=None, forbid_tgt=None):
    # relax_reg [B,N] bool (optional): per-region "lift the work-cap MOVE restriction"
    # flag. Where set, this region's labourers may also be ordered to move (surplus =
    # ALL stationary warriors, not just those beyond work_cap). Used only for the
    # opponent in full-mobilisation games; the training agent always passes None.
    # forbid_tgt [B] long (optional): per-game token index to mask out of EVERY move
    # source's target distribution (-1 = none). Used by the turn-1 opening split so the
    # second inference can't re-pick the first inference's target.
    B, T = o['tmask'].shape
    dev = o['t1'].device
    h1, head5 = t1net(o['t1'], o['glob'], o['tmask'])

    committed = o['hq_commit']                        # [B] saving for an HQ upgrade
    not_committed = ~committed

    # HQ-upgrade handling, split by TARGET level:
    #  * reaching level 2 or 3 (current level < 3): behaves like a base upgrade --
    #    sampleable only when affordable + legal this turn, no commitment -- EXCEPT it
    #    stays exempt from the free-worker gate (the HQ is always staffed at home).
    #  * reaching level 4 or 5 (current level >= 3): keeps the save-commit MACRO --
    #    sampleable even when unaffordable / while an enemy occupies the HQ, entering a
    #    "saving" mode until affordable, then emitted (deferred while illegal).
    # can_up_hq_room marks the (upgradeable, owned) HQ token ignoring occupancy;
    # can_up_hq is the same token but ALSO judge-legal (friendly present, no enemy).
    can_up_hq_room = o['can_up_hq_room']               # [B,T]
    hq_room = can_up_hq_room.any(1)                     # [B] HQ has room to upgrade
    hq_legal = o['can_up_hq'].any(1)                    # [B] legal to upgrade this turn
    hq_tok = can_up_hq_room.float().argmax(1)          # [B] (valid where hq_room)
    hq_cost = o['hq_upcost']                            # [B] next-level cost (occupancy-free)
    hq_afford = hq_room & (o['gold'] >= hq_cost)
    hq_macro = hq_room & (o['hq_level'] >= 3)           # [B] level 3->4 / 4->5: macro
    hq_normal = hq_room & (o['hq_level'] < 3)           # [B] level 1->2 / 2->3: base-like

    # ---------------- BUILD ----------------
    # Base builds/upgrades need affordability + legality + a free worker. HQ upgrades
    # are exempt from the free-worker gate at EVERY level; the macro HQ token is further
    # exempt from the affordability/occupancy check (to commit to saving). While
    # committed (a level 4/5 save), every build is masked off.
    afford_legal = (o['build_cand'] & (o['build_cost'] <= o['gold'][:, None]) & o['tmask'])
    normal_build_mask = afford_legal & (o['free_total'][:, None] >= 1)
    normal_hq_mask = afford_legal & can_up_hq_room & hq_normal[:, None]  # base-like HQ (no worker gate)
    macro_hq_mask = can_up_hq_room & hq_macro[:, None]                   # macro HQ (ignore afford/occupancy)
    build_mask = (normal_build_mask | normal_hq_mask | macro_hq_mask) & not_committed[:, None]
    p_build = torch.sigmoid(head5[:, :, 0]) * build_mask.float()
    outcome = torch.bernoulli(p_build)
    build_logp = (build_mask.float() * bern_logp(torch.sigmoid(head5[:, :, 0]), outcome)).sum(1)

    hq_sampled = outcome.bool().gather(1, hq_tok[:, None]).squeeze(1) & hq_room
    # macro path: intend (committed or sampled) + affordable, emit only if legal this
    # turn; otherwise keep/enter saving mode (defer while unaffordable or enemy on HQ).
    sampled_macro = hq_sampled & hq_macro
    do_hq_macro = hq_afford & (committed | sampled_macro) & hq_legal
    new_commit = (committed | sampled_macro) & hq_macro & (~do_hq_macro)
    # normal path: only sampleable when already affordable + legal, so just execute.
    do_hq_normal = hq_sampled & hq_normal
    do_hq_now = do_hq_macro | do_hq_normal

    # HQ upgrade takes gold priority; the rest goes to the sampled non-HQ builds.
    gold_after_hq = o['gold'] - do_hq_now.long() * hq_cost
    non_hq_outcome = outcome.bool() & (~can_up_hq_room)
    perm = torch.argsort(torch.rand(B, T, device=dev), dim=1)
    exec_build, gold1 = _greedy(non_hq_outcome, o['build_cost'], gold_after_hq, perm, T)

    do_hq_tok = torch.zeros(B, T, dtype=torch.bool, device=dev)
    rows_hq = do_hq_now.nonzero(as_tuple=True)[0]
    if rows_hq.numel() > 0:
        do_hq_tok[rows_hq, hq_tok[rows_hq]] = True
    exec_all = exec_build | do_hq_tok

    # post-build quantities
    wc_pb = torch.where(exec_all, o['wc_after'], o['wc_cur'])
    # full-mobilisation: drop the move-source work-cap (-> surplus = all stationary)
    # for flagged regions, so labourers there become movable. Affects moves only; the
    # build free-worker gate above (o['free_total']) is unchanged.
    if relax_reg is not None:
        relax_tok = relax_reg.gather(1, o['tok_ids'].clamp(max=N - 1)) & o['tmask']
        wc_pb = torch.where(relax_tok, torch.zeros_like(wc_pb), wc_pb)
    surplus_pb = (o['stat_cnt'] - wc_pb).clamp(min=0)
    owner_me_pb = o['owner_me'] | (exec_all & o['build_new'])
    hq_after = (o['hq_level'] + do_hq_now.long()).clamp(max=HQ_MAXLEVEL)

    # ---------------- MOVE (T2) ----------------
    # While committed only free moves are allowed: targets restricted to our own
    # building tokens (the HQ is always one, so a surplus source is never starved).
    # A source needs actual surplus (stationary beyond work_cap) to be a valid move
    # source at all -- labourers can never be ordered out (no full-mobilisation
    # action).
    valid_src = (o['tmask'] & (surplus_pb > 0)
                 & (MOVE_COST * surplus_pb <= gold1[:, None]))
    tgt_allowed = torch.where(committed[:, None], o['tmask'] & owner_me_pb, o['tmask'])
    logits = t2_logits_sources(t2net, h1, o['extra4'], surplus_pb, o['tok_dist'],
                               o['normx'], o['normy'], o['tmask'], valid_src)
    logits = logits.masked_fill(~tgt_allowed[:, None, :], -1e9)
    if forbid_tgt is not None:
        fmask = torch.zeros(B, T, dtype=torch.bool, device=dev)
        frows = (forbid_tgt >= 0).nonzero(as_tuple=True)[0]
        if frows.numel() > 0:
            fmask[frows, forbid_tgt[frows].clamp(max=T - 1)] = True
            logits = logits.masked_fill(fmask[:, None, :], -1e9)
    logp_tok = F.log_softmax(logits, dim=2)                       # [B,src,tok]
    src_idx = torch.arange(T, device=dev)[None, :].expand(B, T)
    tgt = src_idx.clone()                                         # default: self (no move)
    qb, qs = valid_src.nonzero(as_tuple=True)
    if qb.numel() > 0:                                            # sample only valid sources
        rp = logp_tok[qb, qs].exp()                              # masked targets are ~0
        tgt[qb, qs] = torch.multinomial(rp, 1).squeeze(1)
    move_logp_src = logp_tok.gather(2, tgt[:, :, None]).squeeze(2)
    move_logp = (move_logp_src * valid_src.float()).sum(1)

    tgt_is_self = tgt == src_idx
    tgt_mine = owner_me_pb.gather(1, tgt)
    move_cost = torch.where(tgt_mine, torch.zeros_like(surplus_pb), MOVE_COST * surplus_pb)
    move_item = valid_src & (~tgt_is_self)
    perm2 = torch.argsort(torch.rand(B, T, device=dev), dim=1)
    exec_move, gold2 = _greedy(move_item, move_cost, gold1, perm2, T)

    # ---------------- TRAIN ----------------
    tl = train_logits_from_head(head5, o['tmask'])
    cats = torch.arange(4, device=dev)
    cap = env_traincap(t1net, hq_after)
    tmask_train = (cats[None, :] <= cap[:, None]) & (cats[None, :] * TRAIN_COST <= gold2[:, None])
    # while committed: no training -> only category 0 allowed
    tmask_train = torch.where(committed[:, None], cats[None, :] == 0, tmask_train)
    tl_m = tl.masked_fill(~tmask_train, -1e9)
    logp_tr = F.log_softmax(tl_m, dim=1)
    train_cat = torch.multinomial(logp_tr.exp(), 1).squeeze(1)
    train_logp = logp_tr.gather(1, train_cat[:, None]).squeeze(1)

    old_logp = build_logp + move_logp + train_logp

    # ---------------- env action tensors ----------------
    tok_ids = o['tok_ids']
    build_env = torch.zeros(B, N, dtype=torch.bool, device=dev)
    rb, tb = exec_build.nonzero(as_tuple=True)
    build_env[rb, tok_ids[rb, tb]] = True
    force_build_env = torch.zeros(B, N, dtype=torch.bool, device=dev)
    if rows_hq.numel() > 0:                                       # forced (committed) HQ upgrade
        hq_region = tok_ids.gather(1, hq_tok[:, None]).squeeze(1)
        force_build_env[rows_hq, hq_region[rows_hq]] = True
    move_env = torch.full((B, N), -1, dtype=torch.long, device=dev)
    rm, sm = exec_move.nonzero(as_tuple=True)
    src_reg = tok_ids[rm, sm]
    tgt_reg = tok_ids[rm, tgt[rm, sm]]
    move_env[rm, src_reg] = tgt_reg
    action = {'build': build_env, 'move': move_env, 'train': train_cat,
              'force_build': force_build_env}

    store = dict(
        t1=o['t1'], glob=o['glob'],
        t1_crit=o['t1_crit'], glob_crit=o['glob_crit'], tmask=o['tmask'],
        extra4=o['extra4'], tok_dist=o['tok_dist'],
        normx=o['normx'], normy=o['normy'],
        build_mask=build_mask, build_outcome=outcome,
        train_mask=tmask_train, train_cat=train_cat,
        valid_src=valid_src, tgt=tgt, surplus_pb=surplus_pb,
        tgt_allowed=tgt_allowed,
        old_logp=old_logp,
    )
    return action, store, old_logp, new_commit


_TRAINCAP = None
def env_traincap(t1net, hq_after):
    # HQ train caps by level: [0,1,1,2,2,3]
    global _TRAINCAP
    if _TRAINCAP is None or _TRAINCAP.device != hq_after.device:
        _TRAINCAP = torch.tensor([0, 1, 1, 2, 2, 3], device=hq_after.device)
    return _TRAINCAP[hq_after.clamp(max=HQ_MAXLEVEL)]


_GREEDY_BIG = 1 << 60          # cost that is never affordable (gold fits in int64)


def _greedy(item_mask, cost, gold, perm, T):
    """Greedy gold allocation: in random order, take an item if affordable.

    Sequential by nature (skipping an unaffordable item leaves gold for later
    ones), but it is the single hottest kernel-launch site in the rollout -- it
    runs T times inside every sample_policy call, twice per call, for the agent
    AND for every distinct opponent. So the per-step work is kept to the bone:
    the two gathers are hoisted out of the loop (permute cost/mask ONCE), items
    that aren't candidates are folded into the cost as "infinite" so the loop
    needs no boolean AND, and the takes are written back with a single scatter
    instead of T of them. ~3x faster than the naive form, bit-identical output."""
    c = cost.gather(1, perm)                                   # costs in draw order
    m = item_mask.gather(1, perm)
    cm = torch.where(m, c, torch.full_like(c, _GREEDY_BIG))     # non-items: unaffordable
    remaining = gold.clone()
    takes = []
    for k in range(T):
        ck = cm[:, k]
        take = ck <= remaining
        remaining = remaining - torch.where(take, ck, torch.zeros_like(ck))
        takes.append(take)
    exec_ = torch.zeros_like(item_mask)
    exec_.scatter_(1, perm, torch.stack(takes, 1))
    return exec_, remaining


# --------------------------------------------------------------------------- #
# PPO re-evaluation of stored decisions
# --------------------------------------------------------------------------- #
def masked_token_mse(pred, target, tmask):
    """MSE of a per-token [B,T,C] prediction vs target, averaged over valid tokens
    (tmask [B,T] bool) and channels."""
    m = tmask.float()[:, :, None]
    se = ((pred - target) ** 2) * m
    return se.sum() / (m.sum().clamp(min=1) * pred.shape[-1])


def aux_losses(aux_pred, gold_aux, gold_glob, tmask):
    """Combined auxiliary loss for the [B,T,7] aux head:
      - channels 0..5: per-token MSE vs ``gold_aux`` [B,T,6] (enemy garrison +
        reachable-within-1..5-turns, log1p'd), masked over valid tokens.
      - channel 6: a GLOBAL scalar per game, formed by masked-averaging the 7th
        channel across tokens, regressed (MSE) onto ``gold_glob`` [B] = ln(1 +
        opp_gold/100).
    Returned as their sum; the caller scales by aux_coef."""
    tok_loss = masked_token_mse(aux_pred[:, :, :6], gold_aux, tmask)
    m = tmask.float()
    gpred = (aux_pred[:, :, 6] * m).sum(1) / m.sum(1).clamp(min=1)      # [B]
    glob_loss = F.mse_loss(gpred, gold_glob)
    return tok_loss + glob_loss


def evaluate_policy(t1net, t2net, b):
    h1, head5 = t1net(b['t1'], b['glob'], b['tmask'])
    aux_pred = t1net.aux(h1)                 # [B,T,7] enemy-count + opp-gold aux
    B, T = b['tmask'].shape

    # build
    logit0 = head5[:, :, 0]
    p = torch.sigmoid(logit0)
    bm = b['build_mask'].float()
    build_logp = (bm * bern_logp(p, b['build_outcome'])).sum(1)
    pe = p.clamp(1e-6, 1 - 1e-6)
    build_ent = (bm * -(pe * torch.log(pe) + (1 - pe) * torch.log(1 - pe))).sum(1)

    # train
    tl = train_logits_from_head(head5, b['tmask']).masked_fill(~b['train_mask'], -1e9)
    logp_tr = F.log_softmax(tl, dim=1)
    train_logp = logp_tr.gather(1, b['train_cat'][:, None]).squeeze(1)
    train_ent = -(logp_tr.exp() * logp_tr).sum(1)

    # move
    logits = t2_logits_sources(t2net, h1, b['extra4'], b['surplus_pb'],
                               b['tok_dist'], b['normx'], b['normy'],
                               b['tmask'], b['valid_src'])
    logits = logits.masked_fill(~b['tgt_allowed'][:, None, :], -1e9)
    logp_tok = F.log_softmax(logits, dim=2)
    vs = b['valid_src'].float()
    move_logp = (logp_tok.gather(2, b['tgt'][:, :, None]).squeeze(2) * vs).sum(1)
    move_ent = ((-(logp_tok.exp() * logp_tok).sum(2)) * vs).sum(1)

    logp = build_logp + train_logp + move_logp
    ent = build_ent + train_ent + move_ent
    return logp, ent, aux_pred


# --------------------------------------------------------------------------- #
# reward / done
# --------------------------------------------------------------------------- #
def reward_done(env):
    alive = env.hq_alive()
    ag, op = alive[:, 0], alive[:, 1]
    ag_hp = env.b_hp.gather(1, env.hq_region[:, 0:1]).squeeze(1)
    op_hp = env.b_hp.gather(1, env.hq_region[:, 1:2]).squeeze(1)
    done = (~ag) | (~op) | (env.day >= MAX_DAYS)
    r = torch.zeros(env.B, device=env.device)
    r = torch.where((~op) & ag, torch.full_like(r, 10.0), r)
    r = torch.where((~ag) & op, torch.full_like(r, -10.0), r)
    tl = (env.day >= MAX_DAYS) & ag & op
    r = torch.where(tl & (ag_hp > op_hp), torch.full_like(r, 10.0), r)
    r = torch.where(tl & (ag_hp < op_hp), torch.full_like(r, -10.0), r)
    return r, done


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    B: int = 256
    # 1e6 steps/iter is workable but coarse: it means only ~50 policy-refresh
    # cycles and a big, increasingly-stale on-policy batch. A smaller rollout with
    # more iterations updates the policy more often and uses less buffer RAM.
    steps_per_iter: int = 200_000
    iters: int = 250
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 5e-4
    epochs: int = 4
    minibatch: int = 4096
    # KL early stopping: stop an iter's PPO epochs once the policy has drifted
    # this far from the data-collection policy (approx_kl, Schulman estimator).
    # null/None disables it. Lets you raise lr/epochs without overshooting.
    target_kl: Optional[float] = 0.02
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    # auxiliary task (BOTH actor & critic, shapes the encoders): per-거점, predict
    # ln(1 + next-turn enemy garrison) and ln(1 + enemy reachable-within-1..5-turns),
    # plus a GLOBAL ln(1 + opp_gold/100) from the token-averaged 7th aux channel.
    # aux_coef scales the summed aux loss. 0 disables.
    aux_coef: float = 0.25
    max_grad_norm: float = 1.0
    d_model: int = 64
    # Rollout-buffer location: "cpu" (host RAM), "cuda", or "auto" (cuda when the
    # estimated buffer fits in store_vram_frac of free VRAM). Measured at B=2048:
    # no difference either way -- the per-minibatch gather + H2D overlaps with GPU
    # compute -- so the default leaves the ~2.7 GiB of VRAM free for a larger B or
    # minibatch, which do matter. See the note in config.yaml.
    store_device: str = "cpu"
    # Fraction of FREE VRAM the rollout buffer may claim under store_device=auto.
    store_vram_frac: float = 0.45
    # ---- throughput ----
    # Background processes that pre-generate maps for episode regeneration, PER
    # RANK. One map costs ~25 ms of single-core Python and a finished episode
    # needs one, so without this the rollout stalls on map generation at large B.
    # 0 = generate inline (deterministic; the old behaviour).
    map_workers: int = 6
    map_queue: int = 512             # ready maps kept buffered per process
    torch_threads: Optional[int] = None   # None -> min(8, cpu_count // 2)
    # opponent pool
    opp_ema_alpha: float = 0.02      # EMA rate for per-opponent win rate
    pool_add_threshold: float = 0.6  # add agent to pool when min win rate exceeds this
    pool_max_size: int = 6           # BASE total pool cap (incl. fixed rusher+japper).
                                     # +1 per PERMANENT opponent added later, so the
                                     # evictable room stays at pool_max_size - N_SCRIPTED.
    opp_sample_floor: float = 0.05   # min sampling weight per opponent
    # ---- PERMANENT SELF-SNAPSHOTS ----
    # Every perm_snapshot_every iterations, the CURRENT main actor is frozen into the
    # pool as a PERMANENT opponent (never evicted; the cap grows by 1). Unlike the
    # ordinary win-rate-triggered snapshots -- which are evicted oldest-first and so
    # only ever cover the recent past -- these accumulate into a permanent ladder of
    # past selves, so the agent cannot drift back into a style it already beat.
    perm_snapshot_every: int = 500   # 0 = disabled
    # "full-mobilisation" opponent games: a fraction of games where the opponent may
    # ignore the work-cap move restriction (labourers CAN be ordered to move) so the
    # agent learns to predict opponents that mobilise every warrior. Per such game,
    # each opponent base/HQ independently drops the restriction with opp_relax_prob
    # per turn. These games' win/loss do NOT feed the EMA win rate.
    opp_relax_frac: float = 0.20     # fraction of games flagged full-mobilisation
    opp_relax_prob: float = 0.10     # per-base/HQ per-turn relax prob in those games
    # turn-1 opening split: on each game's first turn, infer the actor twice so the HQ's
    # two spare warriors head to DIFFERENT strongholds (faster 2-base opening; matches
    # the submission bots). Applies to the agent and to NET opponents (not rusher/japper).
    opening_split: bool = True
    use_wandb: bool = True
    # checkpointing
    ckpt_path: str = "checkpoint.pt"  # saved at the start of every iter
    resume: bool = True               # resume from ckpt_path if it exists


def load_config(path):
    """Load a Config from a YAML file, keeping defaults for any missing field."""
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    fields = {f.name for f in dataclasses.fields(Config)}
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    return Config(**d)


def _buffer_bytes_per_sample(T):
    """Bytes one stored transition occupies, matching the `keys` list below.

    Used only to decide whether the iteration's rollout buffer fits in VRAM."""
    f, i8, b1 = 4, 8, 1
    return (
        T * TOK_FEAT * f * 2          # t1, t1_crit
        + GLOB_FEAT * f * 2           # glob, glob_crit
        + T * T * f                   # tok_dist
        + T * 4 * f                   # extra4
        + T * f * 2                   # normx, normy
        + T * f                       # build_outcome
        + T * i8 * 2                  # tgt, surplus_pb
        + T * b1 * 4                  # tmask, build_mask, valid_src, tgt_allowed
        + T * 6 * f                   # gold_aux
        + 4 * b1 + i8                 # train_mask, train_cat
        + 5 * f                       # old_logp, adv, ret, value, gold_glob
    )


def save_ckpt(path, obj):
    """Atomically write a checkpoint (temp file + replace)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_maps(B, seed, workers=0):
    """The initial batch of B random maps.

    Delegates to map_gen so the draw (and the retry on the generator's
    "could only place k/K strongholds" failure -- which the old inline version
    did NOT handle, and which is hit within a few hundred maps) is shared with
    the background factory used for per-episode regeneration. ``workers > 0``
    generates them across processes: at 25 ms each, B = 2048 is 50 s serially."""
    return map_gen.make_maps(B, seed, workers=workers)


# --------------------------------------------------------------------------- #
# opponent pool helpers
# --------------------------------------------------------------------------- #
def frozen_copy(net):
    c = copy.deepcopy(net).eval()
    for p in c.parameters():
        p.requires_grad_(False)
    return c


def frozen_from_state(make_net, state, device):
    net = make_net().to(device)
    # strict=False: pre-aux checkpoints lack the (inference-unused) aux head;
    # pooled opponents only run the policy, so a fresh aux head is harmless.
    net.load_state_dict(state, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class Learner:
    """An actor (t1+t2) + critic with their optimizers and flat parameter lists.
    `pa`/`pc` are cached because the gradient all-reduce and the clip walk them
    once per minibatch."""

    def __init__(self, t1, t2, cr, lr, on_cuda):
        self.t1, self.t2, self.cr = t1, t2, cr
        self.pa = list(t1.parameters()) + list(t2.parameters())
        self.pc = list(cr.parameters())
        adam_kw = dict(fused=True) if on_cuda else {}
        self.opt_a = torch.optim.Adam(self.pa, lr=lr, **adam_kw)
        self.opt_c = torch.optim.Adam(self.pc, lr=lr, **adam_kw)

    def set_lr(self, lr):
        for opt in (self.opt_a, self.opt_c):
            for g in opt.param_groups:
                g['lr'] = lr

    def train_mode(self):
        self.t1.train(); self.t2.train(); self.cr.train()


class RolloutState:
    """Per-game state that carries ACROSS steps within a rollout: each side's
    previous-turn enemy reachability (an input feature), plus the scripted bots'
    state machines. Reset per game at episode boundaries -- a fresh map means
    there is no previous turn."""

    def __init__(self, B, T, device):
        self.reach_ag = torch.zeros(B, T, 5, device=device)
        self.reach_op = torch.zeros(B, T, 5, device=device)
        self.rstate = RusherState(B, device)
        self.jstate = JapperState(B, device)
        self._all = torch.arange(B, device=device)

    def reset_rows(self, rows):
        self.reach_ag[rows] = 0
        self.reach_op[rows] = 0
        self.rstate.reset_rows(rows)
        self.jstate.reset_rows(rows)

    def reset_all(self):
        self.reset_rows(self._all)


def slice_obs(o, rows):
    """Slice every per-batch tensor in an observation dict along dim 0."""
    return {k: (v[rows] if torch.is_tensor(v) else v) for k, v in o.items()}


def sample_opponents(n, pool_wr, gen, floor=0.05):
    """Sample n opponent indices, weighted toward lower EMA win rate (harder)."""
    w = (1.0 - pool_wr).clamp(min=floor)
    probs = w / w.sum()
    return torch.multinomial(probs, n, replacement=True, generator=gen)


# The first N_SCRIPTED unified opponent indices are FIXED scripted bots (never
# evicted): index 0 = rusher, index 1 = japper. Net snapshot j is unified index
# j + N_SCRIPTED.
RUSHER_IDX = 0
JAPPER_IDX = 1
N_SCRIPTED = 2

# rush-bot strategy knobs (mirror rush_bot.py). The three turn-count knobs are
# scaled by MAX_DAYS/200 so the scripted opponent's phase transitions still land
# at the same FRACTION of the game now that MAX_DAYS is 400, not the original 200
# (an un-scaled bot would "give up" expansion or turn fully passive far too early
# on the longer clock, making it a much weaker/worse training partner).
_DAY_SCALE = MAX_DAYS / 200
RUSH_SIZE = 6               # warriors massed at home before the wave launches
KEEP_HOME = 1               # workers kept home when the wave launches / while sieging
PASSIVE_UPGRADE_TURN = round(100 * _DAY_SCALE)  # passive opponent by this turn -> economy/UPGRADE
SIEGE_MARGIN = round(20 * _DAY_SCALE)           # turns past ETA before a launched wave is "spent"
_MODE_MASS, _MODE_RUSH_SENT, _MODE_UPGRADE = 0, 1, 2

# japper-bot strategy knobs (mirror the CURRENT japper_bot.py).
JAP_WAVE_TRIGGER = 6        # warriors gathered at the rally before a wave launches
JAP_GOLD_RESERVE = 30       # gold kept back (upkeep buffer) before training
JAP_SETTLE_GIVEUP = round(30 * _DAY_SCALE)  # give up the 2-base expansion after this turn -> just train
_JBIG = 1 << 30


class RusherState:
    """Per-game persistent state for the scripted rush opponent -- a batched port of
    rush_bot.py's Brain state machine. All tensors are [B]; ``reset_rows`` clears a
    game's state at its episode boundary. Not checkpointed: episodes never resume
    across runs (the env is freshly reset on resume), so a fresh state always aligns
    with a fresh env."""

    def __init__(self, B, device):
        self.mode = torch.zeros(B, dtype=torch.long, device=device)            # MASS
        self.send_turn = torch.full((B,), -1, dtype=torch.long, device=device)  # None
        self.opp_attacked = torch.zeros(B, dtype=torch.bool, device=device)
        self.HQ_WCAP = torch.tensor(fe.HQ_WCAP, device=device)
        self.HQ_TRAINCAP = torch.tensor(fe.HQ_TRAINCAP, device=device)
        self.HQ_UPCOST = torch.tensor(fe.HQ_UPCOST, device=device)
        self.HQ_HP = torch.tensor(fe.HQ_HP, device=device)

    def reset_rows(self, rows):
        self.mode[rows] = _MODE_MASS
        self.send_turn[rows] = -1
        self.opp_attacked[rows] = False


def rusher_action(env, side, rstate):
    """Scripted fixed opponent: a batched port of rush_bot.py. State machine over
    MASS (mass RUSH_SIZE warriors at home, defend an inbound attack, then launch the
    wave once the opponent commits to economy or has attacked), RUSH_SENT (the wave
    is marching -- keep one worker home for income), and UPGRADE (pure economy: keep
    the HQ work slots filled and pour gold into HQ upgrades; also the fallback once an
    early rush is spent or the opponent stays passive). Persistent per-game state
    lives in ``rstate`` (reset at episode boundaries). Returns a full-batch action
    dict (build / move / train / force_build); the caller copies only its rows.
    The HQ upgrade is emitted on the ``force_build`` channel so it bypasses the
    env's free-worker build gating, matching rush_bot's "upgrade whenever a friendly
    is home, no enemy is home, and gold suffices"."""
    B, N, dev = env.B, env.N, env.device
    W, T = env.W, env.mb.T
    opp = 1 - side
    me_own = OWN_LEFT if side == 0 else OWN_RIGHT
    op_own = OWN_RIGHT if side == 0 else OWN_LEFT
    turn = env.day                                       # [B] per-game turn
    my_hq = env.hq_region[:, side]                       # [B]
    opp_hq = env.hq_region[:, opp]                       # [B]

    sd = env.slot_side[None, :].expand(B, W)
    alive = env.w_hp > 0
    mine = alive & (sd == side)
    enemy = alive & (sd == opp)
    my_count = mine.sum(1)                               # [B]
    stat_mine = mine & (~env.w_move)

    my_reg = env._scatter_region(mine)                   # [B,N]
    opp_reg = env._scatter_region(enemy)                 # [B,N]
    home_stat = env._scatter_region(stat_mine).gather(1, my_hq[:, None]).squeeze(1)
    my_at_hq = my_reg.gather(1, my_hq[:, None]).squeeze(1)
    away_count = my_count - my_at_hq
    opp_at_ownhq = opp_reg.gather(1, opp_hq[:, None]).squeeze(1)
    opp_off_base = enemy.sum(1) > opp_at_ownhq           # opp warrior off its HQ
    opp_at_myhq = opp_reg.gather(1, my_hq[:, None]).squeeze(1)
    opp_has_base = ((env.b_owner == op_own) & (env.b_kind == KIND_BASE)).any(1)
    enemy_hq_alive = env.b_owner.gather(1, opp_hq[:, None]).squeeze(1) == op_own
    hq_level = env.b_level.gather(1, my_hq[:, None]).squeeze(1).clamp(max=HQ_MAXLEVEL)

    # ---- distances (in turns) via the token travel table ----
    tok = env.mb.token_ids                              # [B,T] sorted ascending
    tt = env.mb.travel_turns                            # [B,N,T]

    def tok_idx(region):                                # region [B] -> token index [B]
        q = region[:, None].contiguous()
        return torch.searchsorted(tok, q).clamp(max=T - 1).squeeze(1)

    opp_hq_ti = tok_idx(opp_hq)
    my_hq_ti = tok_idx(my_hq)
    eta = tt.gather(1, my_hq[:, None, None].expand(B, 1, T)).squeeze(1) \
            .gather(1, opp_hq_ti[:, None]).squeeze(1)   # [B] dist my_hq -> opp_hq
    defend_hops = torch.clamp((eta + 1) // 2, min=3)
    tt_to_myhq = tt.gather(2, my_hq_ti[:, None, None].expand(B, N, 1)).squeeze(2)  # [B,N]
    w_dist = tt_to_myhq.gather(1, env.w_region)         # [B,W] dist of each warrior -> my_hq
    off_opphq = env.w_region != opp_hq[:, None]
    enemy_threat = (enemy & off_opphq & (w_dist <= defend_hops[:, None])).any(1)  # [B]

    # ---- transitions ----
    rstate.opp_attacked |= enemy_threat
    mode = rstate.mode
    to_up_passive = (mode == _MODE_MASS) & (turn >= PASSIVE_UPGRADE_TURN) \
        & (~opp_has_base) & (~rstate.opp_attacked) & (~enemy_threat)
    mode = torch.where(to_up_passive, torch.full_like(mode, _MODE_UPGRADE), mode)
    st = rstate.send_turn
    spent = (my_count == 0) \
        | ((away_count == 0) & (st >= 0) & (turn > st + 1)) \
        | ((st >= 0) & (turn > st + eta + SIEGE_MARGIN))
    to_up_spent = (mode == _MODE_RUSH_SENT) & enemy_hq_alive & spent
    mode = torch.where(to_up_spent, torch.full_like(mode, _MODE_UPGRADE), mode)

    gold = env.gold[:, side]
    traincap = rstate.HQ_TRAINCAP[hq_level]
    build = torch.zeros(B, N, dtype=torch.bool, device=dev)
    move = torch.full((B, N), -1, dtype=torch.long, device=dev)
    force = torch.zeros(B, N, dtype=torch.bool, device=dev)
    train = torch.zeros(B, dtype=torch.long, device=dev)

    # ---- MASS: launch the wave when ready, else mass toward RUSH_SIZE ----
    is_mass = mode == _MODE_MASS
    ready = is_mass & (~enemy_threat) & (home_stat >= RUSH_SIZE) \
        & (opp_has_base | rstate.opp_attacked)
    lrows = ready.nonzero(as_tuple=True)[0]
    if lrows.numel() > 0:
        # per-region move keeps the HQ's work_cap (=1 at level 1) worker home and
        # sends the rest at the enemy HQ -- matches rush_bot's send-all-but-KEEP_HOME.
        move[lrows, my_hq[lrows]] = opp_hq[lrows]
    train_mass = is_mass & (~ready) & (my_count < RUSH_SIZE)
    n_mass = torch.minimum(torch.minimum(traincap, (RUSH_SIZE - my_count).clamp(min=0)),
                           gold // TRAIN_COST)
    train = torch.where(train_mass, n_mass, train)

    # ---- RUSH_SENT: keep one worker home for income ----
    is_rs = mode == _MODE_RUSH_SENT
    train_rs = is_rs & (my_at_hq < KEEP_HOME)
    n_rs = torch.minimum(torch.minimum(traincap, (KEEP_HOME - my_at_hq).clamp(min=0)),
                         gold // TRAIN_COST)
    train = torch.where(train_rs, n_rs, train)

    # ---- UPGRADE: keep the work slots filled, pour gold into the HQ ----
    is_up = mode == _MODE_UPGRADE
    work_cap = rstate.HQ_WCAP[hq_level]
    friendly_home = my_at_hq > 0
    enemy_at_home = opp_at_myhq > 0
    hq_can_up = hq_level < HQ_MAXLEVEL
    hq_upcost = rstate.HQ_UPCOST[(hq_level + 1).clamp(max=HQ_MAXLEVEL)]
    do_up = is_up & friendly_home & (~enemy_at_home) & hq_can_up & (gold >= hq_upcost)
    # heal a maxed, damaged HQ (rush_bot keeps a 200-gold cushion past the heal cost)
    hq_hp = env.b_hp.gather(1, my_hq[:, None]).squeeze(1)
    hq_maxhp = rstate.HQ_HP[hq_level]
    do_heal = is_up & friendly_home & (~enemy_at_home) & (~hq_can_up) \
        & (hq_hp < hq_maxhp) & (gold >= HQ_HEAL + 200)
    do_hq = do_up | do_heal
    hrows = do_hq.nonzero(as_tuple=True)[0]
    if hrows.numel() > 0:
        force[hrows, my_hq[hrows]] = True
    # the env charges the build (HQ upgrade) before training, so size training from
    # the gold left after the upgrade to avoid driving gold negative.
    spend = torch.where(do_up, hq_upcost,
                        torch.where(do_heal, torch.full_like(gold, HQ_HEAL),
                                    torch.zeros_like(gold)))
    gold_after = gold - spend
    train_up = is_up & (my_at_hq < work_cap)
    n_up = torch.minimum(torch.minimum(traincap, (work_cap - my_at_hq).clamp(min=0)),
                         gold_after // TRAIN_COST)
    train = torch.where(train_up, n_up, train)

    # ---- commit the launch transition ----
    rstate.mode = torch.where(ready, torch.full_like(mode, _MODE_RUSH_SENT), mode)
    rstate.send_turn = torch.where(ready, turn, st)
    return {'build': build, 'move': move, 'train': train, 'force_build': force}


class JapperState:
    """Per-game state for the scripted japper opponent -- a batched port of the CURRENT
    japper_bot.py. That strategy is region-based enough that the per-warrior settler /
    attacker bookkeeping is realized POSITIONALLY here (fast_env moves per region: one
    target per source, moving every stationary warrior beyond a region's work-cap):
    warriors sitting on the HQ / own bases gather & expand, those stranded on a razed
    field region march on the weakest-then-nearest enemy building, those already on an
    enemy building siege it. Everything is a deterministic function of the map + current
    board (nearest-2 strongholds, rally = own base nearest the enemy HQ), so no
    persistent state is needed beyond the train-cap table; reset_rows is a no-op."""

    def __init__(self, B, device):
        self.HQ_TRAINCAP = torch.tensor(fe.HQ_TRAINCAP, device=device)

    def reset_rows(self, rows):
        pass


def japper_action(env, side, jstate):
    """Scripted fixed opponent: a batched, region-based port of the CURRENT japper_bot.py.
      1) expand to the 2 nearest strongholds (build a base once a warrior stands on an
         empty one, no enemy present, gold >= 300);
      2) train up to the HQ cap while keeping a small gold reserve, once expansion is done;
      3) funnel spare warriors to the RALLY (own base nearest the enemy HQ, else the HQ);
      4) when >= WAVE_TRIGGER gather at the rally, launch its surplus (~5, work-cap stays
         home) at the enemy building with the FEWEST enemies, then nearest, then lowest id;
      5) field warriors chain to the next such enemy building; siegers hold;
      6) if an enemy reaches the HQ or an adjacent region, recall gatherers home to defend
         and hold the waves (attackers already out keep attacking).
    Returns a full-batch action dict; bases go on the force_build channel (bypasses the
    free-worker gate)."""
    B, N, dev = env.B, env.N, env.device
    W = env.W
    opp = 1 - side
    me_own = OWN_LEFT if side == 0 else OWN_RIGHT
    op_own = OWN_RIGHT if side == 0 else OWN_LEFT
    turn = env.day                                       # [B] per-game turn
    my_hq = env.hq_region[:, side]
    opp_hq = env.hq_region[:, opp]
    tt = env.mb.travel_turns                             # [B,N,T] hops region->token
    tok = env.mb.token_ids                               # [B,T] ascending
    T = env.mb.T
    reg_ids = torch.arange(N, device=dev)[None, :]       # [1,N]

    sd = env.slot_side[None, :].expand(B, W)
    alive = env.w_hp > 0
    mine = alive & (sd == side)
    enemy = alive & (sd == opp)
    stat_mine = mine & (~env.w_move)
    stat_reg = env._scatter_region(stat_mine)            # [B,N] my stationary per region
    opp_reg = env._scatter_region(enemy)                 # [B,N] enemy per region
    gold = env.gold[:, side]
    hq_level = env.b_level.gather(1, my_hq[:, None]).squeeze(1).clamp(max=HQ_MAXLEVEL)
    traincap = jstate.HQ_TRAINCAP[hq_level]
    owner = env.b_owner

    def tok_idx(region):                                 # region [B] -> token index [B]
        q = region.clamp(min=0)[:, None].contiguous()
        return torch.searchsorted(tok, q).clamp(max=T - 1).squeeze(1)

    def dist_to(region):                                 # [B,N] hops each region -> `region`
        ti = tok_idx(region)
        return tt.gather(2, ti[:, None, None].expand(B, N, 1)).squeeze(2)

    def at(field, region):                               # field[B,N] gathered at region[B]
        return field.gather(1, region.clamp(min=0)[:, None]).squeeze(1)

    tt_to_myhq = dist_to(my_hq)                          # [B,N]
    tt_to_opphq = dist_to(opp_hq)                        # [B,N]

    build = torch.zeros(B, N, dtype=torch.bool, device=dev)
    move = torch.full((B, N), -1, dtype=torch.long, device=dev)
    force = torch.zeros(B, N, dtype=torch.bool, device=dev)
    train = torch.zeros(B, dtype=torch.long, device=dev)

    def emit_force(mask, reg):
        rows = mask.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            force[rows, reg.clamp(min=0)[rows]] = True

    # ---- static expansion targets: the 2 strongholds nearest my HQ -------------
    is_strong = env.mb.is_stronghold
    d_sh = torch.where(is_strong, tt_to_myhq, torch.full_like(tt_to_myhq, _JBIG))
    sa = d_sh.argmin(1)                                  # nearest stronghold
    d_sh2 = d_sh.scatter(1, sa[:, None], _JBIG)
    sb = d_sh2.argmin(1)                                 # 2nd nearest
    has_sh = is_strong.any(1)
    sa_open = has_sh & (at(owner, sa) == 0)              # empty (OWN_NONE == 0)
    sb_open = has_sh & (at(owner, sb) == 0)
    expand_tgt = torch.where(sa_open, sa, torch.where(sb_open, sb, torch.full_like(sa, -1)))
    expanding = expand_tgt >= 0
    expansion_done = ((~sa_open) & (~sb_open)) | (turn >= JAP_SETTLE_GIVEUP)

    # ---- rally: my BASE nearest the enemy HQ (tie: lower region), else the HQ ---
    owned_base = (owner == me_own) & (env.b_kind == KIND_BASE)
    d_rb = torch.where(owned_base, tt_to_opphq * 256 + reg_ids,
                       torch.full_like(tt_to_opphq, _JBIG))
    rally = torch.where(owned_base.any(1), d_rb.argmin(1), my_hq)

    # ---- defense: an enemy on the HQ or an adjacent (<=1 hop) region -----------
    defending = (opp_reg * (tt_to_myhq <= 1).long()).sum(1) > 0
    gather_to = torch.where(defending, my_hq, rally)

    # ---- pick_target(from_region): per-SOURCE weakest->nearest->lowest enemy bldg
    enemy_bldg = owner == op_own
    enemy_base = enemy_bldg & (env.b_kind == KIND_BASE)
    pool = torch.where(enemy_base.any(1)[:, None], enemy_base, enemy_bldg)  # [B,N]
    has_pool = pool.any(1)                               # [B]
    tokreg = tok.clamp(max=N - 1)                        # [B,T] candidate region ids
    pool_tok = pool.gather(1, tokreg)                    # [B,T]
    cnt_tok = opp_reg.gather(1, tokreg)                  # [B,T] enemy count at candidate
    # lexicographic key (count, hops-from-src, region) packed into one int64 per (src,tok)
    score = cnt_tok[:, None, :] * 65536 + tt * 256 + tokreg[:, None, :]     # [B,N,T]
    score = torch.where(pool_tok[:, None, :], score, torch.full_like(score, _JBIG))
    best_tok = score.argmin(2)                           # [B,N] chosen candidate token per src
    tgt_by_src = tok.gather(1, best_tok.clamp(max=T - 1))  # [B,N] chosen enemy-building region

    # ---- build the per-region move plan (later layers override earlier) --------
    # 1) own bases (not the rally): funnel their surplus to the gather point
    ob = owned_base & (reg_ids != rally[:, None])
    move = torch.where(ob, gather_to[:, None].expand(B, N), move)
    # 2) HQ: settle the next open stronghold, else funnel to the gather point. Matches
    # japper_bot: expansion (step 1) runs unconditionally, so the HQ keeps pushing a
    # settler toward an open target even while defending (gather-to-HQ is a no-op for
    # the HQ garrison anyway).
    hq_tgt = torch.where(expanding, expand_tgt, gather_to)
    move = torch.where(reg_ids == my_hq[:, None], hq_tgt[:, None].expand(B, N), move)
    # 3) field (stationary warriors on a razed / non-building region): chain to a target
    my_bldg = owner == me_own
    field = (stat_reg > 0) & (~my_bldg) & (~enemy_bldg)
    move = torch.where(field & has_pool[:, None], tgt_by_src, move)
    # 4) rally: recall to HQ while defending, else launch a wave at the trigger
    rally_stat = at(stat_reg, rally)
    wave_tgt = tgt_by_src.gather(1, rally[:, None]).squeeze(1)
    do_rally = defending | ((rally_stat >= JAP_WAVE_TRIGGER) & has_pool)
    rally_tgt = torch.where(defending, my_hq, wave_tgt)
    move = torch.where((reg_ids == rally[:, None]) & do_rally[:, None],
                       rally_tgt[:, None].expand(B, N), move)

    # ---- build a base on the current open expansion target (one per turn) -------
    et_can = (expanding & (at(stat_reg, expand_tgt) > 0)
              & (at(opp_reg, expand_tgt) == 0) & (gold >= fe.BASE_COST[1]))
    emit_force(et_can, expand_tgt)

    # ---- train once the 2 bases are secured (or given up), keeping the reserve --
    n_afford = ((gold - JAP_GOLD_RESERVE) // TRAIN_COST).clamp(min=0)
    n = torch.minimum(traincap, n_afford)
    hq_mine = at(owner, my_hq) == me_own
    train = torch.where(expansion_done & hq_mine, n, train)

    return {'build': build, 'move': move, 'train': train, 'force_build': force}


def region_to_token(token_ids, region):
    """Map a region id [B] to its token index [B] via searchsorted on the ascending
    per-game token_ids [B,T]. The region is assumed to BE a token (move targets always
    are); non-token/-1 inputs give a harmless clamped index the caller masks off."""
    q = region.clamp(min=0)[:, None]
    T = token_ids.shape[1]
    return torch.searchsorted(token_ids, q).clamp(max=T - 1).squeeze(1)


def opponent_actions(pool_t1, pool_t2, o_op, opp_assign, env, side, B, N, dev,
                     rstate, jstate, relax_reg=None, forbid_tgt=None):
    """Sample opponent actions, grouping batch slots by their assigned opponent so
    each opponent runs once over its subset of games. The first N_SCRIPTED unified
    indices are fixed scripted bots (0 = rusher, 1 = japper); index p >= N_SCRIPTED
    maps to net snapshot pool_t1[p-N_SCRIPTED]/pool_t2[p-N_SCRIPTED]. Returns
    (action, new_hq_commit) -- the scripted bots never commit."""
    build_full = torch.zeros(B, N, dtype=torch.bool, device=dev)
    move_full = torch.full((B, N), -1, dtype=torch.long, device=dev)
    train_full = torch.zeros(B, dtype=torch.long, device=dev)
    force_full = torch.zeros(B, N, dtype=torch.bool, device=dev)
    commit_full = torch.zeros(B, dtype=torch.bool, device=dev)
    for p in torch.unique(opp_assign).tolist():
        rows = (opp_assign == p).nonzero(as_tuple=True)[0]
        if p < N_SCRIPTED:
            act = (rusher_action(env, side, rstate) if p == RUSHER_IDX
                   else japper_action(env, side, jstate))
            build_full[rows] = act['build'][rows]
            move_full[rows] = act['move'][rows]
            train_full[rows] = act['train'][rows]
            force_full[rows] = act['force_build'][rows]
        else:
            sub = slice_obs(o_op, rows)
            sub_relax = relax_reg[rows] if relax_reg is not None else None
            sub_forbid = forbid_tgt[rows] if forbid_tgt is not None else None
            act, _, _, ncommit = sample_policy(pool_t1[p - N_SCRIPTED],
                                               pool_t2[p - N_SCRIPTED], sub, N,
                                               relax_reg=sub_relax,
                                               forbid_tgt=sub_forbid)
            build_full[rows] = act['build']
            move_full[rows] = act['move']
            train_full[rows] = act['train']
            force_full[rows] = act['force_build']
            commit_full[rows] = ncommit
    return ({'build': build_full, 'move': move_full, 'train': train_full,
             'force_build': force_full},
            commit_full)


def train(cfg: Config, device=None, seed=0, log_every=1):
    dd = Dist.init()
    tune_backend(cfg.torch_threads)
    if device is None:
        n_gpu = torch.cuda.device_count()
        device = f"cuda:{min(dd.local_rank, n_gpu - 1)}" if n_gpu else "cpu"
    on_cuda = torch.device(device).type == "cuda"

    # cfg.B is the TOTAL number of parallel games; each rank simulates its share,
    # so an N-GPU run collects exactly the same data per iteration as a 1-GPU run
    # with the same config (same horizon, same minibatch count, same lr).
    if cfg.B % dd.world:
        raise ValueError(f"B={cfg.B} must be divisible by world size {dd.world}")
    B_loc = cfg.B // dd.world
    if cfg.minibatch % dd.world:
        raise ValueError(f"minibatch={cfg.minibatch} must be divisible by "
                         f"world size {dd.world}")
    mb_loc = cfg.minibatch // dd.world

    # Every rank must build IDENTICAL nets but simulate DIFFERENT games. Seed the
    # net init from `seed` alone (weights are broadcast from rank 0 right after,
    # so this is belt-and-braces), then offset the stream per rank.
    torch.manual_seed(seed)
    map_workers = cfg.map_workers          # PER RANK (see config.yaml)
    maps = make_maps(B_loc, seed + 7919 * dd.rank, workers=map_workers)
    # reserve capacity for the largest possible map so per-episode regen fits any size
    env = fe.FastEnv(maps, device=device, n_cap=249, t_cap=21)
    env._map_rng = __import__("random").Random(seed + 12345 + 7919 * dd.rank)
    # background map generation for regen() -- see map_gen.MapFactory
    map_factory = None
    if map_workers > 0:
        map_factory = map_gen.MapFactory(workers=map_workers,
                                         seed=seed + 31 * dd.rank + 1,
                                         depth=cfg.map_queue)
        env.map_factory = map_factory
    N = env.N

    actor_t1 = ActorT1(d=cfg.d_model).to(device)
    actor_t2 = ActorT2(cfg.d_model + T2_EXTRA, d=cfg.d_model).to(device)
    critic = Critic(d=cfg.d_model).to(device)
    dd.broadcast_params([actor_t1, actor_t2, critic])
    torch.manual_seed(seed + 104729 * dd.rank)     # diverge the per-rank streams

    mk_t1 = lambda: ActorT1(d=cfg.d_model)
    mk_t2 = lambda: ActorT2(cfg.d_model + T2_EXTRA, d=cfg.d_model)

    # opponent pool. Unified index 0 is a FIXED scripted rusher (never evicted,
    # tracked with its own EMA win rate); index i>=1 is net snapshot pool_t1[i-1].
    # It starts with the rusher + a frozen copy of the initial policy. The total pool
    # cap is cfg.pool_max_size plus one slot per PERMANENT opponent added, so the evictable
    # room stays constant: ordinary win-rate-triggered self-snapshots are all evictable
    # (oldest first), while the scripted bots and the periodic perm_snapshot_every
    # self-snapshots are permanent. pool_perm[j] marks net snapshot j as permanent.
    # pool_ids names each opponent: 'rusher'/'japper' for the fixed ones, 'main@<it>'
    # for the periodic permanent self-snapshots, and an int for the ordinary evictable
    # ones (those are logged to wandb by SLOT -- opponent1..N -- so the number of
    # curves stays bounded).
    pool_t1 = [frozen_copy(actor_t1)]
    pool_t2 = [frozen_copy(actor_t2)]
    # EMA win rate, indexed by unified opponent index: [rusher, japper, net0]
    pool_wr = torch.full((N_SCRIPTED + 1,), 0.5)
    pool_ids = ['rusher', 'japper', 0]
    pool_perm = [False]                 # per net snapshot: never-evict?
    next_opp_id = 1
    n_perm_snaps = 0                    # periodic permanent self-snapshots taken so far
    # last iteration whose permanent self-snapshot already completed. Checkpointed, so
    # a resume that lands back on a boundary iteration (the checkpoint is written at
    # the START of each iteration) does not redo it.
    last_perm_it = -1
    opp_gen = torch.Generator().manual_seed(seed + 777 + 7919 * dd.rank)
    opp_assign = sample_opponents(B_loc, pool_wr, opp_gen,
                                  cfg.opp_sample_floor).to(device)
    # full-mobilisation flag per game (see opp_relax_frac): these games let the opponent
    # ignore the work-cap move restriction, and their results are excluded from the EMA
    # win rate. Constant within an episode; resampled when a game regenerates. Ephemeral
    # (not checkpointed) -- the env resets on resume, so a fresh draw is consistent.
    relaxed_game = torch.rand(B_loc, device=device) < cfg.opp_relax_frac

    main_L = Learner(actor_t1, actor_t2, critic, cfg.lr, on_cuda)
    opt_actor, opt_critic = main_L.opt_a, main_L.opt_c

    alpha = cfg.opp_ema_alpha

    # ---- where the rollout buffer lives ---------------------------------- #
    bytes_per_sample = _buffer_bytes_per_sample(env.mb.T)
    buf_bytes = bytes_per_sample * (cfg.steps_per_iter // dd.world)
    sdev = cfg.store_device
    spilled = False
    if sdev == "auto":
        sdev = "cpu"
        if on_cuda:
            free, _tot = torch.cuda.mem_get_info(torch.device(device))
            sdev = device if buf_bytes < cfg.store_vram_frac * free else "cpu"
            spilled = sdev == "cpu"
    elif sdev == "cuda":
        sdev = device
    if dd.is_main:
        print(f"rollout buffer: ~{buf_bytes / 2**30:.2f} GiB/rank on "
              f"{'GPU' if sdev != 'cpu' else 'host RAM'}"
              f"  ({cfg.steps_per_iter // dd.world:,} samples x {bytes_per_sample / 1024:.1f} KiB)")
        if spilled:
            print("  (does not fit VRAM -- the per-minibatch gather and H2D copy "
                  "will dominate the PPO update; lower steps_per_iter, raise "
                  "--gpus, or accept it)")

    # ---- resume from checkpoint if present ----
    start_iter = 0
    if cfg.resume and os.path.exists(cfg.ckpt_path):
        ck = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        # strict=False so checkpoints predating the aux heads still load; also drop
        # shape-mismatched keys so a checkpoint with the OLD 2-dim aux head (now 7)
        # transfers its backbone/policy/critic and re-inits the aux head fresh.
        def _compat(model, sd):
            cur = model.state_dict()
            return {k: v for k, v in sd.items()
                    if k in cur and cur[k].shape == v.shape}
        a1_miss = actor_t1.load_state_dict(_compat(actor_t1, ck['actor_t1']), strict=False)
        actor_t2.load_state_dict(ck['actor_t2'])
        c_miss = critic.load_state_dict(_compat(critic, ck['critic']), strict=False)
        try:
            opt_actor.load_state_dict(ck['opt_actor'])
            opt_critic.load_state_dict(ck['opt_critic'])
        except (ValueError, KeyError) as e:
            # param set changed (aux heads added) -> Adam moments can't map; reset.
            if dd.is_main:
                print(f"  optimizer state incompatible ({e}); reinitializing optimizers")
        if (a1_miss.missing_keys or c_miss.missing_keys) and dd.is_main:
            print(f"  loaded pre-aux checkpoint; aux heads initialized fresh "
                  f"(actor missing {len(a1_miss.missing_keys)}, "
                  f"critic missing {len(c_miss.missing_keys)})")
        pool_t1 = [frozen_from_state(mk_t1, sd, device) for sd in ck['pool_t1']]
        pool_t2 = [frozen_from_state(mk_t2, sd, device) for sd in ck['pool_t2']]
        pool_wr = ck['pool_wr'].cpu()   # kept on CPU (used with the CPU opp_gen)
        pool_ids = ck['pool_ids']
        # checkpoints written before permanent snapshots existed have no pool_perm ->
        # every net snapshot in them is an ordinary (evictable) one.
        pool_perm = list(ck.get('pool_perm', [False] * len(pool_t1)))
        if len(pool_perm) != len(pool_t1):        # legacy/mismatched -> rebuild
            pool_perm = [False] * len(pool_t1)
        n_perm_snaps = int(ck.get('n_perm_snaps', 0))
        last_perm_it = int(ck.get('last_perm_it', -1))
        next_opp_id = ck['next_opp_id']
        # The checkpoint stores rank 0's assignment for B_loc games. A resume with a
        # different B (or on another rank) just redraws it -- the env is rebuilt from
        # scratch on resume anyway, so nothing is carried over that it must match.
        opp_assign = ck['opp_assign'].to(device)
        if opp_assign.numel() != B_loc or dd.rank != 0:
            opp_assign = sample_opponents(B_loc, ck['pool_wr'].cpu(), opp_gen,
                                          cfg.opp_sample_floor).to(device)
        # backward-compat: older checkpoints have fewer scripted slots than the
        # current N_SCRIPTED (rusher=0, japper=1). The scripted bots occupy the
        # leading unified indices ahead of the net snapshots; insert the MISSING
        # ones at their positions (right after the present scripted, before the
        # nets) and shift only the opp_assign references at/after that position.
        n_have = pool_wr.numel() - len(pool_t1)            # scripted slots present
        if n_have < N_SCRIPTED:
            missing = ['rusher', 'japper'][n_have:N_SCRIPTED]
            k = len(missing)
            pool_wr = torch.cat([pool_wr[:n_have],
                                 torch.full((k,), 0.5), pool_wr[n_have:]])
            pool_ids = list(pool_ids[:n_have]) + missing + list(pool_ids[n_have:])
            opp_assign = torch.where(opp_assign >= n_have, opp_assign + k, opp_assign)
            if dd.is_main:
                print(f"  migrated checkpoint: inserted scripted opponents {missing}")
        # RNG states must be CPU ByteTensors (map_location may have moved them). Each
        # rank offsets them so the ranks keep simulating DIFFERENT games after a resume.
        opp_gen.set_state(ck['opp_gen'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck.get('cuda_rng') is not None and torch.cuda.is_available():
            # only this rank's device (a multi-GPU run must not stamp its peers')
            torch.cuda.set_rng_state(ck['cuda_rng'][0].cpu(), device=device)
        if dd.rank:
            torch.manual_seed(int(torch.randint(1 << 30, (1,)).item()) + 7919 * dd.rank)
            opp_gen.manual_seed(seed + 777 + 104729 * dd.rank + ck['iter'])
        start_iter = ck['iter']
        if dd.is_main:
            print(f"resumed from {cfg.ckpt_path} at iter {start_iter} "
                  f"(pool size {len(pool_t1) + N_SCRIPTED})")

    run = None
    wstep = start_iter
    if cfg.use_wandb and dd.is_main:
        import wandb
        os.environ["WANDB_API_KEY"] = (
            "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")
        run = wandb.init(project="nypc2026-selfplay", config=vars(cfg))
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")

    def wlog(d):
        """Log one point on the monotone wandb step (rank 0 / wandb only)."""
        nonlocal wstep
        if run is None:
            return
        import wandb
        wandb.log(d, step=wstep)
        wstep += 1

    # Per-game rollout state: each side's previous-turn reachability + own aux gold
    # prediction (both are input features) and the scripted bots' state machines.
    # Fresh, not checkpointed: episodes never resume across runs (the env is reset on
    # resume), so a fresh state always aligns with a fresh env.
    rs = RolloutState(B_loc, env.mb.T, device)

    # map-factory starvation counter: a "miss" is an episode that had to generate
    # its map inline (~25 ms on the rollout's critical path) because the queue was
    # empty. A steady stream of them means map_workers is too low.
    prev_misses, warned_maps = 0, False

    # ------------------------------------------------------------------ #
    # ONE PPO ITERATION (rollout + update): which nets learn (L), which nets play
    # the other side (p_t1/p_t2 + opp_assign), and the hyperparameters (hp) are
    # all parameters so the main self-play loop is the only caller.
    #
    # pool_wr / opp_assign / relaxed_game are passed in AND returned, because the
    # EMA update and the per-episode redraws replace them.
    # ------------------------------------------------------------------ #
    def run_iteration(L, rs, p_t1, p_t2, pool_wr, opp_assign, relaxed_game, hp,
                      resample_opp=True):
        steps = hp['steps']
        epochs_i, ent_i, clip = hp['epochs'], hp['ent_coef'], hp['clip']
        relax_frac = hp['relax_frac']
        # local aliases so the body below reads exactly as it did when the main
        # agent was the only learner
        actor_t1, actor_t2, critic = L.t1, L.t2, L.cr
        opt_actor, opt_critic = L.opt_a, L.opt_c
        actor_params, critic_params = L.pa, L.pc
        pool_t1, pool_t2 = p_t1, p_t2
        prev_reach_ag, prev_reach_op = rs.reach_ag, rs.reach_op
        rstate, jstate = rs.rstate, rs.jstate

        buf = []
        # Episode bookkeeping is accumulated on the DEVICE and read once per
        # iteration: the old per-finished-game Python loop forced a GPU->CPU sync
        # on every step that ended an episode, which at large B is every step.
        n_opp = pool_wr.numel()
        ep_stat = torch.zeros(2, device=device)              # [sum reward, count]
        wr_sum = torch.zeros(n_opp, device=device)           # per-opponent result sum
        wr_cnt = torch.zeros(n_opp, device=device)           # ...and game count
        for s in range(steps):
            o_ag = extract(env, 0, prev_reach_ag)
            with torch.no_grad():
                act_ag, store, _, ncommit_ag = sample_policy(actor_t1, actor_t2, o_ag, N)
                val = critic.value(o_ag['t1_crit'], o_ag['glob_crit'], o_ag['tmask'])
                o_op = extract(env, 1, prev_reach_op)
                # per-region full-mobilisation mask for the opponent this turn: only in
                # flagged games, each opponent region independently with opp_relax_prob.
                relax_reg = (relaxed_game[:, None]
                             & (torch.rand(B_loc, N, device=device) < cfg.opp_relax_prob))
                act_op, ncommit_op = opponent_actions(
                    pool_t1, pool_t2, o_op, opp_assign, env, 1, B_loc, N, device,
                    rstate, jstate, relax_reg=relax_reg)
            # ---------------- turn-1 opening split ----------------
            # On a game's first turn (env.day==0) our region-move action space would send
            # the HQ's two spare warriors to ONE stronghold. Re-infer with warrior #1
            # already en route (gold / HQ movable-count / target arrivals updated in the
            # features) and its target MASKED, so warrior #2 goes to a DIFFERENT stronghold
            # -- both moves execute this turn (the normal registration below skips the
            # already-moving #1 via ~w_move). The STORED transition stays the FIRST
            # inference (trained as usual, o_ag/store/val untouched); the second move is
            # folded into the executed action, matching how the submission bots auto-split.
            act_ag_exec = act_ag
            if cfg.opening_split:
                turn1 = env.day == 0
                with torch.no_grad():
                    hq0 = env.hq_region[:, 0]
                    a_reg0 = act_ag['move'].gather(1, hq0[:, None]).squeeze(1)
                    split0 = turn1 & (a_reg0 >= 0)
                    if bool(split0.any()):
                        env.opening_premove(0, hq0, a_reg0, split0)          # warrior #1 -> A
                        o2 = extract(env, 0, prev_reach_ag)
                        forbid0 = torch.where(split0, region_to_token(env.mb.token_ids, a_reg0),
                                              torch.full_like(a_reg0, -1))
                        act2, _, _, _ = sample_policy(actor_t1, actor_t2, o2, N,
                                                       forbid_tgt=forbid0)
                        b_reg0 = act2['move'].gather(1, hq0[:, None]).squeeze(1)
                        mv = act_ag['move'].clone()
                        r0 = split0.nonzero(as_tuple=True)[0]
                        mv[r0, hq0[r0]] = b_reg0[r0]                          # execute HQ -> B
                        act_ag_exec = {**act_ag, 'move': mv}
                    # net opponents only (scripted rusher/japper keep their region moves)
                    hq1 = env.hq_region[:, 1]
                    a_reg1 = act_op['move'].gather(1, hq1[:, None]).squeeze(1)
                    split1 = turn1 & (opp_assign >= N_SCRIPTED) & (a_reg1 >= 0)
                    if bool(split1.any()):
                        env.opening_premove(1, hq1, a_reg1, split1)
                        o2op = extract(env, 1, prev_reach_op)
                        forbid1 = torch.where(split1, region_to_token(env.mb.token_ids, a_reg1),
                                              torch.full_like(a_reg1, -1))
                        act_op2, _ = opponent_actions(
                            pool_t1, pool_t2, o2op, opp_assign, env, 1, B_loc, N, device,
                            rstate, jstate, relax_reg=relax_reg, forbid_tgt=forbid1)
                        b_reg1 = act_op2['move'].gather(1, hq1[:, None]).squeeze(1)
                        r1 = split1.nonzero(as_tuple=True)[0]
                        act_op['move'][r1, hq1[r1]] = b_reg1[r1]             # execute HQ -> B
            # this turn's reach becomes next turn's feature
            prev_reach_ag = o_ag['reach_raw'].clone()
            prev_reach_op = o_op['reach_raw'].clone()
            env.step({'left': act_ag_exec, 'right': act_op}, relax_right=relax_reg)
            # carry the HQ-upgrade commitment to next turn (regen resets it for
            # finished games below).
            env.hq_commit[:, 0] = ncommit_ag
            env.hq_commit[:, 1] = ncommit_op
            r, done = reward_done(env)
            # auxiliary labels: what the agent (side 0) will SEE next turn -- per
            # 거점 enemy garrison + reachable-within-1..5-turns, and the opp gold.
            # Computed on the post-step state (before regen), attached to o_ag.
            tok_tgt, opp_gold = env.aux_label(0)
            gold_aux = torch.log1p(tok_tgt)                 # [B,T,6] ln(1+count)
            gold_glob = torch.log1p(opp_gold / 100.0)       # [B]     ln(1+gold/100)
            dn = done.float()
            ep_stat[0] += (r * dn).sum()
            ep_stat[1] += dn.sum()
            # per-opponent win/count tallies, scattered on the GPU (drawn down into
            # the EMA once, after the rollout). Full-mobilisation games are excluded
            # (their relaxed opponent isn't representative of real play).
            res = torch.where(r > 0, torch.ones_like(r),
                              torch.where(r < 0, torch.zeros_like(r),
                                          torch.full_like(r, 0.5)))
            cnt_m = dn * (~relaxed_game).float()
            wr_sum.scatter_add_(0, opp_assign, res * cnt_m)
            wr_cnt.scatter_add_(0, opp_assign, cnt_m)
            # On a GPU-resident buffer .to() is a no-op alias: the tensors are
            # already there and freshly allocated every step, so nothing copies.
            rec = {k: v.to(sdev) for k, v in store.items()}
            rec['value'] = val.to(sdev)
            rec['reward'] = r.to(sdev)
            rec['done'] = dn.to(sdev)
            rec['gold_aux'] = gold_aux.to(sdev)
            rec['gold_glob'] = gold_glob.to(sdev)
            buf.append(rec)
            if bool(done.any()):
                drows = done.nonzero(as_tuple=True)[0]
                # fresh opponent for each finished game, then fresh map + reset.
                # pool_wr is the value from the START of this iteration (it is only
                # advanced once per iteration now, so ranks stay in lockstep); with
                # alpha=0.02 and hundreds of games per iter this is indistinguishable
                # from the old update-as-you-go sampling.
                if resample_opp:
                    opp_assign[drows] = sample_opponents(
                        drows.numel(), pool_wr, opp_gen, cfg.opp_sample_floor).to(device)
                # re-roll the full-mobilisation flag for each regenerated game
                relaxed_game[drows] = (torch.rand(drows.numel(), device=device)
                                       < relax_frac)
                env.regen(done)
                # new map -> no prior turn; baseline the reach deltas at 0
                prev_reach_ag[drows] = 0
                prev_reach_op[drows] = 0
                # restart the scripted opponents' state machines for the new games
                rstate.reset_rows(drows)
                jstate.reset_rows(drows)

        with torch.no_grad():
            # bootstrap value on the post-rollout state; same prev_reach the next
            # iter's first step will use (no env step happened in between).
            o_ag = extract(env, 0, prev_reach_ag)
            last_val = critic.value(o_ag['t1_crit'], o_ag['glob_crit'], o_ag['tmask']).to(sdev)

        # ---- opponent-pool EMA win rates ------------------------------------ #
        # Applied once per iteration, from tallies all-reduced across ranks, so
        # every rank ends the iteration with a bit-identical pool_wr (and thus
        # makes the same add/evict decision below). n results with mean r for one
        # opponent collapse to the closed form of n successive EMA steps with a
        # shared value: wr <- (1-a)^n wr + (1 - (1-a)^n) r. Unlike the old
        # game-at-a-time loop this is order-independent, which is what lets the
        # two ranks agree.
        dd.all_reduce_(wr_sum); dd.all_reduce_(wr_cnt); dd.all_reduce_(ep_stat)
        wc = wr_cnt.cpu(); ws = wr_sum.cpu()
        decay = (1.0 - alpha) ** wc
        mean_res = ws / wc.clamp(min=1)
        pool_wr = torch.where(wc > 0, decay * pool_wr + (1 - decay) * mean_res, pool_wr)
        ep_rewards, ep_count = float(ep_stat[0]), int(ep_stat[1])

        # ---- GAE ----
        adv = [None] * steps
        gae = torch.zeros(B_loc, device=sdev)
        for t in reversed(range(steps)):
            nonterm = 1.0 - buf[t]['done']
            nextv = last_val if t == steps - 1 else buf[t + 1]['value']
            delta = buf[t]['reward'] + cfg.gamma * nextv * nonterm - buf[t]['value']
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae.clone()
        for t in range(steps):
            buf[t]['adv'] = adv[t]
            buf[t]['ret'] = adv[t] + buf[t]['value']

        # ---- flatten ----
        keys = ['t1', 'glob', 't1_crit', 'glob_crit', 'tmask', 'extra4', 'tok_dist', 'normx', 'normy',
                'build_mask', 'build_outcome', 'train_mask', 'train_cat',
                'valid_src', 'tgt', 'surplus_pb', 'tgt_allowed', 'old_logp',
                'adv', 'ret', 'value', 'gold_aux', 'gold_glob']
        # Concatenate key by key and drop each key's per-step tensors as we go: a
        # dict-comprehension would hold the whole buffer AND its copy at once,
        # which on a GPU-resident buffer means peaking at twice the VRAM.
        flat = {}
        for k in keys:
            flat[k] = torch.cat([buf[t][k] for t in range(steps)], dim=0)
            for t in range(steps):
                buf[t].pop(k, None)
        buf.clear()
        Ntot = flat['t1'].shape[0]
        # advantage whitening uses the GLOBAL batch statistics, so an N-GPU run
        # normalizes exactly like a 1-GPU run on the same data
        a = flat['adv']
        s = torch.stack([a.sum(), (a * a).sum(),
                         torch.tensor(float(Ntot), device=a.device)]).to(device)
        dd.all_reduce_(s)
        a_mean = s[0] / s[2]
        a_std = (s[1] / s[2] - a_mean * a_mean).clamp(min=0).sqrt()
        flat['adv'] = (a - a_mean.to(a.device)) / (a_std.to(a.device) + 1e-8)

        # ---- PPO epochs ----
        # Metrics are accumulated on the DEVICE: a .item() per minibatch would sync
        # the stream ~250 times per iteration for numbers only printed once.
        acc = torch.zeros(6, device=device)     # ploss, vloss, ent, kl, aux_a, aux_c
        nb = 0
        epochs_run = 0
        idx_dev = "cpu" if sdev == "cpu" else device
        for _ in range(epochs_i):
            perm = torch.randperm(Ntot, device=idx_dev)
            ep_kl = torch.zeros((), device=device)
            ep_nb = 0
            for i in range(0, Ntot, mb_loc):
                idx = perm[i:i + mb_loc]
                mb = {k: flat[k][idx].to(device, non_blocking=True) for k in keys}
                # ---- actor update (policy + entropy + gold-production aux) ----
                logp, ent, aux_a = evaluate_policy(actor_t1, actor_t2, mb)
                logratio = logp - mb['old_logp']
                ratio = torch.exp(logratio)
                with torch.no_grad():
                    # Schulman's positive KL estimator E[(r-1) - log r] ~ KL(old||new)
                    approx_kl = ((ratio - 1) - logratio).mean()
                adv_b = mb['adv']
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b
                ploss = -torch.min(s1, s2).mean()
                eloss = -ent.mean()
                aux_loss_a = aux_losses(aux_a, mb['gold_aux'], mb['gold_glob'], mb['tmask'])
                actor_loss = ploss + ent_i * eloss + cfg.aux_coef * aux_loss_a
                opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                # average gradients across ranks BEFORE clipping, so the clip acts
                # on the gradient the optimizer will actually apply
                dd.all_reduce_grads(actor_params)
                nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                opt_actor.step()

                # ---- critic update (value regression + gold-production aux) ----
                value, aux_c = critic.value_aux(mb['t1_crit'], mb['glob_crit'], mb['tmask'])
                vloss = F.mse_loss(value, mb['ret'])
                aux_loss_c = aux_losses(aux_c, mb['gold_aux'], mb['gold_glob'], mb['tmask'])
                critic_loss = vloss + cfg.aux_coef * aux_loss_c
                opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                dd.all_reduce_grads(critic_params)
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                opt_critic.step()

                with torch.no_grad():
                    acc += torch.stack([ploss.detach(), vloss.detach(), ent.mean(),
                                        approx_kl, aux_loss_a.detach(),
                                        aux_loss_c.detach()])
                    ep_kl += approx_kl
                nb += 1
                ep_nb += 1

            epochs_run += 1
            # KL early stopping: if this epoch's mean drift exceeds the target,
            # stop refreshing on this (now-stale) batch before overshooting. The
            # (single, per-epoch) sync only happens when the check is enabled.
            if cfg.target_kl is not None:
                kl_ep = (ep_kl / max(ep_nb, 1)).reshape(1)
                dd.all_reduce_(kl_ep, "mean")
                if float(kl_ep) > cfg.target_kl:
                    break

        # ---- value-net explained variance ----
        with torch.no_grad():
            ret_f, val_f = flat['ret'].to(device), flat['value'].to(device)
            evs = torch.stack([ret_f.sum(), (ret_f * ret_f).sum(),
                               (ret_f - val_f).sum(), ((ret_f - val_f) ** 2).sum(),
                               torch.tensor(float(Ntot), device=device)])
            dd.all_reduce_(evs)
            n_ev = evs[4]
            var_ret = evs[1] / n_ev - (evs[0] / n_ev) ** 2
            var_err = evs[3] / n_ev - (evs[2] / n_ev) ** 2
            ev = float(1.0 - var_err / (var_ret + 1e-8))
            acc_m = dd.all_reduce_(acc / max(nb, 1), "mean").tolist()
        pl, vl, el, kl, ax_a, ax_c = acc_m
        del flat

        # the rollout rebinds these locals every step; hand them back so the next
        # iteration continues from this one's last turn
        rs.reach_ag, rs.reach_op = prev_reach_ag, prev_reach_op

        return dict(pool_wr=pool_wr, opp_assign=opp_assign, relaxed_game=relaxed_game,
                    ep_rewards=ep_rewards, ep_count=ep_count,
                    wr_sum=ws, wr_cnt=wc, pl=pl, vl=vl, el=el, kl=kl,
                    ax_a=ax_a, ax_c=ax_c, ev=ev,
                    epochs_run=epochs_run, steps=steps)

    def snapshot(it):
        """Crash-safe checkpoint written before each iteration, rank 0 only: every
        rank holds identical weights/pool, and the rollout state that does differ is
        regenerated on resume anyway."""
        if not dd.is_main:
            return
        save_ckpt(cfg.ckpt_path, {
            'iter': it,
            'actor_t1': actor_t1.state_dict(),
            'actor_t2': actor_t2.state_dict(),
            'critic': critic.state_dict(),
            'opt_actor': opt_actor.state_dict(),
            'opt_critic': opt_critic.state_dict(),
            'pool_t1': [n.state_dict() for n in pool_t1],
            'pool_t2': [n.state_dict() for n in pool_t2],
            'pool_wr': pool_wr,
            'pool_ids': pool_ids,
            'pool_perm': pool_perm,
            'n_perm_snaps': n_perm_snaps,
            'last_perm_it': last_perm_it,
            'next_opp_id': next_opp_id,
            'opp_assign': opp_assign.cpu(),
            'opp_gen': opp_gen.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': ([torch.cuda.get_rng_state(device)]
                         if on_cuda else None),
            'cfg': vars(cfg),
        })

    steps = max(1, cfg.steps_per_iter // cfg.B)
    for it in range(start_iter, cfg.iters):
        # ---- checkpoint before starting this iter (for crash-safe resume) ----
        snapshot(it)

        # ---- every cfg.perm_snapshot_every iters: freeze the CURRENT main actor
        # into the pool as a PERMANENT opponent (never evicted, cap grows by 1).
        if (cfg.perm_snapshot_every and it > 0
                and it % cfg.perm_snapshot_every == 0
                and it != last_perm_it):
            last_perm_it = it
            pool_t1.append(frozen_copy(actor_t1))
            pool_t2.append(frozen_copy(actor_t2))
            pool_wr = torch.cat([pool_wr, torch.full((1,), 0.5)])
            pool_ids.append(f'main@{it}')
            pool_perm.append(True)
            n_perm_snaps += 1
            if dd.is_main:
                print(f"== iter {it}: main actor frozen into the pool as PERMANENT "
                      f"opponent 'main@{it}' -> pool {len(pool_t1) + N_SCRIPTED}/"
                      f"{cfg.pool_max_size + sum(pool_perm)}")
            wlog({'iter': it, 'n_perm_snaps': n_perm_snaps})
            # no opp_assign redraw: games in flight keep their opponent and pick the
            # new one up when they finish (same as an ordinary pool add).
            snapshot(it)    # persist the pool addition

        t0 = time.time()
        res = run_iteration(main_L, rs, pool_t1, pool_t2, pool_wr, opp_assign,
                            relaxed_game,
                            dict(steps=steps, epochs=cfg.epochs, ent_coef=cfg.ent_coef,
                                 clip=cfg.clip, relax_frac=cfg.opp_relax_frac))
        pool_wr, opp_assign = res['pool_wr'], res['opp_assign']
        relaxed_game = res['relaxed_game']
        ep_rewards, ep_count = res['ep_rewards'], res['ep_count']
        pl, vl, el, kl = res['pl'], res['vl'], res['el'], res['kl']
        ax_a, ax_c, ev = res['ax_a'], res['ax_c'], res['ev']
        epochs_run = res['epochs_run']

        added = False

        # ---- grow the pool when the agent beats even the hardest opponent ----
        # Cap = cfg.pool_max_size + one slot per PERMANENT opponent, so the evictable
        # room is constant. When full, evict the oldest EVICTABLE net; the scripted bots
        # (unified indices < N_SCRIPTED) and the periodic permanent self-snapshots are
        # never touched. The `while` also trims a checkpoint loaded with more
        # snapshots than fit.
        pool_cap = cfg.pool_max_size + sum(pool_perm)
        if float(pool_wr.min()) > cfg.pool_add_threshold:
            while (len(pool_t1) + N_SCRIPTED >= pool_cap
                   and not all(pool_perm) and pool_t1):
                evict_pos = next(i for i, p in enumerate(pool_perm) if not p)
                ev_idx = evict_pos + N_SCRIPTED
                pool_t1.pop(evict_pos); pool_t2.pop(evict_pos)
                pool_wr = torch.cat([pool_wr[:ev_idx], pool_wr[ev_idx + 1:]])
                pool_ids.pop(ev_idx)
                pool_perm.pop(evict_pos)
                # rows on the evicted opponent fall back to the rusher (index 0);
                # higher unified indices shift down one. Reassign BEFORE shifting.
                opp_assign = torch.where(opp_assign == ev_idx,
                                         torch.zeros_like(opp_assign), opp_assign)
                opp_assign = torch.where(opp_assign > ev_idx,
                                         opp_assign - 1, opp_assign)
            pool_t1.append(frozen_copy(actor_t1))
            pool_t2.append(frozen_copy(actor_t2))
            pool_wr = torch.cat([pool_wr, torch.full((1,), 0.5)])
            pool_ids.append(next_opp_id)
            pool_perm.append(False)
            next_opp_id += 1
            added = True

        wr = (ep_rewards / max(ep_count, 1))
        dt = time.time() - t0
        misses = 0
        if map_factory is not None:
            misses, prev_misses = map_factory.misses - prev_misses, map_factory.misses
            if (not warned_maps and dd.is_main and it > 0
                    and misses > 0.2 * max(ep_count // max(dd.world, 1), 1)):
                warned_maps = True
                print(f"  note: {misses} of this rank's episodes generated their map "
                      f"inline (queue starved) -- raise map_workers above "
                      f"{cfg.map_workers} to keep it off the rollout's critical path")
        if it % log_every == 0 and dd.is_main:
            print(f"iter {it:3d} | "
                  f"eps {ep_count:5d} | avg_ep_R {wr:+6.2f} | "
                  f"ploss {pl:+.4f} vloss {vl:.3f} ent {el:.3f} kl {kl:.4f} "
                  f"aux {ax_a:.3f}/{ax_c:.3f} ep {epochs_run}/{cfg.epochs} | "
                  f"ev {ev:+.3f} "
                  f"pool {len(pool_t1)+N_SCRIPTED}/{pool_cap} "
                  f"wr_min {float(pool_wr.min()):.2f} "
                  f"rush_wr {float(pool_wr[RUSHER_IDX]):.2f} "
                  f"jap_wr {float(pool_wr[JAPPER_IDX]):.2f} | "
                  f"{steps*cfg.B/dt:,.0f} steps/s ({dt:.1f}s)")

        if run is not None:
            log = {
                'iter': it,
                'avg_ep_R': wr,
                'episodes': ep_count,
                'ploss': pl,
                'vloss': vl,
                'entropy': el,
                'approx_kl': kl,
                'aux_loss_actor': ax_a,
                'aux_loss_critic': ax_c,
                'epochs_run': epochs_run,
                'lr': cfg.lr,
                'ent_coef': cfg.ent_coef,
                'steps_per_iter': cfg.steps_per_iter,
                'value_ev': ev,
                'pool_size': len(pool_t1) + N_SCRIPTED,
                'pool_cap': pool_cap,
                'opp_winrate_min': float(pool_wr.min()),
                'opp_winrate_mean': float(pool_wr.mean()),
                'opp_winrate_rusher': float(pool_wr[RUSHER_IDX]),
                'opp_winrate_japper': float(pool_wr[JAPPER_IDX]),
                'pool_added': int(added),
                'steps_per_s': steps * cfg.B / dt,
                'world_size': dd.world,
                'map_queue_misses': misses,
            }
            # Per-opponent EMA win rate. The EVICTABLE snapshots are keyed by their
            # SLOT ('opponent1' = oldest surviving one), not by identity, so the set of
            # curves is bounded by cfg.pool_max_size - N_SCRIPTED however long the run
            # goes: on an eviction each remaining snapshot shifts down one slot and
            # opponentN simply continues with the next one. Only the permanent entries
            # (rusher/japper, exp@<it>, main@<it>) get a curve of their own, and those
            # are added on a fixed schedule.
            slot = 0
            for k, oid in enumerate(pool_ids):
                if k >= N_SCRIPTED and not pool_perm[k - N_SCRIPTED]:
                    slot += 1
                    oid = f'opponent{slot}'
                log[f'opp_winrate/{oid}'] = float(pool_wr[k])
            wlog(log)

    if run is not None:
        run.finish()
    if map_factory is not None:
        map_factory.close()
    dd.shutdown()
    return actor_t1, actor_t2, critic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    ap.add_argument("--config", default="config.yaml",
                    help="YAML hyperparameter file (used if it exists)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    ap.add_argument("--no-resume", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--gpus", type=int, default=None,
                    help="data-parallel training across N GPUs (default: all visible "
                         "ones; 1 disables). cfg.B and cfg.minibatch are the TOTALS "
                         "and are split evenly, so the data per iteration is identical "
                         "to a single-GPU run.")
    ap.add_argument("--map-workers", type=int, default=None,
                    help="background processes pre-generating maps (0 = inline)")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (overrides config)")
    args = ap.parse_args()

    if args.smoke:
        cfg = Config(B=8, steps_per_iter=2000, iters=2, minibatch=512, d_model=32,
                     use_wandb=False, resume=False, ckpt_path="checkpoint_smoke.pt",
                     map_workers=0, store_device="cpu")
    else:
        cfg = load_config(args.config) if os.path.exists(args.config) else Config()
        if args.B is not None: cfg.B = args.B
        if args.steps is not None: cfg.steps_per_iter = args.steps
        if args.iters is not None: cfg.iters = args.iters
        if args.no_wandb: cfg.use_wandb = False
        if args.no_resume: cfg.resume = False
    if args.map_workers is not None:
        cfg.map_workers = args.map_workers
    if args.ckpt is not None:
        cfg.ckpt_path = args.ckpt

    # ---- multi-GPU launch ------------------------------------------------- #
    # Already inside a torchrun/elastic launch? Just train; Dist.init() picks the
    # rank up from the environment. Otherwise spawn one worker per GPU ourselves
    # so `python ppo_selfplay.py --gpus 2` is all the user has to type. Nothing
    # above has touched CUDA yet, which is what makes spawning safe here.
    if "WORLD_SIZE" in os.environ:
        train(cfg, device=args.device)
        return
    n_gpu = args.gpus
    if n_gpu is None:
        n_gpu = torch.cuda.device_count() if args.device is None and not args.smoke else 1
    n_gpu = max(1, min(n_gpu, max(1, torch.cuda.device_count())))
    if n_gpu <= 1:
        train(cfg, device=args.device)
        return
    print(f"launching data-parallel training on {n_gpu} GPUs "
          f"(B {cfg.B} -> {cfg.B // n_gpu}/rank, minibatch {cfg.minibatch} -> "
          f"{cfg.minibatch // n_gpu}/rank)")
    import torch.multiprocessing as tmp
    port = str(29500 + (os.getpid() % 2000))
    tmp.spawn(_dist_worker, args=(n_gpu, cfg, port), nprocs=n_gpu, join=True)


def _dist_worker(rank, world, cfg, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    train(cfg)


if __name__ == "__main__":
    main()
