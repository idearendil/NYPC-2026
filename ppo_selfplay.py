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
from fast_env import (OWN_LEFT, OWN_RIGHT, KIND_HQ, KIND_BASE, MOVE_COST,
                      TRAIN_COST, HQ_MAXLEVEL, BASE_MAXLEVEL, HQ_HEAL, BASE_HEAL,
                      MAX_DAYS)

TOK_FEAT = 31          # 14 scalars + 5 arrive + 5 reach (all log1p) + 2 norm coords
                       #                                   + 5 reach-delta vs prev turn
GLOB_FEAT = 12         # 11 + HQ-upgrade "turns to afford" (log1p)
T2_EXTRA = 8           # 4 logged + surplus + travel + 2 norm coord diffs
COST_INF = 1_000_000_000


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
        # auxiliary head: per-token next-turn gold-production change (/WORK_INCOME)
        # for [me, opp]. Shapes the shared encoder; unused at inference.
        self.aux = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.d = d

    def forward(self, t1, glob, tmask):
        B, T, _ = t1.shape
        x = torch.cat([t1, glob[:, None, :].expand(B, T, GLOB_FEAT)], dim=2)
        kpm = ~tmask
        h = self.enc(x, kpm)
        return h, self.head(h)


class ActorT2(nn.Module):
    def __init__(self, d_in, d=64, heads=4, ff=128, layers=2):
        super().__init__()
        self.enc = Encoder(d_in, d, heads, ff, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, tmask):
        h = self.enc(x, ~tmask)
        return self.head(h).squeeze(-1)        # [Q,T]


class Critic(nn.Module):
    def __init__(self, d=64, heads=4, ff=128, layers=3):
        super().__init__()
        self.enc = Encoder(TOK_FEAT + GLOB_FEAT, d, heads, ff, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # auxiliary head: per-token next-turn gold-production change (/WORK_INCOME)
        # for [me, opp]. Shapes the critic's encoder.
        self.aux = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

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
    raw24 = slog1p(tokens[:, :, :24].float())          # the 24 log1p features
    # raw (un-log'd) enemy-reachable-within-1..5-turns counts, packed at [19:24];
    # returned so the caller can feed it back next turn as prev_reach for the delta.
    reach_raw = tokens[:, :, 19:24].float()            # [B,T,5]

    g = glob.float()
    glob_t = torch.stack([
        g[:, 0] / 10 - 10,
        plog1p(g[:, 1] / 10), plog1p(g[:, 2] / 10),
        plog1p(g[:, 3]),      plog1p(g[:, 4]),
        plog1p(g[:, 5] / 100), plog1p(g[:, 6] / 100),
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

    tmask = info['token_mask']
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
    t1 = torch.cat([raw24, normx[:, :, None], normy[:, :, None], reach_delta],
                   dim=2)  # [B,T,31]

    cnt, sumhp = env._side_counts()
    turret, workcap = env._turret(), env._workcap()

    def gtok(field):
        return field.gather(1, tok_g)

    my_cnt = gtok(cnt[:, :, side]); op_cnt = gtok(cnt[:, :, opp_idx])
    my_hps = gtok(sumhp[:, :, side]); op_hps = gtok(sumhp[:, :, opp_idx])
    own_t = gtok(env.b_owner); kind_t = gtok(env.b_kind)
    lvl_t = gtok(env.b_level); bhp_t = gtok(env.b_hp)
    tur_t = gtok(turret); wc_t = gtok(workcap)
    me_b = own_t == me; opp_b = (own_t != 0) & (own_t != me)

    my_tur = torch.where(me_b, tur_t, torch.zeros_like(tur_t))
    op_tur = torch.where(opp_b, tur_t, torch.zeros_like(tur_t))
    my_bhp = torch.where(me_b, bhp_t, torch.zeros_like(bhp_t))
    op_bhp = torch.where(opp_b, bhp_t, torch.zeros_like(bhp_t))
    my_wc = torch.where(me_b, wc_t, torch.zeros_like(wc_t))

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

    # T2 token-specific extra features (logged)
    e1 = plog1p((op_cnt + op_tur).float())
    e2 = plog1p((op_hps + op_bhp).float() / 5)
    e3 = plog1p((my_cnt + my_tur).float())
    e4 = plog1p((my_hps + my_bhp).float() / 5)
    extra4 = torch.stack([e1, e2, e3, e4], dim=2)

    tok_dist = tokens[:, :, 24:24 + T].float()       # observe packs dist after the 24 feats

    return dict(
        t1=t1, glob=glob_t, tmask=tmask, tok_ids=tok_ids,
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
def sample_policy(t1net, t2net, o, N):
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
    surplus_pb = (o['stat_cnt'] - wc_pb).clamp(min=0)
    owner_me_pb = o['owner_me'] | (exec_all & o['build_new'])
    hq_after = (o['hq_level'] + do_hq_now.long()).clamp(max=HQ_MAXLEVEL)

    # ---------------- MOVE (T2) ----------------
    # While committed only free moves are allowed: targets restricted to our own
    # building tokens (the HQ is always one, so a surplus source is never starved).
    # Sources still require surplus (valid_src), unchanged.
    valid_src = o['tmask'] & (surplus_pb > 0) & (MOVE_COST * surplus_pb <= gold1[:, None])
    tgt_allowed = torch.where(committed[:, None], o['tmask'] & owner_me_pb, o['tmask'])
    logits = t2_logits_sources(t2net, h1, o['extra4'], surplus_pb, o['tok_dist'],
                               o['normx'], o['normy'], o['tmask'], valid_src)
    logits = logits.masked_fill(~tgt_allowed[:, None, :], -1e9)
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
        t1=o['t1'], glob=o['glob'], tmask=o['tmask'],
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


def _greedy(item_mask, cost, gold, perm, T):
    """Greedy gold allocation: in random order, take an item if affordable."""
    remaining = gold.clone()
    exec_ = torch.zeros_like(item_mask)
    for k in range(T):
        idx = perm[:, k:k + 1]
        c = cost.gather(1, idx).squeeze(1)
        it = item_mask.gather(1, idx).squeeze(1)
        take = it & (c <= remaining)
        remaining = remaining - take.long() * c
        exec_.scatter_(1, idx, take[:, None])
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


def evaluate_policy(t1net, t2net, b):
    h1, head5 = t1net(b['t1'], b['glob'], b['tmask'])
    aux_pred = t1net.aux(h1)                 # [B,T,2] gold-production change pred
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
    lr: float = 3e-4
    epochs: int = 3
    minibatch: int = 4096
    # KL early stopping: stop an iter's PPO epochs once the policy has drifted
    # this far from the data-collection policy (approx_kl, Schulman estimator).
    # null/None disables it. Lets you raise lr/epochs without overshooting.
    target_kl: Optional[float] = 0.02
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    # auxiliary task: predict each 거점's next-turn gold-production change
    # (/WORK_INCOME, for [me, opp]) from the token encodings, on BOTH the actor and
    # critic. Shapes the encoders with a dense economy signal. 0 disables.
    aux_coef: float = 0.25
    max_grad_norm: float = 1.0
    d_model: int = 64
    store_device: str = "cpu"
    # opponent pool
    opp_ema_alpha: float = 0.02      # EMA rate for per-opponent win rate
    pool_add_threshold: float = 0.6  # add agent to pool when min win rate exceeds this
    pool_max_size: int = 7           # INITIAL total pool cap (incl. fixed rusher+japper);
                                     # grows +1 each time a permanent snapshot is added
    pool_snapshot_every: int = 100   # every N iters, snapshot the current actor as a
                                     # PERMANENT (never-evicted) opponent and bump the cap
    opp_sample_floor: float = 0.05   # min sampling weight per opponent
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


def save_ckpt(path, obj):
    """Atomically write a checkpoint (temp file + replace)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_maps(B, seed):
    specs = []
    rng = torch.Generator().manual_seed(seed)
    for _ in range(B):
        NP = int(torch.randint(25, 55, (1,), generator=rng))
        # legal KP range for this N
        N = 2 * NP + 1
        K_lo = (3 * N + 19) // 20
        K_hi = N // 5
        klo = K_lo + 1 if K_lo % 2 == 0 else K_lo
        khi = K_hi - 1 if K_hi % 2 == 0 else K_hi
        kp_lo, kp_hi = (klo - 1) // 2, (khi - 1) // 2
        KP = int(torch.randint(kp_lo, kp_hi + 1, (1,), generator=rng))
        specs.append((NP, KP))
    maps = []
    for i, (NP, KP) in enumerate(specs):
        maps.append(fe.tt.read_map(fe.tt.generate_map(fe.tt.XoShiro256(seed * 99991 + i + 1), NP, KP)))
    return maps


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

# rush-bot strategy knobs (mirror rush_bot.py).
RUSH_SIZE = 6               # warriors massed at home before the wave launches
KEEP_HOME = 1               # workers kept home when the wave launches / while sieging
PASSIVE_UPGRADE_TURN = 100  # passive opponent by this turn -> economy/UPGRADE
SIEGE_MARGIN = 20           # turns past ETA before a launched wave is "spent"
_MODE_MASS, _MODE_RUSH_SENT, _MODE_UPGRADE = 0, 1, 2

# japper-bot strategy knobs (mirror japper_bot.py).
JAP_DETECT_TURN = 6         # turn we read the enemy posture on
JAP_GROUP_MIN = 5           # enemy warriors off their HQ at turn 6 => all-in
JAP_WAVE_TRIGGER = 6        # rally garrison that launches a wave
JAP_STALL = 3               # turns without the strike group closing in => it stopped
_JSETUP, _JDEFENSE, _JTRANS, _JMAIN = 0, 1, 2, 3
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
    """Per-game persistent state for the scripted japper opponent -- a batched port
    of japper_bot.py's Brain. All tensors are [B]; ``reset_rows`` clears a game's
    state at its episode boundary. Not checkpointed (episodes never resume across
    runs). Because fast_env's move channel is per-region (one target per source
    region, moving every stationary warrior beyond a region's work-cap), the batched
    port cannot split the two starting warriors to two strongholds in one turn like
    the per-warrior submission does. Instead it expands sequentially: send the HQ
    surplus to the nearest stronghold (sa), build there, then push its surplus on to
    the second stronghold (sb) -- reaching the same turn-6 posture (HQ + sa base +
    sb warrior) a couple turns later. Everything else mirrors japper_bot."""

    def __init__(self, B, device):
        f = lambda v: torch.full((B,), v, dtype=torch.long, device=device)
        self.mode = torch.zeros(B, dtype=torch.long, device=device)      # SETUP
        self.assigned = torch.zeros(B, dtype=torch.bool, device=device)
        self.sa = f(-1); self.sb = f(-1); self.rally = f(-1)
        self.n = torch.zeros(B, dtype=torch.long, device=device)
        self.gathered = torch.zeros(B, dtype=torch.bool, device=device)
        self.defend_hops = f(3)
        self.threat_min = f(_JBIG); self.threat_prev = f(_JBIG)
        self.threat_stall = torch.zeros(B, dtype=torch.long, device=device)
        self.HQ_TRAINCAP = torch.tensor(fe.HQ_TRAINCAP, device=device)

    def reset_rows(self, rows):
        self.mode[rows] = _JSETUP
        self.assigned[rows] = False
        self.sa[rows] = -1; self.sb[rows] = -1; self.rally[rows] = -1
        self.n[rows] = 0
        self.gathered[rows] = False
        self.defend_hops[rows] = 3
        self.threat_min[rows] = _JBIG; self.threat_prev[rows] = _JBIG
        self.threat_stall[rows] = 0


def japper_action(env, side, jstate):
    """Scripted fixed opponent: a batched port of japper_bot.py. State machine over
    SETUP (double expansion), DEFENSE (mass n-1 and hold while the enemy all-in
    approaches, leave once it stalls/diverts/is repelled), TRANSITION (push back
    out to a stronghold) and MAIN (rally-point war machine: build the rally base,
    funnel every surplus warrior to it keeping the HQ/base work slots filled, and
    send waves of ~5 at the nearest enemy building, chaining while >=5 survive).
    Persistent per-game state lives in ``jstate``. Returns a full-batch action dict
    (build / move / train / force_build); bases are emitted on the force_build
    channel (bypasses the free-worker gate; the env auto-staffs the new base)."""
    B, N, dev = env.B, env.N, env.device
    W = env.W
    opp = 1 - side
    me_own = OWN_LEFT if side == 0 else OWN_RIGHT
    op_own = OWN_RIGHT if side == 0 else OWN_LEFT
    turn = env.day                                       # [B] per-game turn
    my_hq = env.hq_region[:, side]
    opp_hq = env.hq_region[:, opp]
    tt = env.mb.travel_turns                             # [B,N,T]
    tok = env.mb.token_ids                               # [B,T] ascending
    T = env.mb.T

    sd = env.slot_side[None, :].expand(B, W)
    alive = env.w_hp > 0
    mine = alive & (sd == side)
    enemy = alive & (sd == opp)
    stat_mine = mine & (~env.w_move)
    my_reg = env._scatter_region(mine)                   # [B,N]
    stat_reg = env._scatter_region(stat_mine)            # [B,N]
    opp_reg = env._scatter_region(enemy)                 # [B,N]
    home_stat = stat_reg.gather(1, my_hq[:, None]).squeeze(1)
    opp_at_myhq = opp_reg.gather(1, my_hq[:, None]).squeeze(1)
    gold = env.gold[:, side]
    hq_level = env.b_level.gather(1, my_hq[:, None]).squeeze(1).clamp(max=HQ_MAXLEVEL)
    traincap = jstate.HQ_TRAINCAP[hq_level]

    def tok_idx(region):                                 # region [B] -> token index [B]
        q = region.clamp(min=0)[:, None].contiguous()
        return torch.searchsorted(tok, q).clamp(max=T - 1).squeeze(1)

    def dist_to(region):                                 # [B,N] travel turns region->`region`
        ti = tok_idx(region)
        return tt.gather(2, ti[:, None, None].expand(B, N, 1)).squeeze(2)

    tt_to_myhq = dist_to(my_hq)                          # [B,N]
    w_dist = tt_to_myhq.gather(1, env.w_region)          # [B,W] each warrior -> my_hq
    eta = tt_to_myhq.gather(1, opp_hq[:, None]).squeeze(1)
    defend = torch.clamp((eta + 1) // 2, min=3)

    is_strong = env.mb.is_stronghold
    empty = is_strong & (env.b_owner == 0)               # OWN_NONE == 0
    has_empty = empty.any(1)
    d_empty = torch.where(empty, tt_to_myhq, torch.full_like(tt_to_myhq, _JBIG))
    near1 = d_empty.argmin(1)                            # nearest empty stronghold
    d_empty2 = d_empty.scatter(1, near1[:, None], _JBIG)
    near2 = d_empty2.argmin(1)                           # 2nd nearest

    build = torch.zeros(B, N, dtype=torch.bool, device=dev)
    move = torch.full((B, N), -1, dtype=torch.long, device=dev)
    force = torch.zeros(B, N, dtype=torch.bool, device=dev)
    train = torch.zeros(B, dtype=torch.long, device=dev)

    mode0 = jstate.mode.clone()                          # snapshot: one block per row
    m_setup = mode0 == _JSETUP
    m_def = mode0 == _JDEFENSE
    m_trans = mode0 == _JTRANS
    m_main = mode0 == _JMAIN

    def emit_move(mask, src, tgt):                        # move[src]=tgt for masked rows
        rows = mask.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            move[rows, src.clamp(min=0)[rows]] = tgt[rows]

    def emit_force(mask, reg):
        rows = mask.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            force[rows, reg.clamp(min=0)[rows]] = True

    def owner_at(reg):
        return env.b_owner.gather(1, reg.clamp(min=0)[:, None]).squeeze(1)

    # ---- assignment (turn 1): pick sa/sb, dispatch HQ surplus toward sa --------
    need = ~jstate.assigned
    jstate.sa = torch.where(need, near1, jstate.sa)
    jstate.sb = torch.where(need, near2, jstate.sb)
    jstate.defend_hops = torch.where(need, defend, jstate.defend_hops)
    emit_move(need & has_empty, my_hq, jstate.sa)
    jstate.assigned = jstate.assigned | need

    # ---- SETUP -----------------------------------------------------------------
    sa = jstate.sa
    sa_stat = stat_reg.gather(1, sa.clamp(min=0)[:, None]).squeeze(1)
    sa_owner = owner_at(sa)
    sa_enemy = opp_reg.gather(1, sa.clamp(min=0)[:, None]).squeeze(1)
    do_sa_build = m_setup & (sa >= 0) & (sa_stat > 0) & (sa_owner == 0) \
        & (gold >= fe.BASE_COST[1]) & (sa_enemy == 0)
    emit_force(do_sa_build, sa)
    # once sa is ours, push its surplus on to sb
    emit_move(m_setup & (sa_owner == me_own) & (jstate.sb >= 0), sa, jstate.sb)
    # turn-6 posture read
    enemy_off = (enemy & (env.w_region != opp_hq[:, None])).sum(1)
    at6 = m_setup & (turn >= JAP_DETECT_TURN)
    to_def = at6 & (enemy_off >= JAP_GROUP_MIN)
    to_main = at6 & (~to_def)
    jstate.n = torch.where(to_def, enemy_off, jstate.n)
    jstate.gathered = torch.where(to_def, torch.zeros_like(jstate.gathered),
                                  jstate.gathered)
    jstate.threat_min = torch.where(to_def, torch.full_like(jstate.threat_min, _JBIG),
                                    jstate.threat_min)
    jstate.threat_prev = torch.where(to_def, torch.full_like(jstate.threat_prev, _JBIG),
                                     jstate.threat_prev)
    jstate.threat_stall = torch.where(to_def, torch.zeros_like(jstate.threat_stall),
                                      jstate.threat_stall)
    emit_move(to_def & (jstate.sb >= 0), jstate.sb, my_hq)   # recall the sb warrior
    jstate.rally = torch.where(to_main, jstate.sb, jstate.rally)
    jstate.mode = torch.where(to_def, torch.full_like(jstate.mode, _JDEFENSE),
                              jstate.mode)
    jstate.mode = torch.where(to_main, torch.full_like(jstate.mode, _JMAIN),
                              jstate.mode)

    # ---- DEFENSE ---------------------------------------------------------------
    emit_move(m_def & (jstate.sb >= 0), jstate.sb, my_hq)    # keep recalling
    tgt_home = (jstate.n - 1).clamp(min=0)
    need_tr = (tgt_home - home_stat).clamp(min=0)
    n_tr = torch.minimum(torch.minimum(traincap, need_tr), gold // TRAIN_COST)
    train = torch.where(m_def & (~jstate.gathered) & (home_stat < tgt_home), n_tr, train)
    jstate.gathered = jstate.gathered | (m_def & (home_stat >= tgt_home))
    # watch the strike group by its closest distance to our HQ
    off = enemy & (env.w_region != opp_hq[:, None])
    cur = torch.where(off, w_dist, torch.full_like(w_dist, _JBIG)).min(1).values  # [B]
    upd = m_def & jstate.gathered
    jstate.threat_min = torch.where(upd, torch.minimum(jstate.threat_min, cur),
                                    jstate.threat_min)
    approach = (cur < _JBIG) & (jstate.threat_prev < _JBIG) & (cur >= jstate.threat_prev)
    jstate.threat_stall = torch.where(upd & approach, jstate.threat_stall + 1,
                                      jstate.threat_stall)
    jstate.threat_stall = torch.where(upd & (cur < _JBIG) & (~approach),
                                      torch.zeros_like(jstate.threat_stall),
                                      jstate.threat_stall)
    jstate.threat_prev = torch.where(upd & (cur < _JBIG), cur, jstate.threat_prev)
    enemy_at_hq = opp_at_myhq > 0
    engaged = jstate.threat_min <= jstate.defend_hops
    leave = upd & (~enemy_at_hq) & (
        (cur >= _JBIG)
        | (engaged & (cur > jstate.defend_hops))
        | ((~engaged) & (jstate.threat_stall >= JAP_STALL)))
    jstate.mode = torch.where(leave, torch.full_like(jstate.mode, _JTRANS), jstate.mode)
    jstate.rally = torch.where(leave, torch.full_like(jstate.rally, -1), jstate.rally)

    # ---- TRANSITION ------------------------------------------------------------
    set_rally = m_trans & has_empty & (jstate.rally < 0)
    jstate.rally = torch.where(set_rally, near1, jstate.rally)
    rally_t = jstate.rally.clamp(min=0)
    emit_move(m_trans & (jstate.rally >= 0) & (home_stat >= 2), my_hq, jstate.rally)
    need_t2 = (2 - home_stat).clamp(min=0)
    n_tr2 = torch.minimum(torch.minimum(traincap, need_t2), gold // TRAIN_COST)
    train = torch.where(m_trans & (home_stat < 2), n_tr2, train)
    at_rally = stat_reg.gather(1, rally_t[:, None]).squeeze(1)
    arrived = m_trans & (jstate.rally >= 0) & (at_rally > 0)
    jstate.mode = torch.where(arrived, torch.full_like(jstate.mode, _JMAIN), jstate.mode)

    # ---- MAIN ------------------------------------------------------------------
    rally_t = jstate.rally.clamp(min=0)
    rb_mine = owner_at(jstate.rally) == me_own
    rally_stat = stat_reg.gather(1, rally_t[:, None]).squeeze(1)
    rally_enemy = opp_reg.gather(1, rally_t[:, None]).squeeze(1)
    have_rally = m_main & (jstate.rally >= 0)
    # priority: build the rally base (economic engine) before spending on training
    do_rb = have_rally & (~rb_mine) & (rally_stat > 0) & (gold >= fe.BASE_COST[1]) \
        & (rally_enemy == 0)
    emit_force(do_rb, jstate.rally)
    # funnel HQ surplus: prefer restaffing an owned BASE (not the rally) that has no
    # friendly warrior stationed nor inbound; else feed the rally. (Per-region moves
    # can pick only one target per source, so an empty base takes the whole HQ
    # surplus that turn -- a couple over-sends vs the submission's exact 1-per-base,
    # self-correcting once the base is no longer empty.)
    owned_base = (env.b_owner == me_own) & (env.b_kind == KIND_BASE)
    inc = torch.zeros(B, N, dtype=torch.long, device=dev)
    inc.scatter_add_(1, env.w_tgt.clamp(min=0), (mine & env.w_move).long())
    rally_oh = torch.zeros(B, N, dtype=torch.bool, device=dev)
    rally_oh.scatter_(1, rally_t[:, None], True)
    empty_base = owned_base & (~rally_oh) & ((stat_reg + inc) == 0)
    d_eb0 = torch.where(empty_base, tt_to_myhq, torch.full_like(tt_to_myhq, _JBIG))
    eob = d_eb0.argmin(1)
    funnel_tgt = torch.where(d_eb0.min(1).values < _JBIG, eob, rally_t)
    emit_move(have_rally, my_hq, funnel_tgt)
    base_up = have_rally & rb_mine
    train = torch.where(base_up, torch.minimum(traincap, gold // TRAIN_COST), train)
    # launch: EVERY turn the rally reaches the trigger, send its surplus (per-region
    # move keeps the base work_cap home) at the nearest enemy building. No single-
    # wave gate -> concurrent waves, matching japper_bot.
    enemy_bldg = env.b_owner == op_own
    d_reb = torch.where(enemy_bldg, dist_to(rally_t), torch.full_like(tt_to_myhq, _JBIG))
    launch_tgt = d_reb.argmin(1)
    has_eb = d_reb.min(1).values < _JBIG
    do_launch = base_up & (rally_stat >= JAP_WAVE_TRIGGER) & has_eb
    emit_move(do_launch, rally_t, launch_tgt)
    # react to warriors left standing on razed targets (regions that are neither my
    # building nor a live enemy building). Treat them as ONE body PER GAME: if >=5
    # remain in total they ALL chain to the nearest enemy building (from where most
    # of them stand), else they ALL retreat to the rally -- never split forward/back.
    my_bldg = env.b_owner == me_own
    field = (stat_reg > 0) & (~my_bldg) & (~enemy_bldg)             # [B,N] razed spots
    field_pop = stat_reg * field.long()
    field_cnt = field_pop.sum(1)                                    # [B] total survivors
    main_reg = field_pop.argmax(1)                                  # [B] main-body region
    d_main = torch.where(enemy_bldg, dist_to(main_reg),
                         torch.full_like(tt_to_myhq, _JBIG))        # [B,N]
    chain_tgt = d_main.argmin(1)
    has_eb2 = d_main.min(1).values < _JBIG
    chain_game = base_up & (field_cnt >= 5) & has_eb2              # whole-game decision
    dest = torch.where(chain_game, chain_tgt, rally_t)            # [B] one dest per game
    apply_field = base_up[:, None] & field
    move = torch.where(apply_field, dest[:, None].expand(B, N), move)

    return {'build': build, 'move': move, 'train': train, 'force_build': force}


def opponent_actions(pool_t1, pool_t2, o_op, opp_assign, env, side, B, N, dev,
                     rstate, jstate):
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
            act, _, _, ncommit = sample_policy(pool_t1[p - N_SCRIPTED],
                                               pool_t2[p - N_SCRIPTED], sub, N)
            build_full[rows] = act['build']
            move_full[rows] = act['move']
            train_full[rows] = act['train']
            force_full[rows] = act['force_build']
            commit_full[rows] = ncommit
    return ({'build': build_full, 'move': move_full, 'train': train_full,
             'force_build': force_full}, commit_full)


def train(cfg: Config, device=None, seed=0, log_every=1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    maps = make_maps(cfg.B, seed)
    # reserve capacity for the largest possible map so per-episode regen fits any size
    env = fe.FastEnv(maps, device=device, n_cap=109, t_cap=23)
    env._map_rng = __import__("random").Random(seed + 12345)
    N = env.N

    actor_t1 = ActorT1(d=cfg.d_model).to(device)
    actor_t2 = ActorT2(cfg.d_model + T2_EXTRA, d=cfg.d_model).to(device)
    critic = Critic(d=cfg.d_model).to(device)

    mk_t1 = lambda: ActorT1(d=cfg.d_model)
    mk_t2 = lambda: ActorT2(cfg.d_model + T2_EXTRA, d=cfg.d_model)

    # opponent pool. Unified index 0 is a FIXED scripted rusher (never evicted,
    # tracked with its own EMA win rate); index i>=1 is net snapshot pool_t1[i-1].
    # It starts with the rusher + a frozen copy of the initial policy. The total pool
    # cap starts at cfg.pool_max_size (=6) and grows by 1 whenever a PERMANENT snapshot
    # is added (every cfg.pool_snapshot_every iters), so the evictable capacity stays
    # constant. pool_perm[j] marks net snapshot j as permanent (never evicted, like the
    # rusher). pool_ids gives each opponent a stable id (survives eviction index shifts)
    # so wandb curves track the same opponent over time ('rusher' for the fixed one).
    pool_t1 = [frozen_copy(actor_t1)]
    pool_t2 = [frozen_copy(actor_t2)]
    # EMA win rate, indexed by unified opponent index: [rusher, japper, net0]
    pool_wr = torch.full((N_SCRIPTED + 1,), 0.5)
    pool_ids = ['rusher', 'japper', 0]
    pool_perm = [False]                                 # per net snapshot: never-evict?
    next_opp_id = 1
    opp_gen = torch.Generator().manual_seed(seed + 777)
    opp_assign = sample_opponents(cfg.B, pool_wr, opp_gen,
                                  cfg.opp_sample_floor).to(device)

    actor_params = list(actor_t1.parameters()) + list(actor_t2.parameters())
    critic_params = list(critic.parameters())
    opt_actor = torch.optim.Adam(actor_params, lr=cfg.lr)
    opt_critic = torch.optim.Adam(critic_params, lr=cfg.lr)

    steps = max(1, cfg.steps_per_iter // cfg.B)
    sdev = cfg.store_device
    alpha = cfg.opp_ema_alpha

    # ---- resume from checkpoint if present ----
    start_iter = 0
    if cfg.resume and os.path.exists(cfg.ckpt_path):
        ck = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        # strict=False so checkpoints predating the gold-production aux heads still
        # load (the backbone/policy/critic transfer; the new aux heads start fresh).
        a1_miss = actor_t1.load_state_dict(ck['actor_t1'], strict=False)
        actor_t2.load_state_dict(ck['actor_t2'])
        c_miss = critic.load_state_dict(ck['critic'], strict=False)
        try:
            opt_actor.load_state_dict(ck['opt_actor'])
            opt_critic.load_state_dict(ck['opt_critic'])
        except (ValueError, KeyError) as e:
            # param set changed (aux heads added) -> Adam moments can't map; reset.
            print(f"  optimizer state incompatible ({e}); reinitializing optimizers")
        if a1_miss.missing_keys or c_miss.missing_keys:
            print(f"  loaded pre-aux checkpoint; aux heads initialized fresh "
                  f"(actor missing {len(a1_miss.missing_keys)}, "
                  f"critic missing {len(c_miss.missing_keys)})")
        pool_t1 = [frozen_from_state(mk_t1, sd, device) for sd in ck['pool_t1']]
        pool_t2 = [frozen_from_state(mk_t2, sd, device) for sd in ck['pool_t2']]
        pool_wr = ck['pool_wr'].cpu()   # kept on CPU (used with the CPU opp_gen)
        pool_ids = ck['pool_ids']
        # pre-permanent-snapshot checkpoints have no pool_perm -> all nets evictable
        pool_perm = list(ck.get('pool_perm', [False] * len(pool_t1)))
        next_opp_id = ck['next_opp_id']
        opp_assign = ck['opp_assign'].to(device)
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
            print(f"  migrated checkpoint: inserted scripted opponents {missing}")
        # RNG states must be CPU ByteTensors (map_location may have moved them)
        opp_gen.set_state(ck['opp_gen'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        start_iter = ck['iter']
        print(f"resumed from {cfg.ckpt_path} at iter {start_iter} "
              f"(pool size {len(pool_t1) + N_SCRIPTED})")

    run = None
    if cfg.use_wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = (
            "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")
        run = wandb.init(project="nypc2026-selfplay", config=vars(cfg))

    # per-token realized work 'take' from the previous step, for the gold-production
    # aux label (Δtake = this turn's production change). Episodes span iters, so this
    # persists across iterations; reset to 0 per game at episode boundaries.
    prev_take = torch.zeros(cfg.B, env.mb.T, 2, device=device)

    # previous turn's raw enemy-reachability (per side) for the reach-delta token
    # feature; reset to 0 per game at episode boundaries (fresh map = no prior turn).
    prev_reach_ag = torch.zeros(cfg.B, env.mb.T, 5, device=device)
    prev_reach_op = torch.zeros(cfg.B, env.mb.T, 5, device=device)

    # per-game state for the scripted opponents (rusher=index 0, japper=index 1).
    # Fresh, not checkpointed: episodes never resume across runs (env is reset on
    # resume), so a fresh state always aligns with a fresh env.
    rstate = RusherState(cfg.B, device)
    jstate = JapperState(cfg.B, device)

    for it in range(start_iter, cfg.iters):
        # ---- checkpoint before starting this iter (for crash-safe resume) ----
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
            'next_opp_id': next_opp_id,
            'opp_assign': opp_assign.cpu(),
            'opp_gen': opp_gen.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            'cfg': vars(cfg),
        })

        t0 = time.time()
        buf = []
        ep_rewards, ep_count = 0.0, 0
        for s in range(steps):
            o_ag = extract(env, 0, prev_reach_ag)
            with torch.no_grad():
                act_ag, store, _, ncommit_ag = sample_policy(actor_t1, actor_t2, o_ag, N)
                val = critic.value(o_ag['t1'], o_ag['glob'], o_ag['tmask'])
                o_op = extract(env, 1, prev_reach_op)
                act_op, ncommit_op = opponent_actions(pool_t1, pool_t2, o_op, opp_assign,
                                                      env, 1, cfg.B, N, device,
                                                      rstate, jstate)
            # this turn's reach becomes next turn's baseline for the delta feature
            prev_reach_ag = o_ag['reach_raw'].clone()
            prev_reach_op = o_op['reach_raw'].clone()
            env.step({'left': act_ag, 'right': act_op})
            # carry the HQ-upgrade commitment to next turn (regen resets it for
            # finished games below).
            env.hq_commit[:, 0] = ncommit_ag
            env.hq_commit[:, 1] = ncommit_op
            r, done = reward_done(env)
            # gold-production change this turn (per 거점, [me, opp]); label for o_ag
            cur_take = env.token_take(0)
            gold_aux = cur_take - prev_take
            prev_take = cur_take
            if done.any():
                ep_rewards += float(r[done].sum()); ep_count += int(done.sum())
            rec = {k: v.to(sdev) for k, v in store.items()}
            rec['value'] = val.to(sdev)
            rec['reward'] = r.to(sdev)
            rec['done'] = done.float().to(sdev)
            rec['gold_aux'] = gold_aux.to(sdev)
            buf.append(rec)
            if done.any():
                drows = done.nonzero(as_tuple=True)[0]
                # EMA win rate per opponent (agent's win rate; draw = 0.5)
                rr = r[drows]
                res = torch.where(rr > 0, torch.ones_like(rr),
                                  torch.where(rr < 0, torch.zeros_like(rr),
                                              torch.full_like(rr, 0.5)))
                assigned = opp_assign[drows].cpu()
                res_c = res.cpu()
                for j in range(drows.numel()):
                    p = int(assigned[j])
                    pool_wr[p] = (1 - alpha) * pool_wr[p] + alpha * float(res_c[j])
                # fresh opponent for each finished game, then fresh map + reset
                opp_assign[drows] = sample_opponents(
                    drows.numel(), pool_wr, opp_gen, cfg.opp_sample_floor).to(device)
                env.regen(done)
                # new map -> no prior turn; baseline the aux + reach deltas at 0
                prev_take[drows] = 0
                prev_reach_ag[drows] = 0
                prev_reach_op[drows] = 0
                # restart the scripted opponents' state machines for the new games
                rstate.reset_rows(drows)
                jstate.reset_rows(drows)

        with torch.no_grad():
            # bootstrap value on the post-rollout state; same prev_reach the next
            # iter's first step will use (no env step happened in between).
            o_ag = extract(env, 0, prev_reach_ag)
            last_val = critic.value(o_ag['t1'], o_ag['glob'], o_ag['tmask']).to(sdev)

        # ---- GAE ----
        adv = [None] * steps
        gae = torch.zeros(cfg.B, device=sdev)
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
        keys = ['t1', 'glob', 'tmask', 'extra4', 'tok_dist', 'normx', 'normy',
                'build_mask', 'build_outcome', 'train_mask', 'train_cat',
                'valid_src', 'tgt', 'surplus_pb', 'tgt_allowed', 'old_logp',
                'adv', 'ret', 'value', 'gold_aux']
        flat = {k: torch.cat([buf[t][k] for t in range(steps)], dim=0) for k in keys}
        Ntot = flat['t1'].shape[0]
        a = flat['adv']
        flat['adv'] = (a - a.mean()) / (a.std() + 1e-8)

        # ---- PPO epochs ----
        pl = vl = el = kl = 0.0
        ax_a = ax_c = 0.0
        nb = 0
        epochs_run = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(Ntot)
            ep_kl = 0.0
            ep_nb = 0
            for i in range(0, Ntot, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                mb = {k: flat[k][idx].to(device) for k in keys}
                # ---- actor update (policy + entropy + gold-production aux) ----
                logp, ent, aux_a = evaluate_policy(actor_t1, actor_t2, mb)
                logratio = logp - mb['old_logp']
                ratio = torch.exp(logratio)
                with torch.no_grad():
                    # Schulman's positive KL estimator E[(r-1) - log r] ~ KL(old||new)
                    approx_kl = ((ratio - 1) - logratio).mean()
                adv_b = mb['adv']
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_b
                ploss = -torch.min(s1, s2).mean()
                eloss = -ent.mean()
                aux_loss_a = masked_token_mse(aux_a, mb['gold_aux'], mb['tmask'])
                actor_loss = ploss + cfg.ent_coef * eloss + cfg.aux_coef * aux_loss_a
                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                opt_actor.step()

                # ---- critic update (value regression + gold-production aux) ----
                value, aux_c = critic.value_aux(mb['t1'], mb['glob'], mb['tmask'])
                vloss = F.mse_loss(value, mb['ret'])
                aux_loss_c = masked_token_mse(aux_c, mb['gold_aux'], mb['tmask'])
                critic_loss = vloss + cfg.aux_coef * aux_loss_c
                opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                opt_critic.step()

                pl += ploss.item(); vl += vloss.item(); el += ent.mean().item()
                kl += approx_kl.item(); nb += 1
                ax_a += aux_loss_a.item(); ax_c += aux_loss_c.item()
                ep_kl += approx_kl.item(); ep_nb += 1

            epochs_run += 1
            # KL early stopping: if this epoch's mean drift exceeds the target,
            # stop refreshing on this (now-stale) batch before overshooting.
            if cfg.target_kl is not None and ep_kl / max(ep_nb, 1) > cfg.target_kl:
                break

        # ---- value-net explained variance ----
        with torch.no_grad():
            ret_f, val_f = flat['ret'], flat['value']
            ev = float(1.0 - (ret_f - val_f).var() / (ret_f.var() + 1e-8))

        added = False

        # ---- every N iters: snapshot the current actor as a PERMANENT opponent ----
        # Like the rusher it is never evicted; the total cap grows by 1 (implicitly,
        # via sum(pool_perm) below) so the evictable capacity is unchanged.
        if cfg.pool_snapshot_every and it > 0 and it % cfg.pool_snapshot_every == 0:
            pool_t1.append(frozen_copy(actor_t1))
            pool_t2.append(frozen_copy(actor_t2))
            pool_wr = torch.cat([pool_wr, torch.full((1,), 0.5)])
            pool_ids.append(next_opp_id)
            pool_perm.append(True)
            next_opp_id += 1
            added = True

        # ---- grow the pool when the agent beats even the hardest opponent ----
        # Total cap = base + one slot per permanent snapshot (so evictable room is
        # constant at pool_max_size - N_SCRIPTED). When full, evict the oldest
        # EVICTABLE net; the scripted bots (indices < N_SCRIPTED) and permanent
        # snapshots are never touched.
        pool_cap = cfg.pool_max_size + sum(pool_perm)
        if float(pool_wr.min()) > cfg.pool_add_threshold:
            if len(pool_t1) + N_SCRIPTED >= pool_cap:
                # oldest non-permanent net (list pos -> unified index pos+N_SCRIPTED)
                evict_pos = next(i for i, perm in enumerate(pool_perm) if not perm)
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
        if it % log_every == 0:
            print(f"iter {it:3d} | eps {ep_count:5d} | avg_ep_R {wr:+6.2f} | "
                  f"ploss {pl/nb:+.4f} vloss {vl/nb:.3f} ent {el/nb:.3f} kl {kl/nb:.4f} "
                  f"aux {ax_a/nb:.3f}/{ax_c/nb:.3f} ep {epochs_run}/{cfg.epochs} | "
                  f"ev {ev:+.3f} pool {len(pool_t1)+N_SCRIPTED}/{pool_cap} "
                  f"wr_min {float(pool_wr.min()):.2f} "
                  f"rush_wr {float(pool_wr[RUSHER_IDX]):.2f} "
                  f"jap_wr {float(pool_wr[JAPPER_IDX]):.2f} | "
                  f"{steps*cfg.B/dt:,.0f} steps/s ({dt:.1f}s)")

        if run is not None:
            import wandb
            log = {
                'iter': it,
                'avg_ep_R': wr,
                'episodes': ep_count,
                'ploss': pl / nb,
                'vloss': vl / nb,
                'entropy': el / nb,
                'approx_kl': kl / nb,
                'aux_loss_actor': ax_a / nb,
                'aux_loss_critic': ax_c / nb,
                'epochs_run': epochs_run,
                'value_ev': ev,
                'pool_size': len(pool_t1) + N_SCRIPTED,
                'pool_cap': pool_cap,
                'pool_perm_count': N_SCRIPTED + sum(pool_perm),
                'opp_winrate_min': float(pool_wr.min()),
                'opp_winrate_mean': float(pool_wr.mean()),
                'opp_winrate_rusher': float(pool_wr[RUSHER_IDX]),
                'opp_winrate_japper': float(pool_wr[JAPPER_IDX]),
                'pool_added': int(added),
                'steps_per_s': steps * cfg.B / dt,
            }
            # per-opponent EMA win rate, keyed by stable id (survives eviction)
            for k, oid in enumerate(pool_ids):
                log[f'opp_winrate/{oid}'] = float(pool_wr[k])
            wandb.log(log, step=it)

    if run is not None:
        run.finish()
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
    args = ap.parse_args()

    if args.smoke:
        cfg = Config(B=8, steps_per_iter=2000, iters=2, minibatch=512, d_model=32,
                     use_wandb=False, resume=False, ckpt_path="checkpoint_smoke.pt")
    else:
        cfg = load_config(args.config) if os.path.exists(args.config) else Config()
        if args.B is not None: cfg.B = args.B
        if args.steps is not None: cfg.steps_per_iter = args.steps
        if args.iters is not None: cfg.iters = args.iters
        if args.no_wandb: cfg.use_wandb = False
        if args.no_resume: cfg.resume = False
    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
