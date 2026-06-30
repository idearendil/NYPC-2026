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
from fast_env import (OWN_LEFT, OWN_RIGHT, KIND_HQ, MOVE_COST, TRAIN_COST,
                      HQ_MAXLEVEL, BASE_MAXLEVEL, HQ_HEAL, BASE_HEAL, MAX_DAYS)

TOK_FEAT = 26          # 14 scalars + 5 arrive + 5 reach (all log1p) + 2 norm coords
GLOB_FEAT = 11
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
def extract(env, side):
    B, N, T = env.B, env.N, env.mb.T
    me = OWN_LEFT if side == 0 else OWN_RIGHT          # owner code (1/2)
    opp_idx = 1 - side                                  # side index (0/1)
    tokens, glob, info = env.observe(side)
    raw24 = slog1p(tokens[:, :, :24].float())          # the 24 log1p features

    g = glob.float()
    glob_t = torch.stack([
        g[:, 0] / 10 - 10,
        plog1p(g[:, 1] / 10), plog1p(g[:, 2] / 10),
        plog1p(g[:, 3]),      plog1p(g[:, 4]),
        plog1p(g[:, 5] / 100), plog1p(g[:, 6] / 100),
        plog1p(g[:, 7] / 10), plog1p(g[:, 8] / 10),
        plog1p(g[:, 9] / 5),  plog1p(g[:, 10] / 5),
    ], dim=1)

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
    t1 = torch.cat([raw24, normx[:, :, None], normy[:, :, None]], dim=2)  # [B,T,26]

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
    can_up = me_b & (lvl_t < maxlev)
    can_heal = me_b & (lvl_t >= maxlev)
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
        owner_me=me_b, build_new=build_new,
        extra4=extra4, tok_dist=tok_dist, normx=normx, normy=normy,
        # free workers = non-moving friendly not currently labouring (surplus
        # beyond each region's work_cap); used to gate builds.
        free_total=(env._scatter_region(my_stat)
                    - torch.where(env.b_owner == me, workcap,
                                  torch.zeros_like(workcap))).clamp(min=0).sum(1),
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

    # ---------------- BUILD ----------------
    # Gating: building/upgrading requires >=1 free worker -- a non-moving friendly
    # warrior not currently labouring -- who can be sent to staff the base. The
    # per-build cap and the force-staffing move are applied inside the env
    # (fast_env._phase_build); dropped builds still count as taken for PPO since
    # build_logp uses the Bernoulli outcome.
    build_mask = (o['build_cand'] & (o['build_cost'] <= o['gold'][:, None]) & o['tmask']
                  & (o['free_total'][:, None] >= 1))
    p_build = torch.sigmoid(head5[:, :, 0]) * build_mask.float()
    outcome = torch.bernoulli(p_build)
    build_logp = (build_mask.float() * bern_logp(torch.sigmoid(head5[:, :, 0]), outcome)).sum(1)

    perm = torch.argsort(torch.rand(B, T, device=dev), dim=1)
    exec_build, gold1 = _greedy(outcome.bool(), o['build_cost'], o['gold'], perm, T)

    # post-build quantities
    wc_pb = torch.where(exec_build, o['wc_after'], o['wc_cur'])
    surplus_pb = (o['stat_cnt'] - wc_pb).clamp(min=0)
    owner_me_pb = o['owner_me'] | (exec_build & o['build_new'])
    hq_up = (exec_build & o['can_up_hq']).any(1)
    hq_after = (o['hq_level'] + hq_up.long()).clamp(max=HQ_MAXLEVEL)

    # ---------------- MOVE (T2) ----------------
    valid_src = o['tmask'] & (surplus_pb > 0) & (MOVE_COST * surplus_pb <= gold1[:, None])
    logits = t2_logits_sources(t2net, h1, o['extra4'], surplus_pb, o['tok_dist'],
                               o['normx'], o['normy'], o['tmask'], valid_src)
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
    tl_m = tl.masked_fill(~tmask_train, -1e9)
    logp_tr = F.log_softmax(tl_m, dim=1)
    train_cat = torch.multinomial(logp_tr.exp(), 1).squeeze(1)
    train_logp = logp_tr.gather(1, train_cat[:, None]).squeeze(1)

    old_logp = build_logp + move_logp + train_logp

    # ---------------- build env action tensors ----------------
    tok_ids = o['tok_ids']
    build_env = torch.zeros(B, N, dtype=torch.bool, device=dev)
    rb, tb = exec_build.nonzero(as_tuple=True)
    build_env[rb, tok_ids[rb, tb]] = True
    move_env = torch.full((B, N), -1, dtype=torch.long, device=dev)
    rm, sm = exec_move.nonzero(as_tuple=True)
    src_reg = tok_ids[rm, sm]
    tgt_reg = tok_ids[rm, tgt[rm, sm]]
    move_env[rm, src_reg] = tgt_reg
    action = {'build': build_env, 'move': move_env, 'train': train_cat}

    store = dict(
        t1=o['t1'], glob=o['glob'], tmask=o['tmask'],
        extra4=o['extra4'], tok_dist=o['tok_dist'],
        normx=o['normx'], normy=o['normy'],
        build_mask=build_mask, build_outcome=outcome,
        train_mask=tmask_train, train_cat=train_cat,
        valid_src=valid_src, tgt=tgt, surplus_pb=surplus_pb,
        old_logp=old_logp,
    )
    return action, store, old_logp


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
    pool_max_size: int = 10          # cap pool; evict oldest (FIFO) when full
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


def rusher_action(env, side):
    """Scripted fixed opponent embodying a pure 'rush' strategy (a batched port of
    the supplied rush bot): never builds or upgrades; trains 1 warrior/turn; once
    >=6 non-moving warriors have gathered at its HQ, hurls them at the enemy HQ
    (the per-region move keeps the HQ's work_cap=1 worker and sends the rest).
    Returns a full-batch action dict; the caller copies only its assigned rows."""
    B, N, dev = env.B, env.N, env.device
    my_hq = env.hq_region[:, side]            # [B]
    opp_hq = env.hq_region[:, 1 - side]       # [B]
    sd = env.slot_side[None, :].expand(B, env.W)
    stat = (env.w_hp > 0) & (sd == side) & (~env.w_move)
    hq_cnt = env._scatter_region(stat).gather(1, my_hq[:, None]).squeeze(1)   # [B]

    move = torch.full((B, N), -1, dtype=torch.long, device=dev)
    rows = (hq_cnt >= 6).nonzero(as_tuple=True)[0]
    if rows.numel() > 0:
        move[rows, my_hq[rows]] = opp_hq[rows]
    return {'build': torch.zeros(B, N, dtype=torch.bool, device=dev),
            'move': move,
            'train': torch.ones(B, dtype=torch.long, device=dev)}   # env caps by cap/gold


# Unified opponent index 0 is the fixed rusher; net snapshot i is unified index i+1.
RUSHER_IDX = 0


def opponent_actions(pool_t1, pool_t2, o_op, opp_assign, env, side, B, N, dev):
    """Sample opponent actions, grouping batch slots by their assigned opponent so
    each opponent runs once over its subset of games. Unified index 0 is the fixed
    scripted rusher; index p>=1 maps to net snapshot pool_t1[p-1]/pool_t2[p-1]."""
    build_full = torch.zeros(B, N, dtype=torch.bool, device=dev)
    move_full = torch.full((B, N), -1, dtype=torch.long, device=dev)
    train_full = torch.zeros(B, dtype=torch.long, device=dev)
    for p in torch.unique(opp_assign).tolist():
        rows = (opp_assign == p).nonzero(as_tuple=True)[0]
        if p == RUSHER_IDX:
            act = rusher_action(env, side)
            build_full[rows] = act['build'][rows]
            move_full[rows] = act['move'][rows]
            train_full[rows] = act['train'][rows]
        else:
            sub = slice_obs(o_op, rows)
            act, _, _ = sample_policy(pool_t1[p - 1], pool_t2[p - 1], sub, N)
            build_full[rows] = act['build']
            move_full[rows] = act['move']
            train_full[rows] = act['train']
    return {'build': build_full, 'move': move_full, 'train': train_full}


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
    # It starts with the rusher + a frozen copy of the initial policy. Net snapshots
    # are capped at cfg.pool_max_size (=10) so the total pool is at most 11.
    # pool_ids gives each opponent a stable id (survives eviction index shifts) so
    # wandb curves track the same opponent over time ('rusher' for the fixed one).
    pool_t1 = [frozen_copy(actor_t1)]
    pool_t2 = [frozen_copy(actor_t2)]
    pool_wr = torch.full((2,), 0.5)                     # EMA win rate: [rusher, net0]
    pool_ids = ['rusher', 0]
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
        next_opp_id = ck['next_opp_id']
        opp_assign = ck['opp_assign'].to(device)
        # backward-compat: checkpoints predating the fixed rusher have no rusher
        # slot (pool_wr/pool_ids align 1:1 with the net snapshots, opp_assign is
        # net-0-based). Prepend the rusher at unified index 0 and shift indices.
        if pool_wr.numel() == len(pool_t1):
            pool_wr = torch.cat([torch.full((1,), 0.5), pool_wr])
            pool_ids = ['rusher'] + list(pool_ids)
            opp_assign = opp_assign + 1
            print("  migrated pre-rusher checkpoint: prepended fixed rusher opponent")
        # RNG states must be CPU ByteTensors (map_location may have moved them)
        opp_gen.set_state(ck['opp_gen'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        start_iter = ck['iter']
        print(f"resumed from {cfg.ckpt_path} at iter {start_iter} "
              f"(pool size {len(pool_t1) + 1})")

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
            o_ag = extract(env, 0)
            with torch.no_grad():
                act_ag, store, _ = sample_policy(actor_t1, actor_t2, o_ag, N)
                val = critic.value(o_ag['t1'], o_ag['glob'], o_ag['tmask'])
                o_op = extract(env, 1)
                act_op = opponent_actions(pool_t1, pool_t2, o_op, opp_assign,
                                          env, 1, cfg.B, N, device)
            env.step({'left': act_ag, 'right': act_op})
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
                # new map -> no prior production; baseline the aux delta at 0
                prev_take[drows] = 0

        with torch.no_grad():
            o_ag = extract(env, 0)
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
                'valid_src', 'tgt', 'surplus_pb', 'old_logp', 'adv', 'ret', 'value',
                'gold_aux']
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
                critic_loss = cfg.vf_coef * vloss + cfg.aux_coef * aux_loss_c
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

        # ---- grow the pool when the agent beats even the hardest opponent ----
        added = False
        if float(pool_wr.min()) > cfg.pool_add_threshold:
            if len(pool_t1) >= cfg.pool_max_size:
                # evict the oldest NET snapshot (unified index 1); the fixed rusher
                # at index 0 is never evicted. Remaining net indices shift down one.
                pool_t1.pop(0); pool_t2.pop(0)
                pool_wr = torch.cat([pool_wr[:1], pool_wr[2:]])
                pool_ids.pop(1)
                opp_assign = torch.where(opp_assign >= 1,
                                         (opp_assign - 1).clamp(min=0), opp_assign)
            pool_t1.append(frozen_copy(actor_t1))
            pool_t2.append(frozen_copy(actor_t2))
            pool_wr = torch.cat([pool_wr, torch.full((1,), 0.5)])
            pool_ids.append(next_opp_id)
            next_opp_id += 1
            added = True

        wr = (ep_rewards / max(ep_count, 1))
        dt = time.time() - t0
        if it % log_every == 0:
            print(f"iter {it:3d} | eps {ep_count:5d} | avg_ep_R {wr:+6.2f} | "
                  f"ploss {pl/nb:+.4f} vloss {vl/nb:.3f} ent {el/nb:.3f} kl {kl/nb:.4f} "
                  f"aux {ax_a/nb:.3f}/{ax_c/nb:.3f} ep {epochs_run}/{cfg.epochs} | "
                  f"ev {ev:+.3f} pool {len(pool_t1)+1} wr_min {float(pool_wr.min()):.2f} "
                  f"rush_wr {float(pool_wr[RUSHER_IDX]):.2f} | "
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
                'pool_size': len(pool_t1) + 1,
                'opp_winrate_min': float(pool_wr.min()),
                'opp_winrate_mean': float(pool_wr.mean()),
                'opp_winrate_rusher': float(pool_wr[RUSHER_IDX]),
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
