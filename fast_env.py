#!/usr/bin/env python3
"""GPU-accelerated, batched re-implementation of the board-game simulator.

The goal is raw throughput for RL data collection: `B` independent games are
stepped in parallel on the GPU. The dynamics are a faithful, bit-exact copy of
`testing-tool.py` (verified by `test_fast_env.py`).

Design notes
------------
* Map generation is *not* re-derived; we call the original ``generate_map`` so
  boards come from the exact same distribution. One batch shares a single
  ``(N, K)`` size (each game still gets an independent random layout). Mixing
  sizes in one batch = run several ``FastEnv`` instances, or pad to ``Nmax``.
* Warriors are stored in per-side pools ``[B, Wside]``. The *slot index is the
  suffix* (creation order), which is exactly what the rules need for the two
  order-dependent mechanics: combat focus-fires the lowest-HP / lowest-suffix
  warrior, and hunger is charged in suffix order.
* Every order-dependent rule is reduced to one primitive: ``_seg_stats`` returns,
  for each warrior, its rank and the exclusive HP-prefix within its group,
  ordered by (hp asc, suffix asc). Built from a single global sort + cummax.

Action interface (per side), all batched tensors on ``self.device``:
    build : bool [B, N]   -- regions to build/upgrade/heal (build candidates)
    move  : long [B, N]   -- move[b, src] = target region, or -1 for "no move"
                             (encodes the "one source -> one target" rule)
    train : long [B]      -- 0..3 warriors to train at the HQ
"""
from __future__ import annotations

import importlib.util
import math
import os
from typing import Optional

import torch

# --------------------------------------------------------------------------- #
# Load the original simulator as a module (single source of truth for map gen
# and for the reference dynamics used by the parity test).
# --------------------------------------------------------------------------- #
_TT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing-tool.py")
if os.path.exists(_TT_PATH):
    _spec = importlib.util.spec_from_file_location("tt_reference", _TT_PATH)
    tt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tt)
else:
    # Only map generation / the parity tests need the reference simulator. The
    # submission bot reuses the encoder but never generates maps, so allow import
    # to succeed without testing-tool.py present.
    tt = None

MAX_DAYS = 200
START_GOLD = 500
START_WARRIORS = 3
MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2

# Level tables (index = level; 0 = "no building"). Mirrors testing-tool.py.
HQ_HP       = [0, 10, 15, 20, 25, 30]
HQ_TURRET   = [0, 1,  2,  2,  3,  3]
HQ_WCAP     = [0, 1,  2,  3,  4,  5]
HQ_WHP      = [0, 4,  5,  6,  7,  8]
HQ_TRAINCAP = [0, 1,  1,  2,  2,  3]
HQ_UPCOST   = [0, 0,  600, 1200, 2400, 3600]   # cost to *reach* level i
HQ_MAXLEVEL = 5
HQ_HEAL     = 1000

BASE_HP     = [0, 6, 12, 18]
BASE_TURRET = [0, 1, 1,  2]
BASE_WCAP   = [0, 1, 2,  3]
BASE_COST   = [0, 300, 600, 1000]              # cost to *reach* level i (1 = build)
BASE_MAXLEVEL = 3
BASE_HEAL   = 500

OWN_NONE, OWN_LEFT, OWN_RIGHT = 0, 1, 2
KIND_NONE, KIND_HQ, KIND_BASE = 0, 1, 2


# --------------------------------------------------------------------------- #
# Per-map precompute (CPU edge weights -> exact; GPU Floyd-Warshall + next-step)
# --------------------------------------------------------------------------- #
def _edge_weights_cpu(m) -> list[list[int]]:
    """Adjacency edge weights = ceil(euclidean), computed with math (double) to
    match the reference's ``edge_weight`` exactly. Non-edges = -1."""
    N = m.N
    W = [[-1] * N for _ in range(N)]
    for u in range(N):
        for v in m.adj[u]:
            dx = m.x[u] - m.x[v]
            dy = m.y[u] - m.y[v]
            W[u][v] = math.ceil(math.sqrt(dx * dx + dy * dy))
    return W


def compute_map_tensors(maps: list, N: int, T: int, device: torch.device) -> dict:
    """Build the static per-map tensors for a list of maps, padded to (N, T).
    Reused both for the initial batch and for regenerating individual slots."""
    B = len(maps)
    INF = 1 << 50
    wW = torch.full((B, N, N), INF, dtype=torch.int64)
    adj_mask = torch.zeros((B, N, N), dtype=torch.bool)
    is_stronghold = torch.zeros((B, N), dtype=torch.bool)
    cx = torch.zeros((B, N), dtype=torch.int64)
    cy = torch.zeros((B, N), dtype=torch.int64)
    tok_ids = torch.full((B, T), N, dtype=torch.int64)
    tok_valid = torch.zeros((B, T), dtype=torch.bool)
    n_regions = torch.tensor([m.N for m in maps], dtype=torch.int64)

    for b, m in enumerate(maps):
        nb = m.N
        cx[b, :nb] = torch.tensor(m.x, dtype=torch.int64)
        cy[b, :nb] = torch.tensor(m.y, dtype=torch.int64)
        W = _edge_weights_cpu(m)
        for u in range(nb):
            for v in m.adj[u]:
                wW[b, u, v] = W[u][v]
                adj_mask[b, u, v] = True
        for r in m.strongholds:
            is_stronghold[b, r] = True
        for i in range(N):
            wW[b, i, i] = 0
        ids = sorted(set(m.strongholds) | {0, nb - 1})
        tok_ids[b, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
        tok_valid[b, :len(ids)] = True

    wW = wW.to(device)
    adj_mask = adj_mask.to(device)
    cx, cy = cx.to(device), cy.to(device)
    is_stronghold = is_stronghold.to(device)
    token_ids = tok_ids.to(device)
    token_valid = tok_valid.to(device)

    # Floyd-Warshall (exact int64)
    dist = wW.clone()
    for k in range(N):
        dist = torch.minimum(dist, dist[:, :, k].unsqueeze(2) + dist[:, k, :].unsqueeze(1))

    # next-hop table (ties -> smallest neighbour id)
    nxt = torch.full((B, N, N), -1, dtype=torch.int32, device=device)
    best = torch.full((B, N, N), INF, dtype=torch.int64, device=device)
    for nb in range(N):
        score = wW[:, :, nb].unsqueeze(2) + dist[:, nb, :].unsqueeze(1)
        cand = adj_mask[:, :, nb].unsqueeze(2) & (score < best)
        best = torch.where(cand, score, best)
        nxt = torch.where(cand, torch.full_like(nxt, nb), nxt)
    diag = torch.arange(N, device=device)
    nxt[:, diag, diag] = diag.to(torch.int32)

    # travel time in turns to each token region
    tt_turns = torch.full((B, N, T), 1 << 20, dtype=torch.int64, device=device)
    b_ar = torch.arange(B, device=device)
    allr = torch.arange(N, device=device)
    for ti in range(T):
        tgt = token_ids[:, ti].clamp(max=N - 1)
        cur = torch.full((B, N), 1 << 20, dtype=torch.int64, device=device)
        cur[b_ar, tgt] = 0
        nxt_to_tgt = nxt[b_ar[:, None].expand(B, N), allr[None, :].expand(B, N),
                         tgt[:, None].expand(B, N)].long()
        valid_nb = nxt_to_tgt >= 0
        nxt_safe = nxt_to_tgt.clamp(min=0)
        for _ in range(N):
            cand = cur.gather(1, nxt_safe) + 1
            cand = torch.where(valid_nb, cand, torch.full_like(cand, 1 << 20))
            upd = torch.minimum(cur, cand)
            upd[b_ar, tgt] = 0
            if torch.equal(upd, cur):
                break
            cur = upd
        tt_turns[:, :, ti] = cur

    # bit-packed target-inference table for the hidden-info opp-gold reconstruction
    # (mirrors the submission bot's `tvia`): tvia_bits[b, a, nb] = bitmask over targets t of
    # (nxt[a][t] == nb) -- the set of targets whose next hop from region a is nb. So a
    # warrior stepping a->nb is consistent with exactly the targets in that mask.
    # Packed 62 real bits per int64 word (avoids the sign bit); W64 = ceil(N/62).
    BITS = 62
    W64 = (N + BITS - 1) // BITS
    reg = torch.arange(N, device=device)
    reg_word = reg // BITS                                    # [N] word index per region
    reg_bit = (torch.ones(N, dtype=torch.int64, device=device) << (reg % BITS))  # [N]
    tvia_bits = torch.zeros((B, N, N, W64), dtype=torch.int64, device=device)
    valid_hop = nxt >= 0                                      # [B,N,N] (a, t)
    dest_nb = nxt.clamp(min=0).long()                        # [B,N,N] -> next hop nb
    for w in range(W64):
        in_word = (reg_word == w)                            # [N] over targets t
        srcbit = torch.where(in_word[None, None, :] & valid_hop,
                             reg_bit[None, None, :].expand(B, N, N),
                             torch.zeros((), dtype=torch.int64, device=device))
        # scatter targets onto their next-hop nb (dim=2): tvia_bits[b,a,nb,w] |= bit(t)
        tvia_bits[:, :, :, w].scatter_add_(2, dest_nb, srcbit)

    return dict(cx=cx, cy=cy, is_stronghold=is_stronghold, next_step=nxt,
                token_ids=token_ids, token_valid=token_valid, travel_turns=tt_turns,
                tvia_bits=tvia_bits,
                n_regions=n_regions.to(device), hq_right=(n_regions - 1).to(device))


class MapBatch:
    """Holds the static, per-map tensors for a batch of maps (padded to N,T).

    n_cap / t_cap let you reserve capacity larger than the initial maps so that
    games can later be regenerated with bigger maps (used for per-episode maps)."""

    def __init__(self, maps: list, device: torch.device, n_cap=None, t_cap=None):
        self.device = device
        self.B = len(maps)
        self.N = max([m.N for m in maps] + ([n_cap] if n_cap else []))
        self.T = max([m.K + 2 for m in maps] + ([t_cap] if t_cap else []))
        self.K = max(m.K for m in maps)
        d = compute_map_tensors(maps, self.N, self.T, device)
        for k, v in d.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------- #
# Segmented order statistics (the one primitive that captures every tie-break)
# --------------------------------------------------------------------------- #
def _seg_stats(group: torch.Tensor, hp: torch.Tensor, slot: torch.Tensor,
               valid: torch.Tensor):
    """For each entry, within its ``group`` and ordered by (hp asc, slot asc),
    return (rank, exclusive-hp-prefix). Invalid entries get (0, 0).

    Shapes are [B, W]. Sorting is done **per row** (dim=1) -- a batched sort,
    which is dramatically faster on GPU than one flat sort over B*W. ``group``
    ids only need to be unique within a row. Requires group < 2^24, hp < 64,
    slot < 2048 (all satisfied: region < 110, hp <= ~30, slot < 1206).
    """
    B, W = group.shape
    dev = group.device
    HUGE_G = 1 << 24
    gg = torch.where(valid, group, torch.full_like(group, HUGE_G))
    key = (gg * 64 + hp.to(torch.int64)) * 2048 + slot.to(torch.int64)
    order = torch.argsort(key, dim=1)
    gs = torch.gather(gg, 1, order)
    hs = torch.gather(hp.to(torch.int64), 1, order)

    idx = torch.arange(W, device=dev)[None, :].expand(B, W)
    seg_change = torch.ones((B, W), dtype=torch.bool, device=dev)
    seg_change[:, 1:] = gs[:, 1:] != gs[:, :-1]

    seg_first = torch.where(seg_change, idx, torch.zeros_like(idx))
    seg_first = torch.cummax(seg_first, dim=1).values
    rank_sorted = idx - seg_first

    cum = torch.cumsum(hs, dim=1)
    cum_excl = cum - hs
    base = torch.where(seg_change, cum_excl, torch.full_like(cum_excl, -1))
    base = torch.cummax(base, dim=1).values
    prefix_sorted = cum_excl - base

    out_rank = torch.empty_like(rank_sorted).scatter_(1, order, rank_sorted)
    out_pref = torch.empty_like(prefix_sorted).scatter_(1, order, prefix_sorted)
    out_rank = torch.where(valid, out_rank, torch.zeros_like(out_rank))
    out_pref = torch.where(valid, out_pref, torch.zeros_like(out_pref))
    return out_rank, out_pref


# --------------------------------------------------------------------------- #
# The batched environment
# --------------------------------------------------------------------------- #
class FastEnv:
    def __init__(self, maps: list, device: Optional[str] = None,
                 max_warriors_per_side: Optional[int] = None,
                 n_cap: Optional[int] = None, t_cap: Optional[int] = None):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mb = MapBatch(maps, self.device, n_cap=n_cap, t_cap=t_cap)
        self._map_rng = None
        # Optional map_gen.MapFactory: a background process pool that keeps fresh
        # maps queued up. Generating one costs ~25 ms of single-core Python, and
        # regen() needs one per finished episode, so at large B this is the
        # difference between the rollout waiting on a core and not. None -> maps
        # are generated inline from _map_rng (deterministic).
        self.map_factory = None
        B, N = self.mb.B, self.mb.N
        self.B, self.N = B, N
        dev = self.device

        # bit-packing helpers for the opp-gold move-cost target inference (see
        # _update_est_gold): 62 real region-bits per int64 word, W64 = ceil(N/62).
        self.BITS = 62
        self.W64 = (N + self.BITS - 1) // self.BITS
        reg = torch.arange(N, device=dev)
        self._reg_word = reg // self.BITS                       # [N]
        self._reg_bit = (torch.ones(N, dtype=torch.int64, device=dev) << (reg % self.BITS))

        # constant lookup tables on device
        self.HQ_HP = torch.tensor(HQ_HP, device=dev)
        self.HQ_TURRET = torch.tensor(HQ_TURRET, device=dev)
        self.HQ_WCAP = torch.tensor(HQ_WCAP, device=dev)
        self.HQ_WHP = torch.tensor(HQ_WHP, device=dev)
        self.HQ_TRAINCAP = torch.tensor(HQ_TRAINCAP, device=dev)
        self.HQ_UPCOST = torch.tensor(HQ_UPCOST, device=dev)
        self.BASE_HP = torch.tensor(BASE_HP, device=dev)
        self.BASE_TURRET = torch.tensor(BASE_TURRET, device=dev)
        self.BASE_WCAP = torch.tensor(BASE_WCAP, device=dev)
        self.BASE_COST = torch.tensor(BASE_COST, device=dev)

        # Pool width per side. Default is the theoretical max (training 3/day for
        # all 200 days) so the simulation can never overflow -> bit-exact. For
        # throughput you can lower it (gold limits real games to far fewer
        # warriors); a smaller pool means a narrower per-row sort.
        full = START_WARRIORS + 3 * MAX_DAYS
        self.Wside = full if max_warriors_per_side is None else max_warriors_per_side
        self.W = 2 * self.Wside
        self.left_base = 0
        self.right_base = self.Wside
        # side per slot: 0 left, 1 right
        slot_side = torch.zeros(self.W, dtype=torch.int64, device=dev)
        slot_side[self.right_base:] = 1
        self.slot_side = slot_side                    # [W]
        self.slot_idx = torch.arange(self.W, device=dev)  # [W]
        self.b_ar = torch.arange(B, device=dev)

        self.hq_region = torch.zeros((B, 2), dtype=torch.int64, device=dev)
        self.hq_region[:, 0] = 0
        self.hq_region[:, 1] = self.mb.hq_right          # per-game right HQ
        allr = torch.arange(N, device=dev)
        self.is_hq_mask = (allr[None, :] == 0) | (allr[None, :] == self.mb.hq_right[:, None])

        self.reset()

    # ----------------------------------------------------------------- reset
    def _alloc(self):
        B, N, W, dev = self.B, self.N, self.W, self.device
        z = lambda *s: torch.zeros(s, dtype=torch.int64, device=dev)
        self.day = z(B)
        self.gold = z(B, 2)
        self.prev_income = z(B, 2)
        self.n_created = z(B, 2)
        self.b_owner = z(B, N); self.b_kind = z(B, N)
        self.b_level = z(B, N); self.b_hp = z(B, N)
        self.w_hp = z(B, W); self.w_region = z(B, W)
        self.w_move = torch.zeros((B, W), dtype=torch.bool, device=dev)
        self.w_tgt = z(B, W)
        # per-region realized work 'take' (= gold/WORK_INCOME) from the last work
        # phase, per side; used to build the gold-production aux-prediction label.
        self.last_region_take = z(B, N, 2)
        # per-side "saving to upgrade the HQ" commitment flag (set by the policy's
        # HQ-upgrade macro; while True, costly actions are masked off in sampling).
        self.hq_commit = torch.zeros((B, 2), dtype=torch.bool, device=dev)
        # ---- hidden-info opponent-gold ESTIMATE (mirrors vanilla_bot) ---------
        # est_gold[b, s] = side s's gold as RECONSTRUCTED by its opponent from the
        # visible economy: builds/trains/income/upkeep are exact, only move cost is
        # inferred (target hidden). Fed to the net as the opp-gold feature so training
        # matches inference (the bot never sees exact opp gold). Per-warrior move-
        # inference state: tgt_set (bitmask of still-consistent targets), move_chg
        # (an outstanding refundable 10-charge), moved_last (moved on the prev turn).
        self.est_gold = torch.full((B, 2), START_GOLD, dtype=torch.int64, device=dev)
        self.build_spend = z(B, 2)                      # per-side build spend this turn
        self.prev_region = z(B, W)                      # warrior regions before _phase_move
        self.w_tgt_set = torch.zeros((B, W, self.W64), dtype=torch.int64, device=dev)
        self.w_move_chg = torch.zeros((B, W), dtype=torch.bool, device=dev)
        self.w_moved_last = torch.zeros((B, W), dtype=torch.bool, device=dev)

    def reset(self, mask=None):
        """Reset all games (mask=None) or only the games where mask[b] is True
        (same maps reused). Returns self."""
        dev = self.device
        if not hasattr(self, "day"):
            self._alloc()
        m = (torch.ones(self.B, dtype=torch.bool, device=dev) if mask is None
             else mask.to(dev).bool())
        mc = m[:, None]
        self.day = torch.where(m, torch.zeros_like(self.day), self.day)
        self.gold = torch.where(mc, torch.full_like(self.gold, START_GOLD), self.gold)
        self.prev_income = torch.where(mc, torch.zeros_like(self.prev_income), self.prev_income)
        self.n_created = torch.where(mc, torch.full_like(self.n_created, START_WARRIORS), self.n_created)
        for t in (self.b_owner, self.b_kind, self.b_level, self.b_hp):
            t[m] = 0
        for t in (self.w_hp, self.w_region, self.w_tgt):
            t[m] = 0
        self.w_move[m] = False
        self.last_region_take[m] = 0
        self.hq_commit[m] = False
        # reset the opp-gold estimate + move-inference state for the (re)started games
        self.est_gold = torch.where(mc, torch.full_like(self.est_gold, START_GOLD), self.est_gold)
        self.build_spend[m] = 0
        self.prev_region[m] = 0
        self.w_tgt_set[m] = 0
        self.w_move_chg[m] = False
        self.w_moved_last[m] = False
        rows = m.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            rhq = self.mb.hq_right[rows]
            self.b_owner[rows, 0] = OWN_LEFT; self.b_kind[rows, 0] = KIND_HQ
            self.b_level[rows, 0] = 1; self.b_hp[rows, 0] = HQ_HP[1]
            self.b_owner[rows, rhq] = OWN_RIGHT; self.b_kind[rows, rhq] = KIND_HQ
            self.b_level[rows, rhq] = 1; self.b_hp[rows, rhq] = HQ_HP[1]
            for j in range(START_WARRIORS):
                self.w_hp[rows, self.left_base + j] = HQ_WHP[1]
                self.w_region[rows, self.left_base + j] = 0
                self.w_hp[rows, self.right_base + j] = HQ_WHP[1]
                self.w_region[rows, self.right_base + j] = rhq
        return self

    def _random_map(self):
        import random
        if self.map_factory is not None:
            return self.map_factory.get()
        if self._map_rng is None:
            self._map_rng = random.Random()
        r = self._map_rng
        while True:
            NP = r.randint(25, 54)
            N = 2 * NP + 1
            klo = (3 * N + 19) // 20
            khi = N // 5
            klo = klo + 1 if klo % 2 == 0 else klo
            khi = khi - 1 if khi % 2 == 0 else khi
            KP = r.randint((klo - 1) // 2, (khi - 1) // 2)
            try:
                return tt.read_map(tt.generate_map(tt.XoShiro256(r.getrandbits(63)), NP, KP))
            except (ValueError, RuntimeError):
                continue

    def regen(self, mask):
        """Generate brand-new random maps for the masked game slots and reset
        them (used to give each new episode a fresh map). Capacity (n_cap/t_cap)
        must accommodate the largest possible map (N=109, T=23)."""
        rows = mask.to(self.device).bool().nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return self
        maps = [self._random_map() for _ in range(rows.numel())]
        d = compute_map_tensors(maps, self.N, self.mb.T, self.device)
        for k, v in d.items():
            getattr(self.mb, k)[rows] = v
        self.hq_region[rows, 1] = d['hq_right']
        allr = torch.arange(self.N, device=self.device)
        self.is_hq_mask[rows] = (allr[None, :] == 0) | (allr[None, :] == d['hq_right'][:, None])
        self.reset(mask)
        return self

    # ----------------------------------------------------- building stat maps
    def _turret(self):
        lev = self.b_level
        hq = self.HQ_TURRET[lev.clamp(max=HQ_MAXLEVEL)]
        ba = self.BASE_TURRET[lev.clamp(max=BASE_MAXLEVEL)]
        return torch.where(self.b_kind == KIND_HQ, hq,
                           torch.where(self.b_kind == KIND_BASE, ba,
                                       torch.zeros_like(lev)))

    def _workcap(self):
        lev = self.b_level
        hq = self.HQ_WCAP[lev.clamp(max=HQ_MAXLEVEL)]
        ba = self.BASE_WCAP[lev.clamp(max=BASE_MAXLEVEL)]
        return torch.where(self.b_kind == KIND_HQ, hq,
                           torch.where(self.b_kind == KIND_BASE, ba,
                                       torch.zeros_like(lev)))

    def _maxhp(self):
        lev = self.b_level
        hq = self.HQ_HP[lev.clamp(max=HQ_MAXLEVEL)]
        ba = self.BASE_HP[lev.clamp(max=BASE_MAXLEVEL)]
        return torch.where(self.b_kind == KIND_HQ, hq,
                           torch.where(self.b_kind == KIND_BASE, ba,
                                       torch.zeros_like(lev)))

    # ------------------------------------------------- per-region warrior maps
    def _scatter_region(self, mask, values=None):
        """Sum ``values`` (or counts) over warriors selected by ``mask``,
        bucketed by (game, region). Returns [B, N] int64."""
        B, N = self.B, self.N
        flat = self.b_ar[:, None] * N + self.w_region   # [B,W]
        out = torch.zeros(B * N, dtype=torch.int64, device=self.device)
        src = values if values is not None else torch.ones_like(self.w_region)
        out.scatter_add_(0, flat[mask], src[mask])
        return out.view(B, N)

    def _side_counts(self):
        """alive counts and hp sums per side: returns cnt[B,N,2], sumhp[B,N,2]."""
        alive = self.w_hp > 0
        side = self.slot_side[None, :].expand(self.B, self.W)
        is_l = alive & (side == 0)
        is_r = alive & (side == 1)
        cnt = torch.stack([self._scatter_region(is_l),
                           self._scatter_region(is_r)], dim=2)
        sumhp = torch.stack([self._scatter_region(is_l, self.w_hp),
                             self._scatter_region(is_r, self.w_hp)], dim=2)
        return cnt, sumhp

    # --------------------------------------------------------------- step
    def step(self, actions: dict, apply_agent_rules: bool = True, relax_right=None):
        """Advance every game one day. ``actions`` = {'left':{...}, 'right':{...}}
        each with build [B,N] bool, move [B,N] long (-1 none), train [B] long, and
        optionally mobilize [B,N] bool -- regions whose commanded move takes EVERY
        stationary warrior (labourers included) instead of keeping work_cap home.

        ``apply_agent_rules`` (default True) enables the agent-side build policy
        baked into the env: gate builds on having a non-moving worker, cap builds
        to the worker count, and auto-staff understaffed new/upgraded bases. Set
        it False for a pure-rules step (used by the testing-tool parity tests).

        ``relax_right`` [B,N] bool (optional): per-region "lift the work-cap move
        restriction" flag for the RIGHT player (side 1). Where set, every commanded
        warrior in that region moves (no work_cap kept home). Used for the opponent
        in full-mobilisation training games; None = normal rules for both sides."""
        self._phase_build(actions, apply_agent_rules)
        self._phase_register_moves(actions, relax_right)
        self._phase_train_charge(actions)
        # snapshot for the opp-gold move inference: regions BEFORE movement, and who
        # was alive to move this turn (spawns happen after, so they can't have moved).
        self.prev_region = self.w_region.clone()
        self._alive_before = self.w_hp > 0
        self._phase_move()
        self._phase_spawn()
        self._phase_combat()
        self._phase_work()
        self._phase_upkeep()
        self._update_est_gold()
        self.day += 1
        return self

    def _update_est_gold(self):
        """Update est_gold[b, s] = each side's gold as its OPPONENT would reconstruct
        it from visible info. Builds/trains/income are charged exactly; move cost is
        inferred by the bot's nxt-path TARGET INFERENCE (a new move costs 10, and it
        is refunded once the inferred target is known to be one of that side's OWN
        buildings -- a free garrison move). Vectorized over all warriors, both sides at
        once; each warrior contributes to its OWN side's estimate."""
        B, N, W = self.B, self.N, self.W
        dev = self.device
        prev = self.prev_region                          # [B,W] region before move
        cur = self.w_region                              # [B,W] region after move
        moved = self._alive_before & (cur != prev)       # [B,W] made a step this turn
        alive_now = self.w_hp > 0                         # [B,W] survivors (post-combat)
        sd = self.slot_side[None, :].expand(B, W)         # [B,W] 0/1 side per slot

        # per-side owned-building bitmask [B,W64] (targets that are free to move onto)
        def pack(mask):                                   # [B,N] bool -> [B,W64] int64
            contrib = mask.long() * self._reg_bit[None, :]           # [B,N]
            return torch.stack([(contrib * (self._reg_word == w)[None, :]).sum(1)
                                for w in range(self.W64)], dim=1)     # [B,W64]
        owned0 = self.b_owner == OWN_LEFT                 # [B,N]
        owned1 = self.b_owner == OWN_RIGHT
        obits0, obits1 = pack(owned0), pack(owned1)       # [B,W64] each
        owned_w = torch.where(sd[:, :, None] == 0,
                              obits0[:, None, :], obits1[:, None, :])  # [B,W,W64] per warrior
        prev_owned = torch.where(sd == 0, owned0.gather(1, prev), owned1.gather(1, prev))  # [B,W]
        cur_owned = torch.where(sd == 0, owned0.gather(1, cur), owned1.gather(1, cur))     # [B,W]

        # cons[b,w] = targets consistent with the step prev->cur = tvia_bits[b,prev,cur]
        tvia = self.mb.tvia_bits.view(B, N * N, self.W64)  # [B,N*N,W64]
        cons = torch.gather(tvia, 1, (prev * N + cur)[:, :, None].expand(B, W, self.W64))

        tgt = self.w_tgt_set                             # [B,W,W64] carried from last turn
        mchg = self.w_move_chg                           # [B,W]
        mlast = self.w_moved_last                        # [B,W]
        tgt_nonempty = (tgt != 0).any(dim=2)             # [B,W]
        fits_prev = ((tgt & cons) != 0).any(dim=2)       # step fits the running target set
        is_new = moved & ((~mlast) | (~fits_prev))       # fresh dispatch or re-dispatch
        is_cont = moved & (~is_new)                      # continuation of the same move

        new_tgt = torch.where(is_new[:, :, None], cons,
                  torch.where(is_cont[:, :, None], tgt & cons, tgt))     # [B,W,W64]
        new_mchg = torch.where(is_new, torch.ones_like(mchg), mchg)      # cont keeps old chg
        # free detection: all remaining possible targets are owned -> refund the charge
        subset_owned = ((new_tgt & ~owned_w) == 0).all(dim=2)
        free_moved = moved & new_mchg & (new_tgt != 0).any(dim=2) & subset_owned
        new_mchg = new_mchg & (~free_moved)
        # a mid-move warrior that did NOT step this turn -> its move concluded (stopped)
        concluded = alive_now & (~moved) & tgt_nonempty
        refund_concl = concluded & mchg & cur_owned

        # per-warrior net move charge (units of MOVE_COST): +new, -(re-dispatch/free/stop refunds)
        delta_w = (is_new.long()
                   - (is_new & mchg & prev_owned).long()   # re-dispatch off an owned building
                   - free_moved.long()
                   - refund_concl.long()) * MOVE_COST       # [B,W]
        move_cost = torch.stack([delta_w[:, :self.Wside].sum(1),
                                 delta_w[:, self.Wside:].sum(1)], dim=1)  # [B,2]

        # commit per-warrior inference state (moved -> updated; concluded -> cleared;
        # else carried; dead slots cleared so they can never trigger a stray refund)
        self.w_tgt_set = torch.where(moved[:, :, None], new_tgt,
                         torch.where(concluded[:, :, None], torch.zeros_like(tgt), tgt))
        self.w_move_chg = torch.where(moved, new_mchg,
                          torch.where(concluded, torch.zeros_like(mchg), mchg))
        dead = ~alive_now
        self.w_tgt_set = torch.where(dead[:, :, None], torch.zeros_like(self.w_tgt_set), self.w_tgt_set)
        self.w_move_chg = self.w_move_chg & (~dead)
        self.w_moved_last = moved & (~dead)

        # economy: exact income/build/train, estimated moves, estimate-based upkeep
        train_spend = self.pending_train * TRAIN_COST                  # [B,2] exact
        est = self.est_gold - self.build_spend - move_cost - train_spend + self.prev_income
        alive_cnt = torch.stack([(alive_now & (sd == 0)).sum(1),
                                 (alive_now & (sd == 1)).sum(1)], dim=1)  # [B,2]
        fed = torch.minimum(alive_cnt, est.clamp(min=0) // UPKEEP_PER_WARRIOR)
        self.est_gold = est - UPKEEP_PER_WARRIOR * fed

    # build -> register moves -> train (morning, in that order)
    def _phase_build(self, actions, apply_agent_rules: bool = True):
        B, N, W = self.B, self.N, self.W
        cnt, _ = self._side_counts()
        for side in (0, 1):
            me = OWN_LEFT if side == 0 else OWN_RIGHT
            mask = actions['left' if side == 0 else 'right']['build']
            fr = cnt[:, :, side]
            en = cnt[:, :, 1 - side]
            legal = mask & (fr > 0) & (en == 0)

            if apply_agent_rules:
                # Gating uses *free* workers: non-moving friendly warriors that are
                # not currently labouring (surplus beyond a region's work_cap, plus
                # any stationary warriors off the buildings) -- the same pool that
                # can actually be dispatched to staff a base. The cap still uses the
                # plain non-moving count (n_nonmoving).
                sd = self.slot_side[None, :].expand(B, W)
                stat = (self.w_hp > 0) & (sd == side) & (~self.w_move)
                n_nonmoving = stat.sum(dim=1)
                keepcap = torch.where(self.b_owner == me, self._workcap(),
                                      torch.zeros_like(self.b_level))
                n_free = (self._scatter_region(stat) - keepcap).clamp(min=0).sum(dim=1)
                legal = legal & (n_free[:, None] >= 1)

            is_mine = self.b_owner == me
            is_empty = self.b_owner == OWN_NONE
            is_strong = self.mb.is_stronghold
            is_hq = self.is_hq_mask

            maxlev = torch.where(self.b_kind == KIND_HQ, HQ_MAXLEVEL, BASE_MAXLEVEL)

            build_new = legal & is_empty & is_strong & (~is_hq)
            can_up = legal & is_mine & (self.b_level < maxlev)
            can_heal = legal & is_mine & (self.b_level >= maxlev)

            if apply_agent_rules:
                # Cap new-build/upgrade actions (heal exempt) to n_nonmoving: if
                # more were requested, keep a uniformly random subset of that many.
                # The dropped ones are intentionally NOT executed here; for PPO they
                # still count as taken (build_logp uses the pre-exec Bernoulli outcome).
                nu = build_new | can_up
                if bool((nu.sum(dim=1) > n_nonmoving).any()):
                    rnd = torch.where(nu, torch.rand(B, N, device=self.device),
                                      torch.full((B, N), -1.0, device=self.device))
                    order = torch.argsort(rnd, dim=1, descending=True)
                    pos = torch.empty_like(order)
                    pos.scatter_(1, order,
                                 torch.arange(N, device=self.device)[None, :].expand(B, N))
                    keep = pos < n_nonmoving[:, None]
                    build_new = build_new & keep
                    can_up = can_up & keep

            # forced builds (the committed HQ-upgrade macro): bypass the n_free
            # gating and the cap, respecting only the game rules (own warrior present,
            # no enemy in the region). Still subject to force-staffing below.
            fb = actions['left' if side == 0 else 'right'].get('force_build') \
                if apply_agent_rules else None
            if fb is not None:
                f_legal = fb & (fr > 0) & (en == 0)
                build_new = build_new | (f_legal & is_empty & is_strong & (~is_hq))
                can_up = can_up | (f_legal & is_mine & (self.b_level < maxlev))
                can_heal = can_heal | (f_legal & is_mine & (self.b_level >= maxlev))

            # costs
            upcost = torch.where(
                self.b_kind == KIND_HQ,
                self.HQ_UPCOST[(self.b_level + 1).clamp(max=HQ_MAXLEVEL)],
                self.BASE_COST[(self.b_level + 1).clamp(max=BASE_MAXLEVEL)])
            healcost = torch.where(self.b_kind == KIND_HQ, HQ_HEAL, BASE_HEAL)
            cost = torch.zeros((B, N), dtype=torch.int64, device=self.device)
            cost = torch.where(build_new, torch.full_like(cost, BASE_COST[1]), cost)
            cost = torch.where(can_up, upcost, cost)
            cost = torch.where(can_heal, healcost, cost)
            spend = cost.sum(dim=1)
            self.gold[:, side] -= spend
            self.build_spend[:, side] = spend        # exact, visible -> opp-gold estimate

            # apply: build new
            self.b_owner = torch.where(build_new, torch.full_like(self.b_owner, me), self.b_owner)
            self.b_kind = torch.where(build_new, torch.full_like(self.b_kind, KIND_BASE), self.b_kind)
            self.b_level = torch.where(build_new, torch.ones_like(self.b_level), self.b_level)
            # upgrade level
            self.b_level = torch.where(can_up, self.b_level + 1, self.b_level)
            # recompute hp where changed (new build, upgrade, heal)
            newhp = self._maxhp()
            changed = build_new | can_up | can_heal
            self.b_hp = torch.where(changed, newhp, self.b_hp)

            # auto-staff understaffed new/upgraded BASES (heal and the HQ excluded:
            # the HQ is always staffed at home, so it neither needs force-staffing
            # nor the free-worker gate).
            if apply_agent_rules:
                self._force_staff(side, me, (build_new | can_up) & (~self.is_hq_mask))

    def _force_staff(self, side, me, nu_mask):
        """For each region in ``nu_mask`` (this side's new builds / upgrades this
        turn) whose non-moving friendly count is below its post-build work_cap,
        force one nearest *surplus* friendly warrior to move there. The move is
        free (we own the building) and is registered here -- before
        _phase_register_moves -- so the policy's own move commands never touch the
        staffed warrior. Surplus = friendly stationary warriors beyond a region's
        keep-cap (work_cap where we own a building, else 0): exactly the pool the
        policy may move, so staffing simply claims from it first."""
        if not bool(nu_mask.any()):
            return
        B, N, W = self.B, self.N, self.W
        dev = self.device
        workcap = self._workcap()                                   # post-build [B,N]
        sd = self.slot_side[None, :].expand(B, W)
        cand = (self.w_hp > 0) & (sd == side) & (~self.w_move)       # stationary friendly
        stat_cnt = self._scatter_region(cand)                       # [B,N]
        needing = nu_mask & (stat_cnt < workcap)
        if not bool(needing.any()):
            return

        keepcap = torch.where(self.b_owner == me, workcap, torch.zeros_like(workcap))
        keep_w = keepcap.gather(1, self.w_region)                   # [B,W]
        group = self.b_ar[:, None] * N + self.w_region
        slot = self.slot_idx[None, :].expand(B, W)
        rank, _ = _seg_stats(group, self.w_hp, slot, cand)
        is_surplus = cand & (rank >= keep_w)                        # movable pool
        surplus_rem = (stat_cnt - keepcap).clamp(min=0)            # per-region surplus count

        tt = self.mb.travel_turns                                   # [B,N,T]
        tok = self.mb.token_ids
        tvalid = self.mb.token_valid
        assigned = torch.zeros((B, W), dtype=torch.bool, device=dev)
        force_tgt = torch.zeros((B, W), dtype=torch.int64, device=dev)
        BIGD = 1 << 30
        for ti in range(self.mb.T):
            tokreg = tok[:, ti].clamp(max=N - 1)                    # [B]
            need_t = tvalid[:, ti] & needing.gather(1, tokreg[:, None]).squeeze(1)
            if not bool(need_t.any()):
                continue
            md = torch.where(surplus_rem > 0, tt[:, :, ti],
                             torch.full_like(tt[:, :, ti], BIGD))   # [B,N]
            r_star = md.argmin(dim=1)                               # [B] nearest src region
            src_ok = need_t & ((surplus_rem.gather(1, r_star[:, None]).squeeze(1)) > 0)
            at = ((self.w_region == r_star[:, None]) & is_surplus
                  & (~assigned) & src_ok[:, None])
            wkey = torch.where(at, rank, torch.full_like(rank, BIGD))
            wsel = wkey.argmin(dim=1)                               # [B] chosen warrior slot
            chose = at.gather(1, wsel[:, None]).squeeze(1)
            rows = chose.nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                sl = wsel[rows]
                assigned[rows, sl] = True
                force_tgt[rows, sl] = tokreg[rows]
                surplus_rem[rows, r_star[rows]] -= 1

        self.w_move = torch.where(assigned, torch.ones_like(self.w_move), self.w_move)
        self.w_tgt = torch.where(assigned, force_tgt, self.w_tgt)

    def _phase_register_moves(self, actions, relax_right=None):
        B, N = self.B, self.N
        cnt, _ = self._side_counts()
        workcap = self._workcap()
        for side in (0, 1):
            me = OWN_LEFT if side == 0 else OWN_RIGHT
            move_tgt = actions['left' if side == 0 else 'right']['move']  # [B,N]
            base = self.left_base if side == 0 else self.right_base
            sl = slice(base, base + self.Wside)

            w_hp = self.w_hp[:, sl]
            w_region = self.w_region[:, sl]
            w_move = self.w_move[:, sl]
            slot = self.slot_idx[None, base:base + self.Wside].expand(B, self.Wside)

            tgt_w = move_tgt.gather(1, w_region)              # [B,Wside]
            has_cmd = tgt_w >= 0
            candidate = (w_hp > 0) & (~w_move) & has_cmd

            keepcap = torch.where(self.b_owner == me, workcap,
                                  torch.zeros_like(workcap))  # [B,N]
            # full mobilisation, from two independent sources (either one lifts the
            # work-cap for that region, so nobody is kept home and every commanded
            # warrior -- labourers included -- moves):
            #   * actions[side]['mobilize'] [B,N] bool -- the POLICY's own choice
            #     (T2's second head output for the source it ordered to move);
            #   * relax_right [B,N] -- the exogenous full-mobilisation noise applied
            #     to the RIGHT player in a fraction of training games.
            mobil = actions['left' if side == 0 else 'right'].get('mobilize')
            if side == 1 and relax_right is not None:
                mobil = relax_right if mobil is None else (mobil | relax_right)
            if mobil is not None:
                keepcap = torch.where(mobil, torch.zeros_like(keepcap), keepcap)
            keep_w = keepcap.gather(1, w_region)              # [B,Wside]

            group = self.b_ar[:, None] * N + w_region
            rank, _ = _seg_stats(group, w_hp, slot, candidate)
            move_flag = candidate & (rank >= keep_w)

            # charge: 10 per warrior, free if target has my building
            tgt_clamped = tgt_w.clamp(min=0)
            tgt_mine = self.b_owner.gather(1, tgt_clamped) == me
            cost_w = torch.where(tgt_mine, torch.zeros_like(tgt_w),
                                 torch.full_like(tgt_w, MOVE_COST))
            self.gold[:, side] -= (cost_w * move_flag).sum(dim=1)

            # apply
            self.w_move[:, sl] = torch.where(move_flag, torch.ones_like(w_move), w_move)
            self.w_tgt[:, sl] = torch.where(move_flag, tgt_w, self.w_tgt[:, sl])

    def opening_premove(self, side, hq_region, target, mask):
        """Turn-1 opening split helper: for each game in ``mask`` [B] bool, dispatch
        EXACTLY ONE warrior -- the first surplus one (lowest hp, then suffix, beyond the
        HQ work_cap) -- from ``hq_region[b]`` to ``target[b]`` for ``side``. Sets
        w_move/w_tgt and charges the move cost (0 onto our own building). Because the
        subsequent normal move registration skips already-moving warriors (~w_move), the
        remaining surplus warrior can then be sent to a DIFFERENT target the same turn."""
        B, N = self.B, self.N
        me = OWN_LEFT if side == 0 else OWN_RIGHT
        base = self.left_base if side == 0 else self.right_base
        sl = slice(base, base + self.Wside)
        w_hp = self.w_hp[:, sl]
        w_region = self.w_region[:, sl]
        slot = self.slot_idx[None, base:base + self.Wside].expand(B, self.Wside)
        workcap = self._workcap()

        at_hq = (w_region == hq_region[:, None]) & mask[:, None] & (target[:, None] >= 0)
        candidate = (w_hp > 0) & (~self.w_move[:, sl]) & at_hq
        group = self.b_ar[:, None] * N + w_region
        rank, _ = _seg_stats(group, w_hp, slot, candidate)
        keep = workcap.gather(1, hq_region.clamp(min=0)[:, None]).squeeze(1)   # [B]
        move_flag = candidate & (rank == keep[:, None])            # exactly the 1st surplus

        tgt_clamped = target.clamp(min=0)
        tgt_mine = (self.b_owner.gather(1, tgt_clamped[:, None]).squeeze(1) == me)
        cost = torch.where(tgt_mine, torch.zeros_like(target), torch.full_like(target, MOVE_COST))
        self.gold[:, side] -= move_flag.any(1).long() * cost
        tgt_b = target[:, None].expand(B, self.Wside)
        self.w_move[:, sl] = torch.where(move_flag, torch.ones_like(self.w_move[:, sl]),
                                         self.w_move[:, sl])
        self.w_tgt[:, sl] = torch.where(move_flag, tgt_b, self.w_tgt[:, sl])

    def _phase_train_charge(self, actions):
        self.pending_train = torch.zeros((self.B, 2), dtype=torch.int64, device=self.device)
        for side in (0, 1):
            n = actions['left' if side == 0 else 'right']['train'].to(self.device)
            hq_r = self.hq_region[:, side]
            hq_lev = self.b_level.gather(1, hq_r[:, None]).squeeze(1)
            cap = self.HQ_TRAINCAP[hq_lev.clamp(max=HQ_MAXLEVEL)]
            affordable = self.gold[:, side] // TRAIN_COST
            n_eff = torch.minimum(torch.minimum(n, cap), affordable)
            self.gold[:, side] -= n_eff * TRAIN_COST
            self.pending_train[:, side] = n_eff

    def _phase_move(self):
        B, N = self.B, self.N
        cnt, _ = self._side_counts()                     # snapshot before moving
        side = self.slot_side[None, :].expand(B, self.W)
        # opponent presence at each warrior's current region
        cnt_l = cnt[:, :, 0]
        cnt_r = cnt[:, :, 1]
        opp_at = torch.where(side == 0,
                             cnt_r.gather(1, self.w_region),
                             cnt_l.gather(1, self.w_region))   # [B,W]
        blocked = opp_at > 0
        moving = (self.w_hp > 0) & self.w_move

        # reference clears the flag for warriors already at their target FIRST,
        # before the enemy-blocking check (e.g. a src==tgt move).
        already = moving & (self.w_region == self.w_tgt)
        self.w_move = torch.where(already, torch.zeros_like(self.w_move), self.w_move)
        moving = moving & (~already)

        idx = self.w_region * N + self.w_tgt
        nxt = self.mb.next_step.view(B, N * N).gather(1, idx.clamp(min=0)).long()  # [B,W]
        do_move = moving & (~blocked)
        self.w_region = torch.where(do_move, nxt, self.w_region)
        arrived = do_move & (self.w_region == self.w_tgt)
        self.w_move = torch.where(arrived, torch.zeros_like(self.w_move), self.w_move)

    def _phase_spawn(self):
        B, N = self.B, self.N
        for side in (0, 1):
            base = self.left_base if side == 0 else self.right_base
            hq_r = self.hq_region[:, side]
            hq_lev = self.b_level.gather(1, hq_r[:, None]).squeeze(1)
            whp = self.HQ_WHP[hq_lev.clamp(max=HQ_MAXLEVEL)]
            n = self.pending_train[:, side]
            for j in range(3):
                make = (n > j) & (self.n_created[:, side] < self.Wside)
                if not bool(make.any()):
                    continue
                slot = base + self.n_created[:, side]     # [B]
                rows = torch.nonzero(make, as_tuple=True)[0]
                sl = slot[rows]
                self.w_hp[rows, sl] = whp[rows]
                self.w_region[rows, sl] = hq_r[rows]
                self.w_move[rows, sl] = False
                self.w_tgt[rows, sl] = 0
                self.n_created[rows, side] += 1

    def _phase_combat(self):
        B, N = self.B, self.N
        cnt, sumhp = self._side_counts()
        cnt_l, cnt_r = cnt[:, :, 0], cnt[:, :, 1]
        shp_l, shp_r = sumhp[:, :, 0], sumhp[:, :, 1]
        turret = self._turret()

        b_left = self.b_owner == OWN_LEFT
        b_right = self.b_owner == OWN_RIGHT
        left_present = (cnt_l > 0) | b_left
        right_present = (cnt_r > 0) | b_right
        both = left_present & right_present

        left_attacks = (cnt_l + torch.where(b_left, turret, torch.zeros_like(turret))) * both
        right_attacks = (cnt_r + torch.where(b_right, turret, torch.zeros_like(turret))) * both

        # focus-fire: left attacks hit right warriors; right attacks hit left.
        side = self.slot_side[None, :].expand(B, self.W)
        slot = self.slot_idx[None, :].expand(B, self.W)
        alive = self.w_hp > 0
        group = self.b_ar[:, None] * N + self.w_region
        valid_r = alive & (side == 1)
        valid_l = alive & (side == 0)
        _, pref_r = _seg_stats(group, self.w_hp, slot, valid_r)
        _, pref_l = _seg_stats(group, self.w_hp, slot, valid_l)

        A_on_r = left_attacks.gather(1, self.w_region)    # attacks vs right warriors
        A_on_l = right_attacks.gather(1, self.w_region)
        A_w = torch.where(side == 1, A_on_r, A_on_l)
        pref_w = torch.where(side == 1, pref_r, pref_l)
        dmg = (A_w - pref_w).clamp(min=0)
        dmg = torch.minimum(dmg, self.w_hp.clamp(min=0))
        new_hp = self.w_hp - dmg
        self.w_hp = torch.where(alive, new_hp, self.w_hp)

        # siege: idle attacks of the side NOT owning the building
        idle_L = (left_attacks - shp_r).clamp(min=0)       # left's leftover -> right bldg
        idle_R = (right_attacks - shp_l).clamp(min=0)      # right's leftover -> left bldg
        siege = torch.zeros((B, N), dtype=torch.int64, device=self.device)
        siege = torch.where(b_right, torch.minimum(idle_L, self.b_hp), siege)
        siege = torch.where(b_left, torch.minimum(idle_R, self.b_hp), siege)
        self.b_hp = self.b_hp - siege

        # remove dead warriors / destroyed buildings
        dead_w = self.w_hp <= 0
        self.w_hp = torch.where(dead_w, torch.zeros_like(self.w_hp), self.w_hp)
        destroyed = self.b_hp <= 0
        self._destroy_buildings(destroyed)

    def _destroy_buildings(self, destroyed):
        self.b_owner = torch.where(destroyed, torch.zeros_like(self.b_owner), self.b_owner)
        self.b_kind = torch.where(destroyed, torch.zeros_like(self.b_kind), self.b_kind)
        self.b_level = torch.where(destroyed, torch.zeros_like(self.b_level), self.b_level)
        self.b_hp = torch.where(destroyed, torch.zeros_like(self.b_hp), self.b_hp)

    def _phase_work(self):
        cnt, _ = self._side_counts()
        workcap = self._workcap()
        income = torch.zeros((self.B, 2), dtype=torch.int64, device=self.device)
        take_bns = torch.zeros((self.B, self.N, 2), dtype=torch.int64, device=self.device)
        for side in (0, 1):
            me = OWN_LEFT if side == 0 else OWN_RIGHT
            mine = self.b_owner == me
            take = torch.minimum(workcap, cnt[:, :, side])
            take = torch.where(mine, take, torch.zeros_like(take))
            take_bns[:, :, side] = take
            income[:, side] = WORK_INCOME * take.sum(dim=1)
        self.gold += income
        self.prev_income = income
        # stash the realized per-region take so the aux label is exact (post-work,
        # pre-upkeep counts) rather than recomputed from end-of-turn state.
        self.last_region_take = take_bns

    def _phase_upkeep(self):
        B = self.B
        side = self.slot_side[None, :].expand(B, self.W)
        slot = self.slot_idx[None, :].expand(B, self.W)
        alive = self.w_hp > 0
        # rank by suffix (slot) within (game, side), among alive
        group = self.b_ar[:, None] * 2 + side
        zero_hp = torch.zeros_like(self.w_hp)
        rank, _ = _seg_stats(group, zero_hp, slot, alive)   # order by slot only

        fed_count = (self.gold // UPKEEP_PER_WARRIOR)        # [B,2] max warriors fed
        fed_w = torch.where(side == 0,
                            fed_count[:, 0:1].expand(B, self.W),
                            fed_count[:, 1:2].expand(B, self.W))
        fed = alive & (rank < fed_w)
        hungry = alive & (rank >= fed_w)

        # deduct gold = 2 * number fed (per side)
        n_fed_l = (fed & (side == 0)).sum(dim=1)
        n_fed_r = (fed & (side == 1)).sum(dim=1)
        self.gold[:, 0] -= UPKEEP_PER_WARRIOR * n_fed_l
        self.gold[:, 1] -= UPKEEP_PER_WARRIOR * n_fed_r

        self.w_hp = torch.where(hungry, self.w_hp - 1, self.w_hp)
        dead = self.w_hp <= 0
        self.w_hp = torch.where(dead, torch.zeros_like(self.w_hp), self.w_hp)

    # --------------------------------------------------------------- outcome
    def hq_alive(self):
        """[B,2] bool: is each side's HQ still standing?"""
        l = (self.b_owner.gather(1, self.hq_region[:, 0:1]).squeeze(1) == OWN_LEFT)
        r = (self.b_owner.gather(1, self.hq_region[:, 1:2]).squeeze(1) == OWN_RIGHT)
        return torch.stack([l, r], dim=1)

    def token_take(self, side: int = 0):
        """[B,T,2] per-거점 realized work 'take' (= gold/WORK_INCOME) from the most
        recent work phase, columns = (me, opp) for ``side``. Gathered onto the same
        token order as ``observe`` (sorted token_ids, padded tokens clamped)."""
        tok_g = self.mb.token_ids.clamp(max=self.N - 1)
        me, opp = side, 1 - side
        lt = self.last_region_take
        return torch.stack([lt[:, :, me].gather(1, tok_g),
                            lt[:, :, opp].gather(1, tok_g)], dim=2).float()

    def aux_label(self, side: int = 0):
        """Auxiliary-prediction labels for ``side`` from the CURRENT (post-step)
        state, i.e. what ``side`` will SEE next turn. Returns:
          - tok [B,T,6] raw counts: col0 = enemy warriors garrisoned at the 거점,
            cols1..5 = enemy warriors that can REACH it within 1..5 turns.
          - opp_gold [B]: the opponent's gold.
        Both are the same quantities ``observe`` packs as features next turn (op_cnt
        and reach[19:24]); the caller applies the log transform. Token order matches
        ``observe``. Padded tokens are ignored by the (tmask-)masked aux loss."""
        B, N, T = self.B, self.N, self.mb.T
        dev = self.device
        opp = 1 - side
        tok_g = self.mb.token_ids.clamp(max=N - 1)
        cnt, _ = self._side_counts()
        op_region_cnt = cnt[:, :, opp]                          # [B,N]
        garrison = op_region_cnt.gather(1, tok_g).float()       # [B,T]
        tt = self.mb.travel_turns                              # [B,N,T]
        reach = torch.empty((B, T, 5), dtype=torch.float32, device=dev)
        for k in range(1, 6):
            reach[:, :, k - 1] = (op_region_cnt[:, :, None] * (tt <= k)).sum(1).float()
        tok = torch.cat([garrison[:, :, None], reach], dim=2)  # [B,T,6]
        return tok, self.gold[:, opp].float()

    # --------------------------------------------------------------- observe
    def observe(self, side: int):
        """Return token features [B,T,F], global features [B,G], and the action
        helper info, all from ``side``'s perspective (0=left, 1=right)."""
        B, N, T = self.B, self.N, self.mb.T
        dev = self.device
        me, opp = side, 1 - side
        tok = self.mb.token_ids                         # [B,T] (sentinel=N)
        tvalid = self.mb.token_valid                    # [B,T] bool
        tok_g = tok.clamp(max=N - 1)                    # safe gather index

        cnt, sumhp = self._side_counts()
        turret = self._turret()
        workcap = self._workcap()
        myo = OWN_LEFT if side == 0 else OWN_RIGHT
        oppo = OWN_RIGHT if side == 0 else OWN_LEFT

        def gtok(field):  # gather a [B,N] region map onto the T token regions
            return field.gather(1, tok_g)

        my_cnt = cnt[:, :, me]
        op_cnt = cnt[:, :, opp]
        my_base_lvl = torch.where((self.b_owner == myo) & (self.b_kind == KIND_BASE),
                                  self.b_level, torch.zeros_like(self.b_level))
        op_base_lvl = torch.where((self.b_owner == oppo) & (self.b_kind == KIND_BASE),
                                  self.b_level, torch.zeros_like(self.b_level))
        my_hq_lvl = torch.where((self.b_owner == myo) & (self.b_kind == KIND_HQ),
                                self.b_level, torch.zeros_like(self.b_level))
        op_hq_lvl = torch.where((self.b_owner == oppo) & (self.b_kind == KIND_HQ),
                                self.b_level, torch.zeros_like(self.b_level))
        my_tur = torch.where(self.b_owner == myo, turret, torch.zeros_like(turret))
        op_tur = torch.where(self.b_owner == oppo, turret, torch.zeros_like(turret))
        my_wc = torch.where(self.b_owner == myo, workcap, torch.zeros_like(workcap))
        op_wc = torch.where(self.b_owner == oppo, workcap, torch.zeros_like(workcap))
        my_bhp = torch.where(self.b_owner == myo, self.b_hp, torch.zeros_like(self.b_hp))
        op_bhp = torch.where(self.b_owner == oppo, self.b_hp, torch.zeros_like(self.b_hp))

        # stationary friendly per region & their hp sum (move-able this turn)
        sd = self.slot_side[None, :].expand(B, self.W)
        alive = self.w_hp > 0
        my_alive = alive & (sd == me)
        my_stat = my_alive & (~self.w_move)
        stat_cnt = self._scatter_region(my_stat)
        stat_hp = self._scatter_region(my_stat, self.w_hp)
        my_wc_region = torch.where(self.b_owner == myo, workcap, torch.zeros_like(workcap))
        surplus = stat_cnt - my_wc_region                 # may be negative

        # arrival of my moving warriors at each token, in 1..5 turns
        my_moving = my_alive & self.w_move
        # per-warrior remaining turns to its (token) target
        # map target region -> token index for this game is not needed; we count
        # arrivals at token t = my movers whose target == tok[t] and tt==k
        tt = self.mb.travel_turns                          # [B,N,T]
        arrive = torch.zeros((B, T, 5), dtype=torch.int64, device=dev)
        # build per (region,target-token) mover counts
        # mover target token index: match w_tgt to tok
        # mcount[b, region, ti]
        mcount = torch.zeros((B, N, T), dtype=torch.int64, device=dev)
        # scatter movers by (region, token-of-target)
        # find token index of each mover's target (==tok); -1 if target not a token
        # vectorized match: compare w_tgt to each token id
        wtgt = self.w_tgt                                  # [B,W]
        # token index via searchsorted per game (tok sorted)
        tok_sorted = tok
        # since tok sorted ascending, use searchsorted
        ti_of = torch.searchsorted(tok_sorted, wtgt.clamp(min=0))
        ti_of = ti_of.clamp(max=T - 1)
        is_tok = tok_sorted.gather(1, ti_of) == wtgt
        valid_mv = my_moving & is_tok
        if bool(valid_mv.any()):
            flat = (self.b_ar[:, None] * N + self.w_region) * T + ti_of
            flat = flat[valid_mv]
            mcount.view(-1).scatter_add_(0, flat, torch.ones_like(flat))
        for k in range(1, 6):
            eqk = (tt == k)                                # [B,N,T]
            arrive[:, :, k - 1] = (mcount * eqk).sum(dim=1)

        # enemy reachable within k turns at each token
        op_region_cnt = cnt[:, :, opp]                     # [B,N]
        reach = torch.zeros((B, T, 5), dtype=torch.int64, device=dev)
        for k in range(1, 6):
            lek = (tt <= k)                                # [B,N,T]
            reach[:, :, k - 1] = (op_region_cnt[:, :, None] * lek).sum(dim=1)

        # distances token->token (turns); zero columns for padded target tokens
        tok_dist = tt.gather(1, tok_g[:, :, None].expand(B, T, T))  # [B,T,T]
        tok_dist = tok_dist * tvalid[:, None, :]

        feats = [
            gtok(my_cnt), gtok(op_cnt),
            gtok(my_base_lvl), gtok(op_base_lvl),
            gtok(my_hq_lvl), gtok(op_hq_lvl),
            gtok(my_tur), gtok(op_tur),
            gtok(my_wc), gtok(op_wc),
            gtok(my_bhp), gtok(op_bhp),
            gtok(surplus),
            gtok(stat_hp),
        ]
        feats = torch.stack(feats, dim=2).to(torch.float32)    # [B,T,14]
        # arrive/reach are already indexed by token (dim 1 == T)
        arrive_t = arrive.to(torch.float32)
        reach_t = reach.to(torch.float32)
        tokens = torch.cat([feats, arrive_t, reach_t, tok_dist.to(torch.float32)], dim=2)
        tokens = tokens * tvalid[:, :, None].to(torch.float32)   # zero padded tokens

        # global features
        my_total = (my_alive).sum(dim=1)
        op_total = (alive & (sd == opp)).sum(dim=1)
        my_hq_level = self.b_level.gather(1, self.hq_region[:, me:me + 1]).squeeze(1)
        op_hq_level = self.b_level.gather(1, self.hq_region[:, opp:opp + 1]).squeeze(1)
        lvl_sum_my = torch.where(self.b_owner == myo, self.b_level, torch.zeros_like(self.b_level)).sum(1)
        lvl_sum_op = torch.where(self.b_owner == oppo, self.b_level, torch.zeros_like(self.b_level)).sum(1)
        glob = torch.stack([
            self.day, my_total, op_total, my_hq_level, op_hq_level,
            self.gold[:, me], self.gold[:, opp],
            self.prev_income[:, me], self.prev_income[:, opp],
            lvl_sum_my, lvl_sum_op,
        ], dim=1).to(torch.float32)

        # action helper info (restricted to valid tokens via gather + mask)
        build_cand = (my_cnt > 0) & (op_cnt == 0)                   # [B,N]
        move_src = (stat_cnt > my_wc_region)                        # [B,N]
        info = {
            'gold': self.gold[:, me],
            'hq_level': my_hq_level,
            'token_ids': tok,
            'token_mask': tvalid,                                   # [B,T] bool
            'build_candidates': gtok(build_cand) & tvalid,          # [B,T] bool
            'move_sources': gtok(move_src) & tvalid,                # [B,T] bool
        }
        return tokens, glob, info
