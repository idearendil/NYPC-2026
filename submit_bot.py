#!/usr/bin/env python3
"""Submission bot (numpy-only) — plays the protocol in `sample-code.py`'s I/O
format, choosing actions by running the trained PPO actor net.

Why numpy: the handshake budget is 1000ms, but importing torch alone is ~2.3s.
numpy imports in ~0.15s, so we run inference with numpy and load the actor
weights from a torch-free `data.bin` (an .npz produced offline by `export_weights.py`).

Correctness: the encoder features and the transformer forward are a faithful port
of `ppo_selfplay.extract` / `fast_env.observe` and the network modules; the port
is checked numerically against the torch path by `verify_np_bot.py`.

Key behaviours (identical to training):
  * The agent remembers destinations of its moving warriors (the protocol never
    reveals a mover's target), used for the "my arrivals in 1..5 turns" features.
  * A (source, target) move keeps the `work_cap` (post-build) lowest-HP stationary
    friendly warriors at the source (ties -> smaller suffix) and moves the rest.
  * Region->거점 travel time (in turns) is precomputed once at map load (<1s).
  * `--stochastic` samples ~ policy probs; default is greedy/argmax.

The opponent's gold and income are never sent by the protocol; we reconstruct them
from the visible economy (their builds/upgrades/heals, trains, moves, work income,
upkeep). This is exact except for rare opponent move-cost edge cases.
"""
from __future__ import annotations

import math
import sys

import numpy as np

# NOTE: keep module-top imports a strict subset of basic_bot.py's ({math, sys,
# numpy}). argparse (~14ms cold, pulls in re/gettext) is replaced by manual argv
# parsing, and os (free, already loaded) is imported lazily -- so nothing extra
# competes with numpy's import inside the tight 1s handshake window.

# ---- game constants (mirror testing-tool.py / fast_env.py) ------------------
HQ_HP       = [0, 10, 15, 20, 25, 30]
HQ_TURRET   = [0, 1,  2,  2,  3,  3]
HQ_WCAP     = [0, 1,  2,  3,  4,  5]
HQ_WHP      = [0, 4,  5,  6,  7,  8]
HQ_TRAINCAP = [0, 1,  1,  2,  2,  3]
HQ_UPCOST   = [0, 0,  600, 1200, 2400, 3600]
HQ_MAXLEVEL = 5
HQ_HEAL     = 1000
BASE_HP     = [0, 6, 12, 18]
BASE_TURRET = [0, 1, 1,  2]
BASE_WCAP   = [0, 1, 2,  3]
BASE_COST   = [0, 300, 600, 1000]
BASE_MAXLEVEL = 3
BASE_HEAL   = 500

MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2
START_GOLD = 500
START_WARRIORS = 3

OWN_LEFT, OWN_RIGHT = 1, 2
KIND_HQ, KIND_BASE = 1, 2

TOK_FEAT = 31           # 14 + 5 arrive + 5 reach + 2 coords + 5 reach-delta vs prev turn
GLOB_FEAT = 12          # 11 + HQ-upgrade "turns to afford" (log1p)
T2_EXTRA = 8
COST_INF = 1_000_000_000
BIG = 1 << 20            # unreachable travel marker (matches fast_env)


# --------------------------------------------------------------------------- #
# numpy network primitives (match torch nn modules used in training)
# --------------------------------------------------------------------------- #
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _erf(x):                                   # Abramowitz-Stegun 7.1.26 (~1.5e-7)
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A[4] * t + _A[3]) * t + _A[2]) * t + _A[1]) * t + _A[0]) * t
    return s * (1.0 - poly * np.exp(-ax * ax))


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def linear(x, w, b):                           # w:[out,in], x:[...,in]
    return x @ w.T + b


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


class Net:
    """Holds weights and runs the T1 / T2 / encoder forward passes in numpy."""
    def __init__(self, npz):
        self.W = {k: npz[k] for k in npz.files}
        self.heads = int(npz["meta.heads"])
        self.d_model = int(npz["meta.d_model"])

    def _mha(self, p, x):                       # x:[B,T,d], no padding mask
        W = self.W
        B, T, d = x.shape
        h, dk = self.heads, d // self.heads
        qkv = linear(x, W[p + ".qkv.weight"], W[p + ".qkv.bias"])    # [B,T,3d]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        def split(t):
            return t.reshape(B, T, h, dk).transpose(0, 2, 1, 3)      # [B,h,T,dk]
        q, k, v = split(q), split(k), split(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dk)         # [B,h,T,T]
        att = softmax(att, axis=-1)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        return linear(out, W[p + ".proj.weight"], W[p + ".proj.bias"])

    def _ff(self, p, x):
        W = self.W
        return linear(gelu(linear(x, W[p + ".0.weight"], W[p + ".0.bias"])),
                      W[p + ".2.weight"], W[p + ".2.bias"])

    def _encoder(self, pre, x):                 # x:[B,T,in]
        W = self.W
        h = linear(x, W[pre + ".embed.weight"], W[pre + ".embed.bias"])
        i = 0
        while pre + f".blocks.{i}.ln1.weight" in W:
            bp = pre + f".blocks.{i}"
            h = h + self._mha(bp + ".attn",
                              layernorm(h, W[bp + ".ln1.weight"], W[bp + ".ln1.bias"]))
            h = h + self._ff(bp + ".ff",
                             layernorm(h, W[bp + ".ln2.weight"], W[bp + ".ln2.bias"]))
            i += 1
        return h

    def t1(self, t1, glob):                      # t1:[T,26], glob:[11]
        W = self.W
        T = t1.shape[0]
        x = np.concatenate([t1, np.broadcast_to(glob, (T, GLOB_FEAT))], axis=1)
        h = self._encoder("t1.enc", x[None])     # [1,T,d]
        head = linear(gelu(linear(h, W["t1.head.0.weight"], W["t1.head.0.bias"])),
                      W["t1.head.2.weight"], W["t1.head.2.bias"])
        return h[0], head[0]                      # [T,d], [T,5]

    def t2(self, x):                             # x:[S,T,d_in] -> logits [S,T]
        W = self.W
        h = self._encoder("t2.enc", x)
        head = linear(gelu(linear(h, W["t2.head.0.weight"], W["t2.head.0.bias"])),
                      W["t2.head.2.weight"], W["t2.head.2.bias"])
        return head[..., 0]


def slog1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def plog1p(x):
    return np.log1p(np.maximum(x, 0.0))


# --------------------------------------------------------------------------- #
# travel-time precompute (region -> 거점, in turns); matches fast_env exactly
# --------------------------------------------------------------------------- #
def compute_travel(N, x, y, adj, tok_ids):
    INF = BIG
    T = len(tok_ids)
    w = np.full((N, N), INF, dtype=np.int64)
    adj_mask = np.zeros((N, N), dtype=bool)
    for u in range(N):
        for v in adj[u]:
            dx, dy = x[u] - x[v], y[u] - y[v]
            w[u, v] = math.ceil(math.sqrt(dx * dx + dy * dy))
            adj_mask[u, v] = True
    for i in range(N):
        w[i, i] = 0

    dist = w.copy()
    for k in range(N):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :][None, :])

    nxt = np.full((N, N), -1, dtype=np.int64)
    best = np.full((N, N), INF, dtype=np.int64)
    for nb in range(N):
        score = w[:, nb][:, None] + dist[nb, :][None, :]
        cand = adj_mask[:, nb][:, None] & (score < best)
        best = np.where(cand, score, best)
        nxt = np.where(cand, nb, nxt)
    di = np.arange(N)
    nxt[di, di] = di

    tt = np.full((N, T), INF, dtype=np.int64)
    allr = np.arange(N)
    for ti in range(T):
        tgt = int(tok_ids[ti])
        cur = np.full(N, INF, dtype=np.int64)
        cur[tgt] = 0
        nxt_to = nxt[allr, tgt]
        valid_nb = nxt_to >= 0
        nxt_safe = np.where(valid_nb, nxt_to, 0)
        for _ in range(N):
            cand = cur[nxt_safe] + 1
            cand = np.where(valid_nb, cand, INF)
            upd = np.minimum(cur, cand)
            upd[tgt] = 0
            if np.array_equal(upd, cur):
                break
            cur = upd
        tt[:, ti] = cur
    return tt


# --------------------------------------------------------------------------- #
# protocol I/O
# --------------------------------------------------------------------------- #
def readln():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.rstrip("\n")


def read_tokens():
    return readln().split()


class Warrior:
    __slots__ = ("side", "num", "region", "hp", "moving", "target",
                 "moved_last", "moved_now")

    def __init__(self, side, num, region, hp):
        self.side, self.num, self.region, self.hp = side, num, region, hp
        self.moving = False
        self.target = 0
        self.moved_last = False
        self.moved_now = False


class Building:
    __slots__ = ("region", "side", "kind", "level", "hp")

    def __init__(self, region, side, kind, level, hp):
        self.region, self.side, self.kind = region, side, kind
        self.level, self.hp = level, hp


def _maxhp(kind, level):
    return HQ_HP[level] if kind == 'HQ' else BASE_HP[level]


def _maxlevel(kind):
    return HQ_MAXLEVEL if kind == 'HQ' else BASE_MAXLEVEL


def _upgrade_cost(kind, level):
    return HQ_UPCOST[level + 1] if kind == 'HQ' else BASE_COST[level + 1]


def _heal_cost(kind):
    return HQ_HEAL if kind == 'HQ' else BASE_HEAL


def _turret(kind, level):
    return HQ_TURRET[level] if kind == 'HQ' else BASE_TURRET[level]


def _workcap(kind, level):
    return HQ_WCAP[level] if kind == 'HQ' else BASE_WCAP[level]


class Bot:
    def __init__(self, weights_path, stochastic=False):
        import os  # already loaded at startup; lazy here to keep module-top imports minimal
        self.debug = bool(os.environ.get("BOT_DEBUG"))
        self.stochastic = stochastic
        self.weights_path = weights_path
        self.net = None                  # loaded lazily, after the handshake (see _ensure_ready)
        self.rng = None                  # numpy.random pulled in lazily too (handshake hygiene)
        self.hq_commit = False           # saving-for-HQ-upgrade macro (see _select_action)
        self.prev_reach = None           # last turn's raw enemy-reachability [T,5] for the delta feature

        self.my_side = 'A'
        self.me_code = OWN_LEFT
        self.opp = 'B'
        self.N = 0
        self.x = self.y = self.adj = None
        self.tok_ids = None
        self.stronghold = None      # bool[N]
        self.is_hq = None           # bool[N]
        self.tt = None              # [N,T] travel turns
        self.tok2idx = {}           # region -> token index

        self.warriors = {}
        self.buildings = {}
        self.gold = {'A': START_GOLD, 'B': START_GOLD}
        self.income = {'A': 0, 'B': 0}
        self.last_actions = None

    # ------------------------------------------------------------------ init
    def parse_init(self):
        t = read_tokens()
        assert t and t[0] == "READY"
        self.my_side = 'A' if t[1] == "LEFT" else 'B'
        self.me_code = OWN_LEFT if self.my_side == 'A' else OWN_RIGHT
        self.opp = 'B' if self.my_side == 'A' else 'A'

        t = read_tokens()
        N, K = int(t[0]), int(t[1])
        self.N, self.K = N, K
        self.x = [int(v) for v in read_tokens()]
        self.y = [int(v) for v in read_tokens()]
        strongholds = sorted(int(v) for v in read_tokens())
        self.adj = [[] for _ in range(N)]
        for r in range(N):
            tok = read_tokens()
            deg = int(tok[0])
            self.adj[r] = sorted(int(v) for v in tok[1:1 + deg])

        # Keep the handshake path pure-Python (no numpy ops). All numpy array
        # construction is deferred to _ensure_ready so the only thing between
        # receiving the map and printing OK is `import numpy` itself -- matching
        # the basic_bot profile that is known to clear the 1s handshake.
        self._strongholds = strongholds
        self.stronghold = self.is_hq = self.tok_ids = None
        self.tok2idx = {}
        self.tt = None       # travel times precomputed lazily after OK (see _ensure_ready)

        for sfx in range(1, START_WARRIORS + 1):
            self.warriors[('A', sfx)] = Warrior('A', sfx, 0, HQ_WHP[1])
            self.warriors[('B', sfx)] = Warrior('B', sfx, N - 1, HQ_WHP[1])
        self.buildings[0] = Building(0, 'A', 'HQ', 1, HQ_HP[1])
        self.buildings[N - 1] = Building(N - 1, 'B', 'HQ', 1, HQ_HP[1])

        print("OK", flush=True)

    # --------------------------------------------------- encoder (numpy port)
    def encode(self, turn):
        N = self.N
        tok = self.tok_ids
        T = len(tok)
        me, opp = self.my_side, self.opp
        me_code = self.me_code
        opp_code = OWN_RIGHT if me_code == OWN_LEFT else OWN_LEFT

        # per-region warrior aggregates
        cnt_me = np.zeros(N); cnt_op = np.zeros(N)
        hp_me = np.zeros(N); hp_op = np.zeros(N)
        stat_cnt_r = np.zeros(N); stat_hp_r = np.zeros(N)
        for w in self.warriors.values():
            if w.hp <= 0:
                continue
            if w.side == me:
                cnt_me[w.region] += 1; hp_me[w.region] += w.hp
                if not w.moving:
                    stat_cnt_r[w.region] += 1; stat_hp_r[w.region] += w.hp
            else:
                cnt_op[w.region] += 1; hp_op[w.region] += w.hp

        # per-region building aggregates
        bowner = np.zeros(N, dtype=np.int64)
        bkind = np.zeros(N, dtype=np.int64)
        blevel = np.zeros(N, dtype=np.int64)
        bhp = np.zeros(N); bturret = np.zeros(N); bwcap = np.zeros(N)
        for b in self.buildings.values():
            r = b.region
            bowner[r] = OWN_LEFT if b.side == 'A' else OWN_RIGHT
            bkind[r] = KIND_HQ if b.kind == 'HQ' else KIND_BASE
            blevel[r] = b.level
            bhp[r] = b.hp
            bturret[r] = _turret(b.kind, b.level)
            bwcap[r] = _workcap(b.kind, b.level)

        # gather region arrays onto token regions
        g = tok
        own_t = bowner[g]; kind_t = bkind[g]; lvl_t = blevel[g]
        bhp_t = bhp[g]; tur_t = bturret[g]; wc_t = bwcap[g]
        me_b = own_t == me_code
        opp_b = (own_t != 0) & (own_t != me_code)

        my_cnt = cnt_me[g]; op_cnt = cnt_op[g]
        my_hps = hp_me[g]; op_hps = hp_op[g]
        my_base_lvl = np.where(me_b & (kind_t == KIND_BASE), lvl_t, 0)
        op_base_lvl = np.where(opp_b & (kind_t == KIND_BASE), lvl_t, 0)
        my_hq_lvl = np.where(me_b & (kind_t == KIND_HQ), lvl_t, 0)
        op_hq_lvl = np.where(opp_b & (kind_t == KIND_HQ), lvl_t, 0)
        my_tur = np.where(me_b, tur_t, 0); op_tur = np.where(opp_b, tur_t, 0)
        my_wc = np.where(me_b, wc_t, 0); op_wc = np.where(opp_b, wc_t, 0)
        my_bhp = np.where(me_b, bhp_t, 0); op_bhp = np.where(opp_b, bhp_t, 0)
        stat_cnt = stat_cnt_r[g]; stat_hp = stat_hp_r[g]
        surplus = stat_cnt - my_wc          # may be negative (matches observe)

        # arrivals of my movers at each token in exactly 1..5 turns
        arrive = np.zeros((T, 5))
        for w in self.warriors.values():
            if w.side != me or w.hp <= 0 or not w.moving:
                continue
            ti = self.tok2idx.get(int(w.target))
            if ti is None:
                continue
            k = int(self.tt[w.region, ti])
            if 1 <= k <= 5:
                arrive[ti, k - 1] += 1
        # enemy reachable within 1..5 turns at each token
        reach = np.zeros((T, 5))
        for r in range(N):
            if cnt_op[r] == 0:
                continue
            d = self.tt[r, :]               # [T]
            for k in range(1, 6):
                reach[:, k - 1] += cnt_op[r] * (d <= k)

        tok_dist = self.tt[g, :].astype(np.float64)    # [T,T]

        feats = np.stack([my_cnt, op_cnt, my_base_lvl, op_base_lvl,
                          my_hq_lvl, op_hq_lvl, my_tur, op_tur, my_wc, op_wc,
                          my_bhp, op_bhp, surplus, stat_hp], axis=1)   # [T,14]
        raw = np.concatenate([feats, arrive, reach], axis=1)          # [T,24]
        raw24 = slog1p(raw)

        # normalized coordinates -> [-10, 10]
        tx = np.array([self.x[int(r)] for r in g], dtype=np.float64)
        ty = np.array([self.y[int(r)] for r in g], dtype=np.float64)
        xmn, xmx = tx.min(), tx.max()
        ymn, ymx = ty.min(), ty.max()
        normx = (tx - xmn) / max(xmx - xmn, 1) * 20 - 10
        normy = (ty - ymn) / max(ymx - ymn, 1) * 20 - 10
        # If assigned RIGHT, feed the canonical (LEFT) orientation: the board is
        # point-symmetric, so reflecting the state is just negating normalized
        # coords (matches ppo_selfplay.extract's side==1 handling).
        if self.my_side == 'B':
            normx = -normx
            normy = -normy
        # per-turn CHANGE in enemy reachability (this turn's raw reach minus last
        # turn's, per token per horizon); on turn 1 prev is 0 (matches training's
        # episode-reset baseline). Orientation-invariant (per-token counts).
        prev = getattr(self, 'prev_reach', None)     # getattr: some harnesses skip __init__
        if prev is None:
            prev = np.zeros((T, 5))
        reach_delta = slog1p(reach - prev)                            # [T,5]
        self.prev_reach = reach.copy()
        t1 = np.concatenate([raw24, normx[:, None], normy[:, None], reach_delta],
                            axis=1)  # [T,31]

        # global features
        my_total = sum(1 for w in self.warriors.values() if w.side == me and w.hp > 0)
        op_total = sum(1 for w in self.warriors.values() if w.side == opp and w.hp > 0)
        my_hq_level = self._hq_level(me)
        op_hq_level = self._hq_level(opp)
        lvl_sum_me = sum(b.level for b in self.buildings.values() if b.side == me)
        lvl_sum_op = sum(b.level for b in self.buildings.values() if b.side == opp)
        day = turn - 1
        glob = np.array([
            day / 10 - 10,
            plog1p(my_total / 10), plog1p(op_total / 10),
            plog1p(my_hq_level), plog1p(op_hq_level),
            plog1p(self.gold[me] / 100), plog1p(self.gold[opp] / 100),
            plog1p(self.income[me] / 10), plog1p(self.income[opp] / 10),
            plog1p(lvl_sum_me / 5), plog1p(lvl_sum_op / 5),
            self._hq_turns_feature(my_hq_level, my_total),
        ])

        # action masking quantities
        maxlev = np.where(kind_t == KIND_HQ, HQ_MAXLEVEL, BASE_MAXLEVEL)
        up_cost = np.where(kind_t == KIND_HQ,
                           np.take(HQ_UPCOST, np.minimum(lvl_t + 1, HQ_MAXLEVEL)),
                           np.take(BASE_COST, np.minimum(lvl_t + 1, BASE_MAXLEVEL)))
        heal_cost = np.where(kind_t == KIND_HQ, HQ_HEAL, BASE_HEAL)
        is_strong = self.stronghold[g]
        is_hq = self.is_hq[g]
        # Upgrade legality mirrors the judge (testing-tool.apply_upgrades): a region is
        # only upgradeable when a friendly warrior is present AND no enemy warrior is on
        # it. up_room = "has room to upgrade" (ignores transient occupancy) is used by
        # the HQ-saving commitment so it can keep saving while an enemy sits on the HQ.
        up_room = me_b & (lvl_t < maxlev)
        can_up = up_room & (my_cnt > 0) & (op_cnt == 0)
        can_heal = me_b & (lvl_t >= maxlev) & (my_cnt > 0) & (op_cnt == 0)
        build_new = (own_t == 0) & is_strong & (~is_hq)
        cost = np.full(T, COST_INF, dtype=np.float64)
        cost = np.where(build_new, BASE_COST[1], cost)
        cost = np.where(can_up, up_cost, cost)
        cost = np.where(can_heal, heal_cost, cost)
        wc_up = np.where(kind_t == KIND_HQ,
                         np.take(HQ_WCAP, np.minimum(lvl_t + 1, HQ_MAXLEVEL)),
                         np.take(BASE_WCAP, np.minimum(lvl_t + 1, BASE_MAXLEVEL)))
        wc_after = my_wc.copy()
        wc_after = np.where(can_up, wc_up, wc_after)
        wc_after = np.where(build_new, BASE_WCAP[1], wc_after)

        e1 = plog1p(op_cnt + op_tur)
        e2 = plog1p((op_hps + op_bhp) / 5)
        e3 = plog1p(my_cnt + my_tur)
        e4 = plog1p((my_hps + my_bhp) / 5)
        extra4 = np.stack([e1, e2, e3, e4], axis=1)

        build_cand = (my_cnt > 0) & (op_cnt == 0)

        return dict(t1=t1, glob=glob, tok_ids=g, gold=self.gold[me],
                    hq_level=my_hq_level, build_cand=build_cand, build_cost=cost,
                    wc_cur=my_wc.astype(np.int64), wc_after=wc_after.astype(np.int64),
                    stat_cnt=stat_cnt.astype(np.int64),
                    is_hq_me=(is_hq & me_b), can_up_hq=(is_hq & can_up),
                    can_up_hq_room=(is_hq & up_room),
                    owner_me=me_b, build_new=build_new, nu=(build_new | can_up),
                    extra4=extra4, tok_dist=tok_dist, normx=normx, normy=normy)

    def _hq_level(self, side):
        r = 0 if side == 'A' else self.N - 1
        b = self.buildings.get(r)
        return b.level if (b is not None and b.kind == 'HQ' and b.side == side) else 0

    def _hq_turns_feature(self, my_hq_level, my_total):
        """log1p of 'turns to afford the next HQ upgrade by saving' = max(0, cost-gold)
        / net_income (last building income - 2*warriors). Denominator floored at 1 so
        net<=0 reads as 'very far'. 0 when the HQ is already max level."""
        if my_hq_level >= HQ_MAXLEVEL:
            return 0.0
        cost = HQ_UPCOST[my_hq_level + 1]
        net = self.income[self.my_side] - 2 * my_total
        need = max(0, cost - self.gold[self.my_side])
        return float(plog1p(need / max(net, 1)))

    # -------------------------------------------------------- action selection
    def _ensure_ready(self):
        """Heavy init deferred out of the 1s handshake into turn 1's budget: load
        the actor weights and precompute region->거점 travel times. Both are only
        needed once we actually choose an action, so keeping them out of the
        handshake gives the unavoidable numpy import the full 1s OK budget."""
        if self.net is None:
            N = self.N
            self.stronghold = np.zeros(N, dtype=bool)
            for r in self._strongholds:
                self.stronghold[r] = True
            self.is_hq = np.zeros(N, dtype=bool)
            self.is_hq[0] = True
            self.is_hq[N - 1] = True
            self.tok_ids = np.array(sorted(set(self._strongholds) | {0, N - 1}), dtype=np.int64)
            self.tok2idx = {int(r): i for i, r in enumerate(self.tok_ids)}
            self.net = Net(np.load(self.weights_path))
            self.tt = compute_travel(N, self.x, self.y, self.adj, self.tok_ids)
            self.rng = np.random.default_rng()

    def decide(self, turn):
        import time as _t
        t0 = _t.perf_counter()
        self._ensure_ready()
        o = self.encode(turn)
        plan = self._select_action(o)
        cmds = self._to_commands(plan, o)
        if self.debug:
            print(f"turn {turn}: decide {(_t.perf_counter()-t0)*1000:.1f}ms",
                  file=sys.stderr, flush=True)
        return cmds

    def _greedy(self, item, cost, gold, order):
        remaining = int(gold)
        ex = np.zeros(len(item), dtype=bool)
        for k in order:
            if item[k] and cost[k] <= remaining:
                remaining -= int(cost[k]); ex[k] = True
        return ex, remaining

    def _select_action(self, o):
        T = len(o['tok_ids'])
        sto = self.stochastic
        h1, head5 = self.net.t1(o['t1'], o['glob'])

        # Non-moving friendly warriors board-wide (caps builds), and of those the
        # "free" ones not currently labouring -- surplus beyond each region's
        # work_cap -- which can actually be dispatched to staff a base (gates builds).
        by_region = {}
        for w in self.warriors.values():
            if w.side == self.my_side and w.hp > 0 and not w.moving:
                by_region.setdefault(w.region, []).append(w)
        n_nonmoving = sum(len(ws) for ws in by_region.values())
        n_free = 0
        for r, ws in by_region.items():
            b = self.buildings.get(r)
            kc = _workcap(b.kind, b.level) if (b is not None and b.side == self.my_side) else 0
            n_free += max(len(ws) - kc, 0)

        # ---------------- BUILD (with level-split HQ-upgrade macro) ----------------
        committed = self.hq_commit
        my_hq_region = 0 if self.my_side == 'A' else self.N - 1
        hq_tok = self.tok2idx.get(my_hq_region)
        # hq_room: HQ has room to upgrade (ignores occupancy). hq_legal: ALSO judge-legal
        # (friendly present, no enemy). Split by target level:
        #  * level < 3 (reach 2 or 3): base-like -- sampleable only when affordable+legal
        #    this turn, no commitment -- but still exempt from the free-worker gate.
        #  * level >= 3 (reach 4 or 5): keep the save-commit macro (sampleable when
        #    unaffordable / enemy on HQ; deferred emission).
        hq_room = (hq_tok is not None) and bool(o['can_up_hq_room'][hq_tok])
        hq_legal = (hq_tok is not None) and bool(o['can_up_hq'][hq_tok])
        hq_cost = HQ_UPCOST[o['hq_level'] + 1] if hq_room else COST_INF
        hq_afford = hq_room and (o['gold'] >= hq_cost)
        hq_macro = hq_room and (o['hq_level'] >= 3)         # level 3->4 / 4->5: macro
        hq_normal = hq_room and (o['hq_level'] < 3)         # level 1->2 / 2->3: base-like
        hq_normal_ok = hq_normal and hq_legal and (o['gold'] >= hq_cost)

        p_build = 1.0 / (1.0 + np.exp(-head5[:, 0]))
        # normal builds need a free worker (gating). The HQ token is overridden per the
        # level rules (worker-gate exempt); macro HQ stays sampleable when unaffordable
        # (to commit to saving). No builds at all while committed.
        normal_build_mask = o['build_cand'] & (o['build_cost'] <= o['gold']) & (n_free >= 1)
        if committed:
            build_mask = np.zeros(T, dtype=bool)
        else:
            build_mask = normal_build_mask.copy()
            if hq_tok is not None:
                build_mask[hq_tok] = hq_macro or hq_normal_ok
        if sto:
            outcome = (self.rng.random(T) < (p_build * build_mask)).astype(bool)
        else:
            outcome = (p_build > 0.5) & build_mask

        hq_sampled = hq_room and bool(outcome[hq_tok])
        # macro path: intend (committed or sampled) + affordable, emit only if legal;
        # otherwise keep/enter saving mode (defer while unaffordable or enemy on HQ).
        sampled_macro = hq_sampled and hq_macro
        do_hq_macro = hq_afford and (committed or sampled_macro) and hq_legal
        self.hq_commit = (committed or sampled_macro) and hq_macro and (not do_hq_macro)
        # normal path: only sampleable when already affordable + legal, so just execute.
        do_hq_normal = hq_sampled and hq_normal
        do_hq_now = do_hq_macro or do_hq_normal

        # HQ upgrade takes gold priority; greedily allocate the rest to non-HQ builds.
        gold_after_hq = int(o['gold']) - (int(hq_cost) if do_hq_now else 0)
        non_hq_outcome = outcome.copy()
        if hq_tok is not None:
            non_hq_outcome[hq_tok] = False
        prio = np.where(build_mask, p_build, -1.0)
        order = np.argsort(-prio, kind='stable')
        exec_build, _ = self._greedy(non_hq_outcome, o['build_cost'], gold_after_hq, order)

        # cap new-build/upgrade actions (heal exempt) to n_nonmoving (HQ excluded)
        nu_exec = exec_build & o['nu']
        if int(nu_exec.sum()) > n_nonmoving:
            idx = np.nonzero(nu_exec)[0]
            keep_idx = self.rng.choice(idx, size=n_nonmoving, replace=False)
            capped = np.zeros(T, dtype=bool)
            capped[keep_idx] = True
            exec_build = (exec_build & ~o['nu']) | capped
        gold1 = gold_after_hq - int(np.where(exec_build, o['build_cost'], 0.0).sum())

        # add the committed/forced HQ upgrade (bypasses our gating) for emission + staffing
        if do_hq_now and hq_tok is not None:
            exec_build[hq_tok] = True

        # plan force-staffing: one nearest surplus warrior per understaffed base
        force_moves, force_ids = self._plan_force_staff(o, exec_build)

        wc_pb = np.where(exec_build, o['wc_after'], o['wc_cur'])
        surplus_pb = np.maximum(o['stat_cnt'] - wc_pb, 0)
        owner_me_pb = o['owner_me'] | (exec_build & o['build_new'])
        hq_after = min(o['hq_level'] + (1 if do_hq_now else 0), HQ_MAXLEVEL)

        # ---------------- MOVE (T2) ----------------
        # while committed: only free moves -> restrict targets to our own buildings
        valid_src = (surplus_pb > 0) & (MOVE_COST * surplus_pb <= gold1)
        tgt_allowed = owner_me_pb if committed else np.ones(T, dtype=bool)
        tgt = np.arange(T)
        src_list = np.nonzero(valid_src)[0]
        if src_list.size > 0:
            X = np.empty((src_list.size, T, h1.shape[1] + T2_EXTRA), dtype=np.float32)
            for j, si in enumerate(src_list):
                sf = np.full((T, 1), plog1p(surplus_pb[si]))
                tv = plog1p(o['tok_dist'][si, :])[:, None]
                dx = (o['normx'] - o['normx'][si])[:, None]
                dy = (o['normy'] - o['normy'][si])[:, None]
                X[j] = np.concatenate([h1, o['extra4'], sf, tv, dx, dy], axis=1)
            logits = self.net.t2(X)                      # [S,T]
            if committed:
                logits = np.where(tgt_allowed[None, :], logits, -1e9)
            for j, si in enumerate(src_list):
                if sto:
                    p = softmax(logits[j])
                    tgt[si] = self.rng.choice(T, p=p)
                else:
                    tgt[si] = int(np.argmax(logits[j]))
            chosen = logits[np.arange(src_list.size), tgt[src_list]]
        tgt_is_self = tgt == np.arange(T)
        tgt_mine = owner_me_pb[tgt]
        move_cost = np.where(tgt_mine, 0, MOVE_COST * surplus_pb)
        move_item = valid_src & (~tgt_is_self)
        prio2 = np.full(T, -1e30)
        if src_list.size > 0:
            prio2[src_list] = np.where(move_item[src_list], chosen, -1e30)
        order2 = np.argsort(-prio2, kind='stable')
        exec_move, gold2 = self._greedy(move_item, move_cost, gold1, order2)

        # ---------------- TRAIN ----------------
        tl = head5[:, 1:5].mean(axis=0)                  # [4]
        cap = HQ_TRAINCAP[hq_after]
        cats = np.arange(4)
        tmask = (cats <= cap) & (cats * TRAIN_COST <= gold2)
        if committed:                                    # no training while saving
            tmask = cats == 0
        tl_m = np.where(tmask, tl, -1e9)
        if sto:
            train_cat = int(self.rng.choice(4, p=softmax(tl_m)))
        else:
            train_cat = int(np.argmax(tl_m))

        return exec_build, exec_move, tgt, wc_pb, train_cat, force_moves, force_ids

    def _plan_force_staff(self, o, exec_build):
        """For each new/upgraded base whose non-moving friendly count is below its
        post-build work_cap, pick the nearest *surplus* friendly warrior to send
        there. Returns (list of (warrior, target_region), set of force-moved ids).
        Surplus = stationary friendly warriors beyond a region's post-build keep-cap
        (work_cap where we own a building, else 0) -- the same pool the policy moves,
        so force_ids are excluded from policy moves in _to_commands."""
        tok = o['tok_ids']
        nu, stat_cnt = o['nu'], o['stat_cnt']
        wc_after, wc_cur = o['wc_after'], o['wc_cur']
        is_hq_me = o['is_hq_me']
        # exclude the HQ: it is always staffed at home, so HQ upgrades never trigger
        # a force-staffing move.
        needing = [ti for ti in range(len(tok))
                   if exec_build[ti] and nu[ti] and stat_cnt[ti] < wc_after[ti]
                   and not is_hq_me[ti]]
        if not needing:
            return [], set()
        # post-build keep-cap per 거점 region (non-거점 regions keep 0)
        keepcap = {int(tok[ti]): (int(wc_after[ti]) if exec_build[ti] else int(wc_cur[ti]))
                   for ti in range(len(tok))}
        by_region = {}
        for w in self.warriors.values():
            if w.side == self.my_side and w.hp > 0 and not w.moving:
                by_region.setdefault(w.region, []).append(w)
        surplus = []
        for r, ws in by_region.items():
            ws.sort(key=lambda w: (w.hp, w.num))
            surplus.extend(ws[keepcap.get(r, 0):])
        force_moves, assigned = [], set()
        for ti in needing:
            best, bestkey = None, None
            for w in surplus:
                if id(w) in assigned:
                    continue
                key = (int(self.tt[w.region, ti]), w.region, w.hp, w.num)
                if bestkey is None or key < bestkey:
                    bestkey, best = key, w
            if best is not None:
                assigned.add(id(best))
                force_moves.append((best, int(tok[ti])))
        return force_moves, assigned

    def _to_commands(self, plan, o):
        exec_build, exec_move, tgt, wc_pb, train_cat, force_moves, force_ids = plan
        tok = o['tok_ids']
        upgrades, moves = [], []
        for t in np.nonzero(exec_build)[0]:
            upgrades.append(int(tok[t]))
        for w, treg in force_moves:          # force-staffing (nearest surplus warrior)
            moves.append((w, treg))
        for s in np.nonzero(exec_move)[0]:
            sreg = int(tok[s]); treg = int(tok[int(tgt[s])]); keep = int(wc_pb[s])
            here = [w for w in self.warriors.values()
                    if w.side == self.my_side and w.hp > 0
                    and not w.moving and w.region == sreg and id(w) not in force_ids]
            here.sort(key=lambda w: (w.hp, w.num))
            for w in here[keep:]:
                moves.append((w, treg))
        return upgrades, moves, int(train_cat)

    def emit(self, commands):
        upgrades, moves, train_n = commands
        out = ["COMMAND"]
        for w, treg in moves:
            out.append(f"MOVE {w.side}{w.num} {treg}")
        for r in upgrades:
            out.append(f"UPGRADE {r}")
        if train_n > 0:
            out.append(f"TRAIN {train_n}")
        out.append("END")
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        self.last_actions = commands

    # ----------------------------------------------------- result processing
    def _apply_my_actions(self):
        upgrades, moves, train_n = self.last_actions
        for region in upgrades:
            b = self.buildings.get(region)
            if b is None:
                self.gold[self.my_side] -= BASE_COST[1]
                self.buildings[region] = Building(region, self.my_side, 'BASE', 1, BASE_HP[1])
            elif b.level >= _maxlevel(b.kind):
                self.gold[self.my_side] -= _heal_cost(b.kind)
                b.hp = _maxhp(b.kind, b.level)
            else:
                self.gold[self.my_side] -= _upgrade_cost(b.kind, b.level)
                b.level += 1
                b.hp = _maxhp(b.kind, b.level)
        for w, treg in moves:
            b = self.buildings.get(treg)
            cost = 0 if (b is not None and b.side == self.my_side) else MOVE_COST
            self.gold[self.my_side] -= cost
            w.moving = True
            w.target = treg
        self.gold[self.my_side] -= TRAIN_COST * train_n

    def _settle_economy(self):
        for s in ('A', 'B'):
            inc = 0
            for b in self.buildings.values():
                if b.side != s:
                    continue
                cnt = sum(1 for w in self.warriors.values()
                          if w.side == s and w.region == b.region and w.hp > 0)
                inc += WORK_INCOME * min(cnt, _workcap(b.kind, b.level))
            self.gold[s] += inc
            self.income[s] = inc
            alive = sum(1 for w in self.warriors.values() if w.side == s and w.hp > 0)
            n_fed = min(alive, self.gold[s] // 2)
            self.gold[s] -= UPKEEP_PER_WARRIOR * n_fed

    def read_turn_result(self):
        self._apply_my_actions()
        opp = self.opp

        line = readln()
        if line == "FINISH":
            sys.exit(0)
        assert line.split()[0] == "TURN"
        read_tokens()                                   # TIME ...

        n = int(read_tokens()[1])                       # UPGRADE
        for _ in range(n):
            r = read_tokens()
            s = 'A' if r[0][0] == 'A' else 'B'
            region = int(r[1])
            b = self.buildings.get(region)
            if b is None:
                if s == opp:
                    self.gold[opp] -= BASE_COST[1]
                self.buildings[region] = Building(region, s, 'BASE', 1, BASE_HP[1])
            elif s == self.my_side:
                pass
            else:
                if b.level >= _maxlevel(b.kind):
                    self.gold[opp] -= _heal_cost(b.kind)
                    b.hp = _maxhp(b.kind, b.level)
                else:
                    self.gold[opp] -= _upgrade_cost(b.kind, b.level)
                    b.level += 1
                    b.hp = _maxhp(b.kind, b.level)

        n = int(read_tokens()[1])                       # TRAIN
        if n > 0:
            ids = read_tokens()
            opp_trained = 0
            for tok in ids:
                s = 'A' if tok[0] == 'A' else 'B'
                num = int(tok[1:])
                hq_region = 0 if s == 'A' else self.N - 1
                hq_b = self.buildings.get(hq_region)
                lvl = hq_b.level if hq_b is not None else 1
                self.warriors[(s, num)] = Warrior(s, num, hq_region, HQ_WHP[lvl])
                if s == opp:
                    opp_trained += 1
            self.gold[opp] -= TRAIN_COST * opp_trained

        for w in self.warriors.values():                # MOVE
            w.moved_now = False
        n = int(read_tokens()[1])
        for _ in range(n):
            r = read_tokens()
            s = 'A' if r[0][0] == 'A' else 'B'
            num = int(r[0][1:])
            region = int(r[1])
            w = self.warriors.get((s, num))
            if w is not None:
                w.moved_now = True
                w.region = region
                if s == self.my_side and w.moving and w.region == w.target:
                    w.moving = False
        opp_new = sum(1 for w in self.warriors.values()
                      if w.side == opp and w.moved_now and not w.moved_last)
        self.gold[opp] -= MOVE_COST * opp_new
        for w in self.warriors.values():
            w.moved_last = w.moved_now

        n = int(read_tokens()[1])                       # DAMAGE
        for _ in range(n):
            r = read_tokens()
            s = 'A' if r[1][0] == 'A' else 'B'
            num = int(r[1][1:])
            w = self.warriors.get((s, num))
            if w is not None:
                w.hp -= int(r[2])
        self.warriors = {k: w for k, w in self.warriors.items() if w.hp > 0}

        n = int(read_tokens()[1])                       # SIEGE
        for _ in range(n):
            r = read_tokens()
            region = int(r[1])
            b = self.buildings.get(region)
            if b is not None:
                b.hp -= int(r[2])
        self.buildings = {k: b for k, b in self.buildings.items() if b.hp > 0}

        readln()                                        # END
        self._settle_economy()

    def run(self):
        self.parse_init()
        while True:
            line = readln()
            if line == "FINISH":
                return
            t = line.split()
            assert t and t[0] == "START"
            turn = int(t[2])
            self.emit(self.decide(turn))
            self.read_turn_result()


def main():
    # Manual argv parse (no argparse) -- supports `--weights X`, `--weights=X`,
    # and `--stochastic`; defaults match the submission (data.bin, argmax).
    weights, stochastic = "data.bin", True
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--stochastic":
            stochastic = True
        elif a == "--weights":
            i += 1
            if i < len(args):
                weights = args[i]
        elif a.startswith("--weights="):
            weights = a.split("=", 1)[1]
        i += 1
    Bot(weights, stochastic=stochastic).run()


if __name__ == "__main__":
    main()
