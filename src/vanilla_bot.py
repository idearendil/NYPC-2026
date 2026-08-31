#!/usr/bin/env python3
"""Submission bot (numpy-only): the trained actor played straight, no search.

Action selection: ONE actor-net inference per turn, then emit a single action
SAMPLED (stochastically) from the resulting policy probabilities -- the raw policy,
played as-is. (`--greedy` switches to argmax.) `decide` -> `_decide_single` ->
encode -> sample -> emit.

FOG OF WAR: the protocol only ever reports THIS side's own build/train/move/damage
events; the opponent is known only through the WARRIOR/BUILDING snapshot sent every
turn (own units always included, enemy units/buildings only where currently visible
-- within 2 hops of one of our own warriors or buildings). This bot keeps a
per-region BELIEF of the opponent (kind/level/hp/warrior-count/warrior-hp, plus an
age = turns since last seen), refreshed from that snapshot inside the locally
computed vision set and otherwise aged -- mirroring fast_env.FastEnv._update_fog
exactly, since that is what the training-side observation is built from. There is
no opponent-gold reconstruction any more: the training encoder never sees the
opponent's gold in any form (see fast_env.observe / ppo_selfplay.extract).
"""
from __future__ import annotations

import math
import sys

import numpy as np

# NOTE: keep module-top imports a strict subset of basic_bot.py's ({math, sys,
# numpy}). argparse (~14ms cold, pulls in re/gettext) is replaced by manual argv
# parsing, and os (free, already loaded) is imported lazily -- so nothing extra
# competes with numpy's import inside the tight 1s handshake window.

# ---- game constants (mirror testing-tool2.py / fast_env.py) -----------------
HQ_HP       = [0, 10, 15, 20, 25, 30]
HQ_TURRET   = [0, 1,  2,  2,  3,  3]
HQ_WCAP     = [0, 1,  2,  3,  4,  5]
HQ_WHP      = [0, 4,  5,  6,  7,  8]
HQ_TRAINCAP = [0, 1,  1,  2,  2,  3]
HQ_UPCOST   = [0, 0,  600, 1000, 2000, 3000]
HQ_MAXLEVEL = 5
HQ_HEAL     = 1000
BASE_HP     = [0, 6, 12, 18]
BASE_TURRET = [0, 1, 1,  2]
BASE_WCAP   = [0, 1, 2,  3]
BASE_COST   = [0, 500, 550, 600]
BASE_MAXLEVEL = 3
BASE_HEAL   = 500

MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2
START_GOLD = 750
START_WARRIORS = 3
MAX_DAYS = 400          # game ends after day 400 (testing-tool2.py); timeout decided by HQ hp
HOP_VISION = 2          # fog of war: visible = within this many (unweighted) hops

OWN_LEFT, OWN_RIGHT = 1, 2
KIND_HQ, KIND_BASE = 1, 2

TOK_FEAT = 32            # 15 scalars (incl. fog-of-war age) + 5 arrive + 5 reach + 2 coords + 5 reach-delta
GLOB_FEAT = 14           # 10 + HQ-turns + 거점-count + x/y map-span (no opponent-gold feature)
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
    """Holds weights and runs the T1 / T2 / encoder forward passes in numpy.
    Actor only -- the critic is not exported (never used at inference; no search)."""
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

    def t1(self, t1, glob):                      # t1:[T,TOK_FEAT], glob:[GLOB_FEAT]
        W = self.W
        T = t1.shape[0]
        x = np.concatenate([t1, np.broadcast_to(glob, (T, GLOB_FEAT))], axis=1)
        h = self._encoder("t1.enc", x[None])     # [1,T,d]
        head = linear(gelu(linear(h, W["t1.head.0.weight"], W["t1.head.0.bias"])),
                      W["t1.head.2.weight"], W["t1.head.2.bias"])
        return h[0], head[0]                      # [T,d], [T,5]

    def t2(self, x):                             # x:[S,T,d_in] -> [S,T,1]
        """Per-source target head: move-target logit (softmax over tokens -> which
        거점 to send the source's surplus to). No full-mobilisation output -- a
        source only ever sends its surplus (stationary beyond work_cap)."""
        W = self.W
        h = self._encoder("t2.enc", x)
        head = linear(gelu(linear(h, W["t2.head.0.weight"], W["t2.head.0.bias"])),
                      W["t2.head.2.weight"], W["t2.head.2.bias"])
        return head

    def t1_batch(self, t1s, globs):              # t1s:[K,T,TOK_FEAT], globs:[K,GLOB] -> ([K,T,d],[K,T,5])
        """Batched T1 over K states (fixed T). Same math as t1(), one encoder pass."""
        W = self.W
        K, T, _ = t1s.shape
        x = np.concatenate([t1s, np.broadcast_to(globs[:, None, :], (K, T, GLOB_FEAT))], axis=2)
        h = self._encoder("t1.enc", x)               # [K,T,d]
        head = linear(gelu(linear(h, W["t1.head.0.weight"], W["t1.head.0.bias"])),
                      W["t1.head.2.weight"], W["t1.head.2.bias"])   # [K,T,5]
        return h, head


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


def hop_set(centers, adj, hops=HOP_VISION):
    """BFS union of all regions within `hops` (unweighted) steps of any region in
    `centers`. Mirrors testing-tool2._hop_set / fast_env's hop2_reach exactly --
    vision ignores the euclidean edge weight used for movement."""
    seen = set(centers)
    frontier = list(seen)
    for _ in range(hops):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return seen


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
    __slots__ = ("side", "num", "region", "hp", "moving", "target")

    def __init__(self, side, num, region, hp):
        self.side, self.num, self.region, self.hp = side, num, region, hp
        self.moving = False
        self.target = 0


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


def _lvl_arr(kind_arr, level_arr, hq_table, base_table):
    """Vectorized per-kind level-table lookup (belief kind/level -> turret/work_cap),
    0 where kind is unknown/none. Mirrors FastEnv._lvl_derived."""
    hq_t = np.asarray(hq_table)
    ba_t = np.asarray(base_table)
    lvl_hq = hq_t[np.clip(level_arr, 0, len(hq_t) - 1)]
    lvl_ba = ba_t[np.clip(level_arr, 0, len(ba_t) - 1)]
    return np.where(kind_arr == KIND_HQ, lvl_hq, np.where(kind_arr == KIND_BASE, lvl_ba, 0))


class Bot:
    def __init__(self, weights_path, stochastic=True):
        import os  # already loaded at startup; lazy here to keep module-top imports minimal
        self.debug = bool(os.environ.get("BOT_DEBUG"))
        self.stochastic = stochastic     # vanilla: sample the single action ~ policy probs
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

        self.warriors = {}          # OWN warriors only (tracked incrementally, see below)
        self.buildings = {}         # OWN buildings only
        self.gold = START_GOLD
        self.income = 0             # own last-turn income

        # ---- fog-of-war belief about the opponent (per region, full N width) ----
        # Refreshed every turn from the WARRIOR/BUILDING snapshot inside this side's
        # own vision (hop_set of its own warrior/building regions); aged elsewhere.
        # Mirrors FastEnv.op_seen_* exactly (same source of truth as training).
        self.op_kind = None         # int[N]  0 = none, 1 = HQ, 2 = BASE
        self.op_level = None        # int[N]
        self.op_bhp = None          # int[N]
        self.op_wcnt = None         # int[N]  believed enemy warrior count
        self.op_whp = None          # int[N]  believed enemy warrior hp sum
        self.op_age = None          # int[N]  turns since this region was last visible

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

        # self.warriors/self.buildings track OWN state only -- the opponent's starting
        # HQ (and everything else about them) is handled through belief, which starts
        # unknown (a real player hasn't seen it yet either) and fills in on first sight.
        my_hq_region = 0 if self.my_side == 'A' else N - 1
        for sfx in range(1, START_WARRIORS + 1):
            self.warriors[(self.my_side, sfx)] = Warrior(self.my_side, sfx, my_hq_region, HQ_WHP[1])
        self.buildings[my_hq_region] = Building(my_hq_region, self.my_side, 'HQ', 1, HQ_HP[1])

        self.op_kind = np.zeros(N, dtype=np.int64)
        self.op_level = np.zeros(N, dtype=np.int64)
        self.op_bhp = np.zeros(N, dtype=np.int64)
        self.op_wcnt = np.zeros(N, dtype=np.int64)
        self.op_whp = np.zeros(N, dtype=np.int64)
        self.op_age = np.full(N, MAX_DAYS, dtype=np.int64)

        print("OK", flush=True)

    # --------------------------------------------------- encoder (numpy port)
    def encode(self, turn):
        N = self.N
        tok = self.tok_ids
        T = len(tok)
        me = self.my_side
        g = tok

        # ---- MY OWN side: exact, from the incrementally-tracked live state ----
        cnt_me = np.zeros(N); hp_me = np.zeros(N)
        stat_cnt_r = np.zeros(N); stat_hp_r = np.zeros(N)
        for w in self.warriors.values():
            if w.hp <= 0:
                continue
            cnt_me[w.region] += 1; hp_me[w.region] += w.hp
            if not w.moving:
                stat_cnt_r[w.region] += 1; stat_hp_r[w.region] += w.hp

        bowner = np.zeros(N, dtype=np.int64)
        bkind = np.zeros(N, dtype=np.int64)
        blevel = np.zeros(N, dtype=np.int64)
        bhp = np.zeros(N); bturret = np.zeros(N); bwcap = np.zeros(N)
        for b in self.buildings.values():
            r = b.region
            bowner[r] = self.me_code
            bkind[r] = KIND_HQ if b.kind == 'HQ' else KIND_BASE
            blevel[r] = b.level
            bhp[r] = b.hp
            bturret[r] = _turret(b.kind, b.level)
            bwcap[r] = _workcap(b.kind, b.level)

        # ---- OPPONENT: belief only (fog of war) --------------------------------
        op_kind_f = self.op_kind.astype(np.int64)
        op_level_f = self.op_level.astype(np.int64)
        op_tur_full = _lvl_arr(op_kind_f, op_level_f, HQ_TURRET, BASE_TURRET)
        op_wc_full = _lvl_arr(op_kind_f, op_level_f, HQ_WCAP, BASE_WCAP)
        cnt_op = self.op_wcnt.astype(np.float64)
        hp_op = self.op_whp.astype(np.float64)

        own_t = bowner[g]; kind_t = bkind[g]; lvl_t = blevel[g]
        bhp_t = bhp[g]; tur_t = bturret[g]; wc_t = bwcap[g]
        me_b = own_t == self.me_code

        my_cnt = cnt_me[g]; op_cnt = cnt_op[g]
        my_hps = hp_me[g]; op_hps = hp_op[g]
        my_base_lvl = np.where(me_b & (kind_t == KIND_BASE), lvl_t, 0)
        op_base_lvl = np.where(op_kind_f[g] == KIND_BASE, op_level_f[g], 0)
        my_hq_lvl = np.where(me_b & (kind_t == KIND_HQ), lvl_t, 0)
        op_hq_lvl = np.where(op_kind_f[g] == KIND_HQ, op_level_f[g], 0)
        my_tur = np.where(me_b, tur_t, 0); op_tur = op_tur_full[g]
        my_wc = np.where(me_b, wc_t, 0); op_wc = op_wc_full[g]
        my_bhp = np.where(me_b, bhp_t, 0); op_bhp = self.op_bhp[g]
        stat_cnt = stat_cnt_r[g]; stat_hp = stat_hp_r[g]
        surplus = stat_cnt - my_wc          # may be negative (matches observe)
        age_t = self.op_age[g].astype(np.float64)

        # arrivals of my movers at each token in exactly 1..5 turns
        arrive = np.zeros((T, 5))
        for w in self.warriors.values():
            if w.hp <= 0 or not w.moving:
                continue
            ti = self.tok2idx.get(int(w.target))
            if ti is None:
                continue
            k = int(self.tt[w.region, ti])
            if 1 <= k <= 5:
                arrive[ti, k - 1] += 1
        # enemy reachable within 1..5 turns at each token, from BELIEVED positions:
        # reach[t,k] = sum over regions r of op_wcnt[r] * (travel(r->t) <= k).
        reach = np.empty((T, 5))
        for k in range(1, 6):
            reach[:, k - 1] = self.op_wcnt.astype(np.float64) @ (self.tt <= k)

        tok_dist = self.tt[g, :].astype(np.float64)    # [T,T]

        feats = np.stack([my_cnt, op_cnt, my_base_lvl, op_base_lvl,
                          my_hq_lvl, op_hq_lvl, my_tur, op_tur, my_wc, op_wc,
                          my_bhp, op_bhp, surplus, stat_hp], axis=1)   # [T,14]
        raw = np.concatenate([feats, age_t[:, None], arrive, reach], axis=1)  # [T,25]
        raw25 = slog1p(raw)
        # the fog-of-war age feature (index 14) gets its OWN transform, ln(x/10+1),
        # not the generic slog1p everything else got -- overwrite it in place.
        raw25[:, 14] = np.log(raw[:, 14] / 10.0 + 1.0)

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
        t1 = np.concatenate([raw25, normx[:, None], normy[:, None], reach_delta],
                            axis=1)  # [T,32]

        # global features
        my_total = sum(1 for w in self.warriors.values() if w.hp > 0)
        op_total = float(self.op_wcnt.sum())
        my_hq_level = self._hq_level(me)
        opp_hq_region = self.N - 1 if me == 'A' else 0
        op_hq_level = int(self.op_level[opp_hq_region])
        lvl_sum_me = sum(b.level for b in self.buildings.values())
        lvl_sum_op = float(self.op_level.sum())
        # opponent's last-turn income is belief-based, NOT the true value (which a
        # real player never learns): 15 x min(believed count, believed work_cap)
        # per region, summed. Mirrors fast_env.observe()'s op_income_est exactly.
        op_income_est = float(np.sum(WORK_INCOME * np.minimum(self.op_wcnt, op_wc_full)))
        day = turn - 1
        glob = np.array([
            day / 10 - 10,
            plog1p(my_total / 10), plog1p(op_total / 10),
            plog1p(my_hq_level), plog1p(op_hq_level),
            plog1p(self.gold / 100),
            plog1p(self.income / 10), plog1p(op_income_est / 10),
            plog1p(lvl_sum_me / 5), plog1p(lvl_sum_op / 5),
            self._hq_turns_feature(my_hq_level, my_total),
            # static map-geometry globals (mirror ppo_selfplay.extract): 거점 count and
            # physical map size (x/y span over ALL regions; /10000 keeps span O(1)).
            plog1p(len(g) / 7.0),
            plog1p((max(self.x) - min(self.x)) / 10000.0),
            plog1p((max(self.y) - min(self.y)) / 10000.0),
        ])

        # action masking quantities (legality uses TRUE own state only -- a build
        # target always has my own warrior present, which is always visible to me,
        # so there is no fog ambiguity here)
        maxlev = np.where(kind_t == KIND_HQ, HQ_MAXLEVEL, BASE_MAXLEVEL)
        up_cost = np.where(kind_t == KIND_HQ,
                           np.take(HQ_UPCOST, np.minimum(lvl_t + 1, HQ_MAXLEVEL)),
                           np.take(BASE_COST, np.minimum(lvl_t + 1, BASE_MAXLEVEL)))
        heal_cost = np.where(kind_t == KIND_HQ, HQ_HEAL, BASE_HEAL)
        is_strong = self.stronghold[g]
        is_hq = self.is_hq[g]
        # Upgrade legality mirrors the judge (testing-tool2.apply_upgrades): a region is
        # only upgradeable when a friendly warrior is present AND no enemy warrior is on
        # it. up_room = "has room to upgrade" (ignores transient occupancy) is used by
        # the HQ-saving commitment so it can keep saving while an enemy sits on the HQ.
        # my_cnt > 0 already guarantees this region is in my own vision, so op_cnt
        # (belief) == true enemy presence there -- safe to use for legality.
        up_room = me_b & (lvl_t < maxlev)
        can_up = up_room & (my_cnt > 0) & (op_cnt == 0)
        can_heal = me_b & (lvl_t >= maxlev) & (my_cnt > 0) & (op_cnt == 0)
        # empty of BOTH my own buildings AND any BELIEVED opponent building. At a
        # region I don't currently occupy (my_cnt == 0, so build_cand is False and
        # this can't be sampled anyway) this can differ from the true global state
        # -- that's the fog approximation, not a bug; it never reaches the network
        # because build_cand already gates every consumer of this flag.
        build_new = (own_t == 0) & (op_kind_f[g] == 0) & is_strong & (~is_hq)
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

        return dict(t1=t1, glob=glob, tok_ids=g, gold=self.gold,
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
        net = self.income - 2 * my_total
        need = max(0, cost - self.gold)
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
        """Vanilla policy: ONE actor-net inference, then emit a single action sampled
        (stochastically) from its probabilities -- no lookahead/rollout."""
        import time as _t
        t0 = _t.perf_counter()
        self._ensure_ready()
        cmds = self._decide_single(turn)
        if self.debug:
            print(f"turn {turn}: decide {(_t.perf_counter()-t0)*1000:.1f}ms "
                  f"gold={self.gold} cmds={cmds} "
                  f"warriors={[(w.side,w.num,w.region,w.hp,w.moving,w.target) for w in self.warriors.values()]}",
                  file=sys.stderr, flush=True)
        return cmds

    def _decide_single(self, turn):
        if turn == 1:
            return self._decide_first_step(turn)
        o = self.encode(turn)
        plan = self._select_action(o)      # self.stochastic drives the sampling
        return self._to_commands(plan, o)

    def _decide_first_step(self, turn):
        """Turn-1 opening split. Our action space moves EVERY surplus warrior at a
        region to a single target, so the opening HQ move sends both spare warriors to
        one stronghold -- slowing the 2-base expansion vs opponents who split them. So
        on turn 1 only we infer the actor TWICE: the 1st inference's HQ move is applied
        to ONE warrior; we then re-encode with that warrior en route (this updates the
        gold, the HQ's movable-warrior count, and the target 거점's incoming-arrival
        feature) and infer again with the first target MASKED, sending a second warrior
        to a DIFFERENT stronghold. Falls back to the normal single action whenever the
        HQ isn't sending >= 2 warriors."""
        pr0 = self.prev_reach                       # reach baseline before any turn-1 encode
        o1 = self.encode(turn)
        plan1 = self._select_action(o1)
        reach1 = self.prev_reach                    # this turn's true reach (for turn-2 delta)
        hq_reg = 0 if self.my_side == 'A' else self.N - 1
        hq_tok = self.tok2idx.get(hq_reg)
        exec_move1, tgt1, wc_pb1 = plan1[1], plan1[2], plan1[3]

        # the surplus warriors the HQ move would dispatch, in the same (hp, num) order
        # _to_commands uses; split only when the HQ actually moves >= 2 of them.
        keep = int(wc_pb1[hq_tok]) if hq_tok is not None else 1
        here = sorted([w for w in self.warriors.values()
                       if w.hp > 0 and not w.moving and w.region == hq_reg],
                      key=lambda w: (w.hp, w.num))
        movable = here[keep:]
        if hq_tok is None or not bool(exec_move1[hq_tok]) or len(movable) < 2:
            return self._to_commands(plan1, o1)

        a_tok = int(tgt1[hq_tok])
        a_reg = int(o1['tok_ids'][a_tok])
        w_a = movable[0]                            # the one warrior sent to target A

        # --- apply the single HQ->A move, re-encode, infer again with A masked ---
        saved_gold = self.gold
        saved_moving, saved_target = w_a.moving, w_a.target
        saved_commit = self.hq_commit
        a_bld = self.buildings.get(a_reg)
        cost = 0 if (a_bld is not None and a_bld.side == self.my_side) else MOVE_COST
        self.gold = saved_gold - cost
        w_a.moving, w_a.target = True, a_reg
        # reset the recurrent reach feature to its turn-0 baseline so o2's reach-delta
        # matches o1's (still "turn 1", not "since the 1st inference").
        self.prev_reach = pr0
        o2 = self.encode(turn)
        plan2 = self._select_action(o2, forbid_tgt=a_tok)
        # _to_commands reads live state: w_a is now moving (excluded), so this dispatches
        # the REMAINING HQ warrior(s) to target B, and picks up the (re-decided) training.
        upgrades, moves2, train_cat = self._to_commands(plan2, o2)

        # restore the mutated state; read_turn_result will apply the authoritative moves.
        # prev_reach is restored to o1's value (the true turn-1 observation) so turn 2's
        # recurrent feature threads from there.
        self.gold = saved_gold
        w_a.moving, w_a.target = saved_moving, saved_target
        self.hq_commit = saved_commit
        self.prev_reach = reach1

        moves = [(w_a.side, w_a.num, a_reg)] + moves2
        return upgrades, moves, train_cat

    def _greedy(self, item, cost, gold, order):
        remaining = int(gold)
        ex = np.zeros(len(item), dtype=bool)
        for k in order:
            if item[k] and cost[k] <= remaining:
                remaining -= int(cost[k]); ex[k] = True
        return ex, remaining

    def _select_action(self, o, t1_cache=None, return_logp=False, forbid_tgt=None):
        """Full action selection = build phase + T2 net + finish phase. Split so the
        T2 pass can be batched across candidates if ever needed.
        With return_logp=True also returns the joint policy log-prob of the sampled
        action. forbid_tgt (token index or None): if set, that target 거점 is masked
        out of every move-source's target distribution (used by the turn-1 opening
        split)."""
        ctx = self._select_build(o, t1_cache)
        X = ctx['X']
        logits = self.net.t2(X) if X is not None else None
        plan, logp = self._select_finish(ctx, logits, forbid_tgt=forbid_tgt)
        return (plan, logp) if return_logp else plan

    def _select_build(self, o, t1_cache=None):
        # t1_cache = (h1, head5) precomputed by net.t1 for THIS obs. The T1 pass depends
        # only on o, so a caller sampling several candidate actions from one state can
        # compute it once and reuse it here instead of re-running the actor net.
        # Runs everything up to (and including) assembling the T2 input X; the actual T2
        # net call + move/train selection happen in _select_finish (so T2 can be batched).
        T = len(o['tok_ids'])
        sto = self.stochastic
        if t1_cache is None:
            h1, head5 = self.net.t1(o['t1'], o['glob'])
        else:
            h1, head5 = t1_cache

        # Non-moving friendly warriors board-wide (caps builds), and of those the
        # "free" ones not currently labouring -- surplus beyond each region's
        # work_cap -- which can actually be dispatched to staff a base (gates builds).
        by_region = {}
        for w in self.warriors.values():
            if w.hp > 0 and not w.moving:
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

        # log-prob of this build Bernoulli sampling path (sampleable tokens only; the
        # deterministic non-sampleable ones contribute 1).
        pb = np.clip(p_build, 1e-6, 1.0 - 1e-6)
        lp_build = float(np.where(build_mask, np.where(outcome, np.log(pb),
                                                       np.log1p(-pb)), 0.0).sum())

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

        # add the committed/forced HQ upgrade (bypasses our gating) for emission
        if do_hq_now and hq_tok is not None:
            exec_build[hq_tok] = True

        wc_pb = np.where(exec_build, o['wc_after'], o['wc_cur'])
        surplus_pb = np.maximum(o['stat_cnt'] - wc_pb, 0)
        owner_me_pb = o['owner_me'] | (exec_build & o['build_new'])
        hq_after = min(o['hq_level'] + (1 if do_hq_now else 0), HQ_MAXLEVEL)

        # ---------------- MOVE (T2 input assembly) ----------------
        # while committed: only free moves -> restrict targets to our own buildings.
        # A source needs actual surplus (stationary beyond work_cap) to be a valid
        # move source at all -- labourers can never be ordered out (no full-
        # mobilisation action; mirrors ppo_selfplay.sample_policy).
        valid_src = (surplus_pb > 0) & (MOVE_COST * surplus_pb <= gold1)
        tgt_allowed = owner_me_pb if committed else np.ones(T, dtype=bool)
        src_list = np.nonzero(valid_src)[0]
        X = None
        if src_list.size > 0:
            X = np.empty((src_list.size, T, h1.shape[1] + T2_EXTRA), dtype=np.float32)
            for j, si in enumerate(src_list):
                sf = np.full((T, 1), plog1p(surplus_pb[si]))
                tv = plog1p(o['tok_dist'][si, :])[:, None]
                dx = (o['normx'] - o['normx'][si])[:, None]
                dy = (o['normy'] - o['normy'][si])[:, None]
                X[j] = np.concatenate([h1, o['extra4'], sf, tv, dx, dy], axis=1)

        return {
            'o': o, 'T': T, 'sto': sto, 'head5': head5, 'committed': committed,
            'exec_build': exec_build, 'wc_pb': wc_pb, 'surplus_pb': surplus_pb,
            'owner_me_pb': owner_me_pb, 'gold1': gold1, 'hq_after': hq_after,
            'valid_src': valid_src, 'tgt_allowed': tgt_allowed, 'src_list': src_list,
            'X': X, 'lp_build': lp_build,
        }

    def _select_finish(self, ctx, logits, forbid_tgt=None):
        """Given the T2 logits (or None if the state has no move-sources), finish move
        target selection + greedy move allocation + training. Mirrors the original
        _select_action tail exactly. Returns the plan tuple.
        forbid_tgt: optional target token index masked out of every source (turn-1 split)."""
        T = ctx['T']; sto = ctx['sto']; committed = ctx['committed']
        src_list = ctx['src_list']; tgt_allowed = ctx['tgt_allowed']
        surplus_pb = ctx['surplus_pb']; owner_me_pb = ctx['owner_me_pb']
        valid_src = ctx['valid_src']; gold1 = ctx['gold1']
        exec_build = ctx['exec_build']; wc_pb = ctx['wc_pb']; hq_after = ctx['hq_after']
        head5 = ctx['head5']

        tgt = np.arange(T)
        lp_move = 0.0                                    # log-prob of the move-target picks
        if src_list.size > 0:
            tgt_lg = logits[:, :, 0]
            if committed:
                tgt_lg = np.where(tgt_allowed[None, :], tgt_lg, -1e9)
            if forbid_tgt is not None:
                tgt_lg = tgt_lg.copy()
                tgt_lg[:, forbid_tgt] = -1e9
            for j, si in enumerate(src_list):
                p = softmax(tgt_lg[j])
                if sto:
                    tgt[si] = self.rng.choice(T, p=p)
                else:
                    tgt[si] = int(np.argmax(tgt_lg[j]))
                lp_move += float(np.log(max(p[tgt[si]], 1e-12)))
        tgt_is_self = tgt == np.arange(T)
        tgt_mine = owner_me_pb[tgt]
        move_cost = np.where(tgt_mine, 0, MOVE_COST * surplus_pb)
        move_item = valid_src & (~tgt_is_self)
        prio2 = np.full(T, -1e30)
        if src_list.size > 0:
            chosen = tgt_lg[np.arange(src_list.size), tgt[src_list]]
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
        sm = softmax(tl_m)
        if sto:
            train_cat = int(self.rng.choice(4, p=sm))
        else:
            train_cat = int(np.argmax(tl_m))
        lp_train = float(np.log(max(sm[train_cat], 1e-12)))

        plan = (exec_build, exec_move, tgt, wc_pb, train_cat)
        logp = ctx['lp_build'] + lp_move + lp_train     # joint policy log-prob of this action
        return plan, logp

    def _to_commands(self, plan, o):
        exec_build, exec_move, tgt, wc_pb, train_cat = plan
        tok = o['tok_ids']
        # moves are keyed (side, num, target_region) -- side-agnostic to the warrior
        # objects, so the same command list drives emission and my-action bookkeeping.
        upgrades, moves = [], []
        for t in np.nonzero(exec_build)[0]:
            upgrades.append(int(tok[t]))
        for s in np.nonzero(exec_move)[0]:
            sreg = int(tok[s]); treg = int(tok[int(tgt[s])]); keep = int(wc_pb[s])
            here = [w for w in self.warriors.values()
                    if w.hp > 0 and not w.moving and w.region == sreg]
            here.sort(key=lambda w: (w.hp, w.num))
            for w in here[keep:]:
                moves.append((w.side, w.num, treg))
        return upgrades, moves, int(train_cat)

    def emit(self, commands):
        upgrades, moves, train_n = commands
        out = ["COMMAND"]
        for side, num, treg in moves:
            out.append(f"MOVE {side}{num} {treg}")
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
                self.gold -= BASE_COST[1]
                self.buildings[region] = Building(region, self.my_side, 'BASE', 1, BASE_HP[1])
            elif b.level >= _maxlevel(b.kind):
                self.gold -= _heal_cost(b.kind)
                b.hp = _maxhp(b.kind, b.level)
            else:
                self.gold -= _upgrade_cost(b.kind, b.level)
                b.level += 1
                b.hp = _maxhp(b.kind, b.level)
        for side, num, treg in moves:
            w = self.warriors.get((side, num))
            if w is None:
                continue
            b = self.buildings.get(treg)
            cost = 0 if (b is not None and b.side == self.my_side) else MOVE_COST
            self.gold -= cost
            w.moving = True
            w.target = treg
        self.gold -= TRAIN_COST * train_n

    def _settle_economy(self):
        """Own economy only -- the opponent's is never exactly knowable under fog of
        war (see encode()'s belief-based op_income_est)."""
        inc = 0
        for b in self.buildings.values():
            cnt = sum(1 for w in self.warriors.values()
                      if w.region == b.region and w.hp > 0)
            inc += WORK_INCOME * min(cnt, _workcap(b.kind, b.level))
        self.gold += inc
        self.income = inc
        alive = sum(1 for w in self.warriors.values() if w.hp > 0)
        n_fed = min(alive, self.gold // 2)
        self.gold -= UPKEEP_PER_WARRIOR * n_fed

    def _update_fog(self):
        """Refresh the opponent belief from the WARRIOR/BUILDING snapshot just read
        into `self._seen_op_warriors` / `self._seen_op_buildings` (region -> data),
        inside this side's own vision (hop_set of its own warrior/building regions),
        aging everything else. Mirrors FastEnv._update_fog exactly."""
        centers = {w.region for w in self.warriors.values() if w.hp > 0}
        centers |= {b.region for b in self.buildings.values()}
        vis = hop_set(centers, self.adj, HOP_VISION)

        wcnt, whp = {}, {}
        for (region, hp) in self._seen_op_warriors:
            wcnt[region] = wcnt.get(region, 0) + 1
            whp[region] = whp.get(region, 0) + hp
        bld = self._seen_op_buildings          # region -> (kind, level, hp)

        for r in vis:
            if r in bld:
                kind, level, hp = bld[r]
                self.op_kind[r] = kind; self.op_level[r] = level; self.op_bhp[r] = hp
            else:
                self.op_kind[r] = 0; self.op_level[r] = 0; self.op_bhp[r] = 0
            self.op_wcnt[r] = wcnt.get(r, 0)
            self.op_whp[r] = whp.get(r, 0)
            self.op_age[r] = 0
        for r in range(self.N):
            if r not in vis:
                self.op_age[r] += 1

    def read_turn_result(self):
        self._apply_my_actions()

        line = readln()
        if line == "FINISH":
            sys.exit(0)
        assert line.split()[0] == "TURN"
        read_tokens()                                   # TIME ...

        # UPGRADE/TRAIN/MOVE/DAMAGE/SIEGE now report THIS SIDE'S OWN events only
        # (the fog-of-war protocol no longer merges in the opponent's).
        n = int(read_tokens()[1])                       # UPGRADE
        for _ in range(n):
            read_tokens()                                # already applied locally above

        n = int(read_tokens()[1])                       # TRAIN
        if n > 0:
            ids = read_tokens()
            hq_region = 0 if self.my_side == 'A' else self.N - 1
            hq_b = self.buildings.get(hq_region)
            lvl = hq_b.level if hq_b is not None else 1
            for tok in ids[:n]:
                num = int(tok[1:])
                self.warriors[(self.my_side, num)] = Warrior(self.my_side, num, hq_region, HQ_WHP[lvl])

        n = int(read_tokens()[1])                       # MOVE
        for _ in range(n):
            r = read_tokens()
            num = int(r[0][1:])
            region = int(r[1])
            w = self.warriors.get((self.my_side, num))
            if w is not None:
                w.region = region
                if w.moving and w.region == w.target:
                    w.moving = False

        n = int(read_tokens()[1])                       # DAMAGE
        for _ in range(n):
            r = read_tokens()
            num = int(r[1][1:])
            w = self.warriors.get((self.my_side, num))
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

        # WARRIOR: every warrior currently in view -- our own (always) plus the
        # opponent's wherever visible. Ours is redundant with the tracking above
        # (used only as the opponent source here); the opponent rows feed belief.
        self._seen_op_warriors = []
        n = int(read_tokens()[1])
        for _ in range(n):
            r = read_tokens()
            side = r[0][0]
            region, hp = int(r[1]), int(r[2])
            if side != self.my_side:
                self._seen_op_warriors.append((region, hp))

        # BUILDING: ditto -- ours always included, opponent's wherever visible.
        self._seen_op_buildings = {}
        n = int(read_tokens()[1])
        for _ in range(n):
            r = read_tokens()
            side, region, kind, level, hp = r[0], int(r[1]), r[2], int(r[3]), int(r[4])
            if side != self.my_side:
                k = KIND_HQ if kind == "HQ" else KIND_BASE
                self._seen_op_buildings[region] = (k, level, hp)

        readln()                                        # END
        self._update_fog()
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
    # `--stochastic` (default) / `--greedy`. No search knobs: this bot always emits a
    # single one-shot policy action. Defaults: data.bin, stochastic sampling.
    weights, stochastic = "data.bin", False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--stochastic":
            stochastic = True
        elif a == "--greedy":
            stochastic = False
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
