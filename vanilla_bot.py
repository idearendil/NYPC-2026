#!/usr/bin/env python3
"""Submission bot (numpy-only): the trained actor played straight, no search.

Action selection: ONE actor-net inference per turn, then emit a single action
SAMPLED (stochastically) from the resulting policy probabilities -- the raw policy,
played as-is. (`--greedy` switches to argmax, and to the p > 0.5 threshold on T2's
full-mobilisation output.) `decide` -> `_decide_single` -> encode -> sample -> emit.

The hidden opponent gold is reconstructed from visible play: `read_turn_result`
charges builds/trains/income exactly and infers per-warrior move costs via the
`_ensure_ready` next-hop / `tvia` precompute. `fast_env` mirrors that same estimate
during training (env.est_gold), so the feature the net sees matches at submission.
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
MAX_DAYS = 200          # game ends after day 200 (testing-tool.py); timeout decided by HQ hp
WIN_REWARD = 10.0       # terminal reward magnitude (matches ppo_selfplay.reward_done)
GAMMA_SEARCH = 0.8      # per-turn discount for combining the lookahead's per-turn values
ROLLOUT_DECAY = 0.8     # per-turn decay weighting the critic values along a rollout trajectory
SEARCH_BUDGET_S = 0.070 # per-turn INTERNAL search budget. Kept well under the judge's 100ms
                        # cap: the judge measures wall time incl. IPC/stdio overhead (~10-20ms)
                        # AND the predictor can overshoot by one candidate-iteration, so 90ms
                        # internal spiked to ~110ms judge-side -> token depletion -> timeout WA.

OWN_LEFT, OWN_RIGHT = 1, 2
KIND_HQ, KIND_BASE = 1, 2

TOK_FEAT = 31           # 14 + 5 arrive + 5 reach + 2 coords + 5 reach-delta vs prev turn
GLOB_FEAT = 16          # 11 + HQ-turns + 거점-count + x/y map-span + own prev aux opp-gold pred
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

    def t2(self, x):                             # x:[S,T,d_in] -> [S,T,2]
        """Per-source target head: [...,0] = move-target logit (softmax over tokens),
        [...,1] = full-mobilisation logit (sigmoid; send the source's labourers too)."""
        W = self.W
        h = self._encoder("t2.enc", x)
        head = linear(gelu(linear(h, W["t2.head.0.weight"], W["t2.head.0.bias"])),
                      W["t2.head.2.weight"], W["t2.head.2.bias"])
        return head

    def t1_batch(self, t1s, globs):              # t1s:[K,T,31], globs:[K,GLOB] -> ([K,T,d],[K,T,5])
        """Batched T1 over K states (fixed T). Same math as t1(), one encoder pass."""
        W = self.W
        K, T, _ = t1s.shape
        x = np.concatenate([t1s, np.broadcast_to(globs[:, None, :], (K, T, GLOB_FEAT))], axis=2)
        h = self._encoder("t1.enc", x)               # [K,T,d]
        head = linear(gelu(linear(h, W["t1.head.0.weight"], W["t1.head.0.bias"])),
                      W["t1.head.2.weight"], W["t1.head.2.bias"])   # [K,T,5]
        return h, head

    def aux_gold_pred(self, h1):                 # h1:[T,d] -> scalar
        """The actor's auxiliary prediction of the opponent's next-turn gold, in the
        aux target space ln(1+gold/100): run the aux head (Linear-GELU-Linear -> [T,7]),
        take channel 6, mean over tokens (all valid here). Mirrors ppo_selfplay's
        sample_policy gold_pred; fed back as glob feature #15 the FOLLOWING turn."""
        W = self.W
        a = linear(gelu(linear(h1, W["t1.aux.0.weight"], W["t1.aux.0.bias"])),
                   W["t1.aux.2.weight"], W["t1.aux.2.bias"])   # [T,7]
        return float(a[:, 6].mean())

    def has_critic(self):
        return "critic.enc.embed.weight" in self.W

    def value(self, t1, glob):                   # t1:[T,31], glob:[12] -> scalar
        """Critic state value from t1/glob (no padding: all T tokens valid). Mirrors
        ppo_selfplay.Critic.value = encoder -> per-token head -> mean over tokens."""
        W = self.W
        T = t1.shape[0]
        x = np.concatenate([t1, np.broadcast_to(glob, (T, GLOB_FEAT))], axis=1)
        h = self._encoder("critic.enc", x[None])     # [1,T,d]
        v = linear(gelu(linear(h, W["critic.head.0.weight"], W["critic.head.0.bias"])),
                   W["critic.head.2.weight"], W["critic.head.2.bias"])   # [1,T,1]
        return float(v[0, :, 0].mean())

    def value_batch(self, t1s, globs):           # t1s:[K,T,31], globs:[K,GLOB] -> [K]
        """Batched critic value over K states (fixed T). Same math as value()."""
        W = self.W
        K, T, _ = t1s.shape
        x = np.concatenate([t1s, np.broadcast_to(globs[:, None, :], (K, T, GLOB_FEAT))], axis=2)
        h = self._encoder("critic.enc", x)           # [K,T,d]
        v = linear(gelu(linear(h, W["critic.head.0.weight"], W["critic.head.0.bias"])),
                   W["critic.head.2.weight"], W["critic.head.2.bias"])   # [K,T,1]
        return v[:, :, 0].mean(axis=1)               # [K]


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


def compute_next_hop(N, x, y, adj):
    """All-pairs next-hop matrix nxt[u][t] = the neighbour a warrior at region u
    steps to when heading for region t, replicating testing-tool.apply_day_movement:
    pick the adjacent v minimising edge_weight(u,v)+dijkstra_dist(v,t), smallest
    index on ties. The board is fixed, so this is precomputed once at map load and
    reused by the lookahead simulator for every rollout step. -1 = no move."""
    INF = BIG
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
    for nb in range(N):                          # nb ascending -> smallest index wins ties
        score = w[:, nb][:, None] + dist[nb, :][None, :]
        cand = adj_mask[:, nb][:, None] & (score < best)
        best = np.where(cand, score, best)
        nxt = np.where(cand, nb, nxt)
    di = np.arange(N)
    nxt[di, di] = di
    return nxt


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
                 "moved_last", "moved_now", "tgt_set", "move_chg")

    def __init__(self, side, num, region, hp):
        self.side, self.num, self.region, self.hp = side, num, region, hp
        self.moving = False
        self.target = 0
        self.moved_last = False
        self.moved_now = False
        self.tgt_set = 0            # consistent-target bitmask for the current move (0 = not moving)
        self.move_chg = False       # this move carries an outstanding (refundable) 10-gold charge


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


class St:
    """A lightweight, cloneable game state for the lookahead simulator. Uses the
    same Warrior/Building objects as the live tracker so encode/_select_action can
    run on it unchanged (via Bot._plan's temp-swap)."""
    __slots__ = ("warriors", "buildings", "gold", "income", "next_sfx")


class Bot:
    def __init__(self, weights_path, stochastic=True):
        import os  # already loaded at startup; lazy here to keep module-top imports minimal
        self.debug = bool(os.environ.get("BOT_DEBUG"))
        self.stochastic = stochastic     # vanilla: sample the single action ~ policy probs
        self.weights_path = weights_path
        self.net = None                  # loaded lazily, after the handshake (see _ensure_ready)
        self.nxt = None                  # all-pairs next-hop, drives the opp move-cost inference
        self.tvia = None                 # tvia[a][b] = bitmask of targets whose next hop from a
                                         # is b; drives opp move-cost target inference (see below)
        self.rng = None                  # numpy.random pulled in lazily too (handshake hygiene)
        self.hq_commit = False           # saving-for-HQ-upgrade macro (see _select_action)
        self.prev_reach = None           # last turn's raw enemy-reachability [T,5] for the delta feature
        self.prev_gold_pred = 0.0        # last turn's actor-aux opp-gold prediction, fed as glob feature #15

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
        # enemy reachable within 1..5 turns at each token: reach[t,k] = sum over regions
        # r of cnt_op[r] * (travel(r->t) <= k+1). Vectorized as cnt_op @ (tt <= k) per
        # horizon (equivalent to the per-region python loop, ~27x faster).
        reach = np.empty((T, 5))
        for k in range(1, 6):
            reach[:, k - 1] = cnt_op @ (self.tt <= k)     # [N] @ [N,T] -> [T]

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
            # static map-geometry globals (mirror ppo_selfplay.extract): 거점 count and
            # physical map size (x/y span over ALL regions; /10000 keeps span O(1)).
            plog1p(len(g) / 7.0),
            plog1p((max(self.x) - min(self.x)) / 10000.0),
            plog1p((max(self.y) - min(self.y)) / 10000.0),
            # this side's OWN previous-turn actor-aux prediction of the opponent's
            # next-turn gold (ln(1+gold/100) space), fed back as a feature. 0 on turn 1.
            getattr(self, 'prev_gold_pred', 0.0),
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
            # all-pairs next-hop, used by BOTH the lookahead simulator and the opp move-cost
            # target inference. Fixed board -> computed once.
            self.nxt = compute_next_hop(N, self.x, self.y, self.adj)
            # tvia[a][b] = bitmask of targets t whose next hop from a is b (i.e. a step a->b
            # is consistent with heading to t). Lets the opp-gold reconstruction infer each
            # opp warrior's move target from its trajectory (see read_turn_result).
            self.tvia = [dict() for _ in range(N)]
            for a in range(N):
                row = self.nxt[a]
                tv = self.tvia[a]
                for t in range(N):
                    nb = int(row[t])
                    if nb >= 0:
                        tv[nb] = tv.get(nb, 0) | (1 << t)
            self.rng = np.random.default_rng()

    def decide(self, turn):
        """Vanilla policy: ONE actor-net inference, then emit a single action sampled
        (stochastically) from its probabilities -- no lookahead/rollout. The opponent-
        gold reconstruction in read_turn_result is the shared visible-play estimate."""
        import time as _t
        t0 = _t.perf_counter()
        self._ensure_ready()
        cmds = self._decide_single(turn)
        if self.debug:
            print(f"turn {turn}: decide {(_t.perf_counter()-t0)*1000:.1f}ms",
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
        pgp0 = self.prev_gold_pred                  # aux opp-gold-pred baseline (glob feat #15)
        o1 = self.encode(turn)
        plan1 = self._select_action(o1)
        reach1 = self.prev_reach                    # this turn's true reach (for turn-2 delta)
        pred1 = self.prev_gold_pred                 # this turn's true aux pred (for turn-2 feat)
        hq_reg = 0 if self.my_side == 'A' else self.N - 1
        hq_tok = self.tok2idx.get(hq_reg)
        exec_move1, tgt1, wc_pb1 = plan1[1], plan1[2], plan1[3]

        # the surplus warriors the HQ move would dispatch, in the same (hp, num) order
        # _to_commands uses; split only when the HQ actually moves >= 2 of them.
        keep = int(wc_pb1[hq_tok]) if hq_tok is not None else 1
        here = sorted([w for w in self.warriors.values()
                       if w.side == self.my_side and w.hp > 0 and not w.moving
                       and w.region == hq_reg], key=lambda w: (w.hp, w.num))
        movable = here[keep:]
        if hq_tok is None or not bool(exec_move1[hq_tok]) or len(movable) < 2:
            return self._to_commands(plan1, o1)

        a_tok = int(tgt1[hq_tok])
        a_reg = int(o1['tok_ids'][a_tok])
        w_a = movable[0]                            # the one warrior sent to target A

        # --- apply the single HQ->A move, re-encode, infer again with A masked ---
        saved_gold = self.gold[self.my_side]
        saved_moving, saved_target = w_a.moving, w_a.target
        saved_commit = self.hq_commit
        a_bld = self.buildings.get(a_reg)
        cost = 0 if (a_bld is not None and a_bld.side == self.my_side) else MOVE_COST
        self.gold[self.my_side] = saved_gold - cost
        w_a.moving, w_a.target = True, a_reg
        # reset the recurrent features to their turn-0 baseline so o2's reach-delta and
        # opp-gold-pred feature match o1's (still "turn 1", not "since the 1st inference").
        self.prev_reach = pr0
        self.prev_gold_pred = pgp0
        o2 = self.encode(turn)
        plan2 = self._select_action(o2, forbid_tgt=a_tok)
        # _to_commands reads live state: w_a is now moving (excluded), so this dispatches
        # the REMAINING HQ warrior(s) to target B, and picks up the (re-decided) training.
        upgrades, moves2, train_cat = self._to_commands(plan2, o2)

        # restore the mutated state; read_turn_result will apply the authoritative moves.
        # prev_reach / prev_gold_pred are restored to o1's values (the true turn-1
        # observation) so turn 2's recurrent features thread from there.
        self.gold[self.my_side] = saved_gold
        w_a.moving, w_a.target = saved_moving, saved_target
        self.hq_commit = saved_commit
        self.prev_reach = reach1
        self.prev_gold_pred = pred1

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
        T2 pass can be batched across candidates in the search (_select_batch).
        With return_logp=True also returns the joint policy log-prob of the sampled
        action (used to weight opponent replies by their probability in the search).
        forbid_tgt (token index or None): if set, that target 거점 is masked out of
        every move-source's target distribution (used by the turn-1 opening split)."""
        ctx = self._select_build(o, t1_cache)
        X = ctx['X']
        logits = self.net.t2(X) if X is not None else None
        plan, logp = self._select_finish(ctx, logits, forbid_tgt=forbid_tgt)
        return (plan, logp) if return_logp else plan

    def _select_build(self, o, t1_cache=None):
        # t1_cache = (h1, head5) precomputed by net.t1 for THIS obs. The T1 pass depends
        # only on o -- so when we sample several candidate actions from one state (search),
        # we compute it once and reuse it here instead of re-running the actor net.
        # Runs everything up to (and including) assembling the T2 input X; the actual T2
        # net call + move/train selection happen in _select_finish (so T2 can be batched).
        T = len(o['tok_ids'])
        sto = self.stochastic
        if t1_cache is None:
            h1, head5 = self.net.t1(o['t1'], o['glob'])
        else:
            h1, head5 = t1_cache
        # this turn's actor-aux prediction of the opponent's next-turn gold becomes
        # NEXT turn's glob feature #15 (mirrors ppo_selfplay.sample_policy's gold_pred
        # threading). encode() already consumed the PREVIOUS value before this call.
        self.prev_gold_pred = self.net.aux_gold_pred(h1)

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

        # log-prob of this build Bernoulli sampling path (sampleable tokens only; the
        # deterministic non-sampleable ones contribute 1). Used to weight opponent
        # replies by their policy probability in the search. (Argmax path: the same
        # per-token prob p if built else 1-p, which is max(p,1-p) for the >0.5 choice.)
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
        # A source only needs a stationary warrior: 거점 whose warriors are ALL
        # labouring (surplus 0) are legal sources that can move only by full
        # mobilisation (T2 head [1]); mirrors ppo_selfplay.sample_policy.
        valid_src = (o['stat_cnt'] > 0) & (MOVE_COST * surplus_pb <= gold1)
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

        stat_cnt = ctx['o']['stat_cnt']
        tgt = np.arange(T)
        mob = np.zeros(T, dtype=bool)                    # full-mobilisation per source
        lp_move = 0.0                                    # log-prob of the move-target picks
        if src_list.size > 0:
            tgt_lg, mob_lg = logits[:, :, 0], logits[:, :, 1]
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
                # the chosen target's second output decides whether the source's
                # labourers march out too (0.5 threshold; sampled when stochastic)
                if tgt[si] != si and surplus_pb[si] < stat_cnt[si]:
                    pm = 1.0 / (1.0 + np.exp(-mob_lg[j, tgt[si]]))
                    mob[si] = (self.rng.random() < pm) if sto else (pm > 0.5)
                    lp_move += float(np.log(max(pm if mob[si] else 1.0 - pm, 1e-12)))
            chosen = tgt_lg[np.arange(src_list.size), tgt[src_list]]
        tgt_is_self = tgt == np.arange(T)
        tgt_mine = owner_me_pb[tgt]
        mov_cnt = np.where(mob, stat_cnt, surplus_pb)    # warriors actually leaving
        move_cost = np.where(tgt_mine, 0, MOVE_COST * mov_cnt)
        move_item = valid_src & (~tgt_is_self) & (mov_cnt > 0)
        prio2 = np.full(T, -1e30)
        if src_list.size > 0:
            prio2[src_list] = np.where(move_item[src_list], chosen, -1e30)
        order2 = np.argsort(-prio2, kind='stable')
        exec_move, gold2 = self._greedy(move_item, move_cost, gold1, order2)
        # a mobilised source keeps nobody home (the env's keep-cap goes to 0)
        wc_pb = np.where(mob, 0, wc_pb)

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
        # objects, so the same command list drives emission, my-action bookkeeping,
        # and the lookahead simulator (which operates on cloned state).
        upgrades, moves = [], []
        for t in np.nonzero(exec_build)[0]:
            upgrades.append(int(tok[t]))
        for s in np.nonzero(exec_move)[0]:
            sreg = int(tok[s]); treg = int(tok[int(tgt[s])]); keep = int(wc_pb[s])
            here = [w for w in self.warriors.values()
                    if w.side == self.my_side and w.hp > 0
                    and not w.moving and w.region == sreg]
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
                self.gold[self.my_side] -= BASE_COST[1]
                self.buildings[region] = Building(region, self.my_side, 'BASE', 1, BASE_HP[1])
            elif b.level >= _maxlevel(b.kind):
                self.gold[self.my_side] -= _heal_cost(b.kind)
                b.hp = _maxhp(b.kind, b.level)
            else:
                self.gold[self.my_side] -= _upgrade_cost(b.kind, b.level)
                b.level += 1
                b.hp = _maxhp(b.kind, b.level)
        for side, num, treg in moves:
            w = self.warriors.get((side, num))
            if w is None:
                continue
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
        # Opp move-cost via nxt-path TARGET INFERENCE (exact movement model). Each opp
        # warrior carries `tgt_set` = the bitmask of targets still consistent with its
        # current move (the judge always steps along a shortest path via nxt, so a genuine
        # continuation keeps the set nonempty). A NEW move is charged 10 (target unknown);
        # a move is FREE iff its target is an opp-owned building, detected when the set
        # collapses inside `owned` or when the move concludes (stop / re-dispatch) on an
        # owned building -> the 10 is refunded. This replaces the old euclid re-dispatch
        # test + land-and-stop refund.
        owned = 0
        for b in self.buildings.values():
            if b.side == opp:
                owned |= 1 << b.region
        opp_delta = 0                                    # net opp gold spent on moves this turn
        moved = set()
        for _ in range(n):
            r = read_tokens()
            s = 'A' if r[0][0] == 'A' else 'B'
            num = int(r[0][1:])
            region = int(r[1])
            w = self.warriors.get((s, num))
            if w is not None:
                old = w.region
                w.moved_now = True
                w.region = region
                if s == self.my_side and w.moving and w.region == w.target:
                    w.moving = False
                if s == opp:
                    moved.add((s, num))
                    cons = self.tvia[old].get(region, 0)     # targets a step old->region fits
                    if (not w.moved_last) or (w.tgt_set & cons) == 0:
                        # NEW move (fresh dispatch, or a re-dispatch: this step fits NO target
                        # of the previous move). If re-dispatching from an owned building, the
                        # concluded previous move was free -> refund it before charging anew.
                        if w.move_chg and (owned >> old) & 1:
                            opp_delta -= MOVE_COST
                        opp_delta += MOVE_COST               # charge the new move
                        w.tgt_set = cons
                        w.move_chg = True
                    else:
                        w.tgt_set &= cons                    # continuation
                    if w.move_chg and w.tgt_set and (w.tgt_set & ~owned) == 0:
                        opp_delta -= MOVE_COST               # every possible target owned -> free
                        w.move_chg = False
        # opp warriors mid-move that did NOT move this turn -> the move concluded (stopped);
        # refund if it stopped on an owned building (was a free garrison move).
        for k, w in self.warriors.items():
            if w.side == opp and w.tgt_set and k not in moved:
                if w.move_chg and (owned >> w.region) & 1:
                    opp_delta -= MOVE_COST
                w.tgt_set = 0
                w.move_chg = False
        self.gold[opp] -= opp_delta
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
    # `--stochastic` (default) / `--greedy`. No search knobs: this bot always emits a
    # single one-shot policy action. Defaults: data.bin, stochastic sampling.
    weights, stochastic = "data.bin", True
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
