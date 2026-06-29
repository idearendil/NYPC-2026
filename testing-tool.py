import argparse
import heapq
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import cmp_to_key
from enum import Enum
from typing import Dict, List, Optional, TextIO, Tuple

START_GOLD      = 500
START_WARRIORS  = 3
TRAIN_COST      = 120
MOVE_COST       = 10
BASE_BUILD_COST = 300
WORK_INCOME     = 15
UPKEEP_PER      = 2
MAX_DAYS        = 200
SOFT_CAP_MS     = 100
START_TOKENS    = 5
HANDSHAKE_MS    = 1.0

HQ_MAX_LEVEL   = 5
BASE_MAX_LEVEL  = 3

HQ_HEAL_COST   = 1000
BASE_HEAL_COST = 500

HqLevel   = namedtuple("HqLevel",   ["upgrade_cost", "warrior_hp", "hp", "turret", "train_cap", "work_cap"])
BaseLevel = namedtuple("BaseLevel", ["cost", "hp", "turret", "work_cap"])

HQ_LEVELS = [
    HqLevel(0,     0, 0,  0, 0, 0),
    HqLevel(0,     4, 10, 1, 1, 1),
    HqLevel(600,   5, 15, 2, 1, 2),
    HqLevel(1200,  6, 20, 2, 2, 3),
    HqLevel(2400,  7, 25, 3, 2, 4),
    HqLevel(3600,  8, 30, 3, 3, 5),
]
BASE_LEVELS = [
    BaseLevel(0,    0,  0, 0),
    BaseLevel(300,  6, 1, 1),
    BaseLevel(600,  12, 1, 2),
    BaseLevel(1000, 18, 2, 3),
]


def hq_warrior_hp(hq_level: int) -> int:
    return HQ_LEVELS[hq_level].warrior_hp

_U64_MASK = (1 << 64) - 1


class XoShiro256:

    def __init__(self, seed: int):
        self.s = self._split_mix_64(seed & _U64_MASK)

    @staticmethod
    def _split_mix_64(z: int) -> List[int]:
        out: List[int] = []
        for _ in range(4):
            z = (z + 0x9E3779B97F4A7C15) & _U64_MASK
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64_MASK
            out.append(z ^ (z >> 31))
        return out

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) & _U64_MASK) | (x >> (64 - k))

    def next_u64(self) -> int:
        s = self.s
        result = (self._rotl((s[0] + s[3]) & _U64_MASK, 23) + s[0]) & _U64_MASK
        t = (s[1] << 17) & _U64_MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] = (s[2] ^ t) & _U64_MASK
        s[3] = self._rotl(s[3], 45)
        return result

    def next_u64_rng(self, n: int) -> int:
        assert n > 0
        x = self.next_u64()
        m = x * n
        m_lo = m & _U64_MASK
        if m_lo < n:
            neg_n = (-n) & _U64_MASK
            t = neg_n % n
            while m_lo < t:
                x = self.next_u64()
                m = x * n
                m_lo = m & _U64_MASK
        return m >> 64

    def next_u64_incl(self, lo: int, hi: int) -> int:
        assert lo <= hi
        if lo == 0 and hi == _U64_MASK:
            return self.next_u64()
        return lo + self.next_u64_rng(hi - lo + 1)

    def next_i64_incl(self, lo: int, hi: int) -> int:
        BIAS = 1 << 63
        u_lo = (lo + BIAS) & _U64_MASK
        u_hi = (hi + BIAS) & _U64_MASK
        u = self.next_u64_incl(u_lo, u_hi)
        v = (u + BIAS) & _U64_MASK
        return v - (1 << 64) if v >= BIAS else v


GEN_L = 10000  
GEN_D = 100    
GEN_A = 24     


def _cross_pt(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _in_circle(ax: int, ay: int, bx: int, by: int, cx: int, cy: int,
               dx: int, dy: int) -> int:
    ax_, ay_ = ax - dx, ay - dy
    bx_, by_ = bx - dx, by - dy
    cx_, cy_ = cx - dx, cy - dy
    aa = ax_ * ax_ + ay_ * ay_
    bb = bx_ * bx_ + by_ * by_
    cc = cx_ * cx_ + cy_ * cy_
    return (ax_ * (by_ * cc - bb * cy_)
            - ay_ * (bx_ * cc - bb * cx_)
            + aa  * (bx_ * cy_ - by_ * cx_))


def _round_div(num: int, den: int) -> int:
    assert den != 0
    an = -num if num < 0 else num
    ad = -den if den < 0 else den
    result = (an + ad // 2) // ad
    positive = (num >= 0) == (den > 0)
    return result if positive else -result


def _circumcenter_int(ax: int, ay: int, bx: int, by: int,
                      cx: int, cy: int) -> Tuple[int, int]:
    det = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    sa = ax * ax + ay * ay
    sb = bx * bx + by * by
    sc = cx * cx + cy * cy
    ux_num = sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)
    uy_num = sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)
    return (_round_div(ux_num, det), _round_div(uy_num, det))


def _polygon_centroid_shoelace(pts: List[Tuple[int, int]]) -> Tuple[int, int]:
    k = len(pts)
    cross_sum = 0
    cx_num = 0
    cy_num = 0
    for i in range(k):
        j = (i + 1) % k
        xi, yi = pts[i]
        xj, yj = pts[j]
        crs = xi * yj - xj * yi
        cross_sum += crs
        cx_num += (xi + xj) * crs
        cy_num += (yi + yj) * crs
    if cross_sum == 0:
        raise RuntimeError(
            "degenerated cell — try a different seed"
        )
    denom = 3 * cross_sum
    return (_round_div(cx_num, denom), _round_div(cy_num, denom))


def _quad(dx: int, dy: int) -> int:
    if dx > 0 and dy >= 0:
        return 0
    if dx <= 0 and dy > 0:
        return 1
    if dx < 0 and dy <= 0:
        return 2
    return 3


def _ccw_cmp(site: Tuple[int, int], a: Tuple[int, int],
             b: Tuple[int, int]) -> int:
    ax, ay = a[0] - site[0], a[1] - site[1]
    bx, by = b[0] - site[0], b[1] - site[1]
    qa, qb = _quad(ax, ay), _quad(bx, by)
    if qa != qb:
        return -1 if qa < qb else 1
    cr = ax * by - ay * bx
    if cr > 0:
        return -1
    if cr < 0:
        return 1
    return 0


def _q_points() -> List[Tuple[int, int]]:
    q: List[Tuple[int, int]] = [(0, 0)] * GEN_A
    R = 1.5 * float(GEN_L)
    for i in range(GEN_A):
        theta = i * (2.0 * math.pi / GEN_A)
        q[i] = (int(math.trunc(R * math.cos(theta))),
                int(math.trunc(R * math.sin(theta))))
    for i in range(GEN_A // 2):
        q[i + GEN_A // 2] = (-q[i][0], -q[i][1])
    return q


def _delaunay_triangulate(pts: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    n = len(pts)
    SUP = 300000
    S0, S1, S2 = n, n + 1, n + 2
    all_pts = list(pts) + [(-SUP, -SUP), (SUP, -SUP), (0, SUP)]
    if _cross_pt(all_pts[S0][0], all_pts[S0][1],
                 all_pts[S1][0], all_pts[S1][1],
                 all_pts[S2][0], all_pts[S2][1]) <= 0:
        S1, S2 = S2, S1

    tris: List[Tuple[int, int, int]] = [(S0, S1, S2)]

    def ekey(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    for pi in range(n):
        px, py = all_pts[pi]
        bad: List[int] = []
        for t_idx, (a, b, c) in enumerate(tris):
            ax, ay = all_pts[a]
            bx, by = all_pts[b]
            cx, cy = all_pts[c]
            if _in_circle(ax, ay, bx, by, cx, cy, px, py) > 0:
                bad.append(t_idx)

        shared: Dict[Tuple[int, int], int] = {}
        for t_idx in bad:
            a, b, c = tris[t_idx]
            for u, v in ((a, b), (b, c), (c, a)):
                k = ekey(u, v)
                shared[k] = shared.get(k, 0) + 1

        boundary: List[Tuple[int, int]] = []
        for t_idx in bad:
            a, b, c = tris[t_idx]
            for u, v in ((a, b), (b, c), (c, a)):
                if shared[ekey(u, v)] == 1:
                    boundary.append((u, v))

        for t_idx in sorted(bad, reverse=True):
            tris[t_idx] = tris[-1]
            tris.pop()

        for u, v in boundary:
            tris.append((u, v, pi))

    return [t for t in tris if t[0] < n and t[1] < n and t[2] < n]


def _voronoi_centroids(p: List[Tuple[int, int]], q: List[Tuple[int, int]],
                       tris: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    np_ = len(p)
    adj_tris: List[List[int]] = [[] for _ in range(np_)]
    for t_idx, (a, b, c) in enumerate(tris):
        for vi in (a, b, c):
            if vi < np_:
                adj_tris[vi].append(t_idx)

    all_pts = list(p) + list(q)
    p_prime: List[Tuple[int, int]] = [(0, 0)] * np_
    for i in range(np_):
        verts: List[Tuple[int, int]] = []
        for t_idx in adj_tris[i]:
            a, b, c = tris[t_idx]
            ax, ay = all_pts[a]
            bx, by = all_pts[b]
            cx, cy = all_pts[c]
            verts.append(_circumcenter_int(ax, ay, bx, by, cx, cy))
        site = p[i]
        verts.sort(key=cmp_to_key(lambda a, b, s=site: _ccw_cmp(s, a, b)))
        p_prime[i] = _polygon_centroid_shoelace(verts)
    return p_prime


def generate_map(rng: XoShiro256, NP: Optional[int] = None,
                 KP: Optional[int] = None) -> List[str]:
    L = GEN_L
    D = GEN_D

    if NP is not None:
        N = 2 * NP + 1
        if N < 51 or N > 109:
            raise ValueError(
                f"N = 2*NP+1 = {N} is out of range [51, 109] (got NP={NP})"
            )
    else:
        NP = rng.next_u64_incl(25, 54)

    N = 2 * NP + 1

    K_lo = (3 * N + 19) // 20   # ceil(0.15N) = ceil(3N/20)
    K_hi = N // 5                # floor(0.2N)
    K_lo_odd = K_lo + 1 if K_lo % 2 == 0 else K_lo
    K_hi_odd = K_hi - 1 if K_hi % 2 == 0 else K_hi
    KP_lo = (K_lo_odd - 1) // 2
    KP_hi = (K_hi_odd - 1) // 2

    if KP is not None:
        K = 2 * KP + 1
        if K < K_lo_odd or K > K_hi_odd:
            raise ValueError(
                f"K = 2*KP+1 = {K} is out of allowed range for N={N}: "
                f"{K_lo_odd} <= K <= {K_hi_odd} (odd), so {KP_lo} <= KP <= {KP_hi} "
                f"(got KP={KP})"
            )
    else:
        KP = rng.next_u64_incl(KP_lo, KP_hi)

    K = 2 * KP + 1

    Q = _q_points()

    D2 = D * D
    P: List[Tuple[int, int]] = [(0, 0)]

    max_attempts = NP * 20000
    attempts = 0
    placed = 0
    while placed < NP:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not draw {NP} non-overlapping points "
                f"(try different --seed or smaller --NP)"
            )
        x = rng.next_i64_incl(-L, L)
        y = rng.next_i64_incl(-L, L)

        if x * x + y * y > L * L:
            continue
        if any(vx == x for vx, _ in P):
            continue
        too_close = False
        for vx, vy in P:
            dx = vx - x
            dy = vy - y
            if dx * dx + dy * dy < D2:
                too_close = True
                break
        if too_close:
            continue

        P.append((x, y))
        P.append((-x, -y))
        placed += 1

    P.sort(key=lambda pt: pt[0])
    assert P[NP] == (0, 0)

    pq = list(P) + list(Q)
    tris1 = _delaunay_triangulate(pq)
    P_prime = _voronoi_centroids(P, Q, tris1)

    seen: set = set()
    for pp in P_prime:
        if pp in seen:
            raise RuntimeError(
                "two centroids collapsed to the same lattice point — try different seed"
            )
        seen.add(pp)
    P = P_prime  

    adj: List[set] = [set() for _ in range(N)]
    pq = list(P) + list(Q)
    tris2 = _delaunay_triangulate(pq)
    for a, b, c in tris2:
        for u, v in ((a, b), (b, c), (c, a)):
            if u < N and v < N:
                adj[u].add(v)
                adj[v].add(u)

    R: set = {NP}
    candidates = list(range(1, NP))
    for i in range(len(candidates) - 1, 0, -1):
        j = rng.next_u64_incl(0, i)
        candidates[i], candidates[j] = candidates[j], candidates[i]

    def mirror(r: int) -> int:
        return N - 1 - r

    for c in candidates:
        if len(R) >= K:
            break
        m = mirror(c)
        if c in R or m in R:
            continue
        adj_to_r = False
        for r in R:
            if r in adj[c] or r in adj[m]:
                adj_to_r = True
                break
        if adj_to_r:
            continue
        if m in adj[c]:
            continue
        R.add(c)
        R.add(m)

    if len(R) < K:
        raise RuntimeError(
            f"could only place {len(R)}/{K} strongholds under adjacency "
            f"constraints; try different --seed or smaller --KP"
        )

    rows: List[str] = []
    rows.append(f"{N} {K}")
    rows.append(" ".join(str(P[i][0]) for i in range(N)))
    rows.append(" ".join(str(P[i][1]) for i in range(N)))
    rows.append(" ".join(str(r) for r in sorted(R)))
    for i in range(N):
        neigh = sorted(adj[i])
        rows.append(" ".join(str(v) for v in [len(neigh), *neigh]))
    return rows


class Side(Enum):
    LEFT  = 0
    RIGHT = 1

    @property
    def opp(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @property
    def letter(self) -> str:
        return "A" if self is Side.LEFT else "B"

    @property
    def name_str(self) -> str:
        return "LEFT" if self is Side.LEFT else "RIGHT"


class BKind(Enum):
    HQ   = "HQ"
    BASE = "BASE"


@dataclass
class Building:
    region: int
    side:   Side
    kind:   BKind
    level:  int
    hp:     int

    def _table(self):
        return HQ_LEVELS if self.kind is BKind.HQ else BASE_LEVELS

    def max_level(self) -> int:
        return HQ_MAX_LEVEL if self.kind is BKind.HQ else BASE_MAX_LEVEL

    def max_hp_now(self) -> int:
        return self._table()[self.level].hp

    def turret(self) -> int:
        return self._table()[self.level].turret

    def work_cap(self) -> int:
        return self._table()[self.level].work_cap

    def train_cap(self) -> int:
        assert self.kind is BKind.HQ
        return HQ_LEVELS[self.level].train_cap

    def heal_cost(self) -> int:
        return HQ_HEAL_COST if self.kind is BKind.HQ else BASE_HEAL_COST

    def next_level_cost(self) -> int:
        if self.level >= self.max_level():
            return -1
        nxt = self._table()[self.level + 1]
        return nxt.upgrade_cost if self.kind is BKind.HQ else nxt.cost


@dataclass
class Warrior:
    key:            int
    side:           Side
    suffix:         int
    region:         int
    hp:             int
    moving_target:  Optional[int] = None


def wkey(side: Side, suffix: int) -> int:
    return -suffix if side is Side.LEFT else +suffix


def side_from_wkey(key: int) -> Side:
    return Side.LEFT if key < 0 else Side.RIGHT


def suffix_from_wkey(key: int) -> int:
    return -key if key < 0 else key


def id_str(key: int) -> str:
    return side_from_wkey(key).letter + str(suffix_from_wkey(key))


def wkey_sort(k: int) -> Tuple[int, int]:
    return (side_from_wkey(k).value, suffix_from_wkey(k))


@dataclass
class MapData:
    N:           int
    K:           int
    x:           List[int]
    y:           List[int]
    strongholds: List[int]
    adj:         List[List[int]]


def hq_region(m: "MapData", side: "Side") -> int:
    return 0 if side is Side.LEFT else m.N - 1


@dataclass
class GameState:
    day:          int = 0
    gold:         List[int] = field(default_factory=lambda: [START_GOLD, START_GOLD])
    warriors:     Dict[int, Warrior] = field(default_factory=dict)
    buildings:    Dict[int, Building] = field(default_factory=dict)
    next_suffix:  List[int] = field(default_factory=lambda: [START_WARRIORS + 1, START_WARRIORS + 1])
    tokens:       List[int] = field(default_factory=lambda: [START_TOKENS, START_TOKENS])


@dataclass
class Submission:
    has_train: bool = False
    train_n:   int  = 0
    upgrades:  List[int] = field(default_factory=list)
    moves:     List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ResultBlock:
    trained_keys: List[int] = field(default_factory=list)
    upgrades:  List[Tuple[int, Side]] = field(default_factory=list)
    moves:     List[Tuple[int, int]] = field(default_factory=list)
    damages:   List[Tuple[str, int, int]] = field(default_factory=list)
    sieges:    List[Tuple[Side, int, int]] = field(default_factory=list)


class WaError(Exception):
    def __init__(self, side: Side, msg: str):
        super().__init__(msg)
        self.side = side
        self.msg  = msg


class Player:
    def __init__(self, no: int, exec_cmd: str, log_stream: TextIO):
        self.name = ["LEFT", "RIGHT"][no]
        self.exec_cmd = exec_cmd
        try:
            self.process = subprocess.Popen(
                exec_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
        except Exception as e:
            print(f"Error: failed to start process {exec_cmd}: {e}")
            sys.exit(1)

        self.reads      = queue.Queue()
        self.writes     = queue.Queue()
        self.log_stream = log_stream

        self._stdin_thread  = threading.Thread(target=self._handle_stdin,  daemon=True)
        self._stdout_thread = threading.Thread(target=self._handle_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._handle_stderr, daemon=True)
        self._stdin_thread.start()
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _handle_stdin(self):
        stdin = self.process.stdin
        assert stdin is not None
        try:
            while True:
                msg = self.writes.get()
                if msg is None:
                    break
                stdin.write(f"{msg}\n")
                stdin.flush()
        finally:
            stdin.close()

    def _handle_stdout(self):
        stdout = self.process.stdout
        assert stdout is not None
        try:
            while True:
                r = stdout.readline()
                if not r:
                    break
                self.reads.put(r)
        except Exception:
            pass
        finally:
            stdout.close()

    def _handle_stderr(self):
        stderr = self.process.stderr
        assert stderr is not None
        try:
            while True:
                r = stderr.readline()
                if not r:
                    break
                self.log_stream.write(f"# Debug {self.name}: {r.rstrip()}\n")
        except Exception:
            pass
        finally:
            stderr.close()

    def send(self, message: str):
        self.writes.put(message)

    def readline(self, timeout: float) -> Optional[str]:
        try:
            return self.reads.get(timeout=timeout)
        except queue.Empty:
            return None

    @classmethod
    def read_all(cls, players: List["Player"], timeout: float) -> List[Optional[str]]:
        results: List[Optional[str]] = [None] * len(players)

        def _read(p: "Player", idx: int):
            results[idx] = p.readline(timeout)

        threads = [threading.Thread(target=_read, args=(p, i)) for i, p in enumerate(players)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def join(self, timeout: Optional[float] = 1.0):
        self.writes.put(None)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def read_map(lines: List[str]) -> MapData:
    it = iter(lines)

    def nxt() -> str:
        return next(it).strip()

    parts = nxt().split()
    N, K = int(parts[0]), int(parts[1])

    x = list(map(int, nxt().split()))
    y = list(map(int, nxt().split()))

    strongholds = list(map(int, nxt().split()))
    if 0 in strongholds:
        raise ValueError("stronghold list must not contain 0 (LEFT HQ)")
    if N - 1 in strongholds:
        raise ValueError(f"stronghold list must not contain {N - 1} (RIGHT HQ)")

    adj = []
    for _ in range(N):
        row = list(map(int, nxt().split()))
        deg = row[0]
        adj.append(sorted(row[1:1 + deg]))

    return MapData(N=N, K=K, x=x, y=y, strongholds=strongholds, adj=adj)


def edge_weight(m: MapData, u: int, v: int) -> int:
    dx = m.x[u] - m.x[v]
    dy = m.y[u] - m.y[v]
    return math.ceil(math.sqrt(dx * dx + dy * dy))


_dijkstra_cache: Dict[int, List[int]] = {}


def dijkstra_from(m: MapData, target: int) -> List[int]:
    if target in _dijkstra_cache:
        return _dijkstra_cache[target]
    dist = [-1] * m.N
    dist[target] = 0
    heap = [(0, target)]
    while heap:
        du, u = heapq.heappop(heap)
        if du != dist[u]:
            continue
        for v in m.adj[u]:
            dv = du + edge_weight(m, u, v)
            if dist[v] < 0 or dv < dist[v]:
                dist[v] = dv
                heapq.heappush(heap, (dv, v))
    _dijkstra_cache[target] = dist
    return dist


def bfs_reachable(m: MapData, src: int, dst: int) -> bool:
    if src == dst:
        return True
    visited = [False] * m.N
    q = [src]
    visited[src] = True
    for u in q:
        for v in m.adj[u]:
            if v == dst:
                return True
            if not visited[v]:
                visited[v] = True
                q.append(v)
    return False


def init_state(m: MapData) -> GameState:
    st = GameState()
    hq_l = 0
    hq_r = m.N - 1

    st.buildings[hq_l] = Building(hq_l, Side.LEFT,  BKind.HQ, 1, HQ_LEVELS[1].hp)
    st.buildings[hq_r] = Building(hq_r, Side.RIGHT, BKind.HQ, 1, HQ_LEVELS[1].hp)

    for sfx in range(1, START_WARRIORS + 1):
        kl = wkey(Side.LEFT,  sfx)
        kr = wkey(Side.RIGHT, sfx)
        st.warriors[kl] = Warrior(kl, Side.LEFT,  sfx, hq_l, hq_warrior_hp(1))
        st.warriors[kr] = Warrior(kr, Side.RIGHT, sfx, hq_r, hq_warrior_hp(1))

    return st


def parse_warrior_token(side: Side, tok: str) -> int:
    if len(tok) < 2:
        raise WaError(side, f"bad warrior id: {tok}")
    letter = tok[0]
    if letter not in ("A", "B"):
        raise WaError(side, f"bad warrior id prefix: {tok}")
    parsed_side = Side.LEFT if letter == "A" else Side.RIGHT
    if parsed_side != side:
        raise WaError(side, f"warrior id from wrong side: {tok}")
    try:
        suffix = int(tok[1:])
    except ValueError:
        raise WaError(side, f"bad warrior id digits: {tok}")
    if suffix <= 0 or suffix > 1_000_000:
        raise WaError(side, f"warrior id out of range: {tok}")
    return suffix


def parse_block(side: Side, lines: List[str]) -> Submission:
    sub = Submission()
    moved_suffixes: set = set()
    upgrade_regions: set = set()

    for raw in lines:
        parts = raw.split()
        if not parts:
            raise WaError(side, "empty command line")
        verb = parts[0]

        if verb == "MOVE":
            if len(parts) != 3:
                raise WaError(side, f"MOVE: expected `<id> <region>`, got: {raw}")
            suffix = parse_warrior_token(side, parts[1])
            try:
                target = int(parts[2])
            except ValueError:
                raise WaError(side, f"MOVE: bad target: {raw}")
            key = wkey(side, suffix)
            if key in moved_suffixes:
                raise WaError(side, f"duplicate MOVE for {parts[1]}")
            moved_suffixes.add(key)
            sub.moves.append((suffix, target))

        elif verb == "TRAIN":
            if len(parts) != 2:
                raise WaError(side, f"TRAIN: expected `<n>`, got: {raw}")
            if sub.has_train:
                raise WaError(side, "duplicate TRAIN line")
            try:
                n = int(parts[1])
            except ValueError:
                raise WaError(side, f"TRAIN: bad n: {raw}")
            sub.has_train = True
            sub.train_n   = n

        elif verb == "UPGRADE":
            if len(parts) != 2:
                raise WaError(side, f"UPGRADE: expected `<region>`, got: {raw}")
            try:
                r = int(parts[1])
            except ValueError:
                raise WaError(side, f"UPGRADE: bad region: {raw}")
            if r in upgrade_regions:
                raise WaError(side, f"duplicate UPGRADE for region {r}")
            upgrade_regions.add(r)
            sub.upgrades.append(r)

        else:
            raise WaError(side, f"unknown command verb: {verb}")

    return sub


def hq_of(st: GameState, side: Side) -> Optional[Building]:
    for b in st.buildings.values():
        if b.kind is BKind.HQ and b.side is side:
            return b
    return None


def any_warrior_at(st: GameState, region: int, side: Side, friendly: bool) -> bool:
    for w in st.warriors.values():
        if w.region == region and w.hp > 0 and ((w.side is side) == friendly):
            return True
    return False


def any_hq_destroyed(st: GameState) -> bool:
    return hq_of(st, Side.LEFT) is None or hq_of(st, Side.RIGHT) is None


def apply_upgrades(st: GameState, m: MapData, side: Side,
                   sub: Submission, rb: ResultBlock):
    u = side.value
    HQS = {0, m.N - 1}
    STRONGHOLD_SET = set(m.strongholds)
    UPGRADEABLE = HQS | STRONGHOLD_SET
    for r in sub.upgrades:
        if r < 0 or r >= m.N:
            raise WaError(side, f"UPGRADE region out of range: {r}")
        if r not in UPGRADEABLE:
            raise WaError(side, f"UPGRADE: not a stronghold or HQ region: {r}")
        if any_warrior_at(st, r, side, False):
            raise WaError(side, f"UPGRADE: enemy warrior present at {r}")
        if not any_warrior_at(st, r, side, True):
            raise WaError(side, f"UPGRADE without friendly warrior at {r}")

        if r not in st.buildings:
            if r in HQS:
                raise WaError(side, f"UPGRADE: cannot BUILD on HQ region {r}")
            if st.gold[u] < BASE_BUILD_COST:
                raise WaError(side, f"UPGRADE (build) insufficient gold")
            st.gold[u] -= BASE_BUILD_COST
            st.buildings[r] = Building(r, side, BKind.BASE, 1, BASE_LEVELS[1].hp)
        else:
            b = st.buildings[r]
            if b.side != side:
                raise WaError(side, f"UPGRADE: enemy-owned building at {r}")
            if b.level >= b.max_level():
                cost = b.heal_cost()
                if st.gold[u] < cost:
                    raise WaError(side, f"UPGRADE (heal): insufficient gold")
                st.gold[u] -= cost
                b.hp = b.max_hp_now()
            else:
                cost = b.next_level_cost()
                if st.gold[u] < cost:
                    raise WaError(side, f"UPGRADE: insufficient gold")
                st.gold[u] -= cost
                b.level += 1
                b.hp = b.max_hp_now()

        rb.upgrades.append((r, side))


def apply_moves(st: GameState, m: MapData, side: Side, sub: Submission):
    u = side.value
    for (suffix, target) in sub.moves:
        key = wkey(side, suffix)
        if key not in st.warriors:
            raise WaError(side, f"MOVE: unknown warrior {id_str(key)}")
        w = st.warriors[key]
        if w.hp <= 0:
            raise WaError(side, f"MOVE: dead warrior {id_str(key)}")
        if w.moving_target is not None:
            raise WaError(side, f"MOVE: warrior already moving {id_str(key)}")
        if target < 0 or target >= m.N:
            raise WaError(side, f"MOVE: target out of range: {target}")
        if not bfs_reachable(m, w.region, target):
            raise WaError(side, f"MOVE: unreachable target {target}")

        b = st.buildings.get(target)
        cost = 0 if (b is not None and b.side == side) else MOVE_COST
        if st.gold[u] < cost:
            raise WaError(side, f"MOVE: insufficient gold")
        st.gold[u] -= cost
        w.moving_target = target


def apply_train_charge(st: GameState, side: Side, sub: Submission) -> int:
    if not sub.has_train:
        return 0
    u = side.value
    n = sub.train_n

    hq = hq_of(st, side)
    if hq is None:
        raise WaError(side, "TRAIN after HQ destroyed")
    if n < 0:
        raise WaError(side, "TRAIN n must be >= 0")
    if n == 0:
        return 0
    if n > hq.train_cap():
        raise WaError(side, "TRAIN exceeds HQ train cap")
    cost = n * TRAIN_COST
    if cost > st.gold[u]:
        raise WaError(side, "TRAIN insufficient gold")

    st.gold[u] -= cost
    return n


def spawn_trained(st: GameState, side: Side, n: int, rb: ResultBlock):
    if n == 0:
        return
    u = side.value
    hq = hq_of(st, side)
    hq_region = hq.region
    whp = hq_warrior_hp(hq.level)
    for _ in range(n):
        sfx = st.next_suffix[u]
        st.next_suffix[u] += 1
        k = wkey(side, sfx)
        st.warriors[k] = Warrior(k, side, sfx, hq_region, whp)
        rb.trained_keys.append(k)


def apply_day_movement(st: GameState, m: MapData, rb_l: ResultBlock, rb_r: ResultBlock):
    enemy_of_left:  set = set()
    enemy_of_right: set = set()
    for w in st.warriors.values():
        if w.hp <= 0:
            continue
        if w.side is Side.RIGHT:
            enemy_of_left.add(w.region)
        else:
            enemy_of_right.add(w.region)

    dist_cache: Dict[int, List[int]] = {}

    for key, w in st.warriors.items():
        if w.hp <= 0 or w.moving_target is None:
            continue
        t = w.moving_target
        if w.region == t:
            w.moving_target = None
            continue

        enemy_set = enemy_of_left if w.side is Side.LEFT else enemy_of_right
        if w.region in enemy_set:
            continue

        if t not in dist_cache:
            dist_cache[t] = dijkstra_from(m, t)
        dist = dist_cache[t]

        best_v     = -1
        best_score = -1
        for v in m.adj[w.region]:
            if dist[v] < 0:
                continue
            score = edge_weight(m, w.region, v) + dist[v]
            if best_score < 0 or score < best_score:
                best_score = score
                best_v = v

        if best_v < 0:
            continue

        w.region = best_v
        rb = rb_l if w.side is Side.LEFT else rb_r
        rb.moves.append((key, best_v))
        if w.region == t:
            w.moving_target = None


def _damage_tick(st: GameState, region: int, side: Side) -> int:
    best_key = 0
    best_hp  = -1
    found    = False
    for k, w in st.warriors.items():
        if w.region != region or w.side is not side or w.hp <= 0:
            continue
        if (not found
                or w.hp < best_hp
                or (w.hp == best_hp
                    and suffix_from_wkey(k) < suffix_from_wkey(best_key))):
            best_hp  = w.hp
            best_key = k
            found    = True
    if not found:
        return 0
    st.warriors[best_key].hp -= 1
    return best_key


def apply_day_combat(st: GameState, rb_l: ResultBlock, rb_r: ResultBlock,
                     siege_damage: Dict[int, int]):
    counts: Dict[int, List[int]] = {}
    for w in st.warriors.values():
        if w.hp > 0:
            counts.setdefault(w.region, [0, 0])[w.side.value] += 1

    for r in st.buildings:
        counts.setdefault(r, [0, 0])

    dmg_accum: Dict[Tuple[str, int], int] = {}  # (cause, key) → total damage

    for region in sorted(counts.keys()):
        lc, rc = counts[region]
        b = st.buildings.get(region)
        b_is_left  = b is not None and b.side is Side.LEFT
        b_is_right = b is not None and b.side is Side.RIGHT
        turret_val = b.turret() if b is not None else 0

        left_present  = lc > 0 or b_is_left
        right_present = rc > 0 or b_is_right
        if not (left_present and right_present):
            continue

        left_cap  = lc + (turret_val if b_is_left  else 0)
        right_cap = rc + (turret_val if b_is_right else 0)

        left_idle = 0
        if b_is_left:
            for i in range(left_cap):
                cause = "TURRET" if i < turret_val else "COMBAT"
                k = _damage_tick(st, region, Side.RIGHT)
                if k == 0:
                    left_idle += 1
                else:
                    dmg_accum[(cause, k)] = dmg_accum.get((cause, k), 0) + 1
        else:
            for _ in range(left_cap):
                k = _damage_tick(st, region, Side.RIGHT)
                if k == 0:
                    left_idle += 1
                else:
                    dmg_accum[("COMBAT", k)] = dmg_accum.get(("COMBAT", k), 0) + 1

        right_idle = 0
        if b_is_right:
            for i in range(right_cap):
                cause = "TURRET" if i < turret_val else "COMBAT"
                k = _damage_tick(st, region, Side.LEFT)
                if k == 0:
                    right_idle += 1
                else:
                    dmg_accum[(cause, k)] = dmg_accum.get((cause, k), 0) + 1
        else:
            for _ in range(right_cap):
                k = _damage_tick(st, region, Side.LEFT)
                if k == 0:
                    right_idle += 1
                else:
                    dmg_accum[("COMBAT", k)] = dmg_accum.get(("COMBAT", k), 0) + 1

        if b is not None:
            attacker_idle = right_idle if b.side is Side.LEFT else left_idle
            if attacker_idle > 0:
                siege_damage[region] = attacker_idle

    for (cause, key), dmg in dmg_accum.items():
        rb = rb_l if key < 0 else rb_r
        rb.damages.append((cause, key, dmg))


def apply_day_siege(st: GameState, rb_l: ResultBlock, rb_r: ResultBlock,
                    siege_damage: Dict[int, int]):
    regions = sorted(st.buildings.keys())
    destroyed = []

    for r in regions:
        dmg = siege_damage.get(r, 0)
        if dmg <= 0:
            continue
        b = st.buildings[r]
        dealt = min(dmg, b.hp)
        b.hp -= dealt
        rb = rb_l if b.side is Side.LEFT else rb_r
        rb.sieges.append((b.side, r, dealt))
        if b.hp <= 0:
            destroyed.append(r)

    for r in destroyed:
        del st.buildings[r]


def apply_evening_work(st: GameState):
    consumed: set = set()
    for r in sorted(st.buildings.keys()):
        b = st.buildings[r]
        eligible = [k for k, w in st.warriors.items()
                    if w.side is b.side and w.region == r and w.hp > 0 and k not in consumed]
        eligible.sort(key=wkey_sort)
        take = min(b.work_cap(), len(eligible))
        for i in range(take):
            consumed.add(eligible[i])
            st.gold[b.side.value] += WORK_INCOME


def apply_evening_upkeep(st: GameState, rb_l: ResultBlock, rb_r: ResultBlock):
    for side in (Side.LEFT, Side.RIGHT):
        alive_keys = sorted(
            [k for k, w in st.warriors.items() if w.side is side and w.hp > 0],
            key=suffix_from_wkey,
        )
        for k in alive_keys:
            w = st.warriors[k]
            if st.gold[side.value] >= UPKEEP_PER:
                st.gold[side.value] -= UPKEEP_PER
            else:
                w.hp -= 1
                rb = rb_l if k < 0 else rb_r
                rb.damages.append(("HUNGER", k, 1))

    dead = [k for k, w in st.warriors.items() if w.hp <= 0]
    for k in dead:
        del st.warriors[k]


def deduct_tokens(st: GameState, u: int, t_used_ms: int):
    if t_used_ms > SOFT_CAP_MS:
        excess = t_used_ms - SOFT_CAP_MS
        debit  = (excess + 99) // 100
        st.tokens[u] -= debit
        if st.tokens[u] < 0:
            st.tokens[u] = 0


def write_map_block(log: TextIO, m: MapData):
    log.write("MAP\n")
    log.write(f"{m.N} {m.K}\n")
    log.write(" ".join(map(str, m.x)) + "\n")
    log.write(" ".join(map(str, m.y)) + "\n")
    log.write("STRONGHOLDS " + " ".join(map(str, m.strongholds)) + "\n")
    for i in range(m.N):
        adj = m.adj[i]
        if adj:
            log.write(f"{len(adj)} " + " ".join(map(str, adj)) + "\n")
        else:
            log.write(f"{len(adj)}\n")
    log.write("END MAP\n")


def write_turn_header(log: TextIO, day: int):
    log.write(f"TURN {day}\n")


def write_command_block(log: TextIO, side: Side, lines: List[str]):
    log.write(f"COMMAND {side.name_str} START\n")
    for ln in lines:
        log.write(ln + "\n")
    log.write(f"COMMAND {side.name_str} END\n")


CAUSE_ORDER = {"TURRET": 0, "COMBAT": 1, "HUNGER": 2}


def _merge_results(rb_a: ResultBlock, rb_b: ResultBlock):
    return (
        sorted(rb_a.trained_keys + rb_b.trained_keys, key=wkey_sort),
        sorted(rb_a.upgrades + rb_b.upgrades, key=lambda r: r[0]),
        sorted(rb_a.moves + rb_b.moves, key=lambda r: wkey_sort(r[0])),
        sorted(rb_a.damages + rb_b.damages,
               key=lambda r: (CAUSE_ORDER[r[0]], wkey_sort(r[1]))),
        sorted(rb_a.sieges + rb_b.sieges, key=lambda r: r[1]),
    )


def write_turn_result(log: TextIO, day: int,
                      t_used_l: int, t_used_r: int,
                      tokens_l: int, tokens_r: int,
                      rb_l: ResultBlock, rb_r: ResultBlock):
    log.write(f"TURN {day} RESULT\n")
    log.write(f"TIME LEFT {t_used_l} {tokens_l} RIGHT {t_used_r} {tokens_r}\n")

    merged_rec, merged_up, merged_mv, merged_dmg, merged_sg = _merge_results(rb_l, rb_r)

    for (region, up_side) in merged_up:
        log.write(f"UPGRADE {up_side.letter} {region}\n")

    if merged_rec:
        log.write("TRAIN " + " ".join(id_str(k) for k in merged_rec) + "\n")

    for (key, new_region) in merged_mv:
        log.write(f"MOVE {id_str(key)} {new_region}\n")

    for (cause, key, dmg) in merged_dmg:
        log.write(f"DAMAGE {cause} {id_str(key)} {dmg}\n")

    for (sg_side, region, dmg) in merged_sg:
        log.write(f"SIEGE {sg_side.letter} {region} {dmg}\n")

    log.write(f"END TURN {day}\n")


def emit(player: Player, side: Side, *args):
    msg = " ".join(str(a) for a in args)
    player.send(msg)


def wa_result(log: TextIO, loser: Side) -> str:
    winner = "RIGHT_WIN" if loser is Side.LEFT else "LEFT_WIN"
    result = f"RESULT {winner} WA"
    log.write(f"{result}\n")
    return result


def draw_wa_result(log: TextIO) -> str:
    result = "RESULT DRAW WA"
    log.write(f"{result}\n")
    return result


def send_ready(player: Player, side: Side, m: MapData):
    emit(player, side, "READY", side.name_str)
    emit(player, side, m.N, m.K)
    emit(player, side, " ".join(map(str, m.x)))
    emit(player, side, " ".join(map(str, m.y)))
    emit(player, side, " ".join(map(str, m.strongholds)))
    for i in range(m.N):
        row = str(len(m.adj[i])) + ("" if not m.adj[i] else " " + " ".join(map(str, m.adj[i])))
        emit(player, side, row)


def send_start_turn(player: Player, side: Side, day: int):
    emit(player, side, "START", "TURN", day)


def send_result_block(player: Player, side: Side, st: GameState,
                      t_used_self: int, t_used_opp: int,
                      rb_self: ResultBlock, rb_opp: ResultBlock):
    u = side.value
    emit(player, side, "TURN", st.day)
    emit(player, side, "TIME", t_used_self, st.tokens[u], t_used_opp, st.tokens[1 - u])

    merged_rec, merged_up, merged_mv, merged_dmg, merged_sg = _merge_results(rb_self, rb_opp)

    emit(player, side, "UPGRADE", len(merged_up))
    for (region, up_side) in merged_up:
        emit(player, side, up_side.letter, region)

    emit(player, side, "TRAIN", len(merged_rec))
    if merged_rec:
        emit(player, side, " ".join(id_str(k) for k in merged_rec))

    emit(player, side, "MOVE", len(merged_mv))
    for (key, new_region) in merged_mv:
        emit(player, side, id_str(key), new_region)

    emit(player, side, "DAMAGE", len(merged_dmg))
    for (cause, key, dmg) in merged_dmg:
        emit(player, side, cause, id_str(key), dmg)

    emit(player, side, "SIEGE", len(merged_sg))
    for (sg_side, region, dmg) in merged_sg:
        emit(player, side, sg_side.letter, region, dmg)

    emit(player, side, "END")


def read_command_block(player: Player, side: Side,
                       timeout_s: float) -> Optional[List[str]]:
    deadline = time.monotonic() + timeout_s
    lines: List[str] = []
    seen_command = False

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        raw = player.readline(remaining)
        if raw is None:
            return None
        line = raw.rstrip("\n")

        if not seen_command:
            if line != "COMMAND":
                raise WaError(side, f"expected `COMMAND`, got: {line}")
            seen_command = True
        elif line == "END":
            return lines
        else:
            lines.append(line)


def read_command_blocks(players: List[Player],
                        budgets: Tuple[float, float]) -> Tuple[
                            Optional[List[str]], Optional[List[str]], int, int
                        ]:
    sides = (Side.LEFT, Side.RIGHT)
    blocks: List[Optional[List[str]]] = [None, None]
    elapsed_ms = [0, 0]
    errors: List[Optional[WaError]] = [None, None]
    start = time.monotonic()

    def _read(idx: int):
        try:
            blocks[idx] = read_command_block(players[idx], sides[idx], budgets[idx])
        except WaError as e:
            errors[idx] = e
        finally:
            elapsed_ms[idx] = round((time.monotonic() - start) * 1000)

    threads = [threading.Thread(target=_read, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for e in errors:
        if e is not None:
            raise e
    return blocks[0], blocks[1], elapsed_ms[0], elapsed_ms[1]


def run_game(m: MapData, players: List[Player], log: TextIO) -> str:
    st = init_state(m)

    write_map_block(log, m)
    log.flush()

    send_ready(players[0], Side.LEFT,  m)
    send_ready(players[1], Side.RIGHT, m)

    for side in (Side.LEFT, Side.RIGHT):
        raw = players[side.value].readline(HANDSHAKE_MS)
        if raw is None:
            return wa_result(log, side)
        line = raw.rstrip("\n")
        if line != "OK":
            return wa_result(log, side)

    for st.day in range(1, MAX_DAYS + 1):
        budget_l = (st.tokens[0] + 1) * (SOFT_CAP_MS / 1000.0)
        budget_r = (st.tokens[1] + 1) * (SOFT_CAP_MS / 1000.0)

        send_start_turn(players[0], Side.LEFT,  st.day)
        send_start_turn(players[1], Side.RIGHT, st.day)

        block_l, block_r, t_used_l, t_used_r = read_command_blocks(
            players, (budget_l, budget_r)
        )

        if block_l is None:
            return wa_result(log, Side.LEFT)
        if block_r is None:
            return wa_result(log, Side.RIGHT)

        deduct_tokens(st, 0, t_used_l)
        deduct_tokens(st, 1, t_used_r)

        write_turn_header(log, st.day)
        write_command_block(log, Side.LEFT,  block_l)
        write_command_block(log, Side.RIGHT, block_r)

        rb_l = ResultBlock()
        rb_r = ResultBlock()

        def run_phase1(side: Side, block, rb: ResultBlock):
            try:
                sub = parse_block(side, block)
                apply_upgrades(st, m, side, sub, rb)
                apply_moves(st, m, side, sub)
                n = apply_train_charge(st, side, sub)
                return (None, n)
            except WaError as e:
                return (e.msg, 0)

        bad_l, train_l = run_phase1(Side.LEFT,  block_l, rb_l)
        bad_r, train_r = run_phase1(Side.RIGHT, block_r, rb_r)

        if bad_l is not None and bad_r is not None:
            return draw_wa_result(log)
        if bad_l is not None:
            return wa_result(log, Side.LEFT)
        if bad_r is not None:
            return wa_result(log, Side.RIGHT)

        siege_damage: Dict[int, int] = {}
        apply_day_movement(st, m, rb_l, rb_r)
        spawn_trained(st, Side.LEFT,  train_l, rb_l)
        spawn_trained(st, Side.RIGHT, train_r, rb_r)
        apply_day_combat(st, rb_l, rb_r, siege_damage)
        apply_day_siege(st, rb_l, rb_r, siege_damage)
        apply_evening_work(st)
        apply_evening_upkeep(st, rb_l, rb_r)

        send_result_block(players[0], Side.LEFT,  st, t_used_l, t_used_r, rb_l, rb_r)
        send_result_block(players[1], Side.RIGHT, st, t_used_r, t_used_l, rb_r, rb_l)

        write_turn_result(log, st.day, t_used_l, t_used_r,
                          st.tokens[0], st.tokens[1], rb_l, rb_r)
        log.flush()

        if any_hq_destroyed(st):
            break

    left_hq  = hq_of(st, Side.LEFT)
    right_hq = hq_of(st, Side.RIGHT)
    left_alive  = left_hq  is not None
    right_alive = right_hq is not None
    left_hp  = left_hq.hp  if left_alive  else 0
    right_hp = right_hq.hp if right_alive else 0

    reason = "TURN_LIMIT" if (left_alive and right_alive) else "HQ_DESTROYED"

    left_score  = (left_alive,  left_hp)
    right_score = (right_alive, right_hp)
    if left_score > right_score:
        outcome = "LEFT_WIN"
    elif left_score < right_score:
        outcome = "RIGHT_WIN"
    else:
        outcome = "DRAW"

    result = f"RESULT {outcome} {reason}"
    log.write(f"{result}\n")
    return result




def read_settings() -> Tuple[TextIO, MapData, str, str]:
    parser = argparse.ArgumentParser(
        prog="testing-tool",
        description="Testing tool",
    )
    parser.add_argument("-c", "--config", type=str, help="config.ini file")
    parser.add_argument("-i", "--input",  type=str, help="Map input file")
    parser.add_argument("-l", "--log",    type=str, help="Log output file")
    parser.add_argument(
        "-s", "--stdio",
        nargs="?", const=True,
        type=lambda x: True if x is None else x.lower() == "true",
        default=False,
        help="Use stdin/stdout for input/log",
    )
    parser.add_argument("-a", "--exec1",  type=str, help="LEFT player command")
    parser.add_argument("-b", "--exec2",  type=str, help="RIGHT player command")
    parser.add_argument("--seed", type=int,
                        help="Map generator seed (random if omitted)")
    parser.add_argument("--NP", type=int,
                        help="Map generator NP (right-half region count). Total regions N = 2*NP + 1.")
    parser.add_argument("--KP", type=int,
                        help="Map generator KP (right-half mid stronghold count). Total strongholds K = 2*KP + 1.")

    args = parser.parse_args()

    input_file = args.input
    log_file   = args.log
    exec1      = args.exec1
    exec2      = args.exec2
    stdio      = args.stdio
    seed       = args.seed
    map_NP     = args.NP
    map_KP     = args.KP

    if args.config:
        try:
            f = open(args.config, "r")
        except FileNotFoundError:
            parser.print_help()
            print(f"\nError: Config file {args.config} not found.", file=sys.stderr)
            sys.exit(1)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                if   key == "INPUT" and input_file is None: input_file = value
                elif key == "LOG"   and log_file   is None: log_file   = value
                elif key == "EXEC1" and exec1      is None: exec1      = value
                elif key == "EXEC2" and exec2      is None: exec2      = value
                elif key == "SEED"  and seed       is None: seed       = int(value)
                elif key == "NP"    and map_NP     is None: map_NP     = int(value)
                elif key == "KP"    and map_KP     is None: map_KP     = int(value)
        f.close()

    if input_file:
        try:
            map_fh = open(input_file, "r")
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found.", file=sys.stderr)
            sys.exit(1)
        map_lines = [l for l in map_fh.read().splitlines() if l.strip()]
        map_fh.close()
    elif stdio:
        map_lines = [l for l in sys.stdin.read().splitlines() if l.strip()]
    else:
        if (map_NP is None) != (map_KP is None):
            parser.print_help()
            print(
                "\nError: --NP and --KP must be provided together.",
                file=sys.stderr,
            )
            sys.exit(1)
        if seed is None:
            seed = random.randint(0, (1 << 63) - 1)
            print(f"# Generated random seed: {seed}", file=sys.stderr)
        rng = XoShiro256(seed)
        try:
            if map_NP is None:
                map_lines = generate_map(rng)
            else:
                map_lines = generate_map(rng, map_NP, map_KP)
        except (ValueError, RuntimeError) as e:
            print(f"Error generating map: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        m = read_map(map_lines)
    except Exception as e:
        print(f"Error parsing map: {e}", file=sys.stderr)
        sys.exit(1)

    if not exec1:
        print("Error: LEFT player command not specified.", file=sys.stderr)
        sys.exit(1)
    if not exec2:
        print("Error: RIGHT player command not specified.", file=sys.stderr)
        sys.exit(1)

    if log_file is None:
        log_stream = sys.stdout
    else:
        try:
            log_stream = open(log_file, "w")
        except OSError as e:
            print(f"Error opening log file {log_file}: {e}", file=sys.stderr)
            sys.exit(1)

    return (log_stream, m, exec1, exec2)


def main():
    log_stream, m, exec1, exec2 = read_settings()

    e1_str = json.dumps(f"COMMAND: {exec1}", ensure_ascii=False)
    e2_str = json.dumps(f"COMMAND: {exec2}", ensure_ascii=False)
    log_stream.write(f'[LEFT {e1_str}]\n[RIGHT {e2_str}]\n')

    players = [
        Player(0, exec1, log_stream),
        Player(1, exec2, log_stream),
    ]

    result = "RESULT DRAW WA"
    try:
        result = run_game(m, players, log_stream)
    finally:
        for p in players:
            p.send("FINISH")
            p.join()
        log_stream.flush()
        if log_stream is not sys.stdout:
            log_stream.close()

    print(result)


if __name__ == "__main__":
    main()
