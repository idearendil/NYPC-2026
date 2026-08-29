#!/usr/bin/env python3
"""
전략 봇 v5 — 확장 우선 → 정비(업그레이드+병력) → 총공격

  1단계 [확장]  맵을 반으로 갈라 내 쪽 거점을 전부 먹는다. 이 단계에서는
                금화를 땅에만 쓴다 — 기지 500금 + 그 자리 일꾼 120금.
                기지 업그레이드는 하지 않고, 본부도 기지 EXPAND_HQ_AFTER개를
                지은 뒤에야 Lv2까지만 올린다. 시작 금화 750을 1일차에 본부
                업그레이드(600)로 써버리면 첫 개척자가 거점에 도착하고도
                기지 값 500이 모일 때까지 70일을 서 있게 된다.
                개척자가 거점에 도착해 기다리고 있으면 기지 값이 최우선이라
                훈련보다 먼저 빠져나간다. 그래야 500금이 실제로 모인다.
                다만 상대 대군이 올 때가 가까워지면(GUARD_RAMP_TURN) 최소
                수비 병력을 빠르게 올린다.

  2단계 [정비]  내 쪽 땅을 다 먹으면 업그레이드와 병력을 같이 키운다.
                최소 수비선까지는 무조건 훈련하고, 그 위로는 다음 업그레이드
                값(본부 Lv3 → 기지 Lv2/Lv3)을 묶어둔 뒤 남는 금화로 훈련한다.
                본부 Lv3이 우선인 이유는 하루 훈련 2명 + 전사 체력 6이라
                병력의 질과 양이 한 번에 오르기 때문이다.

  3단계 [공격]  병력이 목표치를 넘거나, 상대 대군을 한 번 막아냈거나
                (그 순간 상대 진영은 비어 있다) 총공격으로 넘어간다.
                본부 수비 인원(HQ_GARRISON)은 남기고 ATTACK_SIZE명씩 한
                덩어리로 내보낸다. 목표는 본부가 아니라 **상대 기지**다.
                확인된 기지 → 아직 안 훑은 상대 쪽 거점 → 마지막에 본부 순으로
                돌아다니며 부순다. 거점은 건물을 지을 수 있는 유일한 자리라
                상대 기지는 반드시 그중 하나에 있고, 상대도 본부에서 가까운
                거점부터 지으므로 '상대 본부에 가까운 거점'부터 훑는다.
                눈으로 확인한 빈 거점은 표시해 두고 두 번 가지 않는다.

  [수비]        어느 단계에서든 적이 내 땅에 들어오면 즉시 반응한다.
                · 병력은 평소 본부에 모아 둔다. 전방에 두면 상대 대군이 옆을
                  지나 본부로 직행할 때 되돌릴 수 없다(이동 명령은 도착 전까지
                  변경 불가).
                · 우리가 확실히 많으면(HUNT_RATIO) 뭉쳐서 잡으러 가고,
                  아니면 본부에 전부 모여 받아친다. 한 명씩 흘려보내면
                  각개격파당하므로 절대 나눠 보내지 않는다.
                · 큰 무리 근처(EVAC_HOPS) 건물의 일꾼만 빼고 먼 기지는 계속
                  일한다. 수입이 끊기면 식비 부족으로 굶어 죽는다.
                · 침공 중에는 업그레이드를 멈추고 금화를 전부 훈련에 쓴다.

정찰병은 쓰지 않는다. 시야는 내 건물과 병력이 만들어 주는 만큼만 쓰고,
적 건물은 마지막으로 본 위치를 기억해 둔다. 공격대가 거점을 훑는 진군
자체가 정찰이 되어, 가는 길에 기지가 보이면 즉시 그쪽으로 목표를 바꾼다.

400일이 지나면 본부 체력으로 승패가 갈리므로, 후반(LATE_TURN)에는 본부
레벨과 수리를 최우선으로 둔다.

금화는 심판이 알려주지 않으므로 명령 비용/수입/식비를 직접 정산한다.
훈련 당일 전투로 죽은 전사를 살아있다고 세면 금화가 어긋나 반칙이 되므로,
전사는 반드시 WARRIOR 목록으로만 만든다.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

# --- 규칙 상수 -------------------------------------------------------------
MAX_TURN = 400
START_GOLD = 750
START_WARRIORS = 3
MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2
HQ_MAX_LEVEL = 5
BASE_MAX_LEVEL = 3
HQ_HEAL_COST = 1000
BASE_HEAL_COST = 500
VISION = 2

# --- 전략 파라미터 ---------------------------------------------------------
MAX_SETTLERS = 3         # 동시에 내보내는 개척자 수(확장 속도)
HOME_MIN = 2             # 본부에 항상 두는 최소 인원
DEFENSE_RATIO = 2        # 보이는 적 1명당 이만큼 있으면 안전하다고 본다
GOLD_RESERVE = 60        # 식비용으로 항상 남겨둘 금화
UPGRADE_BUFFER = 120     # 업그레이드 후 남겨둘 금화
HQ_SAVE_RATIO = 0.6      # 본부 업그레이드 비용의 이 비율은 훈련에 안 쓴다
HQ_SAVE_TURN = 60        # 이 턴부터 본부 업그레이드 자금을 모은다
INCURSION_MIN = 5        # 내 땅에서 이만큼 보이면 '침공'으로 보고 총수비 태세
BLOB_EVAC = 15           # 이 규모 이상 적 무리가 와야 근처 일꾼까지 뺀다
HUNT_RATIO = 1.8         # 적 무리보다 이 배수 많으면 뭉쳐서 잡으러 간다
EVAC_HOPS = 2            # 적 무리에서 이 거리 안의 건물 일꾼만 뺀다
HQ_RUSH_LEVEL = 3        # 확장이 끝나면 본부를 최소 여기까지는 먼저 올린다
EXPAND_HQ_LEVEL = 2      # 확장 중에는 본부를 여기까지만 올린다(전사 체력 5, 포탑 2)
EXPAND_HQ_AFTER = 2      # 기지를 이만큼 지은 뒤에야 본부에 손을 댄다
                         # (시작 금화 750을 본부에 쓰면 첫 기지가 70일씩 밀린다)
FOOD_MARGIN = 20         # 수입-식비가 이보다 빠듯하면 훈련을 멈춘다
MIL_TURN = 110           # 이 턴부터는 병력이 우선(훈련비를 먼저 떼고 건설한다)
ARMY_BASE = 24           # 수비 단계에서 유지할 기본 예비대 규모
GUARD_WHILE_EXPAND = 3   # 확장 초반 최소 수비 병력
GUARD_GROW = 3           # 확장 후반 수비 병력을 몇 턴마다 1명씩 늘릴지
GUARD_RAMP_TURN = 90     # 이 턴부터 수비 병력을 본격적으로 올린다
UPGRADE_SAVE = 700       # 2단계에서 업그레이드용으로 묶어둘 금화
EXPAND_DEADLINE = 200    # 이 턴까지 확장이 안 끝나면 그냥 다음 단계로 넘어간다
ATTACK_SIZE = 20         # 한 번에 내보낼 공격대 인원
ATTACK_MIN = 12          # 후반 강행 시 최소 인원
HQ_GARRISON = 8          # 공격대를 내보내도 본부에 남겨둘 인원
ATTACK_TURN_LIMIT = 260  # 확장이 안 끝나도 이 턴부터는 공격을 시작한다
LATE_TURN = 320          # 이 턴부터 본부 레벨/체력 최대화 우선
NO_EXPAND_TURN = 300     # 이 턴 이후로는 새 기지를 짓지 않는다


class HqLevelEntry(NamedTuple):
    upgrade_cost: int
    warrior_hp: int
    hp: int
    turret: int
    train_cap: int
    work_cap: int


class BaseLevelEntry(NamedTuple):
    cost: int
    hp: int
    turret: int
    work_cap: int


HQ_LEVELS: tuple[HqLevelEntry, ...] = (
    HqLevelEntry(0,     0, 0,  0, 0, 0),
    HqLevelEntry(0,     4, 10, 1, 1, 1),
    HqLevelEntry(600,   5, 15, 2, 1, 2),
    HqLevelEntry(1000,  6, 20, 2, 2, 3),
    HqLevelEntry(2000,  7, 25, 3, 2, 4),
    HqLevelEntry(3000,  8, 30, 3, 3, 5),
)
BASE_LEVELS: tuple[BaseLevelEntry, ...] = (
    BaseLevelEntry(0,   0,  0, 0),
    BaseLevelEntry(500, 6,  1, 1),
    BaseLevelEntry(550, 12, 1, 2),
    BaseLevelEntry(600, 18, 2, 3),
)


class Side(Enum):
    LEFT = "A"
    RIGHT = "B"

    @property
    def opposite(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @classmethod
    def from_word(cls, w: str) -> "Side":
        return cls.LEFT if w == "LEFT" else cls.RIGHT

    @classmethod
    def from_char(cls, c: str) -> "Side":
        return cls.LEFT if c == "A" else cls.RIGHT


class BType(Enum):
    HQ = "HQ"
    BASE = "BASE"


# 역할
R_SETTLE = "SETTLE"      # 거점으로 가서 기지를 짓는 중
R_WORK = "WORK"          # 건물에 붙어 금화를 버는 중
R_DEFEND = "DEFEND"      # 위협받는 건물로 달려가는 중
R_ARMY = "ARMY"          # 집결지에서 대기(예비대)
R_ATTACK = "ATTACK"      # 상대 건물로 진격 중


@dataclass
class Warrior:
    num: int
    region: int
    hp: int
    moving: bool = False
    target: int = -1
    role: str = R_ARMY
    assign: int = -1


@dataclass
class Building:
    region: int
    side: Side
    type: BType
    level: int = 1
    hp: int = 10

    def max_hp(self) -> int:
        return HQ_LEVELS[self.level].hp if self.type is BType.HQ else BASE_LEVELS[self.level].hp

    def work_cap(self) -> int:
        return (HQ_LEVELS[self.level].work_cap if self.type is BType.HQ
                else BASE_LEVELS[self.level].work_cap)

    def max_level(self) -> int:
        return HQ_MAX_LEVEL if self.type is BType.HQ else BASE_MAX_LEVEL

    def next_cost(self) -> int:
        if self.level >= self.max_level():
            return HQ_HEAL_COST if self.type is BType.HQ else BASE_HEAL_COST
        if self.type is BType.HQ:
            return HQ_LEVELS[self.level + 1].upgrade_cost
        return BASE_LEVELS[self.level + 1].cost


@dataclass
class GameMap:
    N: int = 0
    K: int = 0
    x: list[int] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    strongholds: list[int] = field(default_factory=list)
    adj: list[list[int]] = field(default_factory=list)
    my_side: Side = Side.LEFT
    my_hq: int = 0
    opp_hq: int = 0

    def hq_of(self, s: Side) -> int:
        return 0 if s is Side.LEFT else self.N - 1


@dataclass
class Actions:
    train_n: int = 0
    moves: list[tuple[int, int]] = field(default_factory=list)
    upgrades: list[int] = field(default_factory=list)


SIDE_CHAR = "A"


# ---------------------------------------------------------------------------
# 거리(필요할 때 다익스트라 + BFS, 결과는 캐시)
# ---------------------------------------------------------------------------
class Geo:
    def __init__(self, M: GameMap) -> None:
        self.M = M
        self.wt: list[list[float]] = [
            [math.ceil(math.hypot(M.x[u] - M.x[v], M.y[u] - M.y[v])) for v in M.adj[u]]
            for u in range(M.N)
        ]
        self._d: dict[int, list[float]] = {}
        self._h: dict[int, list[int]] = {}
        self.vis: list[list[int]] = [self._ball(r, VISION) for r in range(M.N)]

    def _ball(self, s: int, radius: int) -> list[int]:
        seen = {s}
        frontier = [s]
        for _ in range(radius):
            nxt = []
            for u in frontier:
                for v in self.M.adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
        return sorted(seen)

    def dfrom(self, s: int) -> list[float]:
        d = self._d.get(s)
        if d is not None:
            return d
        INF = math.inf
        d = [INF] * self.M.N
        d[s] = 0.0
        pq = [(0.0, s)]
        adj, wt = self.M.adj, self.wt
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            wu = wt[u]
            for i, v in enumerate(adj[u]):
                nd = du + wu[i]
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._d[s] = d
        return d

    def dist(self, u: int, v: int) -> float:
        if u in self._d:
            return self._d[u][v]
        if v in self._d:
            return self._d[v][u]
        return self.dfrom(u)[v]

    def hfrom(self, s: int) -> list[int]:
        h = self._h.get(s)
        if h is not None:
            return h
        BIG = 1 << 29
        h = [BIG] * self.M.N
        h[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            hu = h[u] + 1
            for v in self.M.adj[u]:
                if h[v] > hu:
                    h[v] = hu
                    q.append(v)
        self._h[s] = h
        return h

    def hop(self, u: int, v: int) -> int:
        if u in self._h:
            return self._h[u][v]
        if v in self._h:
            return self._h[v][u]
        return self.hfrom(u)[v]


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------
def readln() -> str:
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.rstrip("\n")


def read_tokens() -> list[str]:
    t = readln().split()
    while not t:
        t = readln().split()
    return t


def parse_init() -> GameMap:
    global SIDE_CHAR
    M = GameMap()

    t = read_tokens()
    assert t and t[0] == "READY"
    M.my_side = Side.from_word(t[1])
    SIDE_CHAR = M.my_side.value

    t = read_tokens()
    M.N, M.K = int(t[0]), int(t[1])

    xs: list[int] = []
    while len(xs) < M.N:
        xs.extend(int(v) for v in read_tokens())
    ys: list[int] = []
    while len(ys) < M.N:
        ys.extend(int(v) for v in read_tokens())
    M.x, M.y = xs, ys

    sh: list[int] = []
    while len(sh) < M.K:
        sh.extend(int(v) for v in read_tokens())
    M.strongholds = sorted(sh)

    M.adj = [[] for _ in range(M.N)]
    for r in range(M.N):
        t = read_tokens()
        deg = int(t[0])
        vals = [int(v) for v in t[1:]]
        while len(vals) < deg:
            vals.extend(int(v) for v in read_tokens())
        M.adj[r] = sorted(vals[:deg])

    M.my_hq = M.hq_of(M.my_side)
    M.opp_hq = M.hq_of(M.my_side.opposite)

    print("OK", flush=True)
    return M


def read_turn_start() -> int | None:
    line = readln()
    if line == "FINISH":
        return None
    t = line.split()
    if not t or t[0] != "START":
        return None
    return int(t[2])


def emit(a: Actions) -> None:
    out: list[str] = ["COMMAND"]
    for num, target in a.moves:
        out.append(f"MOVE {SIDE_CHAR}{num} {target}")
    for r in a.upgrades:
        out.append(f"UPGRADE {r}")
    if a.train_n > 0:
        out.append(f"TRAIN {a.train_n}")
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 봇
# ---------------------------------------------------------------------------
class Bot:
    def __init__(self, M: GameMap) -> None:
        self.M = M
        self.geo = Geo(M)
        self.turn = 0
        self.gold = START_GOLD

        self.warriors: dict[int, Warrior] = {
            i: Warrior(i, M.my_hq, HQ_LEVELS[1].warrior_hp)
            for i in range(1, START_WARRIORS + 1)
        }
        self.my_b: dict[int, Building] = {
            M.my_hq: Building(M.my_hq, M.my_side, BType.HQ, 1, HQ_LEVELS[1].hp)
        }
        self.enemy_b: dict[int, Building] = {
            M.opp_hq: Building(M.opp_hq, M.my_side.opposite, BType.HQ, 1, HQ_LEVELS[1].hp)
        }
        self.enemy_w: dict[int, int] = {}          # 지금 보이는 적 전사 수
        self.enemy_mem: dict[int, tuple[int, int]] = {}   # 구역 -> (인원, 관측 턴)

        g = self.geo
        # 내 쪽 거점(내 본부가 더 가까운 거점)을 가까운 순서로 = 확장 목표
        self.claim: list[int] = sorted(
            (r for r in M.strongholds if g.dist(M.my_hq, r) <= g.dist(M.opp_hq, r)),
            key=lambda r: (g.dist(M.my_hq, r), r))
        # 상대 쪽 거점 = 공격 목표 후보(가까운 순)
        mine = set(self.claim)
        self.enemy_sh: list[int] = sorted(
            (r for r in M.strongholds if r not in mine),
            key=lambda r: (g.dist(M.my_hq, r), r))
        self.attacking = False        # 3단계(총공격) 돌입 여부
        self.max_blob = 0             # 지금까지 본 적 최대 무리 규모
        self.cleared: set[int] = set()   # 눈으로 확인한 '빈' 상대 거점
        self.sh_set: set[int] = set(M.strongholds)

    def read_result(self, submitted: Actions) -> None:
        M = self.M

        # 이번 턴에 낸 명령의 비용을 먼저 반영한다(금화는 심판이 알려주지 않는다)
        for region in submitted.upgrades:
            b = self.my_b.get(region)
            self.gold -= BASE_LEVELS[1].cost if b is None else b.next_cost()
        for num, target in submitted.moves:
            self.gold -= 0 if target in self.my_b else MOVE_COST
            w = self.warriors.get(num)
            if w is not None:
                w.moving = True
                w.target = target
        self.gold -= TRAIN_COST * submitted.train_n
        if self.gold < 0:
            self.gold = 0

        line = readln()
        if line == "FINISH":
            sys.exit(0)
        t = line.split()
        if not t or t[0] != "TURN":
            sys.exit(0)

        read_tokens()                      # TIME ...
        n = int(read_tokens()[1])          # UPGRADE
        for _ in range(n):
            read_tokens()
        n = int(read_tokens()[1])          # TRAIN (전사 생성은 WARRIOR 목록으로만 한다)
        if n > 0:
            ids = read_tokens()
            while len(ids) < n:
                ids.extend(read_tokens())
        n = int(read_tokens()[1])          # MOVE
        for _ in range(n):
            read_tokens()
        n = int(read_tokens()[1])          # DAMAGE
        for _ in range(n):
            read_tokens()
        n = int(read_tokens()[1])          # SIEGE
        for _ in range(n):
            read_tokens()

        n = int(read_tokens()[1])          # WARRIOR
        seen_mine: dict[int, tuple[int, int]] = {}
        enemy_now: dict[int, int] = {}
        for _ in range(n):
            r = read_tokens()
            tag, region, hp = r[0], int(r[1]), int(r[2])
            if tag[0] == SIDE_CHAR:
                seen_mine[int(tag[1:])] = (region, hp)
            else:
                enemy_now[region] = enemy_now.get(region, 0) + 1

        n = int(read_tokens()[1])          # BUILDING
        my_now: dict[int, Building] = {}
        eb_now: dict[int, Building] = {}
        for _ in range(n):
            r = read_tokens()
            side = Side.from_char(r[0][0])
            region, kind, level, hp = int(r[1]), r[2], int(r[3]), int(r[4])
            b = Building(region, side, BType(kind), level, hp)
            (my_now if side is M.my_side else eb_now)[region] = b

        readln()                           # END

        # 아군은 보고 기준으로 재구성(역할/이동 상태만 승계)
        new_w: dict[int, Warrior] = {}
        for num, (region, hp) in seen_mine.items():
            w = self.warriors.get(num) or Warrior(num, region, hp)
            w.region, w.hp = region, hp
            if w.moving and w.region == w.target:
                w.moving = False
                w.target = -1
            new_w[num] = w
        self.warriors = new_w
        self.my_b = my_now

        # 적 건물: 보이는 구역은 갱신, 안 보이면 마지막 관측을 유지
        vis: set[int] = set()
        for w in self.warriors.values():
            vis.update(self.geo.vis[w.region])
        for r in self.my_b:
            vis.update(self.geo.vis[r])
        for r in list(self.enemy_b):
            if r in vis and r not in eb_now:
                del self.enemy_b[r]
        self.enemy_b.update(eb_now)
        self.enemy_w = enemy_now
        for r in vis:                       # 보이는데 없으면 기억에서 지운다
            if r in self.enemy_mem and r not in enemy_now:
                del self.enemy_mem[r]
            if r in self.sh_set:            # 상대 거점의 건물 유무를 기록해 둔다
                if r in eb_now or r in self.my_b:
                    self.cleared.discard(r)
                else:
                    self.cleared.add(r)
        for r, c in enemy_now.items():
            self.enemy_mem[r] = (c, self.turn)

        # 수입 → 식비
        income = 0
        for region, b in self.my_b.items():
            cnt = sum(1 for w in self.warriors.values() if w.region == region)
            income += WORK_INCOME * min(cnt, b.work_cap())
        self.gold += income
        alive = len(self.warriors)
        self.gold -= UPKEEP_PER_WARRIOR * min(alive, self.gold // UPKEEP_PER_WARRIOR)

    # ---- 보조 -------------------------------------------------------------
    def enemy_count(self, region: int) -> int:
        """그 구역의 적 전사 수(지금 보이면 실측, 아니면 최근 기억)."""
        if region in self.enemy_w:
            return self.enemy_w[region]
        rec = self.enemy_mem.get(region)
        if rec is None:
            return 0
        cnt, t = rec
        return cnt if self.turn - t <= 20 else 0

    def attack_target(self, from_region: int) -> int:
        """부술 순서: 확인된 적 기지 → 아직 안 훑은 상대 쪽 거점 → 적 본부.

        상대 기지를 먼저 부수는 게 목적이므로, 아는 기지가 없어도 본부로
        직행하지 않고 상대 쪽 '거점'을 가까운 순서로 훑는다. 거점은 건물을
        지을 수 있는 유일한 자리라, 상대 기지는 반드시 그중 하나에 있다.
        빈 거점은 눈으로 확인하는 즉시 cleared로 표시해 두 번 가지 않는다
        (표시가 없으면 도착한 자리에 그대로 눌러앉는 일이 생긴다).
        """
        M, g = self.M, self.geo
        bases = [b for b in self.enemy_b.values() if b.type is BType.BASE]
        if bases:
            return min(bases, key=lambda b: (self.enemy_count(b.region),
                                             g.dist(from_region, b.region),
                                             b.region)).region
        todo = [r for r in self.enemy_sh
                if r not in self.cleared and r != from_region]
        if todo:
            # 상대도 우리처럼 본부에서 가까운 거점부터 짓는다. 그래서 '상대
            # 본부에 가까운 거점'일수록 기지가 있을 확률이 높다. 그쪽부터
            # 훑으면 헛걸음이 줄고, 가는 방향도 자연히 상대 진영 쪽이 된다.
            return min(todo, key=lambda r: (g.dist(M.opp_hq, r),
                                            g.dist(from_region, r), r))
        return M.opp_hq

    # ---- 하루 결정 --------------------------------------------------------
    def decide(self, turn: int) -> Actions:
        self.turn = turn
        M, g = self.M, self.geo
        a = Actions()
        gold = self.gold
        ws = list(self.warriors.values())
        hq = self.my_b.get(M.my_hq)

        if not ws:
            if hq is not None and gold >= TRAIN_COST + GOLD_RESERVE:
                a.train_n = min(HQ_LEVELS[hq.level].train_cap, 1)
            return a

        # --- 0) 역할 정리 --------------------------------------------------
        for w in ws:
            if w.role == R_SETTLE:
                if w.assign in self.my_b:
                    w.role = R_WORK                     # 다 지었으면 그 자리 일꾼
                elif w.assign in self.enemy_b or self.enemy_count(w.assign) > 0:
                    w.role, w.assign = R_ARMY, -1       # 적이 선점 → 포기
            elif w.role == R_WORK and w.assign not in self.my_b:
                w.role, w.assign = R_ARMY, -1
            elif w.role == R_DEFEND:
                if w.assign not in self.my_b:
                    w.role, w.assign = R_ARMY, -1
            elif w.role == R_ATTACK and not w.moving:
                if w.assign not in self.enemy_b or w.region == w.assign:
                    w.assign = self.attack_target(w.region)

        # --- 1) 위협 판정 ---------------------------------------------------
        # 내 건물 시야 안의 적 수. 그 건물에 있는 병력으로 감당이 되면 위협 아님.
        threats: list[tuple[int, int, int]] = []   # (우선순위, 구역, 부족 인원)
        for region, b in self.my_b.items():
            near = sum(self.enemy_w.get(r, 0) for r in g.vis[region])
            if near == 0:
                continue
            here = sum(1 for w in ws if w.region == region)
            need = near * DEFENSE_RATIO + 1 - here
            if need > 0:
                threats.append((0 if b.type is BType.HQ else 1, region, need))
        threats.sort()

        threatened = {t[1] for t in threats}
        for w in ws:                       # 위협이 사라진 곳의 수비병은 예비대로
            if w.role == R_DEFEND and w.assign not in threatened and not w.moving:
                w.role, w.assign = R_ARMY, -1

        hq_enemies = sum(self.enemy_w.get(r, 0) for r in g.vis[M.my_hq])
        home_force = sum(1 for w in ws if w.region == M.my_hq)
        emergency = hq is not None and (hq.hp * 2 < hq.max_hp()
                                        or hq_enemies * DEFENSE_RATIO + 1 > home_force)

        # 내 땅에 들어온 적의 규모. 한 구역에 뭉쳐 있는 최대 인원이 곧 한 번에
        # 상대해야 할 병력이다.
        near_mine: set[int] = set()
        for region in self.my_b:
            near_mine.update(g.vis[region])
        incursion = sum(self.enemy_w.get(r, 0) for r in near_mine)
        blob = max((self.enemy_w.get(r, 0) for r in near_mine), default=0)
        serious = incursion >= INCURSION_MIN or emergency
        self.max_blob = max(self.max_blob, blob)

        # --- 총수비 태세 -----------------------------------------------------
        # 병력을 한 명씩 흘려보내면 그대로 각개격파당한다. 지킬 수 있는 곳
        # 하나를 정해 전부 그리로 모으고, 못 지킬 건물은 일꾼을 빼서 살린다.
        defense_point = -1
        if serious:
            # 방어 거점은 본부로 고정한다. 이유는 두 가지다.
            #  * 상대 대군의 최종 목표는 결국 본부다. 앞에서 막다 밀리면
            #    다시 뒤로 모이라는 명령을 내려도 이미 이동 중이라 안 먹는다
            #    (이동 명령은 도착 전까지 못 바꾼다) → 본부가 텅 빈 채로 뚫린다.
            #  * 본부는 포탑이 가장 세고 일꾼도 상주해 가장 유리한 전장이다.
            # 앞 기지는 그 자리 인원으로 감당되면 그대로 두고, 아니면 내준다.
            defense_point = M.my_hq
            blob_region0 = max(near_mine, key=lambda r: self.enemy_w.get(r, 0),
                               default=M.my_hq)
            mobile_all = sum(1 for w in ws
                             if w.role in (R_ARMY, R_DEFEND, R_ATTACK))
            # 다만 우리가 확실히 많으면 웅크리지 않고 뭉쳐서 잡으러 간다.
            # (본부에 모여 있으니 다 같이 출발해 같이 도착한다)
            if (not emergency and blob >= 3 and blob_region0 != M.my_hq
                    and mobile_all >= blob * HUNT_RATIO):
                defense_point = blob_region0

            blob_region = max(near_mine, key=lambda r: self.enemy_w.get(r, 0),
                              default=M.my_hq)
            for w in ws:
                if w.moving:
                    continue
                if w.role in (R_ARMY, R_DEFEND, R_ATTACK):
                    w.role, w.assign = R_DEFEND, defense_point
                elif w.role == R_WORK and w.assign != defense_point:
                    # 큰 무리가 근처까지 들어온 건물의 일꾼은 빼서 방어에 합류시킨다
                    # (기지 하나 내주는 것보다 각개격파당하는 쪽이 훨씬 나쁘다)
                    if (w.assign in threatened
                            or (blob >= BLOB_EVAC
                                and g.hop(w.assign, blob_region) <= EVAC_HOPS)):
                        w.role, w.assign = R_DEFEND, defense_point

        # --- 2) 단계 판정 -----------------------------------------------------
        #   1단계 확장   : 내 쪽 거점을 전부 먹을 때까지. 업그레이드는 뒤로 미룬다.
        #   2단계 정비   : 땅을 다 먹으면 업그레이드와 병력을 같이 키운다.
        #   3단계 총공격 : 병력이 갖춰지면 상대 진영으로 나간다.
        bases_now = [b for b in self.my_b.values() if b.type is BType.BASE]
        remaining = [r for r in self.claim
                     if r not in self.my_b and r not in self.enemy_b]
        if turn >= NO_EXPAND_TURN:
            remaining = []
        expansion_done = not remaining or turn >= EXPAND_DEADLINE

        mobile_now = sum(1 for w in ws if w.role in (R_ARMY, R_DEFEND, R_ATTACK))
        army_target = max(ARMY_BASE, self.max_blob)
        if (turn >= ATTACK_TURN_LIMIT
                or mobile_now >= ATTACK_SIZE * 2
                or (expansion_done and mobile_now >= army_target
                    and hq is not None and hq.level >= HQ_RUSH_LEVEL)
                or (self.max_blob >= INCURSION_MIN and not serious
                    and mobile_now >= ATTACK_SIZE)):
            # 마지막 조건: 상대 대군을 막아냈다면 지금 상대 진영은 비어 있다.
            self.attacking = True

        # --- 3) 인원 배치 ----------------------------------------------------
        free = [w for w in ws if w.role == R_ARMY and not w.moving]

        def take_nearest(region: int) -> Warrior | None:
            if not free:
                return None
            w = min(free, key=lambda w: (g.dist(w.region, region), w.num))
            free.remove(w)
            return w

        # (a) 본부 일자리 먼저 — 본진이 비면 초반 러시에 그대로 뚫린다
        if hq is not None:
            have = sum(1 for w in ws if w.role == R_WORK and w.assign == M.my_hq)
            # 첫 기지를 세우기 전에는 본부에 1명만 두고 나머지를 개척에 쓴다
            home_need = 1 if not bases_now else max(hq.work_cap(), HOME_MIN)
            for _ in range(home_need - have):
                w = take_nearest(M.my_hq)
                if w is None:
                    break
                w.role, w.assign = R_WORK, M.my_hq

        # (b) 수비: 총수비 태세면 남은 예비대도 전부 방어 거점으로
        if serious and defense_point >= 0:
            for w in list(free):
                free.remove(w)
                w.role, w.assign = R_DEFEND, defense_point
        else:
            for _, region, need in threats:
                for _ in range(need):
                    w = take_nearest(region)
                    if w is None:
                        break
                    w.role, w.assign = R_DEFEND, region

        # (c) 개척: 내 쪽 거점을 가까운 순서로, 동시에 MAX_SETTLERS명까지
        settling = {w.assign for w in ws if w.role == R_SETTLE}
        if not serious:
            max_settlers = min(MAX_SETTLERS, 1 + len(self.my_b))
            for r in remaining:
                if len(settling) >= max_settlers:
                    break
                if r in settling:
                    continue
                # 전원이 개척을 나가면 수입이 0이 되어 식비로 굶어 죽는다.
                # 건물을 지킬(=일할) 인원은 반드시 남긴다.
                if sum(1 for w in ws if w.role != R_SETTLE) <= 1:
                    break
                if len(free) <= 1 and len(ws) <= 3:
                    break
                w = take_nearest(r)
                if w is None:
                    break
                w.role, w.assign = R_SETTLE, r
                settling.add(r)

        # (d) 나머지 건물 일자리 채우기
        for region in sorted(self.my_b):
            if region == M.my_hq:
                continue
            b = self.my_b[region]
            have = sum(1 for w in ws if w.role == R_WORK and w.assign == region)
            for _ in range(b.work_cap() - have):
                w = take_nearest(region)
                if w is None:
                    break
                w.role, w.assign = R_WORK, region

        # 수입이 0이면(건물에 사람이 아무도 없으면) 굶어 죽는다.
        # 가장 가까운 개척자를 불러 세워 일부터 시킨다.
        if self.my_b and not any(w.region in self.my_b for w in ws):
            cand = [w for w in ws if not w.moving] or ws
            w = min(cand, key=lambda w: (min(g.dist(w.region, r) for r in self.my_b), w.num))
            home = min(self.my_b, key=lambda r: (g.dist(w.region, r), r))
            w.role, w.assign = R_WORK, home

        # --- 4) 집결지 -------------------------------------------------------
        # 수비 단계에서는 본부에 모은다. 전방 기지에 모아두면 상대 대군이
        # 그 옆을 지나 본부로 직행할 때 되돌아올 시간이 없다(시야는 2홉뿐).
        # 총공격 단계로 넘어가야 전방 기지로 전진 배치한다.
        # 집결지는 항상 본부. 전방에 모아두면 상대 대군이 옆을 지나 본부로
        # 직행할 때 되돌릴 수가 없다(이동 중에는 명령 변경 불가).
        rally = M.my_hq
        if serious and defense_point >= 0:
            rally = defense_point

        # --- 5) 총공격: 집결지에 ATTACK_SIZE명이 모이면 한 덩어리로 출격 -------
        if self.attacking and not serious:
            size = ATTACK_SIZE if turn < ATTACK_TURN_LIMIT else ATTACK_MIN
            ready = [w for w in ws
                     if w.role == R_ARMY and not w.moving and w.region == rally]
            # 본부를 지킬 인원은 반드시 남기고 내보낸다.
            # 상대 대군을 본 적이 있다면 그만큼은 집에 남겨둔다(원정 중에
            # 본진이 비면 그대로 진다).
            keep = HQ_GARRISON
            if len(ready) >= size + keep and gold >= MOVE_COST * size:
                ready.sort(key=lambda w: (-w.hp, w.num))
                tgt = self.attack_target(rally)
                for w in ready[:size]:
                    w.role, w.assign = R_ATTACK, tgt

        # --- 6) 이동 명령 -----------------------------------------------------
        def order(w: Warrior, dest: int) -> bool:
            nonlocal gold
            if w.moving or w.region == dest:
                return False
            if self.enemy_w.get(w.region, 0) > 0:
                return False        # 교전 중이면 어차피 못 움직인다
            cost = 0 if dest in self.my_b else MOVE_COST
            if gold < cost:
                return False
            gold -= cost
            a.moves.append((w.num, dest))
            w.moving, w.target = True, dest
            return True

        for w in ws:
            if w.moving:
                continue
            if w.role in (R_SETTLE, R_WORK, R_DEFEND, R_ATTACK) and w.assign >= 0:
                order(w, w.assign)
            elif w.role == R_ARMY:
                order(w, rally)

        # --- 7) 건설 / 업그레이드 / 수리 ---------------------------------------
        # 90일이 지나면 상대 대군이 올 때가 됐다. 그때부터는 훈련비를 먼저 떼고
        # 남는 금화로만 건물을 올린다(확장만 하다 병력이 비면 그대로 뚫린다).
        military = (self.attacking or serious
                    or (expansion_done and turn >= MIL_TURN))
        waiting_settler = any(w.role == R_SETTLE and not w.moving
                              and w.region == w.assign and w.assign not in self.my_b
                              for w in ws)
        train_cap_now = HQ_LEVELS[hq.level].train_cap if hq is not None else 0
        build_buffer = UPGRADE_BUFFER + (TRAIN_COST * train_cap_now if military else 0)
        done: set[int] = set()

        # (a) 새 기지
        for w in ws:
            if (w.role == R_SETTLE and not w.moving and w.region == w.assign
                    and w.assign not in self.my_b and w.assign not in done
                    and self.enemy_w.get(w.assign, 0) == 0
                    and gold >= BASE_LEVELS[1].cost + GOLD_RESERVE):
                a.upgrades.append(w.assign)
                done.add(w.assign)
                gold -= BASE_LEVELS[1].cost

        occupied = {r for r in self.my_b if any(w.region == r for w in ws)}

        def can_build(region: int) -> bool:
            return (region in occupied and region not in done
                    and self.enemy_w.get(region, 0) == 0)

        # (b) 얻어맞은 건물 복구 — 만렙이면 수리, 아니면 업그레이드가 곧 완충
        for region, b in sorted(self.my_b.items()):
            if not can_build(region) or b.hp * 2 > b.max_hp():
                continue
            if b.level < b.max_level():
                continue
            cost = b.next_cost()
            if gold >= cost + GOLD_RESERVE:
                a.upgrades.append(region)
                done.add(region)
                gold -= cost

        def try_hq(buffer: int) -> None:
            nonlocal gold
            if hq is None or not can_build(M.my_hq):
                return
            if hq.level >= HQ_MAX_LEVEL:
                if hq.hp >= hq.max_hp() or turn < LATE_TURN:
                    return
            cost = hq.next_cost()
            if gold >= cost + buffer:
                a.upgrades.append(M.my_hq)
                done.add(M.my_hq)
                gold -= cost

        def try_bases() -> None:
            nonlocal gold
            cand = [b for r, b in self.my_b.items()
                    if b.type is BType.BASE and can_build(r) and b.level < BASE_MAX_LEVEL]
            cand.sort(key=lambda b: (b.next_cost(), b.region))
            for b in cand:
                cost = b.next_cost()
                if gold >= cost + build_buffer:
                    a.upgrades.append(b.region)
                    done.add(b.region)
                    gold -= cost

        # (c) 후반에는 본부 체력이 곧 승패다 → 본부 우선.
        #     침공 중에는 업그레이드에 금화를 묶지 않고 병력으로 돌린다.
        if serious:
            pass                       # 침공 중에는 금화를 전부 병력으로
        elif not expansion_done:
            # 1단계: 금화는 땅에 쓴다(기지 500 + 그 자리 일꾼 120).
            # '기지' 업그레이드는 확장이 끝난 뒤로 미루지만, 본부만은 Lv3까지
            # 올려둔다. 본부 레벨이 곧 전사 체력(4→6)과 하루 훈련 수(1→2)라
            # 여기서 밀리면 나중에 병력으로 회복이 안 된다.
            if (hq is not None and hq.level < EXPAND_HQ_LEVEL
                    and len(bases_now) >= EXPAND_HQ_AFTER and not waiting_settler):
                try_hq(UPGRADE_BUFFER)
        elif turn >= LATE_TURN:
            try_hq(0)                  # 400일 판정이 본부 체력이다
            try_bases()
        elif hq is not None and hq.level < HQ_RUSH_LEVEL:
            # 2단계 첫 투자: 본부 Lv3(하루 2명 훈련 + 전사 체력 6)
            try_hq(UPGRADE_BUFFER)
            try_bases()
        else:
            try_bases()
            try_hq(build_buffer)

        # --- 8) 훈련 ----------------------------------------------------------
        if hq is not None:
            work_slots = sum(b.work_cap() for b in self.my_b.values())
            workers = sum(1 for w in ws if w.role in (R_WORK, R_SETTLE))
            mobile = sum(1 for w in ws if w.role in (R_ARMY, R_DEFEND, R_ATTACK))
            if serious or workers < work_slots:
                # 위험할 때 / 빈 일자리를 채울 때는 무조건 뽑는다(최고 투자)
                reserve = GOLD_RESERVE
            elif not expansion_done:
                # 1단계: 다음 기지 값(500)은 항상 남겨둔다. 확장이 먼저다.
                # 다만 최소한의 수비 병력은 확보해 둔다.
                # 확장기의 최소 수비선. 초반에는 아주 작게 잡아 금화를 땅에
                # 몰아주고, 상대 대군이 올 때가 가까워지면 빠르게 올린다.
                guard = GUARD_WHILE_EXPAND
                if turn >= GUARD_RAMP_TURN:
                    guard += (turn - GUARD_RAMP_TURN) // GUARD_GROW
                guard = max(guard, self.max_blob)
                if waiting_settler:
                    # 개척자가 거점에 도착해 기다리는 중이면 기지 값이 최우선.
                    # (훈련이 먼저 빼가면 500금이 영영 안 모인다)
                    reserve = GOLD_RESERVE + BASE_LEVELS[1].cost
                elif mobile < guard:
                    reserve = GOLD_RESERVE
                elif hq.level < EXPAND_HQ_LEVEL and len(bases_now) >= EXPAND_HQ_AFTER:
                    reserve = GOLD_RESERVE + hq.next_cost() + UPGRADE_BUFFER
                else:
                    reserve = GOLD_RESERVE + BASE_LEVELS[1].cost
            elif self.attacking:
                # 3단계: 수비 병력 + 공격대 한 무리를 채울 때까지는 훈련,
                #        그 이상 여유가 생기면 업그레이드에 양보한다.
                if mobile < army_target + ATTACK_SIZE:
                    reserve = GOLD_RESERVE
                else:
                    reserve = GOLD_RESERVE + UPGRADE_SAVE
            else:
                # 2단계: 업그레이드와 병력을 같이 키운다.
                #  · 최소 수비선(guard)까지는 무조건 훈련
                #  · 그 위로는 '다음 업그레이드 값'을 묶어두고 남는 걸로 훈련
                #    (훈련 1명이 120금이라, 안 묶어두면 550~1000금이 영영 안 모인다)
                guard = max(ARMY_BASE // 2, self.max_blob)
                pending = []
                if hq.level < HQ_RUSH_LEVEL:
                    pending.append(hq.next_cost())
                pending += [b.next_cost() for b in bases_now if b.level < BASE_MAX_LEVEL]
                if mobile < guard or not pending:
                    reserve = GOLD_RESERVE
                elif hq.level < HQ_RUSH_LEVEL:
                    # 본부 Lv3(하루 2명 훈련 + 전사 체력 6)이 가장 값어치 있다
                    reserve = GOLD_RESERVE + hq.next_cost() + UPGRADE_BUFFER
                elif mobile < army_target:
                    reserve = GOLD_RESERVE + min(pending) + UPGRADE_BUFFER
                else:
                    reserve = GOLD_RESERVE + UPGRADE_SAVE
            income = 0
            for region, b in self.my_b.items():
                cnt = sum(1 for w in ws if w.region == region)
                income += WORK_INCOME * min(cnt, b.work_cap())
            upkeep = UPKEEP_PER_WARRIOR * (len(ws) + 1)
            cap = HQ_LEVELS[hq.level].train_cap
            if income - upkeep < FOOD_MARGIN:
                cap = 0                      # 더 뽑으면 굶는다
            n = 0
            while n < cap and gold - TRAIN_COST >= reserve:
                n += 1
                gold -= TRAIN_COST
            a.train_n = n

        if os.environ.get("NYPC_DEBUG"):
            roles: dict[str, int] = {}
            for w in ws:
                roles[w.role] = roles.get(w.role, 0) + 1
            print(f"DBG turn={turn} gold={self.gold} warriors={len(ws)} roles={roles} "
                  f"myb={sorted(self.my_b)} 남은거점={remaining} 공격={self.attacking} "
                  f"침공={incursion} 뭉침={blob} 방어점={defense_point} "
                  f"본부Lv={hq.level if hq else 0}", file=sys.stderr)
        return a


def main() -> None:
    M = parse_init()
    bot = Bot(M)
    while True:
        turn = read_turn_start()
        if turn is None:
            return
        a = bot.decide(turn)
        emit(a)
        bot.read_result(a)


if __name__ == "__main__":
    main()
