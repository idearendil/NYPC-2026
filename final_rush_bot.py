#!/usr/bin/env python3
"""
러시 봇 — "본진 근처 기지 2개 → 병사 15명 모아 한 번에 상대 본진 강습"

전략 요약
  1. 시작하자마자 본부에서 가장 가까운 거점 BASE_COUNT(3)곳에 개척자를 보내 기지를 짓는다.
     지은 사람은 그대로 그 기지에 남아 일꾼이 된다(수입원).
  2. 본부에서는 금화가 되는 대로 계속 전사를 훈련한다.
     훈련된 전사는 본부에 그대로 대기(집결)한다. 본부 일자리도 채워서 수입을 번다.
  3. 본부 대기 인원이 RUSH_SIZE(30)명이 되면, 그 30명을 한 번에
     상대 본부로 보낸다. 같은 곳에서 같은 경로로 출발하므로 뭉쳐서 도착한다.
  4. 1차 강습 후에도 계속 훈련해서 30명이 다시 모이면 또 보낸다(파도 반복).
  5. 본부 근처 적이 남는 수비 병력으로 감당이 안 될 때만 출격을 미룬다.
     (적 정찰병 한둘 때문에 영원히 못 나가는 일이 없도록 비율로 판단한다)

조절 가능한 값은 아래 "전략 파라미터"에 모여 있다.
  RUSH_SIZE     : 한 번에 보낼 인원 (기본 30)
  BASE_COUNT    : 본진 근처에 지을 기지 수 (기본 3)
  REPEAT_WAVES  : 1차 강습 후 반복 출격 여부
  MAX_BASE_LEVEL / MAX_HQ_LEVEL : 여유 금화가 있을 때만 여기까지 올린다

시야(안개) 처리와 금화 계산은 전략 봇과 동일한 검증된 방식을 쓴다.
  * 아군 전사/건물은 항상 보고에 들어오므로 매 턴 통째로 재구성한다.
  * 훈련 당일 죽은 전사를 살아있다고 세면 금화가 어긋나 반칙이 되므로,
    TRAIN 목록으로 전사를 만들어내지 않는다.
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
RUSH_SIZE = 30           # 한 번에 상대 본진으로 보낼 인원
BASE_COUNT = 3           # 본진 근처에 지을 기지 수
REPEAT_WAVES = True      # 1차 강습 후에도 15명 모이면 계속 보낸다
HOME_GUARD = 2           # 출격 후에도 본부에 남겨둘 최소 인원(일꾼 포함)
DEFENSE_RATIO = 2        # 본부 근처 적 1명당 이만큼은 남아 있어야 출격한다
FORCE_LAUNCH_TURN = 200  # 이 턴까지 한 번도 못 나갔으면 위협 무시하고 강행
GOLD_RESERVE = 60        # 식비용으로 항상 남겨둘 금화
UPGRADE_BUFFER = 400     # 업그레이드 후 남겨둘 금화(훈련이 멈추지 않도록)
MAX_BASE_LEVEL = int(os.environ.get("RUSH_BASE_LV", 2))   # 기지 목표 레벨(수입)
MAX_HQ_LEVEL = int(os.environ.get("RUSH_HQ_LV", 3))       # 본부 목표 레벨(훈련 2명/턴, 전사 체력 6)
ECON_BUFFER = 100        # 경제 단계에서 업그레이드 후 남겨둘 금화
ECON_SAVE = 700          # 경제 단계에서 업그레이드용으로 묶어둘 금화


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
R_WAIT = "WAIT"          # 본부에서 강습 인원으로 대기 중
R_RUSH = "RUSH"          # 상대 본부로 진격 중


@dataclass
class Warrior:
    num: int
    region: int
    hp: int
    moving: bool = False
    target: int = -1
    role: str = R_WAIT
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
        self.enemy_w: dict[int, int] = {}

        # 본부에서 가까운 거점 BASE_COUNT곳이 개척 목표
        g = self.geo
        self.claim: list[int] = sorted(M.strongholds,
                                       key=lambda r: (g.dist(M.my_hq, r), r))[:BASE_COUNT]
        self.launched = 0        # 지금까지 출격시킨 파도 수

    # ---- 관측/상태 갱신 --------------------------------------------------
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

        # 수입 → 식비
        income = 0
        for region, b in self.my_b.items():
            cnt = sum(1 for w in self.warriors.values() if w.region == region)
            income += WORK_INCOME * min(cnt, b.work_cap())
        self.gold += income
        alive = len(self.warriors)
        self.gold -= UPKEEP_PER_WARRIOR * min(alive, self.gold // UPKEEP_PER_WARRIOR)

    # ---- 하루 결정 --------------------------------------------------------
    def decide(self, turn: int) -> Actions:
        self.turn = turn
        M, g = self.M, self.geo
        a = Actions()
        gold = self.gold
        ws = list(self.warriors.values())
        hq = self.my_b.get(M.my_hq)

        if os.environ.get("NYPC_DEBUG"):
            roles: dict[str, int] = {}
            for w in ws:
                roles[w.role] = roles.get(w.role, 0) + 1
            print(f"DBG turn={turn} gold={self.gold} warriors={len(ws)} roles={roles} "
                  f"myb={sorted(self.my_b)} waves={self.launched}", file=sys.stderr)

        if not ws:
            if hq is not None and gold >= TRAIN_COST + GOLD_RESERVE:
                a.train_n = min(HQ_LEVELS[hq.level].train_cap, 1)
            return a

        # --- 1) 역할 정리 --------------------------------------------------
        for w in ws:
            if w.role == R_SETTLE and w.assign in self.my_b:
                w.role = R_WORK                      # 기지 완성 → 그 자리 일꾼
            elif w.role == R_WORK and w.assign not in self.my_b:
                w.role, w.assign = R_WAIT, -1        # 건물이 사라짐 → 본부 대기로
            elif w.role == R_RUSH:
                w.assign = M.opp_hq                  # 러시는 끝까지 상대 본부

        # --- 2) 개척: 본진 근처 거점 BASE_COUNT곳 ---------------------------
        settle_targets = [r for r in self.claim
                          if r not in self.my_b and r not in self.enemy_b]
        taken = {w.assign for w in ws if w.role == R_SETTLE}
        free = [w for w in ws if w.role == R_WAIT and not w.moving]
        free.sort(key=lambda w: w.num)

        def take_nearest(region: int) -> Warrior | None:
            if not free:
                return None
            w = min(free, key=lambda w: (g.dist(w.region, region), w.num))
            free.remove(w)
            return w

        for r in settle_targets:
            if r in taken:
                continue
            # 본부가 텅 비지 않도록 최소 인원은 남긴다
            if len(free) <= 1:
                break
            w = take_nearest(r)
            if w is None:
                break
            w.role, w.assign = R_SETTLE, r
            taken.add(r)

        # --- 3) 일꾼: 각 건물의 노동 가능 인원만큼 채운다 --------------------
        #        (본부부터 채워서 본진이 비지 않게 한다)
        order_regions = [M.my_hq] + [r for r in sorted(self.my_b) if r != M.my_hq]
        for region in order_regions:
            b = self.my_b.get(region)
            if b is None:
                continue
            have = sum(1 for w in ws if w.role in (R_WORK, R_SETTLE) and w.assign == region)
            for _ in range(b.work_cap() - have):
                w = take_nearest(region)
                if w is None:
                    break
                w.role, w.assign = R_WORK, region

        # --- 4) 강습: 본부에 RUSH_SIZE명이 모이면 한 번에 출발 ---------------
        # 본부 시야 안의 적 수. 정찰병 한둘이 근처에 죽치고 있다고 해서 출격을
        # 영원히 미루면 안 된다(그러면 병력만 쌓다가 무승부로 끝난다).
        # 출격 후 본부에 남는 인원이 눈에 보이는 위협을 감당할 만하면 그냥 간다.
        threat_near_hq = sum(self.enemy_w.get(r, 0) for r in g.vis[M.my_hq])
        ready = [w for w in ws
                 if w.role == R_WAIT and not w.moving and w.region == M.my_hq]
        home_left = sum(1 for w in ws if w.region == M.my_hq) - RUSH_SIZE
        outmatched = home_left < threat_near_hq * DEFENSE_RATIO + HOME_GUARD
        if turn >= FORCE_LAUNCH_TURN and self.launched == 0:
            outmatched = False          # 끝까지 안 나가면 지는 것이나 마찬가지
        can_launch = ((REPEAT_WAVES or self.launched == 0)
                      and home_left >= HOME_GUARD and not outmatched)
        if can_launch and len(ready) >= RUSH_SIZE and gold >= MOVE_COST * RUSH_SIZE:
            ready.sort(key=lambda w: (-w.hp, w.num))   # 체력 높은 순으로 보낸다
            for w in ready[:RUSH_SIZE]:
                w.role, w.assign = R_RUSH, M.opp_hq
            self.launched += 1

        # --- 5) 이동 명령 ----------------------------------------------------
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
            if w.role in (R_SETTLE, R_WORK, R_RUSH) and w.assign >= 0:
                order(w, w.assign)
            elif w.role == R_WAIT and w.region != M.my_hq:
                order(w, M.my_hq)          # 대기 인원은 본부로 모인다

        # --- 6) 건설/업그레이드 ----------------------------------------------
        done: set[int] = set()
        for w in ws:                                   # 새 기지
            if (w.role == R_SETTLE and not w.moving and w.region == w.assign
                    and w.assign not in self.my_b and w.assign not in done
                    and self.enemy_w.get(w.assign, 0) == 0
                    and gold >= BASE_LEVELS[1].cost):
                a.upgrades.append(w.assign)
                done.add(w.assign)
                gold -= BASE_LEVELS[1].cost

        occupied = {r for r in self.my_b if any(w.region == r for w in ws)}
        # 목표 레벨까지 건물을 올린다. 기지 2개가 다 서기 전에는 건설이 우선.
        econ_done = (not settle_targets) and all(
            b.level >= (MAX_HQ_LEVEL if b.type is BType.HQ else MAX_BASE_LEVEL)
            for b in self.my_b.values())
        buffer = UPGRADE_BUFFER if econ_done else ECON_BUFFER
        for region in order_regions:
            b = self.my_b.get(region)
            if (b is None or region in done or region not in occupied
                    or self.enemy_w.get(region, 0) > 0 or settle_targets):
                continue
            limit = MAX_HQ_LEVEL if b.type is BType.HQ else MAX_BASE_LEVEL
            if b.level >= limit:
                continue
            cost = b.next_cost()
            if gold >= cost + buffer:
                a.upgrades.append(region)
                done.add(region)
                gold -= cost

        # --- 7) 훈련 ----------------------------------------------------------
        # 경제 단계에서는 "빈 일자리를 채울 만큼"만 뽑아 업그레이드 자금을 남기고,
        # 경제가 완성되면 그때부터 러시 병력을 최대 속도로 모은다.
        if hq is not None:
            work_slots = sum(b.work_cap() for b in self.my_b.values())
            reserve = GOLD_RESERVE + (BASE_LEVELS[1].cost if settle_targets else 0)
            if not econ_done:
                workers = sum(1 for w in ws if w.role in (R_WORK, R_SETTLE))
                if workers >= work_slots:
                    reserve += ECON_SAVE     # 일자리가 다 찼으면 업그레이드 자금을 모은다
            cap = HQ_LEVELS[hq.level].train_cap
            n = 0
            while n < cap and gold - TRAIN_COST >= reserve:
                n += 1
                gold -= TRAIN_COST
            a.train_n = n

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
