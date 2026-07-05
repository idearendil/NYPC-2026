#!/usr/bin/env python3
"""
전략 봇
1. 시작 시 내 본부와 가까운 거점 2곳에 전사를 보내 기지를 건설한다.
2. 본부에서 금화가 허락하는 한 계속 전사를 훈련한다.
3. 훈련된 전사는 "상대 본부와 가장 가까운 내 기지"(집결지)로 보내 모은다.
   (내 기지가 없으면 본부에서 모은다. 본부에는 노동 담당 1명만 남긴다.)
4. 집결지에 전사가 6명 모이면 5명을 공격대로 보낸다.
   - 공격 목표: 상대 기지 중 "그 구역의 상대 전사 수가 가장 적은 곳" 우선,
     같으면 더 가까운 곳, 그것도 같으면 구역 번호가 작은 곳.
   - 목표 기지를 파괴하면 다음 기지로 이동, 상대 기지가 전부 없어지면 상대 본부 공격.
5. 그동안에도 계속 훈련→집결하다가 다시 6명이 모이면 또 5명을 보낸다. (반복)
6. 본진 방어: 적 전사가 본부 또는 본부 인접 구역에 나타나면, 집결 중이던
   병사를 즉시 본부로 보내 수비한다. 위협이 사라지면 다시 집결지로 돌아간다.
   (이미 출격한 공격대는 그대로 공격을 계속한다.)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

MAX_TURN = 200          # maximum turn (days)
START_GOLD = 500        # initial gold
START_WARRIORS = 3      # initial warriors
MOVE_COST = 10          # move cost
TRAIN_COST = 120        # train cost
WORK_INCOME = 15        # income per warrior
UPKEEP_PER_WARRIOR = 2  # upkeep per warrior
HQ_MAX_LEVEL = 5        # HQ max level
BASE_MAX_LEVEL = 3      # base max level
HQ_HEAL_COST = 1000     # HQ fix cost
BASE_HEAL_COST = 500    # base fix cost

# --- 전략 파라미터 ---
WAVE_TRIGGER = 6        # 본부에 이 인원이 모이면 공격대 출발
WAVE_SIZE = 5           # 공격대 인원
GOLD_RESERVE = 30       # 식비 등을 위해 항상 남겨둘 금화
SETTLE_GIVEUP_TURN = 30 # 이 턴까지 기지 2개를 못 지으면 그냥 훈련 시작


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
    HqLevelEntry(1200,  6, 20, 2, 2, 3),
    HqLevelEntry(2400,  7, 25, 3, 2, 4),
    HqLevelEntry(3600,  8, 30, 3, 3, 5),
)
BASE_LEVELS: tuple[BaseLevelEntry, ...] = (
    BaseLevelEntry(0,    0,  0, 0),
    BaseLevelEntry(300,  6, 1, 1),
    BaseLevelEntry(600,  12, 1, 2),
    BaseLevelEntry(1000, 18, 2, 3),
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


class WState(Enum):
    STATIONARY = 0
    MOVING = 1


@dataclass(frozen=True)
class WarriorId:
    side: Side
    num: int

    def __str__(self) -> str:
        return f"{self.side.value}{self.num}"

    @classmethod
    def parse(cls, tok: str) -> "WarriorId":
        assert tok and tok[0] in ("A", "B")
        return cls(Side.from_char(tok[0]), int(tok[1:]))


@dataclass
class Warrior:
    id: WarriorId
    region: int
    hp: int
    state: WState = WState.STATIONARY
    target: int = 0


@dataclass
class Building:
    region: int
    side: Side
    type: BType
    level: int = 1
    hp: int = 10

    def current_hp(self) -> int:
        return HQ_LEVELS[self.level].hp if self.type is BType.HQ else BASE_LEVELS[self.level].hp

    def work_cap(self) -> int:
        return HQ_LEVELS[self.level].work_cap if self.type is BType.HQ else BASE_LEVELS[self.level].work_cap

    def apply_upgrade(self) -> None:
        self.level += 1
        self.hp = self.current_hp()

    def upgrade_cost(self) -> int:
        if self.type is BType.HQ:
            return HQ_LEVELS[self.level + 1].upgrade_cost
        else:
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
class GameState:
    gold: int = START_GOLD
    my_countdown: int = 5
    opp_countdown: int = 5
    warriors: list[Warrior] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)

    def find_building(self, region: int) -> Building | None:
        return next((b for b in self.buildings if b.region == region), None)

    def find_warrior(self, wid: WarriorId) -> Warrior | None:
        return next((w for w in self.warriors if w.id == wid), None)


@dataclass
class Actions:
    train_n: int = 0
    moves: list[tuple[WarriorId, int]] = field(default_factory=list)
    upgrades: list[int] = field(default_factory=list)


def make_base(region: int, s: Side) -> Building:
    return Building(region, s, BType.BASE, 1, BASE_LEVELS[1].hp)


def readln() -> str:
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.rstrip("\n")


def read_tokens() -> list[str]:
    return readln().split()


def parse_init() -> tuple[GameMap, GameState]:
    M = GameMap()

    t = read_tokens()
    assert len(t) >= 2 and t[0] == "READY"
    M.my_side = Side.from_word(t[1])

    t = read_tokens()
    M.N, M.K = int(t[0]), int(t[1])

    M.x = [int(v) for v in read_tokens()]  # x_0 x_1 ... x_{N-1}
    M.y = [int(v) for v in read_tokens()]  # y_0 y_1 ... y_{N-1}

    M.strongholds = sorted(int(v) for v in read_tokens())  # K strongholds

    M.adj = [[] for _ in range(M.N)]
    for r in range(M.N):
        t = read_tokens()  # deg n_1 n_2 ...
        deg = int(t[0])
        M.adj[r] = sorted(int(v) for v in t[1:1 + deg])

    M.my_hq = M.hq_of(M.my_side)
    M.opp_hq = M.hq_of(M.my_side.opposite)

    S = GameState()
    opp = M.my_side.opposite
    for sfx in range(1, START_WARRIORS + 1):
        S.warriors.append(Warrior(WarriorId(M.my_side, sfx), M.my_hq, HQ_LEVELS[1].warrior_hp))
        S.warriors.append(Warrior(WarriorId(opp, sfx), M.opp_hq, HQ_LEVELS[1].warrior_hp))
    S.buildings.append(
        Building(0, Side.LEFT, BType.HQ, 1, HQ_LEVELS[1].hp)
    )
    S.buildings.append(
        Building(M.N - 1, Side.RIGHT, BType.HQ, 1, HQ_LEVELS[1].hp)
    )

    print("OK", flush=True)
    return M, S


def read_turn_start() -> int | None:
    line = readln()
    if line == "FINISH":
        return None
    t = line.split()
    assert t and t[0] == "START"
    return int(t[2])


def read_turn_result(S: GameState, M: GameMap, submitted: Actions) -> None:
    for region in submitted.upgrades:
        b = S.find_building(region)
        if b is None:
            S.gold -= BASE_LEVELS[1].cost
            S.buildings.append(make_base(region, M.my_side))
        else:
            max_level = HQ_MAX_LEVEL if b.type is BType.HQ else BASE_MAX_LEVEL
            if b.level >= max_level:
                cost = HQ_HEAL_COST if b.type is BType.HQ else BASE_HEAL_COST
                S.gold -= cost
                b.hp = b.current_hp()
            else:
                S.gold -= b.upgrade_cost()
                b.apply_upgrade()

    for wid, target in submitted.moves:
        b = S.find_building(target)
        cost = 0 if (b is not None and b.side is M.my_side) else MOVE_COST
        S.gold -= cost
        w = S.find_warrior(wid)
        if w is not None:
            w.state = WState.MOVING
            w.target = target

    S.gold -= TRAIN_COST * submitted.train_n

    line = readln()
    if line == "FINISH":
        sys.exit(0)
    t = line.split()
    assert t and t[0] == "TURN"

    t = read_tokens()
    S.my_countdown = int(t[2])
    S.opp_countdown = int(t[4])

    # UPGRADE
    t = read_tokens()  # "UPGRADE N"
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()  # "<A|B> <region>"
        s = Side.from_char(r[0][0])
        region = int(r[1])
        b = S.find_building(region)
        if b is None:
            S.buildings.append(make_base(region, s))
        elif b.side is not M.my_side:
            max_level = HQ_MAX_LEVEL if b.type is BType.HQ else BASE_MAX_LEVEL
            if b.level >= max_level:
                b.hp = b.current_hp()
            else:
                b.apply_upgrade()

    # TRAIN
    t = read_tokens()  # "TRAIN N"
    n = int(t[1])
    if n > 0:
        ids = read_tokens()
        for i in range(n):
            wid = WarriorId.parse(ids[i])
            hq_region = M.hq_of(wid.side)
            hq_b = S.find_building(hq_region)
            hq_level = hq_b.level if hq_b is not None else 1
            S.warriors.append(Warrior(wid, hq_region, HQ_LEVELS[hq_level].warrior_hp))

    # MOVE
    t = read_tokens()  # "MOVE N"
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        wid = WarriorId.parse(r[0])
        region = int(r[1])
        w = S.find_warrior(wid)
        if w is not None:
            w.region = region
            if (wid.side is M.my_side
                    and w.state is WState.MOVING
                    and w.region == w.target):
                w.state = WState.STATIONARY

    # DAMAGE
    t = read_tokens()  # "DAMAGE N"
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        wid = WarriorId.parse(r[1])
        damage = int(r[2])
        w = S.find_warrior(wid)
        if w is not None:
            w.hp -= damage
    S.warriors = [w for w in S.warriors if w.hp > 0]

    # SIEGE
    t = read_tokens()  # "SIEGE N"
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        region = int(r[1])
        dmg = int(r[2])
        b = S.find_building(region)
        if b is not None:
            b.hp -= dmg
    S.buildings = [b for b in S.buildings if b.hp > 0]

    readln()  # "END"

    income = 0
    for b in S.buildings:
        if b.side is not M.my_side:
            continue
        count = sum(
            1 for w in S.warriors
            if w.id.side is M.my_side and w.region == b.region
        )
        income += WORK_INCOME * min(count, b.work_cap())
    S.gold += income

    alive = sum(1 for w in S.warriors if w.id.side is M.my_side)
    S.gold = max(0, S.gold - UPKEEP_PER_WARRIOR * alive)


@dataclass
class Paths:
    dist: list[list[float]]
    nxt: list[list[int]]


def euclid_ceil(M: GameMap, u: int, v: int) -> float:
    return math.ceil(math.hypot(M.x[u] - M.x[v], M.y[u] - M.y[v]))


def calculate_paths(M: GameMap) -> Paths:
    INF = math.inf
    N = M.N
    dist = [[INF] * N for _ in range(N)]
    nxt = [[-1] * N for _ in range(N)]

    for i in range(N):
        dist[i][i] = 0.0
        nxt[i][i] = i
    for u in range(N):
        for v in M.adj[u]:
            w = euclid_ceil(M, u, v)
            if w < dist[u][v]:
                dist[u][v] = w

    # Floyd-Warshall
    for k in range(N):
        dk = dist[k]
        for u in range(N):
            du = dist[u]
            duk = du[k]
            if duk == INF:
                continue
            for v in range(N):
                cand = duk + dk[v]
                if cand < du[v]:
                    du[v] = cand

    for u in range(N):
        du = dist[u]
        for v in range(N):
            if u == v or du[v] == INF:
                continue
            best_score = INF
            for nb in M.adj[u]:
                if dist[nb][v] == INF:
                    continue
                score = euclid_ceil(M, u, nb) + dist[nb][v]
                if score < best_score:
                    best_score = score
                    nxt[u][v] = nb
    return Paths(dist, nxt)


def next_step(P: Paths, u: int, v: int) -> int:
    """Returns the next step on the path from u to v. Returns -1 if the path is not reachable."""
    return P.nxt[u][v]


def path(P: Paths, u: int, v: int) -> list[int]:
    """Returns the path from u to v as [u, ..., v]. Returns an empty list if the path is not reachable."""
    if P.nxt[u][v] == -1:
        return []
    out = [u]
    while u != v:
        u = P.nxt[u][v]
        out.append(u)
    return out


def emit(a: Actions) -> None:
    out: list[str] = ["COMMAND"]
    for wid, target in a.moves:
        out.append(f"MOVE {wid} {target}")
    for r in a.upgrades:
        out.append(f"UPGRADE {r}")
    if a.train_n > 0:
        out.append(f"TRAIN {a.train_n}")
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 전략
# ---------------------------------------------------------------------------
class Strategy:
    def __init__(self, M: GameMap, P: Paths) -> None:
        # 내 본부에서 경로 거리가 가장 가까운 거점 2곳 선택
        sh = sorted(M.strongholds, key=lambda r: (P.dist[M.my_hq][r], r))
        self.targets2: list[int] = sh[:2]
        # 거점별 개척 담당 전사
        self.settler: dict[int, WarriorId | None] = {r: None for r in self.targets2}
        # 공격대로 이미 내보낸 전사들
        self.attackers: set[WarriorId] = set()

    # -- 유틸 ---------------------------------------------------------------
    @staticmethod
    def _enemy_counts(S: GameState, M: GameMap) -> dict[int, int]:
        cnt: dict[int, int] = {}
        for w in S.warriors:
            if w.id.side is not M.my_side:
                cnt[w.region] = cnt.get(w.region, 0) + 1
        return cnt

    @staticmethod
    def _target_pool(S: GameState, M: GameMap) -> list[Building]:
        """공격 후보: 상대 기지가 남아 있으면 기지들, 없으면 상대 본부."""
        enemy = [b for b in S.buildings if b.side is not M.my_side]
        bases = [b for b in enemy if b.type is BType.BASE]
        return bases if bases else enemy

    @staticmethod
    def _pick_target(pool: list[Building], counts: dict[int, int],
                     P: Paths, from_region: int) -> Building:
        """인원수 적은 곳 → 가까운 곳 → 번호 작은 곳 순."""
        return min(pool, key=lambda b: (counts.get(b.region, 0),
                                        P.dist[from_region][b.region],
                                        b.region))

    # -- 메인 ---------------------------------------------------------------
    def decide(self, S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
        a = Actions()
        gold = S.gold
        my = M.my_side

        my_ws = [w for w in S.warriors if w.id.side is my]
        alive = {w.id for w in my_ws}
        counts = self._enemy_counts(S, M)
        pool = self._target_pool(S, M)

        # 죽은 전사 정리
        self.attackers &= alive
        for r, wid in self.settler.items():
            if wid is not None and wid not in alive:
                self.settler[r] = None

        settler_ids = {wid for wid in self.settler.values() if wid is not None}
        ordered = set()  # 이번 턴에 명령을 내린 전사

        # 집결지: 상대 본부와 가장 가까운 내 기지 (기지가 없으면 내 본부)
        my_bases = [b for b in S.buildings if b.side is my and b.type is BType.BASE]
        if my_bases:
            rally = min(my_bases,
                        key=lambda b: (P.dist[b.region][M.opp_hq], b.region)).region
        else:
            rally = M.my_hq

        # 본부 노동 담당(지킴이): 공격대/개척자가 아닌 전사 중 번호가 가장 앞선 전사
        guard = min((w for w in my_ws
                     if w.id not in self.attackers and w.id not in settler_ids),
                    key=lambda w: w.id.num, default=None)
        guard_id = guard.id if guard is not None else None

        # 본진 방어 모드: 적 전사가 본부 또는 본부 인접 구역에 있으면 발동
        threat_zone = {M.my_hq, *M.adj[M.my_hq]}
        defending = any(counts.get(r, 0) > 0 for r in threat_zone)
        gather_to = M.my_hq if defending else rally  # 집결 목적지

        # ---- 1) 초반 확장: 가까운 거점 2곳에 기지 건설 -------------------
        for r in self.targets2:
            b = S.find_building(r)
            if b is not None:
                continue  # 이미 건물이 있음(내 기지면 완료, 적 건물이면 포기)

            wid = self.settler[r]
            if wid is None:
                # 본부에 있는 여유 전사 하나를 개척자로 지정
                spare = sorted(
                    (w for w in my_ws
                     if w.state is WState.STATIONARY
                     and w.region == M.my_hq
                     and w.id not in self.attackers
                     and w.id not in settler_ids),
                    key=lambda w: w.id.num,
                )
                if len(spare) >= 2:  # 최소 1명은 본부에 남긴다
                    wid = spare[-1].id
                    self.settler[r] = wid
                    settler_ids.add(wid)

            if wid is None:
                continue
            w = S.find_warrior(wid)
            if w is None or w.state is not WState.STATIONARY:
                continue

            if w.region != r:
                if gold >= MOVE_COST:
                    a.moves.append((wid, r))
                    ordered.add(wid)
                    gold -= MOVE_COST
            else:
                # 도착함: 적 전사가 없고 금화가 충분하면 기지 건설
                if counts.get(r, 0) == 0 and gold >= BASE_LEVELS[1].cost:
                    a.upgrades.append(r)
                    gold -= BASE_LEVELS[1].cost

        # ---- 2) 기존 공격대: 목표 재설정 ---------------------------------
        for w in my_ws:
            if w.id not in self.attackers or w.state is not WState.STATIONARY:
                continue
            if not pool:
                break
            tgt = self._pick_target(pool, counts, P, w.region)
            if w.region != tgt.region and gold >= MOVE_COST:
                a.moves.append((w.id, tgt.region))
                ordered.add(w.id)
                gold -= MOVE_COST
            # 목표 구역에 이미 있으면 그대로 서서 전투/공성

        # ---- 3) 집결/수비: 훈련된 전사를 집결지로, 본진 위협 시 본부로 ----
        for w in my_ws:
            if (w.id in self.attackers or w.id in settler_ids
                    or w.id == guard_id or w.id in ordered
                    or w.state is not WState.STATIONARY
                    or w.region == gather_to):
                continue
            b = S.find_building(gather_to)
            cost = 0 if (b is not None and b.side is my) else MOVE_COST
            if gold >= cost:
                a.moves.append((w.id, gather_to))
                ordered.add(w.id)
                gold -= cost

        # ---- 4) 새 공격대 출발: 집결지에 6명 모이면 5명 출격 --------------
        #        (본진 방어 중에는 출격하지 않는다)
        at_rally = [w for w in my_ws
                    if w.state is WState.STATIONARY
                    and w.region == rally
                    and w.id not in self.attackers
                    and w.id not in ordered]
        if not defending and len(at_rally) >= WAVE_TRIGGER and pool:
            # 개척자(기지 노동)와 본부 지킴이는 남기고 나머지 중 5명 출격
            sendable = sorted(
                (w for w in at_rally
                 if w.id not in settler_ids and w.id != guard_id),
                key=lambda w: w.id.num,
            )
            group = sendable[:WAVE_SIZE]
            if len(group) == WAVE_SIZE and gold >= MOVE_COST * WAVE_SIZE:
                tgt = self._pick_target(pool, counts, P, rally)
                for w in group:
                    a.moves.append((w.id, tgt.region))
                    self.attackers.add(w.id)
                    ordered.add(w.id)
                    gold -= MOVE_COST

        # ---- 5) 훈련: 초반 기지 2개를 확보한 뒤 계속 훈련 -----------------
        expansion_done = all(
            S.find_building(r) is not None or r in a.upgrades
            for r in self.targets2
        ) or turn >= SETTLE_GIVEUP_TURN
        hq = S.find_building(M.my_hq)
        if expansion_done and hq is not None and hq.side is my:
            cap = HQ_LEVELS[hq.level].train_cap
            n = 0
            while n < cap and gold - TRAIN_COST >= GOLD_RESERVE:
                n += 1
                gold -= TRAIN_COST
            a.train_n = n

        return a


def main() -> None:
    M, S = parse_init()
    P = calculate_paths(M)
    strategy = Strategy(M, P)

    while (turn := read_turn_start()) is not None:
        a = strategy.decide(S, M, P, turn)
        emit(a)
        read_turn_result(S, M, a)


if __name__ == "__main__":
    main()
