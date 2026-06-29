#!/usr/bin/env python3
"""Heuristic bot for the board-game contest.

Strategy (as requested):
  * Early game: expand onto strongholds near our own HQ, building bases for income,
    while training warriors at the HQ.
  * Offense: only commit an attack force to an enemy region when, *assuming the
    enemy does not move or reinforce*, our gathered force is guaranteed to capture
    it (verified by a day-by-day combat simulation).
  * Upgrades / repairs: only when the building is judged safe (no enemy in the
    region and no enemy within a couple of hops).
  * Defense: the HQ is sacred -- if an enemy gets close, everything is recalled.

The I/O framework (parsing, pathfinding, state tracking) is taken from the
provided sample code, which already maintains both players' full state.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

MAX_TURN = 200
START_GOLD = 500
START_WARRIORS = 3
MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2
HQ_MAX_LEVEL = 5
BASE_MAX_LEVEL = 3
HQ_HEAL_COST = 1000
BASE_HEAL_COST = 500


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
BASE_BUILD_COST = BASE_LEVELS[1].cost


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

    def turret(self) -> int:
        return HQ_LEVELS[self.level].turret if self.type is BType.HQ else BASE_LEVELS[self.level].turret

    def max_level(self) -> int:
        return HQ_MAX_LEVEL if self.type is BType.HQ else BASE_MAX_LEVEL

    def apply_upgrade(self) -> None:
        self.level += 1
        self.hp = self.current_hp()

    def upgrade_cost(self) -> int:
        if self.type is BType.HQ:
            return HQ_LEVELS[self.level + 1].upgrade_cost
        else:
            return BASE_LEVELS[self.level + 1].cost

    def heal_cost(self) -> int:
        return HQ_HEAL_COST if self.type is BType.HQ else BASE_HEAL_COST


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

    M.x = [int(v) for v in read_tokens()]
    M.y = [int(v) for v in read_tokens()]

    M.strongholds = sorted(int(v) for v in read_tokens())

    M.adj = [[] for _ in range(M.N)]
    for r in range(M.N):
        t = read_tokens()
        deg = int(t[0])
        M.adj[r] = sorted(int(v) for v in t[1:1 + deg])

    M.my_hq = M.hq_of(M.my_side)
    M.opp_hq = M.hq_of(M.my_side.opposite)

    S = GameState()
    opp = M.my_side.opposite
    for sfx in range(1, START_WARRIORS + 1):
        S.warriors.append(Warrior(WarriorId(M.my_side, sfx), M.my_hq, HQ_LEVELS[1].warrior_hp))
        S.warriors.append(Warrior(WarriorId(opp, sfx), M.opp_hq, HQ_LEVELS[1].warrior_hp))
    S.buildings.append(Building(0, Side.LEFT, BType.HQ, 1, HQ_LEVELS[1].hp))
    S.buildings.append(Building(M.N - 1, Side.RIGHT, BType.HQ, 1, HQ_LEVELS[1].hp))

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
    t = read_tokens()
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
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
    t = read_tokens()
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
    t = read_tokens()
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
    t = read_tokens()
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
    t = read_tokens()
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
    return P.nxt[u][v]


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
# Strategy
# ---------------------------------------------------------------------------

# Hop (unweighted) all-pairs distance, computed once. Used to judge "nearness".
_HOP: list[list[int]] | None = None


def _build_hops(M: GameMap) -> list[list[int]]:
    global _HOP
    if _HOP is not None:
        return _HOP
    N = M.N
    hop = [[-1] * N for _ in range(N)]
    for s in range(N):
        d = hop[s]
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in M.adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
    _HOP = hop
    return hop


def _apply_attacks(arr_sorted: list[int], n: int) -> tuple[list[int], int]:
    """Apply `n` single-point attacks to warriors (sorted ascending by hp),
    always hitting the lowest-hp survivor first. Returns (survivors, leftover)."""
    arr = arr_sorted[:]
    i = 0
    while n > 0 and i < len(arr):
        d = min(n, arr[i])
        arr[i] -= d
        n -= d
        if arr[i] == 0:
            i += 1
    return [h for h in arr if h > 0], n


def simulate_capture(my_hps: list[int], enemy_hps: list[int],
                     enemy_turret: int, building_hp: int,
                     max_days: int = 80) -> tuple[bool, int]:
    """Day-by-day combat sim assuming nobody reinforces/moves.

    We are the attacker (no turret of our own at the target). Returns
    (captured, surviving_attackers). `captured` means all enemy warriors are
    dead and the building is razed while we still have at least one warrior."""
    my = sorted(h for h in my_hps if h > 0)
    en = sorted(h for h in enemy_hps if h > 0)
    bhp = building_hp

    for _ in range(max_days):
        if not my:
            break
        if not en and bhp <= 0:
            break
        # counts captured at start of day (simultaneous)
        my_attacks = len(my)
        en_attacks = len(en) + (enemy_turret if bhp > 0 else 0)

        # our attacks: kill warriors first, spill over into the building
        en, leftover = _apply_attacks(en, my_attacks)
        if leftover > 0 and bhp > 0:
            bhp = max(0, bhp - leftover)
        # their attacks (warriors + turret) hit us
        my, _ = _apply_attacks(my, en_attacks)

    captured = (not en) and bhp <= 0 and len(my) > 0
    return captured, len(my)


def decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    try:
        return _decide(S, M, P, turn)
    except Exception:
        # Never crash: a crash means no output -> TLE loss. Fail safe to noop.
        return Actions()


def _decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    a = Actions()
    me = M.my_side
    my_hq = M.my_hq
    opp_hq = M.opp_hq
    hop = _build_hops(M)
    dist = P.dist
    stronghold_set = set(M.strongholds)

    # --- index state -------------------------------------------------------
    region_b: dict[int, Building] = {b.region: b for b in S.buildings}
    my_at: dict[int, list[Warrior]] = defaultdict(list)
    en_at: dict[int, list[Warrior]] = defaultdict(list)
    stat_at: dict[int, list[Warrior]] = defaultdict(list)  # my stationary
    for w in S.warriors:
        if w.id.side is me:
            my_at[w.region].append(w)
            if w.state is WState.STATIONARY:
                stat_at[w.region].append(w)
        else:
            en_at[w.region].append(w)

    enemy_regions = [r for r in en_at]
    alive = sum(len(v) for v in my_at.values())
    food_reserve = 12 + 4 * alive  # keep a couple of days of upkeep on hand

    # Strongholds we are about to own but have not built on yet (a warrior is
    # sitting there, or one is travelling there). We earmark ~one base cost per
    # such claim so training never starves the economy of build money.
    pending_claims = set()
    for w in S.warriors:
        if w.id.side is not me:
            continue
        if w.state is WState.MOVING and w.target in stronghold_set \
                and w.target not in region_b:
            pending_claims.add(w.target)
        elif w.state is WState.STATIONARY and w.region in stronghold_set \
                and w.region not in region_b and not en_at.get(w.region):
            pending_claims.add(w.region)
    train_reserve = food_reserve + BASE_BUILD_COST * len(pending_claims)

    # --- budget bookkeeping (total spend must stay <= gold) ----------------
    gold = S.gold
    spend = 0
    moved: set[WarriorId] = set()
    upg: set[int] = set()

    def can_full(c: int) -> bool:
        return spend + c <= gold

    def can_opt(c: int) -> bool:  # leaves a food reserve for non-critical spend
        return spend + c <= gold - food_reserve

    def commit(c: int) -> None:
        nonlocal spend
        spend += c

    def do_move(w: Warrior, target: int, critical: bool = False) -> bool:
        if target == w.region or w.id in moved:
            return False
        if w.state is not WState.STATIONARY:
            return False
        b = region_b.get(target)
        cost = 0 if (b is not None and b.side is me) else MOVE_COST
        ok = can_full(cost) if critical else can_opt(cost)
        if not ok:
            return False
        commit(cost)
        a.moves.append((w.id, target))
        moved.add(w.id)
        return True

    # --- helper predicates -------------------------------------------------
    def enemy_within(region: int, hops: int) -> bool:
        for r in enemy_regions:
            if hop[r][region] >= 0 and hop[r][region] <= hops:
                return True
        return False

    def is_safe(region: int) -> bool:
        return not en_at.get(region) and not enemy_within(region, 2)

    def enemy_def_at(region: int) -> tuple[list[int], int, int]:
        """(enemy warrior hps, turret, building hp) defending a region."""
        hps = [w.hp for w in en_at.get(region, [])]
        b = region_b.get(region)
        if b is not None and b.side is not me:
            return hps, b.turret(), b.hp
        return hps, 0, 0

    # ======================================================================
    # 1. BUILD / UPGRADE  (economy & tiebreak HP), only on safe buildings
    # ======================================================================
    def building_cost(b: Building) -> tuple[int, bool]:
        if b.level >= b.max_level():
            return b.heal_cost(), True  # heal
        return b.upgrade_cost(), False  # upgrade

    # 1a. build new bases on strongholds we already occupy
    for r in sorted(stronghold_set):
        if r in region_b or r in upg:
            continue
        if not stat_at.get(r) or en_at.get(r):
            continue
        if not is_safe(r):
            continue
        if can_opt(BASE_BUILD_COST):
            commit(BASE_BUILD_COST)
            a.upgrades.append(r)
            upg.add(r)

    # 1b. upgrade the HQ toward level 3 (more train cap + tougher warriors),
    #     and keep it healed late game for the HP tiebreak.
    hq_b = region_b.get(my_hq)
    if hq_b is not None and hq_b.region not in upg and stat_at.get(my_hq) \
            and is_safe(my_hq):
        cost, is_heal = building_cost(hq_b)
        want = False
        if not is_heal and hq_b.level < 3:
            want = True  # train cap / warrior hp
        elif not is_heal and hq_b.level < HQ_MAX_LEVEL and turn > 60:
            want = True  # tougher HQ for the endgame tiebreak
        elif is_heal and hq_b.hp < hq_b.current_hp() and turn > 150:
            want = True  # top up HP near the end
        if want and can_opt(cost) and (gold - spend - cost) >= 100:
            commit(cost)
            a.upgrades.append(my_hq)
            upg.add(my_hq)

    # 1c. upgrade bases for more income when we are clearly rich
    for b in S.buildings:
        if b.side is not me or b.type is not BType.BASE:
            continue
        if b.region in upg or not stat_at.get(b.region) or not is_safe(b.region):
            continue
        if b.level >= BASE_MAX_LEVEL:
            continue
        # only worth it if we have warriors to fill the higher work cap
        if len(my_at.get(b.region, [])) <= b.work_cap():
            continue
        cost, _ = building_cost(b)
        if can_opt(cost) and (gold - spend - cost) >= 350:
            commit(cost)
            a.upgrades.append(b.region)
            upg.add(b.region)

    # ======================================================================
    # Classify warriors into workers / claimers / free pool
    # ======================================================================
    workers: set[WarriorId] = set()
    claimers: set[WarriorId] = set()
    for r, lst in stat_at.items():
        b = region_b.get(r)
        if b is not None and b.side is me:
            for w in lst[:b.work_cap()]:
                workers.add(w.id)
        elif r in stronghold_set and not en_at.get(r):
            # sitting on an unbuilt friendly stronghold -> waiting to build
            for w in lst:
                claimers.add(w.id)

    free_pool = [w for r, lst in stat_at.items() for w in lst
                 if w.id not in workers and w.id not in claimers]

    # ======================================================================
    # 2. DEFENSE: the HQ is non-negotiable
    # ======================================================================
    hq_threatened = enemy_within(my_hq, 3) or bool(en_at.get(my_hq))
    if hq_threatened:
        # recall everyone we can to the HQ (moving onto our own HQ is free)
        for w in S.warriors:
            if w.id.side is me and w.state is WState.STATIONARY \
                    and w.region != my_hq and w.id not in moved:
                do_move(w, my_hq, critical=True)
        # still train if we can afford it; skip expansion / offense this turn
        _train(S, M, a, region_b, my_hq, gold, spend, train_reserve)
        return a

    # 3. Light base defense: if one of our bases is contested, feed in the
    #    nearest free warriors until our defenders (counting the turret) at
    #    least match the attackers.
    for b in S.buildings:
        if b.side is not me or b.region == my_hq:
            continue
        attackers = en_at.get(b.region, [])
        if not attackers:
            continue
        defender_n = len(my_at.get(b.region, [])) + b.turret()
        cands = sorted(
            (w for w in free_pool
             if w.id not in moved and 0 <= hop[w.region][b.region] <= 3),
            key=lambda w: dist[w.region][b.region],
        )
        for w in cands:
            if defender_n > len(attackers):
                break
            if do_move(w, b.region, critical=True):
                defender_n += 1

    free_pool = [w for w in free_pool if w.id not in moved]

    # ======================================================================
    # 4. EXPANSION: claim safe strongholds on our side of the map
    # ======================================================================
    near_side = [r for r in M.strongholds if dist[my_hq][r] <= dist[opp_hq][r]]
    target_bases = len(near_side)
    my_base_count = sum(1 for b in S.buildings if b.side is me and b.type is BType.BASE)
    # strongholds already being claimed (occupied or someone en route)
    claimed_or_enroute = set()
    for b in S.buildings:
        if b.side is me:
            claimed_or_enroute.add(b.region)
    for w in S.warriors:
        if w.id.side is me and w.state is WState.MOVING and w.target in stronghold_set:
            claimed_or_enroute.add(w.target)
    # strongholds we already physically sit on (waiting to build)
    for r, lst in stat_at.items():
        if r in stronghold_set and r not in region_b and not en_at.get(r) and lst:
            claimed_or_enroute.add(r)

    candidates = [
        r for r in near_side
        if r not in region_b and r not in claimed_or_enroute and is_safe(r)
    ]
    candidates.sort(key=lambda r: dist[my_hq][r])

    new_claims = 0
    max_claims = 2
    for r in candidates:
        if new_claims >= max_claims:
            break
        if my_base_count + new_claims >= target_bases:
            break
        # ensure we will be able to pay for the base after travelling
        if not can_opt(MOVE_COST + BASE_BUILD_COST):
            break
        # closest free warrior
        best = None
        best_d = math.inf
        for w in free_pool:
            if w.id in moved:
                continue
            d = dist[w.region][r]
            if d < best_d:
                best_d = d
                best = w
        if best is None:
            break
        if do_move(best, r):
            new_claims += 1

    free_pool = [w for w in free_pool if w.id not in moved]

    # ======================================================================
    # 5. OFFENSE: gather an army at a forward rally point, strike only when a
    #    capture is guaranteed by simulation.
    # ======================================================================
    my_buildings = [b for b in S.buildings if b.side is me]
    if my_buildings:
        rally = min(my_buildings, key=lambda b: dist[b.region][opp_hq]).region
    else:
        rally = my_hq

    army_here = [w for w in free_pool if w.region == rally and w.id not in moved]

    # candidate enemy targets, nearest first from the rally
    targets = sorted(
        (r for r in set(enemy_regions) | {b.region for b in S.buildings if b.side is not me}),
        key=lambda r: dist[rally][r],
    )

    launched = False
    if len(army_here) >= 2:
        army_hps = [w.hp for w in army_here]
        for r in targets:
            ehps, turret, bhp = enemy_def_at(r)
            ok, survivors = simulate_capture(army_hps, ehps, turret, bhp)
            # require a clear win (survive with a margin) so a small mis-estimate
            # does not cost us the force
            if ok and survivors >= 2:
                for w in army_here:
                    do_move(w, r)
                launched = True
                break

    if not launched:
        # not strong enough yet: pull stragglers to the rally (free if our
        # building is there) so the army concentrates for next time.
        for w in free_pool:
            if w.region != rally and w.id not in moved:
                do_move(w, rally)

    # ======================================================================
    # 6. TRAIN at the HQ
    # ======================================================================
    spend = _train(S, M, a, region_b, my_hq, gold, spend, train_reserve)

    return a


def _train(S, M, a, region_b, my_hq, gold, spend, train_reserve):
    """Train as many warriors as we can afford, up to the HQ cap, while keeping
    `train_reserve` gold untouched. The reserve includes funds earmarked for
    bases that claimers are travelling to, so we never train ourselves out of
    being able to build the economy. Returns the new spend total."""
    hq_b = region_b.get(my_hq)
    if hq_b is None or hq_b.type is not BType.HQ:
        return spend
    cap = HQ_LEVELS[hq_b.level].train_cap
    n = 0
    while n < cap and spend + (n + 1) * TRAIN_COST <= gold - train_reserve:
        n += 1
    a.train_n = n
    return spend + n * TRAIN_COST


def main() -> None:
    M, S = parse_init()
    P = calculate_paths(M)

    while (turn := read_turn_start()) is not None:
        a = decide(S, M, P, turn)
        emit(a)
        read_turn_result(S, M, a)


if __name__ == "__main__":
    main()
