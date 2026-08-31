#!/usr/bin/env python3
"""
attack_bot.py - early all-in rush bot.

Strategy (state machine):

  HOLD   : Keep the 3 starting warriors at home, train nothing, gather gold.
           Watch the opponent. On the decision turn, branch:
             - opponent has ANY warrior off its HQ  -> assume "expansion" -> RUSH
             - opponent keeps everyone home          -> assume "mirror"    -> STALL
           If we are attacked early, go straight to RUSH (defend, then counter).

  RUSH   : Mass RUSH_SIZE warriors at home. DEFEND FIRST: while an enemy force
           is inbound on our HQ, hold everyone home (garrison + turret auto-kill
           the attackers) and keep topping up to RUSH_SIZE. Once the threat is
           clear, launch the wave - send all but KEEP_HOME, leaving a worker home
           for income so the wave never starves. (-> RUSH_SENT)

  RUSH_SENT : The wave is marching / sieging. If it breaks the enemy HQ we win.
              A home worker keeps income flowing. If the wave is spent and the
              enemy HQ still stands -> UPGRADE.

  STALL  : Maintain 3 warriors, just bank gold. If the opponent reaches 6
           warriors and keeps them all home for 2 turns, stop guessing -> UPGRADE.
           If the "mirror" turns aggressor and attacks us -> RUSH (defend/counter).

  UPGRADE: Pure economy. Keep warriors == HQ work_cap (fill the work slots) and
           pour gold into upgrading our HQ every turn. Also the fallback once an
           early rush fails to end the game.

Only decide() and the Brain state machine differ from the shared boilerplate.
I/O protocol mirrors sample-code.py / bot.py.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

# ---- constants -------------------------------------------------------------
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

# ---- tunable strategy knobs ------------------------------------------------
RUSH_SIZE = 6           # warriors massed at home before the wave
KEEP_HOME = 1           # warriors kept home (as workers) when the wave launches
PASSIVE_UPGRADE_TURN = 100  # opp never attacks/expands by this turn -> UPGRADE
SIEGE_MARGIN = 20       # turns after ETA before declaring a rush "spent"


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
        return HQ_LEVELS[self.level + 1].upgrade_cost if self.type is BType.HQ else BASE_LEVELS[self.level + 1].cost


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


# ---- I/O -------------------------------------------------------------------
_token_queue: deque = deque()


def next_token() -> str:
    while not _token_queue:
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        line = line.strip()
        if line == "FINISH":
            sys.exit(0)
        if line:
            _token_queue.extend(line.split())
    return _token_queue.popleft()


def parse_init() -> tuple[GameMap, GameState]:
    M = GameMap()
    assert next_token() == "READY"
    M.my_side = Side.from_word(next_token())

    M.N = int(next_token())
    M.K = int(next_token())

    M.x = [int(next_token()) for _ in range(M.N)]
    M.y = [int(next_token()) for _ in range(M.N)]
    M.strongholds = sorted(int(next_token()) for _ in range(M.K))

    M.adj = [[] for _ in range(M.N)]
    for r in range(M.N):
        deg = int(next_token())
        M.adj[r] = sorted(int(next_token()) for _ in range(deg))

    M.my_hq = M.hq_of(M.my_side)
    M.opp_hq = M.hq_of(M.my_side.opposite)

    S = GameState()
    opp = M.my_side.opposite
    for sfx in range(1, START_WARRIORS + 1):
        S.warriors.append(Warrior(WarriorId(M.my_side, sfx), M.my_hq, HQ_LEVELS[1].warrior_hp))
        S.warriors.append(Warrior(WarriorId(opp, sfx), M.opp_hq, HQ_LEVELS[1].warrior_hp))
    S.buildings.append(Building(0, Side.LEFT, BType.HQ, 1, HQ_LEVELS[1].hp))
    S.buildings.append(Building(M.N - 1, Side.RIGHT, BType.HQ, 1, HQ_LEVELS[1].hp))
    return M, S


def read_turn_start() -> int | None:
    t = next_token()
    if t == "FINISH":
        return None
    assert t == "START"
    assert next_token() == "TURN"
    return int(next_token())


def read_turn_result(S: GameState, M: GameMap, submitted: Actions) -> None:
    for region in submitted.upgrades:
        b = S.find_building(region)
        if b is None:
            S.gold -= BASE_LEVELS[1].cost
            S.buildings.append(make_base(region, M.my_side))
        else:
            max_level = HQ_MAX_LEVEL if b.type is BType.HQ else BASE_MAX_LEVEL
            if b.level >= max_level:
                S.gold -= HQ_HEAL_COST if b.type is BType.HQ else BASE_HEAL_COST
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

    assert next_token() == "TURN"
    next_token()

    assert next_token() == "TIME"
    next_token(); next_token(); next_token(); next_token()

    assert next_token() == "UPGRADE"
    for _ in range(int(next_token())):
        s_str = next_token()
        region = int(next_token())
        s = Side.from_char(s_str[0])
        b = S.find_building(region)
        if b is None:
            S.buildings.append(make_base(region, s))
        elif b.side is not M.my_side:
            max_level = HQ_MAX_LEVEL if b.type is BType.HQ else BASE_MAX_LEVEL
            if b.level >= max_level:
                b.hp = b.current_hp()
            else:
                b.apply_upgrade()

    assert next_token() == "TRAIN"
    for _ in range(int(next_token())):
        wid = WarriorId.parse(next_token())
        hq_region = M.hq_of(wid.side)
        hq_b = S.find_building(hq_region)
        hq_level = hq_b.level if hq_b is not None else 1
        S.warriors.append(Warrior(wid, hq_region, HQ_LEVELS[hq_level].warrior_hp))

    assert next_token() == "MOVE"
    for _ in range(int(next_token())):
        wid = WarriorId.parse(next_token())
        region = int(next_token())
        w = S.find_warrior(wid)
        if w is not None:
            w.region = region
            if wid.side is M.my_side and w.state is WState.MOVING and w.region == w.target:
                w.state = WState.STATIONARY

    assert next_token() == "DAMAGE"
    for _ in range(int(next_token())):
        next_token()
        wid = WarriorId.parse(next_token())
        dmg = int(next_token())
        w = S.find_warrior(wid)
        if w is not None:
            w.hp -= dmg
    S.warriors = [w for w in S.warriors if w.hp > 0]

    assert next_token() == "SIEGE"
    for _ in range(int(next_token())):
        next_token()
        region = int(next_token())
        dmg = int(next_token())
        b = S.find_building(region)
        if b is not None:
            b.hp -= dmg
    S.buildings = [b for b in S.buildings if b.hp > 0]

    assert next_token() == "END"

    income = 0
    for b in S.buildings:
        if b.side is M.my_side:
            count = sum(1 for w in S.warriors if w.id.side is M.my_side and w.region == b.region)
            income += WORK_INCOME * min(count, b.work_cap())
    S.gold += income
    alive = sum(1 for w in S.warriors if w.id.side is M.my_side)
    S.gold = max(0, S.gold - UPKEEP_PER_WARRIOR * alive)


@dataclass
class Paths:
    hops: list[list[int]]


def calculate_paths(M: GameMap) -> Paths:
    N = M.N
    BIG = 1 << 30
    hops = [[BIG] * N for _ in range(N)]
    for s in range(N):
        hops[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in M.adj[u]:
                if hops[s][v] == BIG:
                    hops[s][v] = hops[s][u] + 1
                    q.append(v)
    return Paths(hops)


def emit(a: Actions) -> None:
    out = ["COMMAND"]
    for wid, target in a.moves:
        out.append(f"MOVE {wid} {target}")
    for r in a.upgrades:
        out.append(f"UPGRADE {r}")
    if a.train_n > 0:
        out.append(f"TRAIN {a.train_n}")
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


# ==============================================================================
# Strategy
# ==============================================================================
@dataclass
class Brain:
    mode: str = "MASS"          # MASS (build + wait), RUSH_SENT, UPGRADE
    send_turn: int | None = None
    opp_attacked: bool = False  # opponent has pushed a force at our HQ
    eta: int | None = None
    defend_hops: int | None = None  # enemies within this many hops of our HQ = threat


BRAIN = Brain()


def decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    a = Actions()
    mine = M.my_side
    opp = mine.opposite
    my_hq = M.my_hq
    opp_hq = M.opp_hq

    my_w = [w for w in S.warriors if w.id.side is mine]
    opp_w = [w for w in S.warriors if w.id.side is opp]
    my_count = len(my_w)
    opp_count = len(opp_w)
    opp_off_base = any(w.region != opp_hq for w in opp_w)
    opp_all_home = not opp_off_base
    home_w = [w for w in my_w if w.region == my_hq and w.state is WState.STATIONARY]
    hq = S.find_building(my_hq)
    enemy_hq_alive = S.find_building(opp_hq) is not None

    if BRAIN.eta is None:
        d = P.hops[my_hq][opp_hq]
        BRAIN.eta = d if d < (1 << 29) else M.N
        BRAIN.defend_hops = max(3, (BRAIN.eta + 1) // 2)

    # An enemy that has left its HQ and pushed within defend_hops of our HQ
    # is treated as an inbound attack: hold and defend before counter-rushing.
    enemy_threat = any(
        w.region != opp_hq and P.hops[w.region][my_hq] <= BRAIN.defend_hops
        for w in opp_w
    )
    away_count = sum(1 for w in my_w if w.region != my_hq)
    # opponent committed to economy (built a base) or has attacked us?
    opp_has_base = any(b.side is opp and b.type is BType.BASE for b in S.buildings)
    if enemy_threat:
        BRAIN.opp_attacked = True

    # ---- transitions -------------------------------------------------------
    if BRAIN.mode == "MASS":
        # passive opponent (no attack, no base) for long enough -> turtle/upgrade
        if (turn >= PASSIVE_UPGRADE_TURN
                and not opp_has_base and not BRAIN.opp_attacked
                and not enemy_threat):
            BRAIN.mode = "UPGRADE"

    if BRAIN.mode == "RUSH_SENT":
        spent = (my_count == 0) or (
            away_count == 0 and BRAIN.send_turn is not None and turn > BRAIN.send_turn + 1
        ) or (
            BRAIN.send_turn is not None
            and turn > BRAIN.send_turn + (BRAIN.eta or 0) + SIEGE_MARGIN
        )
        if enemy_hq_alive and spent:
            BRAIN.mode = "UPGRADE"

    # ---- actions (track gold locally so we never overspend) ----------------
    g = S.gold

    if BRAIN.mode == "MASS":
        # Always produce immediately (never waste a turn): mass toward RUSH_SIZE
        # while judging. Launch the 5-warrior wave the moment the opponent
        # commits to economy (builds a base) or once we've weathered an attack.
        # Defend first while an enemy force is inbound (garrison + turret kill
        # the attackers), then counter.
        ready = (not enemy_threat
                 and len(home_w) >= RUSH_SIZE
                 and (opp_has_base or BRAIN.opp_attacked))
        if ready:
            # launch: send all but KEEP_HOME, leaving worker(s) home for income
            for w in home_w[KEEP_HOME:]:
                if g >= MOVE_COST:
                    a.moves.append((w.id, opp_hq))
                    g -= MOVE_COST
            BRAIN.mode = "RUSH_SENT"
            BRAIN.send_turn = turn
        elif hq is not None and my_count < RUSH_SIZE:
            cap = HQ_LEVELS[hq.level].train_cap
            n = min(cap, RUSH_SIZE - my_count, g // TRAIN_COST)
            if n > 0:
                a.train_n = n
                g -= n * TRAIN_COST

    elif BRAIN.mode == "RUSH_SENT":
        # wave is marching (engine advances it for free); keep a home worker
        # alive so income covers upkeep and the wave never starves.
        home_count = sum(1 for w in my_w if w.region == my_hq)
        if hq is not None and home_count < KEEP_HOME:
            cap = HQ_LEVELS[hq.level].train_cap
            n = min(cap, KEEP_HOME - home_count, g // TRAIN_COST)
            if n > 0:
                a.train_n = n
                g -= n * TRAIN_COST

    if BRAIN.mode == "UPGRADE":
        if hq is not None:
            work_cap = HQ_LEVELS[hq.level].work_cap
            train_cap = HQ_LEVELS[hq.level].train_cap
            home_count = sum(1 for w in my_w if w.region == my_hq)
            # keep the work slots filled so income is maxed
            if home_count < work_cap:
                n = min(train_cap, work_cap - home_count, g // TRAIN_COST)
                if n > 0:
                    a.train_n = n
                    g -= n * TRAIN_COST
            # pour the rest into the HQ
            enemy_at_home = any(w.region == my_hq for w in opp_w)
            friendly_at_home = home_count > 0
            if friendly_at_home and not enemy_at_home:
                if hq.level < HQ_MAX_LEVEL:
                    cost = HQ_LEVELS[hq.level + 1].upgrade_cost
                    if g >= cost:
                        a.upgrades.append(my_hq)
                        g -= cost
                elif hq.hp < HQ_LEVELS[hq.level].hp and g >= HQ_HEAL_COST + 200:
                    a.upgrades.append(my_hq)
                    g -= HQ_HEAL_COST

    return a


def main() -> None:
    M, S = parse_init()
    P = calculate_paths(M)
    print("OK", flush=True)

    while True:
        turn = read_turn_start()
        if turn is None:
            break
        a = decide(S, M, P, turn)
        emit(a)
        read_turn_result(S, M, a)


if __name__ == "__main__":
    main()
