#!/usr/bin/env python3
"""
japper_bot.py - two-base "japper" expansion bot.

A scripted opponent that opens with a fast double expansion, reads the enemy's
turn-6 posture, defends an early all-in if it comes, then settles into a
rally-point war machine. It is deliberately simple and deterministic so it makes
a stable sparring partner in the self-play opponent pool (mirrored by
``ppo_selfplay.japper_action``; see that batched port for the RL side).

Strategy (state machine):

  SETUP  : Turn 1 keep one starting warrior home and send the other two to the
           two nearest EMPTY strongholds (거점) -- one each. The nearer one is
           reached first: build a base there immediately. The second warrior just
           waits at its stronghold until turn 6.

  turn 6 : Count the enemy warriors that have LEFT their HQ. If >= 5 (call this
           group size n) it is very likely an early all-in at our HQ -> DEFENSE.
           Otherwise -> MAIN (settle the second stronghold as the rally point).

  DEFENSE: Recall the waiting warrior home and train every turn until the HQ
           garrison reaches n-1 (the returning warrior counts). Once massed,
           watch the enemy group: if it stalls or veers off toward some other
           stronghold, or if it crashes into our HQ and is wiped out (garrison +
           turret hold), transition on -> TRANSITION.

  TRANSITION: Push back out. If >= 2 warriors are home, keep one and send the
           rest to the nearest empty stronghold; otherwise train up to two first,
           then send one. When a warrior reaches the stronghold -> MAIN.

  MAIN   : Build a base at the warriors' stronghold: this is the rally point.
           Thereafter train whenever gold allows and funnel every warrior through
           the rally point. When the rally holds >= 6, keep one and send five at
           the nearest enemy base/HQ. If that target falls with >= 5 survivors,
           chain to the next nearest enemy building; with <= 4 survivors, retreat
           to the rally and rebuild the wave. Repeat to the end of the game.

Only the Brain state machine and decide() differ from rush_bot.py; the I/O
protocol, parsing and bookkeeping are the shared boilerplate (mirrors
sample-code.py / rush_bot.py).
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
RUSH_DETECT_TURN = 6    # turn we read the enemy's posture on
RUSH_GROUP_MIN = 5      # enemy warriors off their HQ at turn 6 => treat as all-in
WAVE_SIZE = 5           # warriors sent per attack wave
WAVE_TRIGGER = 6        # rally garrison that launches a wave (keep 1, send 5)


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
    mode: str = "SETUP"             # SETUP, DEFENSE, TRANSITION, MAIN
    sa: int | None = None           # nearest empty stronghold (first base)
    sb: int | None = None           # 2nd nearest empty stronghold (waits/recalled)
    sa_wid: WarriorId | None = None  # warrior sent to sa
    sb_wid: WarriorId | None = None  # warrior sent to sb
    dispatched: bool = False        # turn-1 split issued?
    n: int = 0                      # enemy all-in group size (turn 6)
    gathered: bool = False          # DEFENSE reached n-1 at home?
    defend_hops: int | None = None  # enemies within this many hops of our HQ = threat
    threat_min: int | None = None   # closest the strike group has ever gotten to us
    threat_prev: int | None = None  # its distance last turn (to detect approach/stall)
    threat_stall: int = 0           # turns the group has NOT gotten closer
    trans_tgt: int | None = None    # TRANSITION destination stronghold
    rally: int | None = None        # MAIN rally point (a base region)


BRAIN = Brain()


def decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    a = Actions()
    B = BRAIN
    mine = M.my_side
    opp = mine.opposite
    my_hq = M.my_hq
    opp_hq = M.opp_hq
    my_w = [w for w in S.warriors if w.id.side is mine]
    opp_w = [w for w in S.warriors if w.id.side is opp]
    hq = S.find_building(my_hq)
    g = [S.gold]     # boxed so nested helpers can spend
    moved_ids: set = set()   # at most one move command per warrior per turn

    def hops(u: int, v: int) -> int:
        d = P.hops[u][v]
        return d if d < (1 << 29) else M.N

    if B.defend_hops is None:
        B.defend_hops = max(3, (hops(my_hq, opp_hq) + 1) // 2)

    def move_w(w: Warrior, target: int) -> bool:
        if w.id in moved_ids:
            return False
        b = S.find_building(target)
        cost = 0 if (b is not None and b.side is mine) else MOVE_COST
        if g[0] >= cost:
            a.moves.append((w.id, target))
            g[0] -= cost
            moved_ids.add(w.id)
            return True
        return False

    def train_to(target_home: int) -> None:
        # train toward `target_home` stationary+inbound warriors at our HQ
        if hq is None:
            return
        home = sum(1 for w in my_w if w.region == my_hq
                   and w.state is WState.STATIONARY)
        inbound = sum(1 for w in my_w if w.state is WState.MOVING
                      and w.target == my_hq)
        cap = HQ_LEVELS[hq.level].train_cap
        n = min(cap, target_home - home - inbound, g[0] // TRAIN_COST)
        if n > 0:
            a.train_n += n
            g[0] -= n * TRAIN_COST

    def empty_strongholds() -> list[int]:
        return [r for r in M.strongholds if S.find_building(r) is None]

    def nearest_empty(src: int) -> int | None:
        es = empty_strongholds()
        return min(es, key=lambda r: hops(src, r)) if es else None

    def nearest_enemy_building(src: int) -> int | None:
        eb = [b.region for b in S.buildings if b.side is opp]
        return min(eb, key=lambda r: hops(src, r)) if eb else None

    def home_stationary() -> list[Warrior]:
        return [w for w in my_w if w.region == my_hq and w.state is WState.STATIONARY]

    # ---- SETUP -------------------------------------------------------------
    if B.mode == "SETUP":
        if not B.dispatched:
            # turn 1: keep one warrior home, send the other two to the two
            # nearest empty strongholds (one each). Nearer = sa (builds first).
            es = sorted(empty_strongholds(), key=lambda r: hops(my_hq, r))
            starters = sorted(home_stationary(), key=lambda w: w.id.num)
            if es and len(starters) >= 2:
                B.sa = es[0]
                B.sb = es[1] if len(es) >= 2 else None
                move_w(starters[1], B.sa)
                B.sa_wid = starters[1].id
                if B.sb is not None:
                    move_w(starters[2], B.sb)
                    B.sb_wid = starters[2].id
            B.dispatched = True

        # the first-reached stronghold (sa) builds a base as soon as its warrior
        # arrives; the sb warrior just waits there until turn 6.
        if B.sa is not None and S.find_building(B.sa) is None:
            sa_w = S.find_warrior(B.sa_wid) if B.sa_wid else None
            if sa_w is not None and sa_w.region == B.sa \
                    and sa_w.state is WState.STATIONARY and g[0] >= BASE_LEVELS[1].cost:
                a.upgrades.append(B.sa)
                g[0] -= BASE_LEVELS[1].cost

        if turn >= RUSH_DETECT_TURN:
            n = sum(1 for w in opp_w if w.region != opp_hq)
            if n >= RUSH_GROUP_MIN:
                B.mode = "DEFENSE"
                B.n = n
                B.gathered = False
                B.threat_min = B.threat_prev = None
                B.threat_stall = 0
                sb_w = S.find_warrior(B.sb_wid) if B.sb_wid else None
                if sb_w is not None and sb_w.region != my_hq \
                        and sb_w.state is WState.STATIONARY:
                    move_w(sb_w, my_hq)
            else:
                # settle the second stronghold (where our warrior waits) as rally
                B.mode = "MAIN"
                B.rally = B.sb if B.sb is not None else nearest_empty(my_hq)

    # ---- DEFENSE -----------------------------------------------------------
    # elif: on a turn that changes mode, only the entering block runs; the new
    # mode's block takes over next turn (avoids double-commanding a warrior).
    elif B.mode == "DEFENSE":
        sb_w = S.find_warrior(B.sb_wid) if B.sb_wid else None
        if sb_w is not None and sb_w.region != my_hq \
                and sb_w.state is WState.STATIONARY:
            move_w(sb_w, my_hq)          # keep recalling until home

        home = home_stationary()
        if not B.gathered:
            if len(home) >= B.n - 1:
                B.gathered = True
            else:
                train_to(B.n - 1)
        if B.gathered:
            # Watch the strike group by its distance to our HQ. While it keeps
            # closing in, HOLD. Leave DEFENSE only once the threat resolves: the
            # group reaches us and is wiped/pushed off (came & cleared), it never
            # arrives and stalls for a few turns (stopped or veered to another
            # stronghold), or no off-HQ enemy remains at all.
            offhq = [hops(w.region, my_hq) for w in opp_w if w.region != opp_hq]
            cur = min(offhq) if offhq else None
            enemy_at_hq = any(w.region == my_hq for w in opp_w)
            if cur is not None:
                B.threat_min = cur if B.threat_min is None else min(B.threat_min, cur)
                if B.threat_prev is not None and cur >= B.threat_prev:
                    B.threat_stall += 1
                else:
                    B.threat_stall = 0
                B.threat_prev = cur
            engaged = B.threat_min is not None and B.threat_min <= B.defend_hops
            if enemy_at_hq:
                leave = False                              # battle at the HQ: hold
            elif cur is None:
                leave = True                               # group spent / repelled
            elif engaged and cur > B.defend_hops:
                leave = True                               # wave came and has passed
            elif not engaged and B.threat_stall >= 3:
                leave = True                               # stopped or diverted away
            else:
                leave = False                              # still approaching: hold
            if leave:
                B.mode = "TRANSITION"
                B.trans_tgt = None

    # ---- TRANSITION --------------------------------------------------------
    elif B.mode == "TRANSITION":
        if B.trans_tgt is None or S.find_building(B.trans_tgt) is not None:
            B.trans_tgt = nearest_empty(my_hq)
        tgt = B.trans_tgt
        if tgt is not None:
            home = home_stationary()
            if len(home) >= 2:
                for w in home[1:]:       # keep one home, push the rest out
                    move_w(w, tgt)
            else:
                train_to(2)
            arrived = [w for w in my_w if w.region == tgt
                       and w.state is WState.STATIONARY]
            if arrived:
                B.mode = "MAIN"
                B.rally = tgt

    # ---- MAIN --------------------------------------------------------------
    elif B.mode == "MAIN":
        rally = B.rally
        if rally is not None:
            rb = S.find_building(rally)
            rally_here = [w for w in my_w if w.region == rally
                          and w.state is WState.STATIONARY]

            # funnel HQ warriors to the rally, but keep the HQ work slots filled
            # so income never collapses (the spec sends *trained* warriors to the
            # rally; the home worker that funds them stays put).
            def funnel_hq() -> None:
                hq_keep = HQ_LEVELS[hq.level].work_cap if hq is not None else 1
                hq_here = sorted((w for w in my_w if w.region == my_hq
                                  and w.state is WState.STATIONARY),
                                 key=lambda w: w.id.num)
                surplus = hq_here[hq_keep:]
                if not surplus:
                    return
                # other owned BASES (not the rally) that currently have no friendly
                # warrior stationed and none inbound -> restaff them first (keeps
                # their income up) before feeding the rally.
                empty_bases = []
                for b in S.buildings:
                    if b.side is mine and b.type is BType.BASE and b.region != rally:
                        here = sum(1 for w in my_w if w.region == b.region
                                   and w.state is WState.STATIONARY)
                        inbound = sum(1 for w in my_w if w.state is WState.MOVING
                                      and w.target == b.region)
                        if here + inbound == 0:
                            empty_bases.append(b.region)
                empty_bases.sort(key=lambda r: hops(my_hq, r))
                for i, w in enumerate(surplus):
                    move_w(w, empty_bases[i] if i < len(empty_bases) else rally)

            if rb is None:
                # PRIORITY: stand up the rally base (the economic engine) before
                # spending any gold on training -- otherwise 120-gold trains keep
                # draining us below the 300 the base costs and it never gets built.
                if rally_here and g[0] >= BASE_LEVELS[1].cost:
                    a.upgrades.append(rally)
                    g[0] -= BASE_LEVELS[1].cost
                funnel_hq()
                return a

            # base is up. Do the (cheap) moves FIRST so a wave's move cost
            # (MOVE_COST/warrior onto enemy ground) is funded before training drains
            # gold below 120; only then train with whatever remains -- otherwise the
            # leftover could pay for only part of a wave (a partial launch).
            funnel_hq()

            # attack: EVERY turn the rally reaches the trigger, send all but one at
            # the nearest enemy building (concurrent waves -- no single-wave gate).
            # ALL-OR-NOTHING: only launch if we can pay the whole wave's move cost,
            # so it never dribbles out just some of the group.
            rally_stat = sorted((w for w in my_w if w.region == rally
                                 and w.state is WState.STATIONARY),
                                key=lambda w: w.id.num)
            if len(rally_stat) >= WAVE_TRIGGER:
                wave = rally_stat[1:]               # keep one home to work
                tgt = nearest_enemy_building(rally)
                if tgt is not None and g[0] >= MOVE_COST * len(wave):
                    for w in wave:
                        move_w(w, tgt)

            # warriors left standing on razed targets (regions that are neither my
            # building nor a live enemy building). Treat them as ONE body: if >=5
            # remain in TOTAL they ALL chain to the nearest enemy building (from
            # where most of them stand) -- all-or-nothing, else hold this turn until
            # affordable; otherwise (<=4) they ALL retreat to the rally (a free move).
            my_bldg = {b.region for b in S.buildings if b.side is mine}
            fieldw = []
            for w in my_w:
                if w.state is WState.STATIONARY and w.region not in my_bldg:
                    eb = S.find_building(w.region)
                    if eb is not None and eb.side is opp:
                        continue                    # still besieging a live target
                    fieldw.append(w)
            if fieldw:
                if len(fieldw) >= WAVE_SIZE:
                    counts: dict[int, int] = {}
                    for w in fieldw:
                        counts[w.region] = counts.get(w.region, 0) + 1
                    main_region = max(counts, key=counts.get)
                    nt = nearest_enemy_building(main_region)
                    if nt is not None and g[0] >= MOVE_COST * len(fieldw):
                        for w in fieldw:
                            move_w(w, nt)           # chain the whole body together
                    # can't pay the full chain (or nothing left to hit) -> hold
                else:
                    for w in fieldw:                # <=4 survivors: retreat (free)
                        move_w(w, rally)

            # train with whatever gold remains after funding the moves
            if hq is not None and g[0] >= TRAIN_COST:
                cap = HQ_LEVELS[hq.level].train_cap
                n = min(cap, g[0] // TRAIN_COST)
                if n > 0:
                    a.train_n += n
                    g[0] -= n * TRAIN_COST

    # note: g[] is only a local overspend guard; read_turn_result does the real
    # gold accounting from the emitted actions (never write g[0] back to S.gold).
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
