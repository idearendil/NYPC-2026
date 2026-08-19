# 참조 시뮬레이터 → 배치 GPU 환경 이식 가이드

본선 1단계(주최측 시뮬레이션 코드 + 규칙 → GPU 병렬 환경)를 위한 문서.
**그날 만드는 것 중 가장 위험한 코드**가 여기다. 네트워크보다 위험하다 —
규칙을 잘못 옮기면 크래시도 안 나고 loss 곡선도 멀쩡한데, 아무도 하지 않는 게임을
학습하게 되고 결과는 대전에서야 드러난다.

목표: **`B`개의 게임을 텐서 하나로 동시에 진행**하고, 그 결과가 참조 구현과
**정확히 일치**함을 보이는 것.

---

## 1. 먼저 읽고 적을 것 (10분)

참조 코드를 열고 아래를 종이에 적는다. 이걸 안 적고 코딩하면 반드시 다시 짠다.

1. **상태 변수 목록** — 각각의 dtype과 모양. "게임당 스칼라"인지 "게임당 N개"인지
   "게임당 가변 개수"인지.
2. **한 턴의 페이즈 순서** — 참조 코드에서 턴 함수가 호출하는 단계를 순서대로.
   (예: 건설 → 이동 명령 → 자원 소모 → 이동 → 생성 → 전투 → 수입 → 유지비)
   **이 순서가 곧 배치 env의 함수 목록이 된다.**
3. **동시성 규칙** — 두 플레이어의 행동이 동시에 처리되는가, 선후가 있는가.
   타이브레이크 규칙(같은 id 우선 등)까지 정확히.
4. **가변 크기** — 유닛 수, 지역 수처럼 게임마다 다른 것. 최대치를 정해 **패딩**한다.
5. **랜덤성** — 참조 구현에 난수가 있는가. 있으면 배치에서 재현 가능한가.

---

## 2. 자료 구조: AoS → SoA

참조 구현은 보통 객체 리스트(Array of Structs)다. 배치 env는 **필드별 텐서**
(Struct of Arrays)로 뒤집는다.

```python
# 참조: games[b].units[i].hp
# 배치: self.u_hp[b, i]   # [B, U] int64, 죽은 슬롯은 hp<=0 으로 표시
```

- 게임당 개수가 가변이면 **최대 U개로 패딩**하고 `alive` 마스크로 관리한다.
  슬롯을 지우지 말고 `hp=0`으로 두는 편이 훨씬 싸다 (재정렬이 없다).
- 지역/칸처럼 인덱스가 고정이면 `[B, N]`.
- 유닛→지역 집계는 `scatter_add_`, 지역→유닛 조회는 `gather`.

```python
# 지역별 아군 수: [B,U] -> [B,N]
cnt = torch.zeros(B, N, dtype=torch.long, device=dev)
cnt.scatter_add_(1, u_region, (alive & mine).long())
```

---

## 3. 규칙을 마스크 연산으로

**`if`는 `torch.where`, `for 게임`은 없음.** 게임별 파이썬 루프가 하나라도 남아
있으면 그게 곧 병목이다.

```python
# 나쁨
for b in range(B):
    if gold[b] >= cost: gold[b] -= cost; built[b] = True

# 좋음
can = gold >= cost
gold = gold - torch.where(can, cost, torch.zeros_like(cost))
built = built | can
```

자주 쓰는 패턴:

| 참조 코드 | 배치 |
|---|---|
| `if cond: x = a else: x = b` | `x = torch.where(cond, a, b)` |
| `for u in units: reg[u.r] += 1` | `scatter_add_` |
| `x = table[level]` | `table[level]` (텐서 인덱싱) |
| `min(a, b)` | `torch.minimum(a, b)` |
| `sorted(...)[0]` (최소 하나 고르기) | 키를 정수로 패킹해 `argmin` |
| 순차적 자원 배분 (앞에서 쓰면 뒤는 못 씀) | 순서 텐서 + 길이 T 루프 (게임 루프 아님) |

**타이브레이크를 정수 하나로 패킹**하는 기법은 특히 유용하다:

```python
# "적이 가장 적은 곳 -> 가장 가까운 곳 -> id가 작은 곳"
score = enemy_cnt * 65536 + dist * 256 + region_id
best = torch.where(candidate, score, BIG).argmin(1)
```

**진짜로 순차적인 규칙**(앞의 명령이 자원을 쓰면 뒤 명령이 못 씀)은 항목 수 T만큼
루프를 돈다. 게임 수 B가 아니라 **항목 수 T**만큼이라 B가 커져도 비용이 안 는다.

---

## 4. 에피소드 경계

끝난 게임만 골라 리셋할 수 있어야 한다. 전체 리셋은 쓸모없다 (게임마다 끝나는
시점이 다르다).

```python
def reset(self, rows):        # rows = 끝난 게임 인덱스
    self.hp[rows] = self.hp0[rows]
    ...
```

맵/초기배치가 게임마다 다르면 생성 함수를 **모듈 최상위**에 두고
`rlkit.InstanceFactory`에 넘긴다 (워커 프로세스가 미리 만들어 둔다).

---

## 5. 정합성 검증 — 반드시 한다

`rlkit.parity`가 참조 구현과 배치 env를 **같은 행동으로** 굴려 매 턴 상태를
비교해 준다. 최초 불일치를 턴/게임/필드 단위로 찍어 준다.

```python
import rlkit
refs = [ReferenceGame(seed=s) for s in range(8)]     # 주최측 코드
env  = MyBatchedEnv([g.map for g in refs], device="cuda")

ok = rlkit.parity.run(
    refs, env, turns=200, seed=0,
    sample_actions=lambda e, rng: my_random_actions(e, rng),  # -> (배치용, [게임별])
    step_env=lambda e, a: e.step(a),
    step_ref=lambda r, a: r.play_turn(a),
    snap_ref=lambda r: {"gold": r.gold, "hp": [u.hp for u in sorted(r.units, key=...)]},
    snap_env=lambda e, b: {"gold": int(e.gold[b]), "hp": e.u_hp[b].tolist()},
)
assert ok
```

두 가지만 지키면 된다:

- **전부 비교한다.** 안 본 필드는 안 맞는 필드다. 골드, hp, 소유권, 레벨, 유닛
  위치, 이동 중 여부까지 다 넣는다.
- **순서가 임의인 건 정렬한다.** 참조의 리스트 순서가 의미 없다면 안정적인 키로
  정렬해서 비교한다. 안 그러면 의미 없는 차이를 쫓다가 하네스를 안 믿게 된다.

행동은 **한 번 뽑아서 양쪽에 먹인다.** 각자 뽑으면 불일치인지 난수 차이인지
구분할 수 없다.

시간이 정말 없으면: 랜덤 플레이 20턴 × 4게임만이라도 돌린다. 그것도 못 하면
최소한 한 판을 손으로 굴려 보고 눈으로 확인한다.

---

## 6. 속도 점검

```python
# 대충 이 정도면 정상: 배치 env가 참조 대비 30~100배
python tune_batch.py mygame --B 256 512 1024 2048
```

느리면 의심할 순서:
1. **게임 루프가 남아 있다** (`for b in range(B)`) — 가장 흔하다
2. `.item()` / `.cpu()` / `bool(...)` 이 스텝 안에 있다 (매번 GPU 동기화)
3. 매 스텝 텐서를 새로 만든다 (미리 만들어 두고 `zero_()`)
4. 인덱싱에 `nonzero()`를 남발한다

---

## 7. 체크리스트

- [ ] 상태 변수 전부 `[B, ...]` 텐서, 게임별 파이썬 루프 없음
- [ ] 한 턴 = 참조와 같은 페이즈 순서
- [ ] `reset(rows)`로 끝난 게임만 리셋 가능
- [ ] 승패/무승부 판정이 참조와 일치
- [ ] `rlkit.parity.run`으로 200턴 × 8게임 일치 (최소 20턴 × 4게임)
- [ ] `tune_batch.py`로 초당 스텝 수 확인
- [ ] CPU/CUDA 양쪽에서 도는지 (디버깅은 CPU가 훨씬 편하다)
