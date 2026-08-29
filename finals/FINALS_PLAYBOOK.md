# 본선 절차서

---

## 0. 상황 브리핑

**대회.** NYPC 2026 AI 부문 본선. 1대1 턴제 게임을 플레이하는 AI 프로그램을 만들어
제출하고, 다른 참가자들의 AI와 대전시켜 승패로 순위를 매긴다. 목표는 단 하나,
**남들 것보다 강한 AI를 만드는 것**이다.

**제약.**
- **게임 규칙은 본선이 시작될 때 공개된다.** 주최측이 규칙과 함께 **게임 시뮬레이션
  코드(참조 구현)**를 제공한다. 예선 게임과 완전히 다를 수 있으므로 게임에 대한
  어떤 가정도 미리 코딩해두지 않았다.
- 규칙 공개 후 **6시간** 안에: 환경 구현 → AI 설계 → 학습 → 제출.
- 학습 서버 스펙(GPU 모델·장수·VRAM, CPU 스레드)과 파이썬 가상환경 경로는
  **사용자가 알려준다.** 어디에도 적혀 있지 않으니 받아서 `CLAUDE.md`의 [환경]
  블록에 기록한다.
- 제출(예선 기준, **본선에서 재확인**): 심판과 stdin/stdout으로 통신하는 단일 파일.
  torch 없음 → 가중치를 덤프해 **numpy로 순전파 재구현**.

**방법론.** self-play PPO. 게임 수천 판을 GPU에서 동시에 굴려 데이터를 모으고,
**상대 풀**(휴리스틱 봇 + 과거 자기 스냅샷)과 겨루며 실력을 올린다.

**이미 되어 있는 것.** 게임과 무관한 부분 전부가 `rlkit/`에 구현·검증되어 있다:

> PPO 업데이트 · GAE · 롤아웃 버퍼 · 상대 풀(EMA 승률/성장/축출/영구 스냅샷) ·
> 멀티 GPU 데이터 병렬 · 체크포인트 재개 · 학습 페이즈 스케줄 · 백그라운드 인스턴스
> 생성 · 로깅 · 정합성 하네스

**그날 새로 쓰는 것.** 딱 세 가지.

| 쓸 것 | 무엇 | 어디에 |
|---|---|---|
| `Task` | 배치 환경, 한 턴의 진행, 보상, 에피소드 리셋 | `examples/toy_duel.py` 복사 |
| `Policy` | 네트워크 + 행동 샘플링 + 재평가 | 〃 |
| `ScriptedOpponent` | 휴리스틱 봇 1~2개 (상대 풀의 기준점) | 〃 |

계약은 `rlkit/interfaces.py`, 자세한 설명은 `rlkit/README.md`.

---

## 1. 대회 전에 끝내둘 것

- [ ] 학습 서버와 개발 머신 **양쪽에서** 환경 확인:
      `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"`
- [ ] 두 머신의 **가상환경 경로 / CPU 스레드 수 / GPU 스펙**을 적어 둘 것 (당일 Claude에게 알려줄 값)
- [ ] `python test_rlkit.py` (14개) 통과
- [ ] `python test_rlkit_dist.py` 통과 (멀티 GPU 로직)
- [ ] `python -m examples.toy_duel --smoke` 30초 내 완료
- [ ] GPU가 2장 이상이면 **실제 NCCL 경로** 한 번 확인:
      `python -m examples.toy_duel --gpus 2 --iters 3 --no-wandb --no-resume`
- [ ] wandb 로그인 또는 `use_wandb: false`로 쓸 준비
- [ ] `examples/toy_duel.py` 정독 — 그날은 이 파일을 복사해서 고친다
- [ ] `PORTING_GUIDE.md` 훑어보기 — 1단계에서 그대로 따라간다

---

## 2. 전체 흐름과 시간 배분

```
    사람 ────────[2단계: 설계]────────┐
                                      ↓
Claude ──[1단계: 배치 env]──→ [3단계: Task/Policy]──→[4단계: HW 튜닝]──→[5단계: 학습]
         0:00        ~1:30        ~2:20            ~2:40        2:40~
                                                                  ↑
                                            팀 ──[휴리스틱 봇]────┘ (준비되는 대로)
```

| 시각(목표) | 단계 | 완료 판정 |
|---|---|---|
| 0:00–0:20 | 규칙 정독 / Claude에게 참조 구현·규칙 전달 | Claude가 상태·페이즈 목록을 복창 |
| 0:20–1:30 | **1단계** 배치 GPU 환경 | `rlkit.parity.run` 통과 |
| (병렬) | **2단계** 설계 | obs/action/reward/네트워크 한 장 |
| 1:30–2:20 | **3단계** Task/Policy | `--smoke` 통과 + 20 iter에 avg_ep_R 상승 |
| 2:20–2:40 | **4단계** 하드웨어 튜닝 | `tune_batch.py` 결과로 B 확정 |
| 2:40– | **5단계** 본 학습 | steps/s 안정, wr_min 상승 |
| 학습 중 | 제출봇 작성 + 휴리스틱 봇 추가 | 중간 체크포인트로 대전 검증 |
| 마지막 20분 | 최종 export → 검증 → 제출 | |

**2:40에 학습이 시작되지 않으면 설계를 줄여라** (9번). 학습 시간이 곧 실력이다.

---

## 3. 1단계 — 배치 GPU 환경 (Claude)

**입력:** 주최측 참조 구현 + 규칙 문서 + 사람의 구두 설명.
**출력:** `game.py` (배치 env) + 정합성 통과.

절차는 **`PORTING_GUIDE.md`** 에 전부 있다. 요약하면:

1. 상태 변수·턴 페이즈 순서·동시성 규칙·가변 크기·랜덤성을 먼저 적는다
2. 객체 리스트(AoS) → 필드별 텐서(SoA), 가변 개수는 패딩 + `alive` 마스크
3. 모든 규칙을 `torch.where` / `scatter_add_` / `gather` 로. **게임별 파이썬 루프 금지**
4. `reset(rows)` — 끝난 게임만 리셋
5. **`rlkit.parity.run`으로 참조와 대조** (200턴 × 8게임 목표, 최소 20턴 × 4게임)

> **정합성이 통과하지 않으면 다음 단계로 넘어가지 않는다.** 규칙이 틀린 환경으로
> 학습하면 크래시도 loss 이상도 없이 엉뚱한 게임을 잘하는 AI가 나온다.

---

## 4. 2단계 — 설계 (사람, 1단계와 병렬)

아래 6개를 결정해 Claude에게 넘긴다. 이게 3단계의 입력이다.

1. **관측 (observation)** — 무엇을 텐서로 만들 것인가.
   - 개체(유닛/지역) 단위가 있으면 `[B, T, F]` + 전역 `[B, G]`(트랜스포머),
     없으면 `[B, F]`(MLP). **고민되면 MLP.**
   - 모든 피처는 `log1p` 등으로 O(1) 스케일. 부호 있으면 `sign*log1p(|x|)`.
   - **관점 정규화**: "내 것 / 상대 것" 으로만. "1P / 2P"로 쓰면 네트워크 하나로
     양쪽을 둘 수 없고 self-play가 성립하지 않는다.
2. **행동 (action)** — 어떤 분포로 쪼갤 것인가.
   - 카테고리 + 베르누이 + (개체별 타깃) 조합. 각 factor의 log-prob **합**이 `old_logp`.
   - 불법 행동은 **마스크**로 (logit에 `-1e9`, 베르누이는 확률 0).
3. **보상 (reward)** — 승 +10 / 패 −10 / 무 0 의 **희소 보상** 기본. shaping은
   되도록 넣지 않는다 (그걸 어뷰징하는 정책이 나온다).
4. **에피소드 종료** — 승패 조건 + 턴 제한 + 무승부 타이브레이크.
5. **네트워크** — 폭(`d_model=64` 기본), 층수(MLP 2층 / 트랜스포머 2~3블록),
   행동 헤드 구성. **작게 시작한다.**

---

## 5. 3단계 — Task / Policy 구현 (Claude)

`examples/toy_duel.py`를 `mygame.py`로 복사하고 위에서부터 갈아끼운다.
구조는 그대로, **내용만** 바꾼다.

```
1. 게임 클래스        → 1단계에서 만든 배치 env
2. gen_instance()     → 에피소드 초기조건 (없으면 삭제)
3. Policy             → 네트워크 + act / value / evaluate / evaluate_value
4. ScriptedOpponent   → 휴리스틱 봇 (없으면 5단계에서 추가)
5. Task               → observe / env_step / empty_opponent_out /
                        reward_done / reset_finished
6. Config, build()    → 그대로 두고 필드만 추가
```

`Policy`에서 반드시 지킬 것 (하나라도 어기면 학습이 **조용히** 망가진다):

- `act`의 `old_logp`와 `evaluate`의 `logp`는 **같은 factor를 같은 마스크로 더한 값**
- 마스크로 막힌 factor는 log-prob도 entropy도 **0**
- `store`에 넣는 텐서는 **env 내부 버퍼의 별칭이 아닐 것** (필요하면 `.clone()`)
- `empty_opponent_out`의 기본값은 **진짜 무행동 값** ("이동 없음"이 `-1`이면 0 금지)

검증:

```bash
python -m mygame --smoke                     # 끝까지 도는가 (수십 초)
python test_rlkit.py                         # 프레임워크가 멀쩡한가
python -m mygame --B 256 --steps 20000 --iters 20 --no-wandb --no-resume \
       --ckpt probe.pt                       # 20 iter 안에 학습 신호가 보이는가
```

**20 iter 안에 avg_ep_R이 오르고 ev가 0에서 떨어져 나오면** 진짜 학습이다.
안 오르면 8번 진단표로.

---

## 6. 4단계 — 학습 서버에 맞추기 (Claude)

**입력:** 사용자가 알려주는 학습 서버 스펙 — GPU 모델/장수/장당 VRAM, CPU 스레드 수,
파이썬 가상환경 경로. (받으면 `CLAUDE.md`의 [환경] 블록에 기록한다.)

### 4-1. 스펙에서 시작값 계산

| 설정 | 계산 | 예: 12GB GPU 2장 + 36스레드 |
|---|---|---|
| `--gpus` | GPU 장수 | 2 |
| `B` (총합) | 장당 VRAM ≥10GB → 2048 / ≥6GB → 1024 / 그 미만 → 512 | 2048 |
| `steps_per_iter` | `B * 50` (GAE 지평 50) | 100000 |
| `minibatch` (총합) | 4096 | 4096 |
| `instance_workers` (랭크당) | `clamp(CPU스레드 // (3 * GPU장수), 2, 12)` | 6 |
| `torch_threads` | `null` (자동 `min(8, CPU//2)`) | null |

`B`와 `minibatch`는 **총합**이다 — `--gpus N`이면 랭크당 `B/N`을 돌린다.
`instance_workers`는 **랭크당**이라 위 예에서 총 12개 프로세스가 뜬다.

### 4-2. 실측으로 B 확정

위 표는 출발점일 뿐이다. **서버에서** 실제로 재서 정한다:

```bash
python tune_batch.py mygame --B 512 1024 2048 4096 --steps 40000
```

`steps_per_iter`를 고정한 채 B만 바꿔가며 **실측 steps/s와 peak VRAM**을 찍고,
GAE 지평(`steps_per_iter / B`)이 20 밑으로 떨어지는 B는 후보에서 제외한 뒤 최적
B를 추천한다.

- 배치 env는 보통 **커널 런치 바운드**라 B를 올리면 처리량이 거의 선형으로 는다.
  멈추는 건 (a) VRAM, (b) GAE 지평 둘 중 하나다.
- 표의 마지막 B가 최적으로 나오면 곡선이 아직 안 꺾인 것이다 — 더 큰 B와 그에
  비례하는 `--steps`로 한 번 더 돌린다.
- GPU가 여러 장이면 본 학습 전에 짧게 확인:
  `python -m mygame --gpus 2 --iters 3 --no-wandb --no-resume --ckpt probe.pt`

---

## 7. 5단계 — 본 학습 + 휴리스틱 봇 추가

```bash
nohup python -m mygame --config mygame.yaml --gpus 2 > train.log 2>&1 &
tail -f train.log
```

- 크래시/OOM 나도 `checkpoint.pt`에서 자동 재개된다 (매 iter 시작 시 저장).
- 같은 명령을 다시 치면 그대로 이어서 학습한다.

### 학습 도중 휴리스틱 봇 추가하기 (계획 5단계)

팀이 휴리스틱 봇을 만들어 오면 **학습을 처음부터 다시 할 필요가 없다.**

1. `mygame.py`에 `ScriptedOpponent`를 추가하고 `build()`의 `scripted=[...]` 리스트에 넣는다
2. 학습을 멈췄다가 **같은 명령으로 재개**한다

상대 풀은 스크립트 봇 슬롯을 **이름으로** 정렬해 복원한다. 기존 봇은 측정된 EMA
승률을 그대로 유지하고, 새 봇은 0.5에서 시작하며, 저장돼 있던 상대 배정 인덱스가
알아서 재매핑된다 (`test_add_scripted_opponent_midrun`이 이 경로를 검증한다).

> **주의: 봇의 `name`은 한 번 정하면 바꾸지 마라.** 이름이 곧 슬롯의 신원이다.
> 이름을 바꾸면 그 봇은 "삭제 후 새로 추가"로 취급되어 승률 이력이 사라진다.

시작 하이퍼파라미터 (`B`/`steps_per_iter`/`instance_workers`/`--gpus`는 4단계에서
정한 값을 쓴다):

```yaml
gamma: 0.99              # 게임이 200턴 이상이면 0.997
lam: 0.95
clip: 0.2
minibatch: 4096
target_kl: null
lr: 0.0005
epochs: 4
ent_coef: 0.001
pool_add_threshold: 0.6
pool_max_size: 6
pool_snapshot_every: 60
```

---

## 8. 진단표 (로그 한 줄 읽는 법)

```
iter 42 p1 | eps 3011 | avg_ep_R +3.56 | ploss -0.0010 vloss 19.4 ent 0.26
           kl 0.0020 | ev +0.203 | ep 3/3 | pool 6/6 wr_min 0.55 | 30k steps/s
```

| 증상 | 원인 / 조치 |
|---|---|
| `ev`가 계속 0 근처 | critic이 아무것도 못 배움. 보상 스케일 확인, `gamma` 낮추기, 관측 피처 점검 |
| `avg_ep_R`이 안 움직임 | 보상이 실제로 갈리는지 확인 (`eps`가 0이면 에피소드가 안 끝나는 것) |
| `ent`가 급격히 0으로 | 정책이 조기 수렴. `ent_coef` 올리고 `lr` 낮춘다 |
| `kl`이 0.05 이상 | 한 iter에 너무 많이 움직임. `epochs` 줄이거나 `lr` 낮춤 |
| `vloss` 폭발 | 보상 스케일이 큼(±10 권장), `max_grad_norm` 확인 |
| `wr_min`이 0에 붙음 | 휴리스틱 봇이 너무 셈 → 초반엔 정상. 오래 지속되면 봇을 약화 |
| `wr_min`이 계속 1.0 | 상대 풀이 너무 약함 → `pool_add_threshold` 낮추기 |
| `pool`이 안 자람 | 위와 동일. 풀이 안 자라면 self-play 커리큘럼이 멈춘 것 |
| `steps/s`가 낮음 | `B` 올리기, `instance_workers` 올리기, env에 파이썬 루프가 없는지 |
| `instance_queue_misses`가 큼 | `instance_workers` 부족 |

---

## 9. 제출봇 (학습 도는 동안 작성)

제출 환경에는 보통 torch가 없다. 예선 방식을 그대로 따른다
(예선 저장소의 `export_weights.py`, `vanilla_bot.py` 참고):

1. 학습된 가중치를 **numpy 바이너리 한 덩어리**로 덤프
2. 제출 파일에서 numpy로 순전파를 재구현
3. **actor만** 필요하다. critic은 제출에 들어가지 않는다
4. 학습 때의 관측 계산을 **그대로** 재현해야 한다 — 여기서 학습/추론 괴리가 나면
   승률이 통째로 사라진다. 가능하면 관측 코드를 한 함수로 공유한다
5. 턴 시간 제한 / 첫 핸드셰이크 제한 확인 (예선: 첫 1초 안에 numpy 연산 금지)

**중간 체크포인트로 미리 export→대전을 한 번 돌려서 파이프라인을 검증해 둘 것.**
마지막 20분에 처음 해보면 반드시 실패한다.

---

## 10. 명령어 모음

```bash
# 환경 / 점검
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python test_rlkit.py                         # 프레임워크 테스트 (14개)
python test_rlkit_dist.py                    # 멀티 GPU 랭크 일치
python -m mygame --smoke                     # 게임 통합 스모크

# 튜닝
python tune_batch.py mygame --B 512 1024 2048 4096 --steps 40000

# 학습
python -m mygame --config mygame.yaml --gpus 2
python -m mygame --B 1024 --steps 50000 --iters 50 --no-wandb --no-resume

# 재개 (같은 명령 = checkpoint.pt 에서 자동 재개)
python -m mygame --config mygame.yaml --gpus 2

# 처음부터 다시
python -m mygame --config mygame.yaml --no-resume --ckpt fresh.pt

# 지켜보기
tail -f train.log
nvidia-smi -l 5
```

---

## 11. 자르는 순서 (시간이 모자랄 때)

1. 추가 loss 항 / 예측 헤드 → 애초에 넣지 않는다 (`ActorOut.extra_loss` 미사용)
2. 트랜스포머 → MLP
3. 휴리스틱 봇 2개 → 1개 (0개는 안 된다)
4. 관측 피처 → 승패에 직결되는 것만
5. 정합성 검증 → 200턴 × 8게임 → 20턴 × 4게임 (**0은 안 된다**)
6. 그래도 안 되면 **`B`를 줄이고** 학습 시간을 확보한다

절대 자르지 말 것: **정합성 검증, 관점 정규화, 행동 마스킹, 승/패 보상,
휴리스틱 봇 1개.**

만약 자를 일이 있으면 꼭 잘라도 되는지 먼저 물어보고 자르기.
