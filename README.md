<p align="right">
  <a href="README.md"><img alt="한국어" src="https://img.shields.io/badge/README-%ED%95%9C%EA%B5%AD%EC%96%B4-1f6feb?style=for-the-badge"></a>
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/README-English-6e7681?style=for-the-badge"></a>
</p>

# NYPC 2026 — 자가대전 강화학습 에이전트 (본선)

<img alt="1st place" src="https://img.shields.io/badge/NYPC%202026%20Master%20Track%20Finals-%EC%A2%85%ED%95%A9%201%EC%9C%84-f5b400?style=for-the-badge">

**NYPC 2026 Master Track 1st Team Solution**

NYPC 2026 전략 게임을 푸는 강화학습 에이전트입니다. 시뮬레이터(judge)를 배치 GPU 환경으로
직접 재구현한 뒤 그 위에서 self-play PPO로 처음부터 끝까지 학습시켰고, 제출 봇은
torch 없이 **numpy만으로** 동작합니다.

여기는 `main` 브랜치, 즉 **본선** 룰(400일, 안개, 더 큰 맵) 코드입니다. 예선 코드는
[`qualification_round`](../../tree/qualification_round) 브랜치에 있습니다 — 구조는
같고 룰과 학습된 네트워크가 다릅니다.

대회 리포트: [예선](docs/Qualification%20Round%20Replay.pdf) ·
[본선](docs/Final%20Round%20Replay.pdf)

| 구성 | 파일 | 설명 |
|---|---|---|
| 환경 | `src/fast_env.py` | 심판의 게임 규칙을 배치 텐서 연산으로 재작성. `B`개의 게임을 GPU에서 동시에 진행하며 `judge/testing-tool2.py`와 비트 단위로 일치. |
| 학습기 | `src/ppo_selfplay.py` | 그 환경 위의 self-play PPO. 2단계로 분해된 actor, 특권 정보를 받는 critic, 상대 풀, 멀티 GPU. |
| 제출 봇 | `src/vanilla_bot.py` | 학습된 actor를 순수 numpy로 재구현. 턴당 추론 1회, 탐색 없음. `data.bin`을 읽음. |

## 게임 규칙

두 플레이어가 무작위로 생성된 지역 그래프(`N`은 181~249, 거점 `K`는 대략 √N) 위에서
싸웁니다. 각 진영은 본부 1개, 전사 3명, 금화 750으로 시작하고 **400일**이 주어집니다.

- **건물.** 거점에는 기지를 지을 수 있습니다(500금, 최대 레벨 3). 본부는 레벨 5까지
  올라갑니다(600 / 1000 / 2000 / 3000금). 레벨이 오르면 체력, 포탑 공격력, 훈련
  한도, 그리고 *일자리 수*(그곳에서 금화를 벌 수 있는 전사 수)가 늘어납니다.
- **경제.** 일하는 전사 1명당 하루 15금을 벌고, 전사 1명당 하루 2금이 유지비로
  나갑니다. 훈련은 120금, 이동은 10금입니다.
- **안개.** 적 유닛과 건물은 내 전사나 건물로부터 **2칸 이내**에서만 보입니다. 그
  바깥은 기억하고 추측해야 합니다.
- **승리 조건.** 상대 본부를 파괴하거나, 400일이 지난 시점에 건물 체력 합이 앞서면
  이깁니다.

본선 시뮬레이터는 `judge/testing-tool2.py`입니다. CLI, 맵 포맷, 로그 포맷은
[docs/testing-tool.md](docs/testing-tool.md)를 보세요.

## 빠른 시작

아래 명령들은 모두 **저장소 루트**에서 실행합니다. torch를 **먼저** 각자 CUDA 빌드에
맞춰 설치한 뒤 나머지를 설치합니다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA 버전에 맞게
pip install -r requirements.txt
```

```bash
# 1. 파이프라인 전체 점검 (작은 네트워크로 몇 iteration만)
python src/ppo_selfplay.py --smoke --ckpt smoke.pt

# 2. 실제 학습 (config.yaml을 먼저 수정; 데이터 병렬은 --gpus N)
python src/ppo_selfplay.py --config config.yaml --gpus 1

# 3. 체크포인트를 봇이 읽는 torch-free 가중치로 변환
python src/export_weights.py --ckpt checkpoint.pt --out data.bin

# 4. numpy 봇이 torch 파이프라인을 그대로 재현하는지 검증 (피처 + forward)
python tests/verify_np_bot.py

# 5. 실제로 두어 보고, 세기를 측정
python tools/run_match.py --seed 42                     # replay.log 생성
python tools/power_test.py --games 40 --old-weights old.bin
```

`config.yaml`의 `use_wandb: true`이면 Weights & Biases로 로깅합니다(`--no-wandb`로
끌 수 있음). 인증 정보는 환경변수 `WANDB_API_KEY`나 `wandb login`에서 가져오며,
저장소에는 아무것도 들어 있지 않습니다.

## 동작 원리

**배치 환경.** 강화학습에는 게임 하나당 프로세스 하나를 띄우는 기존 시뮬레이터가 감당할 수 없는
양의 대국이 필요합니다. 그래서 `fast_env.py`가 규칙 전체를 `B`개 게임에 대한 텐서
연산으로 재구현합니다. 심판과 **비트 단위로 일치**함을 검증했습니다 — 금화, 모든
건물의 소유/종류/레벨/체력, 그리고 모든 전사까지. 턴 종료 시점(`test_fast_env.py`)과
매 페이즈 직후(`test_phases.py`) 양쪽에서 확인합니다. 롤아웃은 커널 실행(launch)에
묶여 있어서 처리량이 `B`에 거의 선형으로 비례합니다(기본 설정은 `B: 12288`). 자세한
내용은 [docs/fast_env.md](docs/fast_env.md)에 있습니다.

**분해된 행동 공간.** 한 턴의 행동은 선택 하나가 아니라 명령의 *집합*이라, actor를
둘로 나눴습니다.

- **T1** — 거점 하나당 토큰 하나를 두는 3블록 트랜스포머(토큰 피처 32개 + 전역 피처
  14개, 전부 log1p / 정규화). 5차원 head에서 토큰별 건설 베르누이 확률과, 토큰에 대해
  마스크 평균을 낸 4-way "무엇을 훈련/업그레이드할지" softmax가 나옵니다.
- **T2** — 이동 출발지마다 다시 도는 2블록 트랜스포머. T1의 토큰 출력에 추가 피처
  8개를 붙여 받고, 토큰에 대한 softmax(= 이 병력을 어디로 보낼지)를 냅니다.

금화는 두 번 강제됩니다. 샘플링 *전에* 지불 가능 마스크를 씌우고, 그 뒤 남은 금화를
탐욕적으로 배분합니다(건설 → 이동 → 훈련). 탐욕 배분 단계에서 잘린 명령은 PPO에서
여전히 "선택된 것"으로 세고, 확률 0으로 마스킹된 명령은 세지 않습니다.

**Critic과 보조 과제.** 같은 형태의 별도 인코더가 가치(value)를 예측합니다. actor와
critic 모두 7차원 보조 head를 답니다 — 거점별로 1~5턴 안에 도달 가능한 적 병력, 그리고
상대의 숨겨진 금화에 대한 전역 추정치입니다. 보조 타깃은 인코더를 다듬기 위한 학습
전용 감독 신호이고, 제출 봇은 이 head를 절대 돌리지 않습니다.

**자가 대전.** 에이전트(LEFT)는 상대 풀(RIGHT)과 대전하며, 현재 가장 어려운 상대가
EMA 승률 기준으로 더 자주 뽑힙니다. 풀의 앞 세 자리는 고정된 스크립트 봇 —
`bots/final_rush_bot.py`, `bots/final_rush_bot2.py`, `bots/final_defence_bot.py`를
배치·벡터화해 이식한 것 — 이고 절대 교체되지 않습니다. 덕분에 풀이 자가 대전 동질화로
무너지지 않습니다. 에이전트가 모든 상대를 `pool_add_threshold` 이상으로 이기면 자기
자신을 풀에 스냅샷하고, `perm_snapshot_every` iteration마다 그 스냅샷이 영구 슬롯이
됩니다.

**제출 봇.** torch는 import만으로 약 2.3초라 심판의 1초 핸드셰이크 안에 들어가지
않습니다. 그래서 `export_weights.py`가 actor를 numpy `.npz`(`data.bin`)로 펴내고,
`vanilla_bot.py`가 forward 연산 — layernorm, 멀티헤드 어텐션, GELU — 을 numpy로 직접
구현합니다. 또한 프로토콜이 절대 알려주지 않는 은닉 상태를 복원합니다. 지역별로 상대에
대한 신념(종류, 레벨, 체력, 주둔 병력, 그리고 그 관측의 나이)을 들고 다니면서, 직접
계산한 시야 안에서는 갱신하고 바깥에서는 나이를 먹입니다. `fast_env`의 안개 처리를
그대로 따라 하므로 봇은 학습 때와 똑같은 피처를 보게 됩니다.

## 파일 구성

```
config.yaml              학습 하이퍼파라미터 전부
requirements.txt

src/                     프로젝트 본체 (이 디렉터리가 import 경로)
  fast_env.py            배치 GPU 환경 (+ 관측 인코더)
  map_gen.py             학습용 무작위 맵 백그라운드 생성
  ppo_selfplay.py        self-play PPO 학습기 (네트워크, 롤아웃, 풀, 업데이트 루프)
  export_weights.py      checkpoint.pt -> data.bin (torch 불필요)
  vanilla_bot.py         실제 제출 봇 (numpy 전용)

judge/                   주최측이 배포한 시뮬레이터
  testing-tool2.py       본선 시뮬레이터
  config.ini             시뮬레이터 설정 예시
  sample-code.py         주최측 프로토콜 예제 봇

bots/                    규칙 기반 봇
  final_rush_bot.py      스크립트 상대 봇. ppo_selfplay가 이 셋을 배치 이식해
  final_rush_bot2.py       상대 풀의 고정 슬롯으로 사용
  final_defence_bot.py
  basic_bot.py           예선 시절 베이스라인. 시뮬레이터에 그대로 물려
  rush_bot.py              스파링 상대로 쓸 수 있음
  japper_bot.py

tools/                   실행 / 평가 스크립트
  run_match.py           봇 대 봇 1판 -> 리플레이 로그
  power_test.py          가중치 파일 두 개의 승률 비교 (진영 교대)
  mode_power_test.py     greedy vs stochastic 행동 선택의 승률 비교
  benchmark_env.py       환경 처리량 벤치마크

tests/                   검증 (전부 스크립트로 직접 실행)
  test_fast_env.py       턴 종료 시점 심판과의 비트 단위 일치
  test_phases.py         페이즈마다 비트 단위 일치
  test_observe.py        관측 shape, 토큰 마스크
  test_encoder.py        모든 인코더 피처 재계산 대조
  verify_np_bot.py       numpy 봇 == torch 파이프라인

docs/                    시뮬레이터 문서, fast_env 상세 설명, 예선/본선 리포트 PDF
```

`tests/`와 `tools/`의 스크립트는 실행될 때 스스로 `src/`를 `sys.path`에 넣으므로,
설치 없이 저장소 루트에서 그대로 실행하면 됩니다.

## 테스트

```bash
python tests/test_fast_env.py    # 턴 종료 시점 심판과의 비트 단위 일치 (느림)
python tests/test_phases.py      # 페이즈마다 비트 단위 일치, cpu + cuda
python tests/test_observe.py     # 관측 shape, 크기가 섞인 배치의 토큰 마스크
python tests/test_encoder.py     # 모든 인코더 피처를 독립적으로 재계산해 대조
python tests/verify_np_bot.py    # numpy 봇 == torch 파이프라인 (checkpoint.pt + data.bin 필요)
```

네트워크를 고쳤다면 `verify_np_bot.py`만으로는 **부족합니다**. 이건 forward 연산만
확인할 뿐 행동 선택은 보지 않습니다. `ppo_selfplay.sample_policy`와
`vanilla_bot._select_action`을 같은 상태에 대해 양쪽 모두 결정론적으로 고정해 놓고
비교하세요. 마스크의 불일치, 특히 마스크 *극성*의 뒤집힘은 그 외 어디에서도 드러나지
않습니다. (비교할 때는 금화를 넉넉히 주세요 — 금화가 부족하면 탐욕 배분기가 무작위로
섞은 부분집합만 지불하므로, 예산에 걸린 상태는 정당한 이유로 불일치합니다.)

## 참고

- 가중치는 저장소에 없습니다. 직접 학습시켜야 봇이 읽을 것이 생깁니다.
- 전부 GPU 1장에서 돌아가며, `--gpus N`은 단순 데이터 병렬입니다(iteration당 데이터는
  같고 랭크로 나뉠 뿐). `minibatch`는 GPU 수로 나누어떨어져야 합니다.
- `python src/ppo_selfplay.py --smoke`가 가장 빠른 전 구간 점검입니다. 작은 네트워크,
  적은 게임 수, 2 iteration, wandb 없음.
