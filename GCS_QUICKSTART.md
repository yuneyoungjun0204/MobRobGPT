# GCS 연동 실행 명령어 모음

`run_gcs_bridge.py`(MobRobGPT ↔ gcs 연동)를 실제로 돌릴 때 필요한 터미널 명령어만
모아둔 빠른 참조. 배경/설계는 [`RUN_GUIDE.md` §10](RUN_GUIDE.md#10-gcshomeyunegcs-연동으로-기동-run_gcs_bridgepy)과
`commander/gcs_bridge.py` 모듈독스트링 참고.

## 0. 사전 준비 (최초 1회)

- 파이썬 인터프리터: **`python3.10`** (torch/numpy 설치된 것 — `python3` 기본값이 아닐 수 있음)
- gcs 쪽 `vehicles.yaml`에 조종할 배(`usv1`, `usv2`, `usv3` 등)가 **`role: defender`**로
  등록돼 있어야 한다 (`command/authority.py` — 아니면 매 goto가 `role_mismatch`로 거부됨)
- `--llm openai`를 쓸 거면 `OPENAI_API_KEY` 환경변수 설정
  (`--llm heuristic`은 키 없이 바로 됨 — 기본 동작 확인용으로 추천)

## 1. 터미널 A — GCS 서버 기동 (`/home/yune/gcs`)

```bash
cd /home/yune/gcs
python3 -m server.run_server --mode udp --command
```

- `--mode`는 필수(`udp` 또는 `tcp`, 기체망 구성에 맞춰). `--command`(또는 그걸 내포하는
  `--scenario`)가 없으면 `/api/command/*/goto`가 아예 안 열린다(기본 꺼짐).
- 기본 포트는 `http://127.0.0.1:8080`(레지스트리/CLI로 변경 가능 — `--host`/`--port`).

## 2. 터미널 B — MobRobGPT 연동 실행 (`/home/yune/MobRobGPT`)

```bash
cd /home/yune/MobRobGPT
python3.10 run_gcs_bridge.py \
    --bag /home/yune/Downloads/ros_data/S03-gcs --span 8 \
    --ckpt boatattack_sim/models/u-net_map.pt --llm openai \
    --gcs-url http://127.0.0.1:8080 --ally-ids usv1,usv2,usv3
```

자주 쓰는 변형:

```bash
# API 키 없이 빠르게 동작만 확인 (휴리스틱 배정, LLM 호출 없음)
python3.10 run_gcs_bridge.py --bag /home/yune/Downloads/ros_data/S03-gcs --span 8 \
    --ckpt boatattack_sim/models/u-net_map.pt --llm heuristic \
    --gcs-url http://127.0.0.1:8080 --ally-ids usv1,usv2,usv3

# matplotlib 실시간 시각화 포함
python3.10 run_gcs_bridge.py ... --viz

# 결정 횟수 제한 + 그물 신호를 파일로 남기기
python3.10 run_gcs_bridge.py ... --max-decisions 40 --net-log /tmp/net.jsonl

# 도움말(모든 플래그)
python3.10 run_gcs_bridge.py --help
```

## 3. 단위 테스트만 (GCS 서버 없이, 실 checkpoint 사용)

```bash
cd /home/yune/MobRobGPT
python3.10 -m unittest commander.test_gcs_bridge -v
```

## 4. gcs 쪽 회귀 테스트 (이 연동이 gcs를 건드리지 않았는지 확인)

```bash
cd /home/yune/gcs
/usr/bin/python3 tests/basic/test_coverage.py
```

## 5. GCS 서버 없이 스모크 테스트만 하고 싶을 때

GCS 서버 대신 최소 stdlib `http.server` 스텁을 띄워 `run_gcs_bridge.py`의 goto POST/
`/api/state` 폴링 경로만 빠르게 확인할 수 있다. 이때는 **`--no-realtime --publish-hz 1000`**
조합을 쓴다(`--no-realtime`은 실제 시계가 아니라 CPU 속도로 도는데, publish 스로틀은 항상
실제 시계 기준이라 기본 `--publish-hz 2`로는 짧은 실행에서 goto가 거의 안 나간다 —
`commander/test_gcs_bridge.py`의 환경 테스트가 쓰는 것과 같은 조합).

```bash
python3.10 run_gcs_bridge.py \
    --bag /home/yune/Downloads/ros_data/S03-gcs --span 8 \
    --ckpt boatattack_sim/models/u-net_map.pt --llm heuristic \
    --gcs-url http://127.0.0.1:8080 --ally-ids usv1,usv2,usv3 \
    --no-realtime --publish-hz 1000 --max-decisions 40
```

## 참고 — 문제가 있으면 먼저 볼 것

- **모든 goto가 `role_mismatch`로 거부됨** → §0의 `vehicles.yaml` `role: defender` 확인
- **아무 goto도 안 나감(GCS 로그에 POST가 안 보임)** → §5처럼 `--no-realtime`을 썼는데
  `--publish-hz`를 기본값(2)으로 뒀는지 확인
- **`ptr`이 전진하지 않고 그물도 시작 안 함** → 실행 시 출력되는
  `[gcs_bridge] arrive_radius = ... m` 경고 확인. 이 값이 1 m 미만이면 `--span`이 실제
  운용 박스보다 너무 작다는 뜻
- **`--ally-ids` 개수 오류** → 체크포인트의 아군 수(P)와 정확히 같아야 함(기본
  `u-net_map.pt`는 P=3)
