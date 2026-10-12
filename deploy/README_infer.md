# MobRobGPT 추론 PC 세팅 가이드

시뮬레이터의 ROS2 토픽을 받아 U-Net 정책(`u-net_map.pt`)으로 추론하고, 아군에게 경유점·그물 명령을 보내는 **추론 전용** PC를 세팅하는 절차다.

실행할 명령:
```bash
python3.10 run_sim_bridge.py --llm heuristic --publish --viz --span auto
```

---

## 0. 전제 조건

| 항목 | 요구 사항 |
|---|---|
| OS | **Ubuntu 22.04** (ROS2 Humble 공식 지원 버전) |
| Python | **3.10**. Ubuntu 22.04 기본 버전이다. ROS2 Humble의 `rclpy`가 3.10용이라 다른 버전은 안 된다 |
| GPU | 필요 없음 (CPU로 충분) |
| 네트워크 | 시뮬레이터 PC와 같은 네트워크. ROS2 DDS 멀티캐스트가 통해야 한다 |

---

## 1. 압축 풀기

```bash
mkdir -p ~/MobRobGPT
tar xzf mobrob_infer.tgz -C ~/MobRobGPT
cd ~/MobRobGPT
```

풀면 아래 구조가 된다. **상대 경로를 바꾸면 안 된다.** 실행도 반드시 이 폴더 안에서 한다.
```
MobRobGPT/
├── run_sim_bridge.py        실행 스크립트
├── run_replay_infer.py      시각화·휴리스틱 지휘관 (run_sim_bridge.py가 가져다 씀)
├── boatattack_sim/          시뮬 환경·모델 코드
│   └── models/u-net_map.pt  U-Net 체크포인트
├── commander/               ROS2 입출력·배정·브릿지 코드
└── deploy/                  이 문서, requirements_infer.txt, infer_files.txt
```

---

## 2. ROS2 Humble 설치 (apt)

이미 설치돼 있으면 건너뛴다. 공식 문서: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-ros-base    # rclpy, nav_msgs, std_msgs 포함
```

시각화 창에 한글 글자를 제대로 띄우려면 폰트도 설치한다(선택).
```bash
sudo apt install -y fonts-noto-cjk
```

---

## 3. Python 패키지 설치 (pip)

```bash
cd ~/MobRobGPT
python3.10 -m pip install --user -r deploy/requirements_infer.txt
# torch: CPU 버전 (GPU 버전도 동작하지만 용량이 수 GB라 불필요)
python3.10 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
```

| 패키지 | 용도 |
|---|---|
| `torch` | U-Net 추론 |
| `numpy` (<2) | 전 코드. 2.x는 시험하지 않았다 |
| `scipy` | 휴리스틱 배정(`linear_sum_assignment`) |
| `matplotlib`, `pillow` | `--viz` 시각화 |
| `pydantic` (>=2) | 배정 계획 스키마 |

LLM 관련 패키지(`ollama`, `openai`, `google-genai`)와 API 키는 `--llm heuristic`에서는 **필요 없다**.

---

## 4. 설치 확인

```bash
cd ~/MobRobGPT
source /opt/ros/humble/setup.bash
python3.10 -c "import rclpy, nav_msgs.msg, std_msgs.msg; print('ROS2 OK')"
python3.10 -c "from boatattack_sim.model.cnn_actor import load_cnn_actor; load_cnn_actor('boatattack_sim/models/u-net_map.pt'); print('model OK')"
python3.10 run_sim_bridge.py --help > /dev/null && echo "bridge OK"
```

시뮬레이터가 켜져 있으면 토픽이 보이는지도 확인한다.
```bash
ros2 topic list | grep /usv/
ros2 topic hz /usv/friendly_01/odometry      # 약 10 Hz 정도가 나와야 정상
```

토픽이 안 보이면 두 PC의 `ROS_DOMAIN_ID`가 같은지 확인한다(`echo $ROS_DOMAIN_ID`, 비어 있으면 0). 방화벽(UDP 멀티캐스트)도 확인한다.

---

## 5. 실행

```bash
cd ~/MobRobGPT
source /opt/ros/humble/setup.bash
python3.10 run_sim_bridge.py --llm heuristic --publish --viz --span auto
```

| 옵션 | 의미 |
|---|---|
| `--publish` | `/usv/waypoints`, `/usv/net_trigger`를 **실제로 발행**한다. 빼면 dry-run(콘솔에만 출력) |
| `--viz` | 실시간 시각화 창. 화면이 없는 서버에서는 뺀다 |
| `--span auto` | 첫 웨이브 적의 거리로 스케일을 자동 결정한다. 에피소드 중간에 붙을 때는 `--span 4400`처럼 고정하는 편이 안전하다 |
| `--cmd-log logs/cmds.jsonl` | 보낸 명령을 JSONL로 저장(선택). `mkdir -p logs`를 먼저 할 것 |
| `--out logs/decisions.json` | 결정 로그 저장(선택) |

정상 실행이면 다음 순서로 출력된다.
```
[sim_bridge] 아군 Odometry 대기 중...
[sim_bridge] span auto = ... m
[t=  3.20s] decision=000 phase=running ... 배정=[0, 1, 2] ...
[sim_bridge] PUB /usv/waypoints {...}
```

### 실행 전 체크
- 시뮬레이터 쪽 **기존 경유점 컨트롤러가 꺼져 있어야** 한다. 두 발행자가 동시에 `/usv/waypoints`를 보내면 충돌한다.
- 아군 `friendly_01~03` Odometry가 들어와야 시작한다. 들어오지 않으면 최대 300초 기다린 뒤 종료한다(`--wait-timeout`으로 조절).
- 토픽 형식은 `docs/sim_ros2_interface.md`(원본 저장소)를 참고한다.

---

## 6. 자주 나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: rclpy` | `source /opt/ros/humble/setup.bash`를 하지 않았거나 `python3`가 3.10이 아니다 → `python3.10`으로 실행 |
| `No such file ... u-net_map.pt` | `MobRobGPT/` 폴더 밖에서 실행했다 → `cd ~/MobRobGPT` 후 실행 |
| `아군 Odometry 미수신` | 토픽이 안 보인다 → §4의 `ros2 topic list`, `ROS_DOMAIN_ID`, 네트워크 확인 |
| `적 Odometry 가 대기 한도 안에 안 보였습니다` | 적이 아직 스폰되지 않았거나 에피소드가 이미 끝났다 → 적 스폰 후 실행하거나 `--span 4400` 지정 |
| 시각화 창이 안 뜸 | 화면이 없는 환경이다 → `--viz` 빼고 실행 |
