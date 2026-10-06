# 시뮬레이터 ↔ MobRobGPT 지휘 브릿지: ROS2 인터페이스

MobRobGPT 브릿지(`run_sim_bridge.py`)가 Unreal 시뮬레이터와 주고받는 ROS2 토픽을 정리한 문서다.

- 하는 일: 시뮬레이터가 내보내는 선박 상태를 받아 U-Net 정책(`u-net_map.pt`)으로 추론한 뒤, 아군 3척에게 **x/y 경유점**과 **그물 투하** 명령을 보낸다.
- 기준 데이터: 형식과 QoS는 2026-10-04 녹화본 `usv_20261004_194054`에서 확인한 값이다.

---

## 1. 좌표·단위 규약

| 항목 | 규약 |
|---|---|
| 프레임 | `map` (로컬 평면, 위경도 아님) |
| 원점 | 모선(`mothership`) = (0, 0) |
| 축 | x = 동(East) m, y = 북(North) m |
| 방위(yaw) | Odometry 쿼터니언의 ENU yaw. +x(동)이 0°, 반시계가 +. 브릿지가 내부에서 nav 방위(북 0°, 시계 +)로 바꾼다 |
| 명령 좌표 | `/usv/waypoints`의 `x_m`, `y_m`는 Odometry의 x, y와 **같은 축**이다 |

> ⚠ **스폰 명령은 축이 반대다.** `/usv/commands`의 `spawn.pos`만 Odometry와 x/y가 뒤바뀌어 있다. 예: `{"x_m":1870.2,"y_m":-42.8}`로 스폰된 적의 Odometry는 (x=-42.8, y=1870.2)다. 브릿지는 스폰 명령을 쓰지 않으므로 영향은 없다. 다만 의도된 규약인지 확인 부탁한다.

---

## 2. 브릿지가 **받는** 토픽 (구독)

| 토픽 | 타입 | QoS | 사용하는 필드 | 용도 |
|---|---|---|---|---|
| `/usv/friendly_01/odometry`<br>`/usv/friendly_02/odometry`<br>`/usv/friendly_03/odometry` | `nav_msgs/Odometry` | BEST_EFFORT, VOLATILE | `pose.pose.position.x/y`, `pose.pose.orientation` | 아군 위치·방위 |
| `/usv/enemy_01/odometry` … `/usv/enemy_07/odometry` | `nav_msgs/Odometry` | BEST_EFFORT, VOLATILE | 위와 동일 | 적 위치·방위 |
| `/usv/mothership/odometry` | `nav_msgs/Odometry` | BEST_EFFORT, VOLATILE | `pose.pose.position.x/y` | 맵 중앙(원점) 기준 |
| `/usv/sim_state` | `std_msgs/String` (JSON) | RELIABLE, VOLATILE | `sim_id`, `phase`, `ships[].captured`, `ships[].nets_left` | 포획 여부, 에피소드 단계, 명령에 실을 `sim_id` |
| `/usv/assignment` | `std_msgs/String` (JSON) | RELIABLE, VOLATILE | `by_ship` | **비교 로그용으로만** 쓴다. 기존 컨트롤러 배정과 우리 배정을 나란히 기록한다 |

### 생존 판정 규칙
- **아군**: 최근 **1.0초** 안에 Odometry를 받았으면 살아 있는 것으로 본다.
- **적**: 위 조건을 만족하고, 동시에 `/usv/sim_state`의 `ships[]`에 `"captured": true`로 올라 있지 **않아야** 한다.
  - 포획된 적도 Odometry는 계속 나오기 때문에 이 규칙이 필요하다.

### `/usv/sim_state` 예시 (녹화본)
```json
{"sim_id":"run-bd1fb9c5-...","t_sim":320.29,"t_wall":1791110593.893,"phase":"failure",
 "ships":[{"ship":"mothership"},{"ship":"friendly_01","nets_left":2},
          {"ship":"enemy_04","captured":true,"group":"w1791110497"}],
 "nets":[{"net":"net_01","ship":"friendly_01","state":"deployed",
          "start":{"x_m":8.25,"y_m":300.14},"end":{"x_m":12.06,"y_m":338.66}}]}
```
- `phase`는 `ready` → `running` → `failure` 순서로 확인됐다.
- 성공 시 어떤 값이 나오는지는 아직 못 봤다(아래 §5 질문 1).

---

## 3. 브릿지가 **보내는** 토픽 (발행)

두 토픽 모두 `std_msgs/String`(JSON), RELIABLE / VOLATILE, depth 10이다. 녹화본의 기존 발행자와 같은 QoS다.

### 3-1. `/usv/waypoints`: 배별 경유점

```json
{
  "sim_id": "run-bd1fb9c5-4a44-9008-c8d7-bb978780eee3",
  "ship": "friendly_02",
  "points": [{"x_m": 725.948, "y_m": -243.586}, {"x_m": 626.37, "y_m": -441.603}],
  "cruise_kn": 40.0,
  "stop_at_final": false,
  "accept_radius_m": 4.0
}
```

| 필드 | 의미 |
|---|---|
| `sim_id` | `/usv/sim_state`에서 마지막으로 받은 값. 아직 못 받았으면 `null` |
| `ship` | `friendly_01` / `friendly_02` / `friendly_03` |
| `points` | **남은** 경유점(첫 점부터 순서대로 따라간다). 보통 2개: `[이동점, 그물 끝점]` |
| `cruise_kn` | 순항 속도. 기본 40 kn |
| `stop_at_final` | 기본 `false` |
| `accept_radius_m` | 경유점 도달 판정 반경. 기본 4 m |

`points`가 어떻게 채워지는지:
- **배정된 배**: `[이동점, 그물 끝점]`. 이동점에 도착하면 그 다음 점까지가 그물 구간이다.
- **이동점에 이미 도착한 배**: 남은 `[그물 끝점]` 1개.
- **미배정 배, 또는 그물을 다 쓴 배**: `[]`(빈 목록) = 정지 의도. 다만 이미 그물을 치고 있는 배는 그 구간을 끝까지 간다.

### 3-2. `/usv/net_trigger`: 그물 투하

```json
{"sim_id": "run-bd1fb9c5-4a44-9008-c8d7-bb978780eee3", "ship": "friendly_01", "lay": true}
```

- **그물 구간 1개당 1회만** 보낸다. 투하 1회 = 그물 1장(약 40 m)이고, 배마다 3장이다.
- 보내는 시점: 배가 이동점에 도착 판정을 받고 그물 구간을 시작할 때.
- `lay: false`는 보내지 않는다.

---

## 4. 발행 타이밍·동작 규칙

| 항목 | 값 |
|---|---|
| 결정(재계획) 주기 | 약 3초. 스케일에 따라 정해지며, 이번 녹화본 기준 약 3.2초다 |
| `/usv/waypoints` 최대 빈도 | 배당 2 Hz |
| 중복 억제 | 같은 배에 0.1 m 단위까지 같은 `points`면 **다시 보내지 않는다** |
| `/usv/net_trigger` | 그물 구간 시작 시 1회. 주기 발행 없음 |
| 명령 대상 | 최근 1초 안에 Odometry가 들어온 아군만. 위치를 모르는 배에는 보내지 않는다 |
| 기본 모드 | **dry-run**. 아무것도 발행하지 않고 콘솔·로그에만 남긴다. `--publish`를 줘야 실제로 발행한다 |

### 실행 예
```bash
# 녹화본 재생 + 그림자 추론 (발행 없음)
ros2 bag play usv_20261004_194054
python3.10 run_sim_bridge.py --llm heuristic --viz

# 라이브 시뮬레이터에 실제 명령 발행
python3.10 run_sim_bridge.py --llm heuristic --publish --viz --cmd-log logs/sim_cmds.jsonl
```

---

## 5. 시뮬레이터 측에 확인 부탁할 사항

1. **성공 phase 값**: `phase`가 성공으로 끝날 때 어떤 값이 오나? (현재 `ready`/`running`만 활성 상태로 취급한다)
2. **빈 경유점**: `points: []`를 받으면 그 배가 정지하나? 아니면 마지막 명령을 계속 수행하나?
3. **경유점 재수신**: 내용이 바뀐 `points`를 다시 받으면 첫 점부터 새로 따라가나? 진행 중인 leg 처리 방식이 궁금하다.
4. **`sim_id` 처리**: `sim_id`가 `null`이거나 현재 run과 다르면 명령을 버리나?
5. **그물 투하 동작**: `net_trigger` 1회에 깔리는 그물 길이가 고정(약 40 m)인가? 투하 중 배가 경유점을 따라 계속 움직이나?
6. **기존 컨트롤러 충돌**: 녹화본에서 `/usv/waypoints`·`/usv/net_trigger`·`/usv/assignment`를 내던 기존 컨트롤러를 라이브 시험 때 끌 수 있나? 두 발행자가 동시에 명령하면 충돌한다.
7. **스폰 좌표 축**: §1의 스폰 명령 x/y가 Odometry와 반대인 것이 의도된 규약인가?
