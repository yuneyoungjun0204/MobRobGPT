# CNN 점수맵 모델 이식 설명서 — `u-net_map.pt`

> **이 문서는 상류 저장소(One-Way_Towing/CNN)의 `MODEL_DEPLOY.md` 를 그대로 반입한 것이다.**
> 원본 대상 파일 `boatattack_sim/models/run_20260819-203248/best.pt` 는 이 저장소의
> `boatattack_sim/models/u-net_map.pt` 와 **동일 파일**이다 (md5 `833e0e3905384e130065b10a15d15e6d`).
>
> MobRobGPT 쪽 구현 대응:
> | 문서 내용 | 이 저장소의 구현 |
> |---|---|
> | §7-A 권장 이식(원본 패키지 재사용) | `commander/unet_bridge.py::CommandedCnnEnv` |
> | §10 센서 → 입력 → WP tick 루프 | `commander/ros2_unet_env.py::ROS2CnnEnv` |
> | §6.3 픽셀 2점 → WP + 정지 규칙 | `DefenseVecEnv._apply_cnn_actions` (상류 그대로) |
> | 실행 | `python run_commander_ui.py --unet` / `--unet --ros2` |
>
> 배정(`_assign`)은 이 저장소에서 **LLM 지휘관**이 낸다 — 문서 §4.2 의 휴리스틱 배정 자리에
> `commander/rl_bridge.py::CommandedDefenseEnv._compute_assignment` 가 LLM 계획을 주입한다.

---

이 체크포인트 **하나만** 들고 나가 다른 폴더/다른 저장소에서 실센서 기반 프로젝트를 새로 만들 때
필요한 전부. 입력을 어떻게 만들고, 무엇을 넣어야 하고, 출력을 어떻게 위경도 WP 로 되돌리는가.

> 대상 파일: `boatattack_sim/models/run_20260819-203248/best.pt` (997 KB)
> 작성 기준일: 2026-08-19 · 검증: 이 문서의 모든 수치는 실제 실행으로 확인함

---

## 0. 세 줄 요약

1. 모델은 **50×50 픽셀 · 15채널 래스터**를 보고 **픽셀 2개**를 찍는다. 그 2점을 잇는 선분이 그물 벽이다.
2. 픽셀 ↔ 미터 ↔ 위경도는 **고정 affine** 이다. 학습 가능한 부분도, 상태에 따라 변하는 부분도 없다.
3. 가장 위험한 건 모델이 아니라 **유효 마스크(`valid`)와 배정(`_assign`)** 이다. 이 둘을 대충 만들면
   모델은 조용히 엉뚱한 곳을 찍는다 (에러가 안 난다).
4. **관측이 무차원이라 스케일 불변이다** — 12.6 km 실해역이든 12.6 m 수조든 같은 모델을 쓴다.
   맞출 것은 좌표와 시계뿐 (§11, `tests/test_scale.py` 가 k=1e-3 에서 검증).

---

## 1. 체크포인트 안에 무엇이 있나

```python
ck = torch.load("best.pt", map_location="cpu", weights_only=False)
ck.keys()   # ['model', 'config', 'd', 'width', 'obs_schema', 'cnn_actor']
```

| 키 | 값 | 뜻 |
|---|---|---|
| `model` | state_dict | 가중치 — **244,389 params**, stem in_channels = **15** |
| `config` | dict | **학습 당시 SimConfig 전체**. 이게 입력 계약의 원본이다 |
| `d` | 32 | 점수맵 feature 차원 |
| `width` | 32 | UNetLite 채널 폭 |
| `obs_schema` | 12개 채널명 리스트 | 관측 스키마 지문. **로드 시 반드시 대조** |
| `cnn_actor` | True | 이 체크포인트가 CNN 점수맵 액터임을 표시 |

`obs_schema` 실제 값:

```
['enemy_presence','enemy_vx','enemy_vy','enemy_threat','ally_presence',
 'net_installed','origin','annulus','land',          # 전역 9
 'self_marker','self_intercept','self_valid']        # 배별 3
```

> ⚠ `obs_schema` 에는 CoordConv 채널(`coord_x`,`coord_y`,`coord_r`)이 빠져 있다.
> 이 3개는 config 에서 완전히 파생되므로 스키마에 안 들어간다. **모델 입력은 9+3+3 = 15채널**이다.

### 이 모델의 확정 스펙 (config 에서 뽑은 값)

| 항목 | 값 | 비고 |
|---|---|---|
| `cnn_grid_n` | **50** | 점수맵 50×50 |
| `cnn_extent` | **6300.0** m | 모선 중심 반폭 |
| **픽셀 크기** | **252.0 m** | `2×6300/50` |
| `world_size` | 12600.0 m | 맵 한 변 |
| `center` | (6300.0, 6300.0) | = 모선 위치 = 맵 정중앙 |
| `cnn_nets` (K) | **2** | 결정당 픽셀 2개 |
| `cnn_subpixel` | True | ±반픽셀(±126 m) 연속 보정 |
| `n_allies` (P) | **3** | 아군 척수 |
| `n_enemies` (M) | **10** | 적 최대 척수 |
| `nets_per_ship` | 1 | 배당 그물 1개 |
| `net_max_len` | 450.0 m | 그물 벽 실제 도색 상한 |
| `decision_period` | **25** step | 재계획 주기 (dt=1.0 → 25초) |
| `dt` | 1.0 s | |
| `ally_speed` / `enemy_speed` | 6.0 / 9.0 m/s | `enemy_speed_mult=1.5` |
| `cell_r_min` / `cell_r_max` | 400 / 4500 m | 요격 환형 |
| `cnn_gate_r` | 3000.0 m | 요격점 기준 반경 게이트 |
| `cnn_gate_angle` | **60.0°** | 방위 게이트 (코리도 반각) |
| `cnn_gate_disjoint` | False | Voronoi 분할 끔 = **겹침 허용 레짐** |
| `cnn_min_valid` | 150 | 폴백 하한 |
| `cnn_dup_r` | 200.0 m | 2번째 픽셀 최소 이격 |
| `cnn_reach_mask` / `slack` | True / 1.5 | 2번째 픽셀 상한 = 450×1.5 = **675 m** |
| `cnn_coord` / `cnn_coord_r` | True / **True** | CoordConv 3채널 (x, y, r) |
| `land_obstacle` / `cnn_ch_land` | True / True | 육지 채널 **켜짐** |
| `land_sites` | `'korea'` | ⚠ 다중 해역 로테이션. 실전에선 `''` 로 바꿔야 한다 (§2.4) |
| `geo_lat`, `geo_lon` | 34.625, 128.52 | 학습 앵커 (통영 매물도 서측) |
| `mothership_radius` | 260.0 m | |
| `enemy_spawn_radius` | 5450.0 m | 적 초기 반경 — 스케일 정합의 기준 |

---

## 2. 좌표계 — 3층 구조

```
   WGS84 (lat, lon)          ← 센서(GPS)가 주는 것, WP 로 되돌려줄 것
        ↕  ① 등거리 근사 affine
   sim world (x, y) [m]      ← 0..12600, x=East, y=North, 모선=(6300,6300)
        ↕  ② 고정 격자
   score map (ix, iy)        ← 0..49, flat = ix*50 + iy
```

### 2.1 ① WGS84 ↔ world (미터)

앵커 `(LAT0, LON0)` 가 **맵 정중앙 = 모선 위치**에 대응한다. 원본 구현
(`env/terrain.py::_lonlat_to_local`)과 동일한 등거리 근사:

```python
import math
MPD_LAT = 111_320.0                               # 위도 1° 당 미터
MPD_LON = 111_320.0 * math.cos(math.radians(LAT0))  # 경도 1° 당 미터 (앵커 위도 고정)
HALF    = 12600.0 * 0.5                            # 6300.0

def ll_to_world(lat, lon):
    x = (lon - LON0) * MPD_LON + HALF
    y = (lat - LAT0) * MPD_LAT + HALF
    return x, y

def world_to_ll(x, y):
    lat = LAT0 + (y - HALF) / MPD_LAT
    lon = LON0 + (x - HALF) / MPD_LON
    return lat, lon
```

- **`MPD_LON` 은 앵커 위도로 한 번만 계산하고 고정한다.** 배 위도마다 다시 계산하면
  왕복이 안 맞아 서서히 어긋난다.
- 12.6 km 박스에서 이 근사의 오차는 수 m 수준이라 252 m 픽셀에 비해 무시할 수 있다.
- 헤딩 규약: **0° = North, 시계방향 +**, `bearing = atan2(dx, dy)`. GPS/IMU 의 true heading 과
  같으므로 변환 불필요. 단 자침 편차(magnetic declination)를 쓰는 IMU 라면 진북으로 보정할 것.

### 2.2 ② world ↔ 픽셀 (`env/cnn_map.py` 가 단일 소스)

```python
PX  = 2 * 6300.0 / 50          # 252.0 m
ORG = (6300.0 - 6300.0,) * 2   # = (0.0, 0.0)  ← center - extent

ix = clip(floor((x - 0.0) / 252.0), 0, 49)
iy = clip(floor((y - 0.0) / 252.0), 0, 49)
flat = ix * 50 + iy

x = 0.0 + (ix + 0.5) * 252.0 + dx * 126.0      # dx ∈ [-1, 1] 서브픽셀
y = 0.0 + (iy + 0.5) * 252.0 + dy * 126.0
```

> ★ **축 순서는 `[ix, iy] = [x, y]`** 다. numpy 이미지 관례(`[row, col] = [y, x]`)가 **아니다**.
> 이걸 틀리면 90° 회전 + 상하반전 버그가 확정적으로 난다. 그림 그릴 때만
> `imshow(arr.T, origin="lower")` 로 전치한다.

### 2.3 실세계 스케일 정합 (t=0 에 한 번)

실제 해역이 12.6 km 박스보다 크거나 작으면 모델이 학습한 기하와 어긋난다. 권장 절차:

1. 운용 시작 시점에 모선(또는 방어 목표) 위치를 `(LAT0, LON0)` 로 **고정**한다.
2. 그 시점에 보이는 가장 먼 적까지의 거리를 `enemy_spawn_radius = 5450 m` 근처에 오도록
   전체 스케일을 잡는다. (실거리가 크게 다르면 등배 스케일 계수를 곱해 sim 좌표로 넣는 방법이 있으나,
   **속도·`net_max_len`·`decision_period` 도 같은 계수로 스케일해야** 일관성이 유지된다.)
3. 이 affine 은 **에피소드 내내 고정**한다. 매 tick 다시 잡으면 관측이 흔들려 정책이 진동한다.

### 2.4 `land_sites` 는 반드시 바꾼다

체크포인트 config 는 `land_sites='korea'` (여러 해역 로테이션, 지형 일반화 학습용)이다.
실전 배치에서는 **운용 해역 하나로 고정**해야 한다:

```python
cfg.land_sites = ""            # → 단일 앵커 (cfg.geo_lat, cfg.geo_lon) 사용
cfg.geo_lat, cfg.geo_lon = LAT0, LON0
```

안 바꾸면 `world_site` 가 매 에피소드 무작위로 뽑혀 **화면·마스크·물리가 다른 섬을 가리킨다.**

---

## 3. 입력 계약 — 4개 텐서

`actor(obs)` 가 받는 dict. `N` = 월드 수(실전 = 1), `P` = 아군 수(3), `H=W=50`.

| 키 | shape | dtype | 뜻 |
|---|---|---|---|
| `gmap` | `[N, 9, 50, 50]` | float32 | 전역 채널 (배와 무관, 월드당 1장) |
| `smap` | `[N, P, 3, 50, 50]` | float32 | 배별 채널 |
| `own` | `[N, P, 8]` | float32 | 래스터로 표현하기 낭비인 스칼라 |
| `valid` | `[N, P, 50, 50]` | bool | **행동 마스크** (= `smap[:,:,2]` 와 동일 배열) |

모델 내부에서 `gmap` 을 P축으로 expand 하고 `smap`, CoordConv 3채널을 붙여
`[N*P, 15, 50, 50]` 로 conv 에 넣는다.

### 3.1 전역 채널 9개 (`gmap`) — 정규화까지 포함

| # | 이름 | 만드는 법 | 정규화 |
|---|---|---|---|
| 0 | `enemy_presence` | 픽셀별 **살아있는 적 수** 합산 | `/ n_enemies` (=10) |
| 1 | `enemy_vx` | 픽셀 내 적 헤딩벡터 x(=sin) 합 | `/ 그 픽셀의 적 수` (평균) |
| 2 | `enemy_vy` | 헤딩벡터 y(=cos) 합 | 〃 |
| 3 | `enemy_threat` | 모선 도달 임박도, 픽셀당 **max** | `1 − clip(d /(enemy_speed·max_steps))` = `1 − clip(d/18000)` |
| 4 | `ally_presence` | 픽셀별 살아있는 아군 수 | `/ P` (=3) |
| 5 | `net_installed` | 물리격자 200² 설치그물 → 50² **블록 max** (4×4 max-pool) | 0/1 |
| 6 | `origin` | 모선 disk (`r ≤ 260 m`) 정적 마스크 | 0/1 |
| 7 | `annulus` | 요격 환형 (`400 ≤ r ≤ 4500`) 정적 마스크 | 0/1 |
| 8 | `land` | 육지 마스크 (50² 로 래스터화) | 0/1 |

> **정규화는 전부 고정 상수 나눗셈이다.** RunningMeanStd / VecNormalize / BatchNorm 같은
> **배치 통계 기반 정규화는 하나도 없다.** 그래서 배치 1로 실시간 추론해도 학습과 완전히 동일하다.
> (모델 내부에는 GroupNorm, `/√d` 스코어 스케일링, bottleneck GAP → FiLM 이 있는데 전부 배치 무관이다.)
>
> 픽셀 하나에 적 3척이 겹치면 `enemy_presence = 0.3` 이다. **맵 전체 적 수가 아니라 픽셀 내 적 수**다.

### 3.2 배별 채널 3개 (`smap`)

| # | 이름 | 만드는 법 |
|---|---|---|
| 0 | `self_marker` | 자기 위치 1픽셀 스탬프 (`cnn_marker_r=0` → 단일 픽셀, 값 1.0) |
| 1 | `self_intercept` | **배정된 요격점** 1픽셀 스탬프 (미배정이면 전부 0) |
| 2 | `self_valid` | 유효 마스크 float 캐스팅 (§4) |

### 3.3 `own` 8개 스칼라

```python
own = [ (x - 6300)/6300,          # 0 자기 위치 x (정규화)
        (y - 6300)/6300,          # 1 자기 위치 y
        sin(hdg), cos(hdg),       # 2,3 헤딩 벡터 (0°=North, CW+)
        a_nets / nets_per_ship,   # 4 남은 그물 (0 또는 1)
        float(doing_net),         # 5 지금 그물 도색 중인가
        (Ix - 6300)/6300,         # 6 배정 요격점 x — 미배정이면 0
        (Iy - 6300)/6300 ]        # 7 배정 요격점 y — 미배정이면 0
```

> `own[4]=0` 이고 `own[5]=0` 인 배는 **행동이 세계를 바꾸지 못한다.** 그런 배의 점수맵은
> 학습 신호가 0이라 거의 균등분포로 남는다(실측 top1 = 0.012, 균등 = 1/170). 정상이다.
> 그 배의 출력은 **버리고 정지시켜라** (§6.3).

### 3.4 CoordConv 3채널 — 직접 만들 필요 없음

`cnn_map.coord_channels(cfg)` 가 config 만으로 생성하고 모델이 buffer 로 들고 있다.
`(x−cx)/6300`, `(y−cy)/6300`, `clip(r/6300, 0, 1)`. 정적이라 매 tick 재계산 불필요.

---

## 4. 유효 마스크 `valid` — 여기가 진짜 핵심

`valid` 는 단순한 "그릴 수 있는 영역"이 아니라 **정책의 신뢰영역(trust region)** 이다.
`-inf` 로 로짓을 막아 학습 내내 모델이 그 밖을 본 적이 없다. 대충 만들면 모델이 조용히 무너진다.

### 4.1 구성 (`defense_env.py::_cnn_valid_mask`)

```
valid = annulus ∩ ¬land ∩ (요격점 반경게이트 r ≤ 3000 m) ∩ (방위게이트 |Δbearing| ≤ 60°)
        [ ∩ Voronoi ]   ← cnn_gate_disjoint=False 이므로 이 모델에선 적용 안 함
```

- **방위게이트 중심선** = `모선 → 배정 요격점` 축. 즉 "배정된 클러스터로 뻗는 코리도".
- **폴백 사다리** (후보가 모자랄 때, 순서가 의미를 좌우한다):
  1. soft (`< cnn_min_valid=150`): **반경게이트만** 푼다 → 방위(코리도)는 절대 안 버린다.
  2. hard (`< cnn_nets=2`): 그제서야 방위를 포기 → `annulus∩¬land` → 맵 전체(단 육지는 계속 제외).
- 실측 유효 픽셀 수 (16월드×3시드×12결정): **min 150 / median 241 / max 319** (50² = 2500 중).
  min 이 정확히 150 인 것은 soft 폴백 하한 `cnn_min_valid=150` 이 실제로 작동한다는 뜻이다.

### 4.2 그래서 **배정(assignment)이 선행 조건**이다

`self_intercept` 채널, `own[6:8]`, 방위게이트가 전부 `_assign` / `_assignI` 에 의존한다.
배정은 이렇게 만들어진다 (`_compute_assignment`):

1. 적을 **방위 gap 기준 적응형 클러스터링** (`cluster_gap_deg=11.97`, 최대 `n_clusters=4`).
2. 클러스터 위협도 = `적 수 × (1 − 모선거리/6300)` → 상위 P개 선정.
3. 요격점 `I = centroid + 0.55·(모선 − centroid)` (`assign_intercept_t=0.55`).
4. 배 → 요격점 거리 최소가 되도록 **그리디 1:1 매칭**. 직전 배정 유지 보너스 6000 m (churn 억제).
5. 남는 배는 잉여 배정 (`assign_surplus=True`, 층을 `assign_layer_dbear=40°` 만큼 방위로 벌림).

> 실전에서 이걸 새로 구현하지 마라. **원본 코드를 그대로 재사용하는 것이 §7-A 를 권하는 가장 큰 이유다.**

---

## 5. 실세계에서 반드시 확보해야 하는 상태

위 입력을 만들려면 매 결정 주기(25 s)마다 아래가 필요하다.

| 상태 | shape | 출처 | 없으면 |
|---|---|---|---|
| 아군 위치 | `[P,2]` m | GPS (NavSatFix) → `ll_to_world` | 필수 |
| 아군 헤딩 | `[P]` rad/deg | IMU/COG (진북 기준) | 필수 |
| 아군 생존 | `[P]` bool | 통신 두절/이탈 판정 | 끊기면 False |
| 아군 남은 그물 | `[P]` int | **내부 상태로 직접 관리** | 필수 |
| 아군 도색중 | `[P]` bool | 내부 상태 | 필수 |
| 적 위치 | `[M,2]` m | 레이더/AIS/탐지 융합 | 필수 |
| 적 헤딩 | `[M]` | 트랙 속도벡터에서 유도 | 필수 |
| 적 생존/유효 | `[M]` bool | 트랙 유효성 | 미탐지=False |
| 설치된 그물 | `[200,200]` bool | 내부 상태 (부설 궤적 도색) | 없으면 전부 False |
| 육지 마스크 | `[50,50]`, `[200,200]` bool | `env/terrain.py` 가 OSM 에서 생성·캐시 | 육지 없으면 0 |

> **적 개수는 정확히 M=10 슬롯**이다. 탐지된 적이 10보다 적으면 남는 슬롯을 `e_alive=False` 로,
> 많으면 위협 상위 10개만 넣는다. 래스터 관측이라 순서는 무관하다(순열 불변).
>
> **육지 마스크는 OSM 벡터에서 받는다**(위성영상은 구름을 육지로 오분류한다).
> `terrain.fetch_land_mask(lat, lon, world_m, source="auto")` 가 벡터→타일→위성 순으로 시도하고
> `.npy` 로 캐시하므로, 배치 전에 한 번 받아 **캐시 파일을 같이 들고 나가면 오프라인 동작**한다.

---

## 6. 추론과 출력 디코딩

### 6.1 추론

```python
from boatattack_sim.model.cnn_actor import load_cnn_actor, cnn_obs_to_torch
import torch

actor, cfg = load_cnn_actor("best.pt", device="cpu")   # cfg 는 체크포인트에서 복원됨
actor.eval()

with torch.no_grad():
    p, _ = actor(cnn_obs_to_torch(obs, "cpu"))         # obs = {gmap,smap,own,valid}
    a = actor.greedy(p)                                # 결정적. 학습용 sample() 아님
pix = a["pix"].view(1, P, 2).numpy()                   # int64 flat 픽셀
off = a["offset"].view(1, P, 2, 2).numpy()             # float ∈ [-1,1] 서브픽셀
```

- **평가/운용은 `greedy`**, 학습만 `sample`. 실선에서 확률 샘플링하면 안 된다.
- `greedy_joint(p, N, P, r)` 는 배 사이 겹침을 막는 변형인데, 이 모델은 **겹침 허용 레짐**에서
  학습됐다(`cnn_gate_disjoint=False`). **추론에서 joint 잠금을 켜면 학습 레짐을 되돌려 성능이 떨어진다.**
  기본값(`greedy`)을 써라.
- 자기회귀 2단계다: 첫 픽셀을 뽑고 → 그 위치의 feature 를 query 에 더하고 → 반경 200 m 안쪽(`cnn_dup_r`)과
  675 m 바깥(`reach`)을 마스크에서 뺀 뒤 두 번째를 뽑는다. CNN forward 는 결정당 **1회**뿐이다.

### 6.2 픽셀 → 미터 → 위경도

```python
from boatattack_sim.env import cnn_map as CM

pts = CM.flat_to_world(cfg, pix, off)      # [1,P,2,2] world m — 서브픽셀 반영됨
```

수식으로 풀면:

```
ix, iy = flat // 50, flat % 50
x = (ix + 0.5) * 252.0 + dx * 126.0
y = (iy + 0.5) * 252.0 + dy * 126.0
lat, lon = world_to_ll(x, y)
```

### 6.3 픽셀 2개 → 실제 WP 경로 (여기서 규칙이 하나 더 붙는다)

`env.pix_to_routes(pix, off)` → `route [1,P,6,2]`, `net_mask [1,P,6]`

- 두 점을 **배에서 가까운 순으로 정렬**한다.
- `route[0]` = 가까운 점 (**이동 WP**, 그물 안 침), `route[1]` = 먼 점 (**그물 벽 끝점**).
- `net_mask[1] = True` → **leg `route[0] → route[1]` 구간이 그물 벽**이다.
- `route[2:]` 는 마지막 점 반복(정지)이다. `transit_wp=6` 이라 배열이 6칸일 뿐 의미 있는 건 앞 2개.

**정지 규칙 (반드시 구현):**

```python
done = (a_nets <= 0) | (assign < 0)     # 그물 소진 또는 미배정
route[done] = 현재 위치                  # 그 자리 유지
net_mask[done] = 0
```

이걸 빼면 §3.3 에서 말한 "의견 없는 배"의 균등분포 출력이 그대로 항해 명령이 되어
배가 엉뚱한 데로 간다.

### 6.4 실행 결과 예시 (검증 로그, `enemy=wave, seed=3`)

```
배2: assign=0 nets=1 pos=[6655.4 6724.5]
     raw pts   = [[6669.5, 6665.7], [6170.6, 6925.5]]
     route[0:2]= [[6669.5, 6665.7], [6170.6, 6925.5]]  net_mask=[0,1,0,0,0,0]
     WP0 -> lat 34.628285  lon 128.524034
     WP1 -> lat 34.630619  lon 128.518588
     그물벽 길이 = 562.4 m

배0: assign=-1 nets=1  → route 전체가 현재 위치로 덮임, net_mask 전부 0 (정지)
```

> 벽 길이 562 m 가 `net_max_len=450` 보다 긴 것은 정상이다. `cnn_reach_slack=1.5` 라
> 마스크 상한이 675 m 이고, 실제 도색은 450 m 에서 끊긴다. 실선 제어에서도
> **450 m 까지만 그물을 놓고 나머지는 이동으로 처리**해야 학습과 일치한다.

---

## 7. 이식 전략

### A. 권장 — 원본 패키지를 라이브러리로 재사용하고 상태만 주입

가장 안전하다. 배정·마스크·클러스터링·폴백 사다리·디코딩을 **한 줄도 다시 쓰지 않는다.**

```
new_project/
├─ vendor/boatattack_sim/       # env/, model/ 만 복사 (eval/, train/ 불필요)
├─ geo_bridge.py                # WGS84 ↔ world affine (§2.1). 신규
├─ world_state.py               # 센서 → env 상태 배열 주입. 신규
├─ inference.py                 # obs → actor → route → lat/lon. 신규
└─ models/best.pt
```

핵심 루프:

```python
env = DefenseVecEnv(num_worlds=1, cfg=cfg, enemy_mode="wave", seed=0)
env.reset(seed=0)

def tick(sensors):
    # 1) 센서 → sim 상태 덮어쓰기 (시뮬 물리는 안 돌린다)
    env.a_pos[0]  = ll_to_world_batch(sensors.ally_ll)      # [P,2]
    env.a_hdg[0]  = sensors.ally_hdg
    env.a_alive[0]= sensors.ally_alive
    env.a_nets[0] = my_net_counter                          # 내부 관리
    env.doing_net[0] = my_deploy_flag
    env.e_pos[0]  = ll_to_world_batch(sensors.enemy_ll)     # [M,2], 빈 슬롯은 아무 값
    env.e_hdg[0]  = sensors.enemy_hdg
    env.e_alive[0]= sensors.enemy_valid
    env.net_installed[0] = my_painted_grid                  # [200,200] bool

    # 2) 관측 (내부에서 _compute_assignment + _cnn_valid_mask 를 알아서 부른다)
    obs = env.build_cnn_obs()

    # 3) 추론
    with torch.no_grad():
        p, _ = actor(cnn_obs_to_torch(obs, "cpu"))
        a = actor.greedy(p)
    pix = a["pix"].view(1, env.P, actor.K).numpy()
    off = a["offset"].view(1, env.P, actor.K, 2).numpy()

    # 4) 디코딩
    route, net_mask = env.pix_to_routes(pix, off)
    done = (env.a_nets[0] <= 0) | (env._assign[0] < 0)
    return [ (None if done[s] else
              [world_to_ll(*route[0,s,0]), world_to_ll(*route[0,s,1])])
             for s in range(env.P) ]
```

- 시뮬 물리(`env.step`)는 **호출하지 않는다.** 상태는 실센서가 준다.
- `env.build_cnn_obs()` 내부의 `_compute_assignment()` 는 sticky 보너스 때문에 `_prev_assign` 을
  기억한다. 매 tick 같은 `env` 객체를 쓰면 배정 churn 억제가 그대로 살아난다.
- `DefenseVecEnv.__init__` 이 육지 마스크를 만들며 네트워크를 탈 수 있으므로, 캐시를 미리 채워둘 것.

### B. 최소 vendoring — 의존을 더 줄이고 싶을 때

`env/cnn_map.py`, `env/rasterizer.py`, `env/kinematics.py`, `env/config.py`,
`model/cnn_actor.py`, `model/actor.py`(mlp), `env/terrain.py`(육지), `env/clustering.py`(배정) 만 가져가고
`defense_env` 대신 **duck-typed 상태 객체**를 만든다. `rasterizer` 가 실제로 읽는 속성은 이게 전부다:

```
cfg, N, P, M, center, world_half, G, cell,
a_pos, a_hdg, a_alive, a_nets, doing_net,
e_pos, e_hdg, e_alive,
net_installed, land_map, world_site, has_land,
_assign, _assignI,
_compute_assignment(), _cnn_valid_mask()
```

마지막 두 메서드는 `defense_env.py` 에서 그대로 복사해 붙이면 된다(다른 상태를 안 건드린다).
**직접 새로 구현하지 마라** — §4 의 폴백 사다리 순서 하나만 틀려도 조용히 성능이 무너진다.

---

## 8. 이식 검증 게이트 (구현 순서 그대로)

새 프로젝트가 "맞게" 옮겨졌는지 확인하는 최소 회귀. 순서대로 통과시켜라.

| # | 검증 | 통과 기준 |
|---|---|---|
| 1 | **좌표 왕복** | `world_to_ll(ll_to_world(lat,lon))` 오차 < 0.1 m |
| 2 | **픽셀 왕복** | `world_to_pix(pix_to_world(ix,iy))` == `(ix,iy)` 전 픽셀 |
| 3 | **스키마 대조** | `channel_names(cfg)[:12] == ck['obs_schema']`, 총 15채널 |
| 4 | **관측 allclose** | 동일 상태에서 새 구현 `obs` vs 원본 `env.build_cnn_obs()` → `np.allclose` (rtol 1e-5) |
| 5 | **출력 일치** | 같은 obs → `actor.greedy` 의 `pix` **완전 일치** |
| 6 | **디코딩 일치** | `pix_to_routes` 결과 route/net_mask 일치 |
| 7 | **정지 규칙** | `assign<0` 또는 `nets<=0` 인 배의 WP 가 현재 위치 |
| 8 | **스케일 불변** | `python -m boatattack_sim.tests.test_scale` ALL PASS (다른 스케일로 옮길 때만) |

4번이 통과하지 않으면 5~7 은 볼 필요가 없다. **4번이 이 이식의 유일한 진짜 게이트다.**

---

## 9. 함정 목록 (전부 실제로 밟은 것들)

1. **축 순서** — `[ix, iy] = [x, y]`. numpy 이미지 관례 아님. 그림 그릴 때만 전치.
2. **`obs_schema` 대조를 채널 *개수* 로 하지 마라.** 기본값이 OFF→ON 으로 바뀌면 개수만
   우연히 맞아 구 체크포인트가 **무음으로 손상**된다. 반드시 **이름 리스트**로 대조하라.
3. **`cnn_coord_r`** — 이 체크포인트는 CoordConv 가 3채널(x,y,r)이다. 이 값은 한때 "x,y 에서
   conv 로 합성 가능한 중복"이라 `False`(2채널)로 껐다가 **2026-08-14 에 `True` 로 되돌렸다**
   (채널 1개는 파라미터 288개(0.118%)라 절약이 없는데 챔피언 호환만 깨진다). 지금 저장소
   기본값도 `True` 다. 다만 이 값이 바뀌면 stem in_channels 가 15↔14 로 움직이므로,
   **config 를 직접 만들지 말고** `load_cnn_actor` 를 cfg 없이 불러 체크포인트에서 자동
   복원시켜라. 구 14채널 체크포인트를 15채널 설정으로 warm-start 하는 경로도
   `_drop_coord_r_stem` 로 열려 있다.
4. **`land_sites='korea'` 를 `''` 로 안 바꾸면** 해역이 매 에피소드 무작위로 바뀐다 (§2.4).
5. **`decision_period=25`** — 25초마다 한 번 재계획한다. 매 초 추론하면 학습과 다른 동역학이 된다.
   그물 도색 중(`doing_net=True`)인 배는 원본에서 재계획이 **동결**된다. 같게 구현할 것.
6. **`greedy` 를 쓰고 joint 잠금은 켜지 마라** (§6.1).
7. **미배정 / 그물 소진 배의 출력은 쓰레기다** — 균등분포가 나온다. 정지시켜라 (§6.3).
8. **그물 벽은 450 m 에서 끊긴다** — 마스크는 675 m 까지 허용하지만 실제 도색 상한은
   `net_max_len=450`. 제어측에서 이 상한을 지켜야 학습과 일치한다.
9. **적 슬롯은 10개 고정.** 부족하면 `e_alive=False`, 넘치면 위협 상위 10개.
10. **배치 통계 정규화가 없다** — 배치 1 실시간 추론이 학습과 완전히 동일하다. 워밍업 불필요.
11. **`torch.load(..., weights_only=False)`** 가 필요하다 (config dict 가 들어 있다). 신뢰하는
    체크포인트에만 쓸 것.
12. **스케일을 바꿀 때 `RewardCfg` 의 미터 단위를 빠뜨리기 쉽다** — `assign_sticky_bonus=6000 m`
    가 대표적이다. 안 곱하면 배정 sticky 가 스케일마다 다르게 동작한다 (§11.6).
13. **`cell_size`·`center` 는 property 다.** `dataclasses.replace` 에 넘기면 `TypeError`.
    `world_size`·`grid_size` 만 바꾸면 자동으로 따라온다.

---

## 10. 센서 → 입력 변환 레시피 (실제 코드 순서)

§5 가 "무엇이 필요한가"였다면 여기는 "어떤 순서로 무엇을 호출하는가"다.

### 10.0 초기화 — 운용 시작 시 **한 번만**

```python
import dataclasses, numpy as np, torch
from boatattack_sim.env.config import SimConfig
from boatattack_sim.env.defense_env import DefenseVecEnv
from boatattack_sim.env.scaling import SimScale
from boatattack_sim.model.cnn_actor import load_cnn_actor, cnn_obs_to_torch

actor, cfg = load_cnn_actor("best.pt", device="cpu")   # cfg 는 체크포인트가 복원 (§9-③)
actor.eval()
cfg = dataclasses.replace(cfg,
        land_sites="",                 # 운용 해역 하나로 고정 (§2.4)
        geo_lat=LAT0, geo_lon=LON0,    # 앵커 = 맵 정중앙 = 방어 목표
        spawn_phase_lo=1.0)

scale = SimScale(cfg, span_real=SPAN_M, v_ally_real=V_ALLY, anchor=(LAT0, LON0))
print(scale.report(r_turn_real=R_TURN, v_enemy_real=V_ENEMY, loa_real=LOA))   # ★ §11 점검

env = DefenseVecEnv(num_worlds=1, cfg=cfg, enemy_mode="rotate", seed=0)
env.reset(seed=0)                     # 배열 할당 + 육지 캐시 로드용. 이후 물리는 안 돌린다
```

- `env` 는 **시뮬레이터가 아니라 상태 용기**로 쓴다. `env.step()` 은 부르지 않는다.
- `scale` 과 `env` 는 운용 내내 **같은 객체**를 재사용한다. 배정 sticky(`_prev_assign`)가
  여기 살아 있어야 WP 가 tick 마다 튀지 않는다 (§4.2-4).

### 10.1 매 tick (= `scale.period_real` 초마다)

```python
def tick(sensor):
    # ── ① 시간 동기: 모든 트랙을 같은 시각 t_now 로 외삽한다 ──────────────
    #    레이더 트랙과 GPS 는 도착 시각이 다르다. 그대로 넣으면 적/아군의
    #    상대 기하가 실제와 어긋나 threat·배정이 흔들린다.
    ally  = sensor.allies.extrapolate(t_now)      # lat, lon, cog, sog
    enemy = sensor.enemy_tracks.extrapolate(t_now)

    # ── ② 좌표: WGS84 → sim world ────────────────────────────────────
    env.a_pos[0] = scale.ll_to_sim(ally.lat, ally.lon)        # [P,2]
    env.a_hdg[0] = np.radians(ally.heading_true)              # 0°=North, CW+ (§2.1)
    env.a_alive[0] = ally.link_ok & ~ally.disabled

    # ── ③ 적 슬롯 10개 고정 (§9-⑨) ──────────────────────────────────
    #     위협 = 방어 목표까지의 거리 → 가까운 순 10개. 래스터라 순서는 무관하다.
    M = cfg.n_enemies
    xy = scale.ll_to_sim(enemy.lat, enemy.lon)                # [n,2]
    d  = np.hypot(*(xy - np.asarray(cfg.center)).T)
    keep = np.argsort(d)[:M]
    env.e_pos[0]   = 0.0;  env.e_hdg[0] = 0.0;  env.e_alive[0] = False
    env.e_pos[0, :len(keep)]   = xy[keep]
    env.e_hdg[0, :len(keep)]   = np.radians(enemy.cog_true[keep])
    env.e_alive[0, :len(keep)] = True

    # ── ④ 내부 상태 — 센서가 안 주는 것은 직접 관리한다 ──────────────────
    env.a_nets[0]    = net_counter                # 남은 그물 (부설 완료 시 −1)
    env.doing_net[0] = deploying                  # 지금 부설 중인가
    env.net_installed[0] = painted_grid           # [200,200] bool, ⑦에서 갱신

    # ── ⑤ 관측 (배정·마스크는 내부에서 알아서 계산된다) ──────────────────
    obs = env.build_cnn_obs()

    # ── ⑥ 추론 + 디코딩 ──────────────────────────────────────────────
    with torch.no_grad():
        p, _ = actor(cnn_obs_to_torch(obs, "cpu"))
        a = actor.greedy(p)                       # ★ sample 아님 (§6.1)
    pix = a["pix"].view(1, env.P, actor.K).numpy()
    off = a["offset"].view(1, env.P, actor.K, 2).numpy()
    route, net_mask = env.pix_to_routes(pix, off)

    # ── ⑦ 정지 규칙 + 발행 ───────────────────────────────────────────
    stop = (env.a_nets[0] <= 0) | (env._assign[0] < 0)        # §6.3
    out = []
    for s in range(env.P):
        if stop[s] or not env.a_alive[0, s]:
            out.append(None);  continue
        wp0 = scale.sim_to_ll(route[0, s, 0])     # 이동 WP (그물 안 침)
        wp1 = scale.sim_to_ll(route[0, s, 1])     # 그물 벽 끝점
        out.append((wp0, wp1))                    # leg wp0→wp1 이 그물 (net_mask[1]=True)
    return out
```

### 10.2 `net_installed` 는 직접 칠해야 한다

센서가 주지 않는 유일한 래스터다. 배가 `wp0 → wp1` 구간을 항행하며 실제로 그물을 놓는 동안,
그 궤적을 `[200,200]` 물리격자에 칠한다. 원본은 `env/grid.py` 가 `net_width=4` cell 폭
(≈ 4 × `world_size/200`)으로 선분을 도색하고 `net_max_len` 에서 끊는다.

- 실물 그물이 없는 실험(수조·기동시험)이라면 **가상 그물**로 두고 궤적만 칠해도 관측은 성립한다.
- 칠하지 않으면 `net_installed` 채널이 계속 0이라 정책이 **이미 친 벽을 못 보고 겹쳐 친다.**

### 10.3 센서 품질 요구

| 항목 | 실해역 (12.6 km) | 10 m 수조 | 근거 |
|---|---|---|---|
| 1 픽셀 | 252 m | **0.2 m** | `2·extent/50` |
| 측위 오차 권장 | < 25 m (GPS 충분) | **< 2 cm** (UWB·모션캡처 필요) | 픽셀의 1/10 |
| 헤딩 오차 | < 5° | < 5° | 무차원 — 스케일 무관 |
| 갱신 주기 | ≥ 결정주기 | ≥ 결정주기 | §11.3 |

**측위 오차 / 픽셀 크기**도 무차원 비다. 수조에서 GPS 급(수 m) 정밀도를 쓰면
배 위치가 맵 전체를 튀어다니는 것과 같다.

---

## 11. 스케일 불변성 — 수조에서도 같은 모델을 쓸 수 있는가

### 11.1 결론: 관측은 무차원이다 (검증 완료)

래스터 채널·`own` 스칼라는 전부 `extent`·`n_enemies`·`P`·`enemy_speed·max_steps` 같은
**config 상수로 나눈 비율**이다. 그래서 길이·속도를 전부 k배 하고 상태도 k배 하면
**같은 텐서, 같은 선택 픽셀**이 나온다.

`python -m boatattack_sim.tests.test_scale` 이 **k = 1e-3 (12.6 km → 12.6 m)** 에서 이를 검증한다:

```
[2] 관측 불변          gmap max|Δ| = 5.96e-08 (float32 반올림 한계)
                      smap · own  max|Δ| = 0.00e+00
                      valid 마스크 완전 일치 (유효픽셀 5176 vs 5176)
                      배정 _assign 동일 · _assignI ×k 일치
[3] 휴리스틱·디코딩     픽셀 0개 불일치 · route ×k 일치 · net_mask 일치
[4] 정책 출력          run_20260819-203248 선택 픽셀 0/24 불일치
                      서브픽셀 오프셋 max|Δ| = 2.24e-08
```

**따라서 수조 실험을 위해 재학습할 필요가 없다.** 맞춰야 하는 건 좌표와 시계뿐이다.

### 11.2 두 개의 스케일 계수 — 모델이 아니라 시계를 스케일한다

```
① 길이  S  [sim-m / real-m] = world_size / (실제 운용 박스 한 변)
② 시간  dt [s / sim-step]   = cfg.ally_speed / (실제 아군 속도 × S)
```

②가 ①에서 **따라 나온다.** 아군은 sim 한 스텝에 `ally_speed = 6` sim-m 를 가야 하므로,
실제 속도 `v` 로 그만큼 이동하는 데 걸리는 실시간이 곧 한 스텝이다. `env/scaling.py::SimScale`
이 이걸 계산한다.

10 m 수조 · 모형선 0.4 m/s 예 (`SimScale.report()` 실제 출력):

```
S  = 1260 sim-m/real-m        (맵 12600 sim-m ↔ 실제 10 m)
dt = 0.0119 s/step  ·  결정주기 = 0.2976 s (25 step)
실제 1 픽셀       = 0.2 m
실제 그물 최대길이 = 0.357 m
실제 요격 환형    = 0.318 ~ 3.571 m
```

즉 **0.3초마다 한 번 재계획**하고, 배 두 척이 0.36 m 짜리 그물 벽을 세우는 실험이 된다.

> 추론이 그 예산 안에 드는가 — 실측(CPU 4스레드, N=1·P=3·50²):
> 관측 1.20 ms + 추론 4.71 ms + 디코딩 0.15 ms = **중앙값 6.13 ms (p95 8.71 ms)**.
> 297.6 ms 예산 대비 **34배 여유**. GPU 불필요.

### 11.3 무차원 비(Π) — 이게 실제 "학습된 무대"다

관측이 무차원이라는 말은, 정책이 아래 비들만 본다는 뜻이다. **스케일을 맞춰도 이 비가 다르면
off-distribution 이다.** `SimScale.pi_groups()` 가 찍는 실제 값:

| Π | 값 | 의미 |
|---|---|---|
| `va·T/extent` | 0.02381 | 결정 한 번 사이에 맵의 몇 %를 이동하는가 |
| `ve/va` | **1.5** | 위협이 방어보다 얼마나 빠른가 |
| `선회반경/extent` | 0.006821 | 얼마나 급하게 돌 수 있는가 |
| `선회반경/선체길이` | **0.1868** | 선체 길이의 1/5 로 회두 — 실선박보다 훨씬 민첩한 추상화 |
| `net_max_len/extent` | 0.07143 | 벽 하나가 맵의 몇 %를 막는가 |
| `(r_min, r_max)/extent` | 0.06349, 0.7143 | 요격 환형 |
| `enemy_spawn_radius/extent` | 0.8651 | 위협이 어디서 나타나는가 |
| `px/extent` | 0.04 | 행동 해상도 (= 2/grid_n, 자동) |
| `ship_len/extent` | 0.03651 | 선체가 맵에서 차지하는 크기 |
| `collision_r/extent` | 0.01825 | 충돌 판정 |
| `arrive_radius/extent` | 0.03175 | WP 도착 판정 |
| `ve·max_steps/extent` | 2.857 | `enemy_threat` 채널의 정규화 분모 |

### 11.4 ★ 스케일해도 **안 따라오는** 세 가지

여기가 정직해야 하는 지점이다. 좌표·시계를 맞춰도 아래는 설비의 물리가 정한다.

**① 선회반경.** 이게 가장 크다. sim 의 선회반경은
`ally_speed·dt / radians(ally_max_turn) = 6/0.1396 = 43 sim-m` 로 고정이다.
10 m 수조(S=1260)에서 이건 **실제 3.4 cm**를 뜻한다. 모형선의 선회반경이 0.5 m 라면
**15배 굼뜨다** — WP 를 따라가는 궤적이 학습 때와 달라진다.

**② 선체 길이 비.** `ship_len = 230 sim-m` = 수조에서 **0.18 m**. 모형선이 0.4 m 라면
2.19배 크다 → 충돌·포획 판정 스케일이 어긋난다.

**③ 속도비 `ve/va = 1.5`.** 위협체가 방어체보다 1.5배 빠르다는 전제가 깨지면
요격 가능 영역(`R_FEAS`) 자체가 달라진다.

`SimScale.report(r_turn_real=..., v_enemy_real=..., loa_real=...)` 가 이 셋을 대조해 경고한다:

```
[scale] ⚠ 선회반경 0.5 m > 허용 0.0341 m — WP 추종이 학습과 달라진다
[scale] ⚠ 선체길이 비 2.19× — 충돌·포획 판정 스케일이 어긋난다
```

### 11.5 그래서 추가로 해야 할 작업

우선순위 순. ①②만 해도 "수조에서 돌아간다"는 되고, ③④가 "학습과 같은 무대"를 만든다.

| # | 작업 | 왜 | 규모 |
|---|---|---|---|
| ① | **`SimScale` 을 쓰는 운용 노드 구현** (§10 의 `tick`) | 좌표·시계·슬롯·정지규칙 | 신규 200줄 남짓 |
| ② | **`net_installed` 도색 로직** | 센서가 안 주는 유일한 래스터 (§10.2) | `env/grid.py` 참고 |
| ③ | **선회반경 정합** — 둘 중 하나 | 위 ①번 불일치 해소 | |
| | ⓐ 스케일 기준을 박스가 아니라 **선회반경**으로 잡는다: `S = 43 / r_turn_real`. 0.5 m 선회면 S=86 → 맵 한 변이 **146 m** 가 되어 수조엔 안 들어간다 | 무대가 커진다 | 설비 문제 |
| | ⓑ `ally_max_turn` 을 낮춰 sim 을 실제에 맞추고 **재학습**. 위 예: `8.0° → 0.55°/step` | 정공법 | 학습 1회 |
| ④ | **선체·충돌 치수 정합 후 재학습** — `ship_len 230→504`, `enemy_size`, `ally_collision_radius` 를 설비 비율로 | 포획·충돌 판정 | ③ⓑ와 같이 |
| ⑤ | **Π 도메인 랜덤화 재학습** — `ally_max_turn`·`ship_len`·`enemy_speed_mult` 를 ±범위로 흔들어 학습 | 한 모델로 실해역·수조 **둘 다** 커버. 근본 해법 | `config.domain_rand` 확장 |
| ⑥ | **측위 정밀도 확보** — 수조는 픽셀이 0.2 m 라 cm 급 필요 (§10.3) | 관측 노이즈/픽셀 비 | UWB·모션캡처 |
| ⑦ | **육지 채널 처리** — 수조엔 육지가 없다. `land_source="none"` 으로 0 채널을 넣는다. 벽·구조물을 장애물로 쓰려면 `land_map [50,50]` 을 직접 채운다 | 채널 수 유지(15) 필수 | 수 줄 |
| ⑧ | `test_scale.py` 를 **CI 회귀로 고정** | 앞으로 config 에 길이 필드가 추가될 때 `LENGTH_FIELDS` 누락을 잡는다 | 이미 작성됨 |

> ⑤가 가장 값어치 있다. ③ⓑ·④는 **수조 전용 모델**을 만드는 것이라 실해역 모델과 갈라지지만,
> ⑤는 하나의 모델로 두 무대를 덮는다. 관측이 이미 무차원이므로 남은 건 Π 의 폭뿐이다.

### 11.6 `scale_config` 는 검증용이지 운용용이 아니다

`env/scaling.py::scale_config(cfg, k)` 는 길이·속도 필드를 전부 k배 한 config 를 만든다.
**불변성 증명에만 쓴다.** 운용에서 config 를 물리 단위로 바꿀 이유는 없다 — sim 단위는
어차피 내부 단위이고, 바꾸면 학습된 기하를 건드릴 위험만 생긴다.

주의할 파생 필드: `cell_size`(= `world_size/grid_size`)와 `center`(= `world_size/2`)는
**property 라 자동으로 따라온다.** `LENGTH_FIELDS` 에 넣으면 `TypeError` 가 난다.
반대로 `RewardCfg` 쪽 미터 단위(`assign_sticky_bonus=6000` 등)는 **자동으로 안 따라오므로**
`REWARD_LENGTH_FIELDS` 로 같이 곱해야 배정이 스케일마다 달라지지 않는다.

---

## 12. 참조 원본 파일

| 파일 | 역할 |
|---|---|
| `boatattack_sim/env/cnn_map.py` | **좌표 변환 단일 소스**. world↔pix, 마스크, 다운샘플 |
| `boatattack_sim/env/rasterizer.py` | 상태 → `{gmap, smap, own, valid}` 관측 생성 |
| `boatattack_sim/env/defense_env.py` | `_compute_assignment` (1108~), `_cnn_valid_mask` (1611~), `pix_to_routes` (1691~), `_apply_cnn_actions` (1704~) |
| `boatattack_sim/env/cell_action.py` | `build_routes_from_cells` — 픽셀 2개 → route/net_mask |
| `boatattack_sim/model/cnn_actor.py` | `UNetLite`, `CnnScoreActor`, `load_cnn_actor`, `greedy`, `score_stages` |
| `boatattack_sim/env/terrain.py` | 육지 마스크 (OSM 벡터), `_lonlat_to_local` 좌표 근사 |
| `boatattack_sim/env/clustering.py` | `cluster_by_gaps_vec` — 적응형 방위 gap 클러스터링 |
| `boatattack_sim/env/config.py` | `SimConfig` — 모든 파라미터의 정의와 기본값 |
| `boatattack_sim/env/scaling.py` | **스케일 계약** — `SimScale`(운용), `scale_config`(검증), Π 점검 |
| `boatattack_sim/tests/test_scale.py` | 스케일 불변성 회귀 (k=1e-3 에서 관측·픽셀 동일) |
| `cnn_map.md` | 이 행동공간 설계의 배경 문서 |
