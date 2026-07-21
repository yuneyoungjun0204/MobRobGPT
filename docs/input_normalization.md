# 셀 정책 입력 정규화 (Input Normalization)

셀선택 정책(`CellPointerActor`)에 들어가는 관측을 어떻게 정규화하는지 정리한 문서.
**스케일이 다른 실험장(예: 33m×33m)으로 이식할 때 무엇을 바꿔야 하는지**가 핵심이다.

구현 위치: `boatattack_sim/env/defense_env.py` → `build_cell_obs()` (1449–1527행)

> ⚠️ `boatattack_sim/env/` 는 **.gitignore 대상**이라 이 코드는 리포에 커밋되지 않는다.
> 정규화 규약이 코드로 남지 않으므로 이 문서가 사실상의 사양서 역할을 한다.

---

## 1. 핵심 원칙

**모든 위치는 "모선 중심 상대좌표 ÷ 기준 반폭" → `[-1, 1]`**

```python
c    = self.center              # 모선 위치 = 정규화 원점 (0,0)
half = cfg.action_grid_half     # 기준 반폭 6000 m
pos_n = (world_pos - c) / half  # ∈ [-1, 1]
```

절대 월드좌표는 **정책에 절대 들어가지 않는다.** 모선이 원점이므로
모선 위에 있는 개체의 입력은 `(0, 0)` 이다.

정규화 계열은 3가지뿐이다:

| 계열 | 분모 | 예 |
|---|---|---|
| **거리/위치** | `action_grid_half` (6000 m) | 아군·적·셀 좌표, 반경 |
| **속도** | `cfg.enemy_speed` | 클러스터 접근속도 |
| **개수** | 정원 (`M`, `nets_per_ship`) | 적 수, 남은 그물 |

각도는 나누지 않고 **`heading_vec()` → (sin, cos)** 단위벡터로 넣는다
(불연속 wrap-around 제거).

---

## 2. 토큰별 정규화 상세

### own `[N, P, 9]` — 자기 자신

| # | 필드 | 코드 | 정규화 |
|---|---|---|---|
| 0-1 | 위치 | `(a_pos - c) / half` | `[-1,1]` |
| 2-3 | 헤딩 | `heading_vec(a_hdg)` | (sin, cos) |
| 4 | 남은 그물 | `a_nets / nets_per_ship` | `[0,1]` |
| 5 | 그물 전개중 | `doing_net` | 0/1 |
| 6-7 | **배정 요격점** | `(_assignI - c) / half` | 미배정이면 0 |
| 8 | 배정 플래그 | `_assign >= 0` | 0/1 |

> 6-8번이 **LLM 지휘관의 배정이 정책으로 흘러드는 통로**다.
> `_assignI`(요격점)를 같은 스케일로 정규화해 넣고, 미배정 함선은
> `assigned_f`를 곱해 0으로 지운다 → "예비" 상태가 명확히 구분된다.

### ally `[N, P, A, 6]` — 타 아군
own의 0-5번과 동일 정규화. 사망 시 `ally_mask`로 마스킹.

### enemy `[N, P, K, 6]` — 적 (기본: 클러스터 토큰)
`cell_cluster_obs=True` 기준:

| # | 필드 | 정규화 |
|---|---|---|
| 0-1 | 무리 중심 | `(centroid - c) / half` |
| 2 | 무리 크기 | `count / M` |
| 3 | 퍼짐 | `spread_norm(spread_deg)` |
| 4 | **접근속도** | `approach / cfg.enemy_speed` |
| 5 | 모선거리 | `dist / half` |

`cell_cluster_obs=False`면 raw 적 M개를 쓰고, 도달시간은
`dist / (enemy_speed × max_steps)`로 정규화.

### cell `[N, P, C, 5]` — 후보셀

| # | 필드 | 정규화 |
|---|---|---|
| 0-1 | 셀 좌표 | `(cell_world - c) / half` |
| 2 | 반경 | `cell_polar[:,0] / half` |
| 3 | 적 밀도 | 셀 주변 `dens_r` 내 생존 적 수 `/ M` |
| 4 | net_present | v1에서는 항상 0 |

---

## 3. ⚠️ 분모가 두 개다 (중요)

관측 모드에 따라 **정규화 분모가 다르다.** 혼동하면 정책이 완전히 어긋난다.

| 모드 | 함수 | 분모 | 결과 |
|---|---|---|---|
| **full** (`cell_obs_slim=False`) | `build_cell_obs` | `action_grid_half` = **6000** | 셀이 `[-0.75, 0.75]` 범위만 사용 |
| **slim** (`cell_obs_slim=True`) | `_cell_tokens`, `_nearest_cell_n` | `cell_r_max` = **4500** | 격자가 `[-1, 1]`을 꽉 채움 |

slim 모드는 격자 extent를 분모로 써서 축 끝 셀이 정확히 ±1.0 이 되게 한다
(표현력 낭비 제거). full 모드는 6000을 쓰므로 후보셀이 최대 4500/6000 = **0.75**까지만 뻗는다.

**현재 `30_model` 3종은 모두 `cell_obs_slim=False`(full) → 분모 6000.**
따라서 추론 환경도 반드시 `action_grid_half=6000`이어야 한다.

---

## 4. 학습/추론 설정 일치 점검

체크포인트에 `config`가 함께 저장되므로 대조할 수 있다:

```python
import torch
ck  = torch.load('30_model/wave/best.pt', map_location='cpu', weights_only=False)
cfg = ck['config']          # ★ dict 이다 (객체 아님) — getattr 로는 못 읽는다
for f in ('cell_obs_slim','action_grid_half','cell_r_max','cell_r_min',
          'cell_nets','cell_cluster_obs','n_clusters','nets_per_ship'):
    print(f, cfg.get(f, '-'))
```

> **추론은 `config.py` 기본값이 아니라 이 ckpt config 를 그대로 쓴다.**
> `commander/cell_bridge.py:52` 가 "체크포인트 자체 config 유지 — 학습분포 충실"
> 원칙으로 로드하고, LLM 정합에 필요한 최소 항목만 오버라이드한다
> (`n_clusters`, `nets_per_ship`, 속도, `decision_period`, `mother_keepout`).
> 따라서 격자·정규화 파라미터는 **자동으로 학습값과 일치**한다.

### `--cell` 과 `--cell --specialized` 는 정규화가 같은가? → **같다**

| 실행 | 가중치 | 정규화 |
|---|---|---|
| `--cell` | `best_mixed_far.pt` | `cell_obs_slim=False`, half=**6000** |
| `--cell --specialized` | `30_model/<mode>/best.pt` | `cell_obs_slim=False`, half=**6000** |

두 체크포인트의 격자 config 가 완전히 동일하다. 대형 라우팅은 **어떤 가중치를
쓸지만** 고르고 관측 파이프라인은 건드리지 않으므로, 정규화는 어느 쪽이든 같다.

`30_model` 3종 학습 설정:

| 필드 | 학습값 | 추론값(`CommandedCellEnv`) | 상태 |
|---|---|---|---|
| `cell_obs_slim` | False | False | ✅ |
| `action_grid_half` | 6000.0 | 6000.0 | ✅ |
| `cell_r_max` / `cell_r_min` | 4500.0 / 400.0 | 동일(ckpt 유지) | ✅ |
| `cell_nets` | 2 | 2 | ✅ |
| `cell_cluster_obs` | True | True | ✅ |
| `n_clusters` | **4** | **3** | ⚠️ 불일치 |
| `nets_per_ship` | 1 | **3** | ⚠️ 불일치 |

**`n_clusters` 4→3**: 어텐션은 집합 기반이라 토큰 수가 달라도 구조적으로는 동작하지만,
학습 때 본 적 없는 토큰 분포라 분포 이동(distribution shift)이다.

**`nets_per_ship` 1→3**: own[4] = `a_nets / nets_per_ship` 의 **분모가 달라진다.**
학습 때는 항상 1.0 아니면 0.0 인 이진값이었는데, 추론에서는 3/3, 2/3, 1/3 의
중간값이 들어간다 — 정책이 본 적 없는 입력이다. (재학습 시 정렬 권장 항목)

---

## 5. 다른 스케일 실험장으로 이식하기 (예: 33m × 33m)

정규화가 **전부 상대값**이라 대부분 자동으로 따라온다. 바꿔야 할 것은 소수다.

### 바꿔야 하는 것 — 길이 차원 (m 단위)
현재 스케일 대비 비율 `s = 33 / 12000 ≈ 0.00275` 를 **모든 길이에 동일하게** 곱한다:

| 필드 | 현재 | 역할 |
|---|---|---|
| `action_grid_half` | 6000 | **정규화 분모 — 가장 중요** |
| `cell_r_min` / `cell_r_max` | 400 / 4500 | 요격 환형 |
| `ally_speed` (→ `enemy_speed`) | — | m/step |
| `ally_mother_radius` | 300 | 모선 격침 반경 |
| `net_max_len`, 스폰 반경 등 | — | 길이 계열 전부 |

### 자동으로 따라오는 것 (건드릴 필요 없음)
- **속도 특징** — `approach / cfg.enemy_speed` 로 자기 자신에 정규화됨
- **개수 특징** — `count / M`, `a_nets / nets_per_ship`
- **각도 특징** — heading_vec은 스케일 무관
- **셀 좌표/반경** — `action_grid_half` 하나만 맞추면 전부 따라옴

### 핵심
**길이를 전부 같은 비율로 줄이면 정규화된 관측은 수학적으로 동일**하다.
즉 정책 재학습 없이 이론상 그대로 이식 가능하다.

단, 스케일과 함께 **무차원 비율이 깨지면** 재학습이 필요하다:
- `속도 × decision_period / cell_r_max` (한 결정당 이동 비율)
- `ally_mother_radius / cell_r_min` (위험 반경 비율)
- `net_max_len / 셀 간격` (그물이 덮는 셀 수)

실기 이식 시 **이 비율들을 먼저 계산해 현재 값과 대조**할 것.
길이만 비례 축소하면 이 비율들은 보존되지만, 실기 하드웨어 제약
(최소 선회반경, 실제 속도 하한 등)이 비율을 깨뜨리기 쉽다.

---

## 6. 요약

- 정규화는 `build_cell_obs()` **한 곳**에 집중 — 원점=모선, 분모=`action_grid_half`
- 위치는 `/half`, 속도는 `/enemy_speed`, 개수는 `/정원`, 각도는 `(sin,cos)`
- **full=6000 / slim=4500** 로 분모가 다르니 모드 확인 필수
- 스케일 이식은 **길이 계열만 같은 비율로** 축소 → 나머지는 자동
- 현재 `n_clusters`(4→3), `nets_per_ship`(1→3) 는 학습/추론 불일치 상태
