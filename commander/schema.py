"""지휘관 입출력 스키마 (벤더 독립 전이 계약).

전장상태(BattlefieldState) → 지휘관 → 배정계획(CommanderPlan).
Pydantic 스키마는 Ollama의 `format=` 구조적 출력(JSON Schema 강제)과
Claude의 `messages.parse`(strict)에 그대로 재사용된다 → 모델 교체 = 어댑터 교체.

좌표는 (x, y) [m]. 방위는 모선/정면 기준 상대 각도[deg].

NOTE: BattlefieldState.to_prompt_json()이 파생 기하값(center, per-cluster
approach_dir/perp_dir/reachable_radius, per-ally lane_angle)을 계산해 프롬프트용
JSON으로 내보낸다. LLM의 좌표 암산 부담을 줄이는 것이 목적. 이 파생값들은 입력
전처리일 뿐이며 LLM 출력 스키마(CommanderPlan)에는 영향을 주지 않는다.
"""
import json
import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


# ── 입력: 전장 상태 ────────────────────────────────────────────────
class Mothership(BaseModel):
    pos: Point
    radius: float = Field(..., description="breach 판정 반경[m]")
    threat_level: float = Field(..., ge=0.0, le=1.0, description="현재 모선 위험도 0~1")


class EnemyCluster(BaseModel):
    id: int
    center: Point
    bearing: float = Field(..., description="모선 기준 상대 방위[deg]")
    spread: float = Field(..., description="각 스프레드[deg] — 무리 폭")
    count: int = Field(..., ge=0, description="이 클러스터의 적 척수")
    approach_speed: float = Field(..., description="모선 방향 평균 접근속도[m/step]")
    net_covered: bool = Field(False, description="접근 방위에 이미 설치된 그물이 있어 포획 예상 여부. "
                                                "true면 새 배정 불필요(건너뛰고 예비로).")


class AllyShip(BaseModel):
    id: int
    pos: Point
    heading: float = Field(..., description="선수각[deg]")
    nets_remaining: int = Field(..., ge=0)
    alive: bool = Field(True, description="생존 여부. false면 격침(충돌/그물접촉) — 배정 불가.")
    assigned_cluster: Optional[int] = Field(None, description="직전 배정(연속성 힌트)")
    route: List[Point] = Field(default_factory=list,
                               description="현재 자동조종 경로 WP (경로 중복·충돌 판단용)")
    deploying: bool = Field(False, description="현재 그물 전개 중 여부")


class Constraints(BaseModel):
    net_max_len: float = Field(..., description="그물 최대 길이[m]")
    ally_speed: float
    enemy_speed: float
    world_size: float
    max_intercept_radius: float = Field(
        ..., description="모선 중심에서 이 반경 안에서만 아군이 적보다 먼저 도달 가능(적이 2배 빠름). "
                         "모든 WP는 반드시 이 반경 안에 둘 것 — 밖은 요격 불가.")


class BattlefieldState(BaseModel):
    mothership: Mothership
    enemy_clusters: List[EnemyCluster]
    allies: List[AllyShip]
    constraints: Constraints
    command: Optional[str] = Field(None, description="선택: 인간 지휘관 자연어 지시")

    # ── 파생 기하값 계산 → 프롬프트용 JSON ─────────────────────────
    def to_prompt_json(self) -> str:
        """SYSTEM_PROMPT가 가정하는 파생 기하값을 계산해 JSON 문자열로 반환.

        - center: 모선 중심
        - per-cluster: approach_dir(중심 지향 단위벡터), perp_dir(수직=그물 방향),
          reachable_radius(이 클러스터보다 먼저 도달 가능한 최대 반경),
          distance_to_center
        - per-ally: lane_angle(측방 분리용 고유 방위 섹터[deg])
        """
        cx, cy = self.mothership.pos.x, self.mothership.pos.y
        c = self.constraints
        v_a = max(c.ally_speed, 1e-6)

        clusters = []
        for cl in self.enemy_clusters:
            dx, dy = cx - cl.center.x, cy - cl.center.y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                ux, uy = dx / d, dy / d           # 클러스터 → 중심 (접근 방향)
            else:
                ux, uy = 0.0, 0.0
            px, py = -uy, ux                       # 수직 (그물 벽 방향)

            # 고전적 요격 반경: r <= v_a * d / (v_a + v_e).
            v_e = max(cl.approach_speed, 1e-6)
            r_reach = v_a * d / (v_a + v_e)
            # 전역 상한과 world 경계로 캡.
            r_reach = min(r_reach, c.max_intercept_radius)
            r_reach = max(r_reach, 0.0)

            # 옹기종기 판단용: 가장 가까운 다른 클러스터와의 방위차[deg] (작을수록 밀집)
            others = [o.bearing for o in self.enemy_clusters if o.id != cl.id]
            if others:
                gap = min(abs(((cl.bearing - ob + 180.0) % 360.0) - 180.0) for ob in others)
                nearest_gap = round(gap, 2)
            else:
                nearest_gap = 360.0

            clusters.append({
                "id": cl.id,
                "center": [round(cl.center.x, 2), round(cl.center.y, 2)],
                "bearing": cl.bearing,
                "spread": cl.spread,
                "count": cl.count,
                "approach_speed": cl.approach_speed,
                "distance_to_center": round(d, 2),
                "approach_dir": [round(ux, 4), round(uy, 4)],
                "perp_dir": [round(px, 4), round(py, 4)],
                "reachable_radius": round(r_reach, 2),
                "net_covered": cl.net_covered,
                "nearest_cluster_gap_deg": nearest_gap,
            })

        # lane_angle: 아군을 방위상 균등 분리(고유 섹터 보장). 힌트일 뿐 배정은 LLM이 결정.
        n = max(len(self.allies), 1)
        allies = []
        for i, a in enumerate(self.allies):
            lane_angle = round((360.0 * i) / n, 2)
            allies.append({
                "id": a.id,
                "alive": a.alive,
                "pos": [round(a.pos.x, 2), round(a.pos.y, 2)],
                "heading": a.heading,
                "nets_remaining": a.nets_remaining,
                "assigned_cluster": a.assigned_cluster,
                "lane_angle": lane_angle,
                "deploying": a.deploying,
                # 현재 자동조종 경로(WP) — LLM이 경로 중복/충돌 판단에 사용
                "route": [[round(p.x, 2), round(p.y, 2)] for p in a.route],
            })

        payload = {
            "center": [round(cx, 2), round(cy, 2)],
            "world_size": c.world_size,
            "mothership_radius": self.mothership.radius,
            "mothership_threat_level": self.mothership.threat_level,
            "net_max_len": c.net_max_len,
            "ally_speed": c.ally_speed,
            "enemy_speed": c.enemy_speed,
            "max_intercept_radius": c.max_intercept_radius,
            "enemy_clusters": clusters,
            "allies": allies,
            "command": self.command,
        }
        return json.dumps(payload, ensure_ascii=False)


# ── 출력: 교전 배분 (LLM은 '어느 클러스터에 몇 척'만 결정 — 경로는 시뮬이 기하로 생성) ──
class ClusterDeployment(BaseModel):
    cluster_id: int = Field(..., description="담당 적 클러스터 id")
    ally_ids: List[int] = Field(default_factory=list,
                                description="이 클러스터를 맡을 아군 USV id 목록(네가 직접 선택). "
                                            "효율(요격 위치에 가깝고·선회 적고·그물 보유)이고 안전"
                                            "(다른 배 경로와 교차·충돌 안 함)한 배를 고를 것. 보통 1척. "
                                            "비우면 시스템이 효율/안전 기준으로 대신 고른다.")
    deploy_net: bool = Field(True, description="지금 그물을 투척할지(투척 시점 결정). true=그물 전개, "
                                              "false=요격 위치로 이동만 하고 아직 안 깖(대기). "
                                              "매 재계획(100스텝)마다 다시 결정 가능.")
    net_legs: Optional[List[int]] = Field(
        None, description="그물을 깔 경로 WP 인덱스 목록(그 배 route 기준, 0부터). "
                          "None=자동(요격 링 구간에 기본 전개), []=이번엔 안 깖(대기), "
                          "[3,4,5]=해당 WP 구간에만. deploy_net=false 면 무시(안 깖).")


class CommanderPlan(BaseModel):
    deployments: List[ClusterDeployment] = Field(
        ..., description="클러스터별 투입 선박 수. 합계는 아군 총수 이하 — 남는 선박은 예비(정지).")
    hold_ships: List[int] = Field(
        default_factory=list,
        description="이번 주기에 '제자리 정지(HOLD)'시킬 아군 id 목록. 배정돼 있어도 이 목록에 "
                    "있으면 전진하지 않고 현재 위치에서 대기(전개중 그물은 마저 설치). 다른 배가 "
                    "이미 차단했거나, 두 배가 충돌 직전이거나, 시간차 출격이 필요할 때 사용.")
    rationale: str = Field(..., description="투입 척수·예비·정지(HOLD) 판단 근거(생각 과정)")