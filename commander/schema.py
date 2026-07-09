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


class AllyShip(BaseModel):
    id: int
    pos: Point
    heading: float = Field(..., description="선수각[deg]")
    nets_remaining: int = Field(..., ge=0)
    assigned_cluster: Optional[int] = Field(None, description="직전 배정(연속성 힌트)")


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
            })

        # lane_angle: 아군을 방위상 균등 분리(고유 섹터 보장). 힌트일 뿐 배정은 LLM이 결정.
        n = max(len(self.allies), 1)
        allies = []
        for i, a in enumerate(self.allies):
            lane_angle = round((360.0 * i) / n, 2)
            allies.append({
                "id": a.id,
                "pos": [round(a.pos.x, 2), round(a.pos.y, 2)],
                "heading": a.heading,
                "nets_remaining": a.nets_remaining,
                "assigned_cluster": a.assigned_cluster,
                "lane_angle": lane_angle,
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


# ── 출력: 배별 경로 계획 (LLM 이 WP 좌표와 그물 구간을 직접 생성) ──
class Waypoint(BaseModel):
    x: float = Field(..., description="맵 x 좌표[m] (0~world_size)")
    y: float = Field(..., description="맵 y 좌표[m] (0~world_size)")
    deploy_net: bool = Field(..., description="직전 WP→이 WP 구간에서 그물을 펼칠지")


class ShipRoute(BaseModel):
    ally_id: int = Field(..., description="이 경로를 수행할 아군 id")
    waypoints: List[Waypoint] = Field(
        ..., description="이 배의 경유 좌표. 정확히 6개(WP0=출발 근처 → WP5=가장 바깥).")


class CommanderPlan(BaseModel):
    routes: List[ShipRoute] = Field(
        ..., description="투입(배정)할 배마다 6-WP 경로. 여기 없는 배는 예비(정지).")
    rationale: str = Field(..., description="투입 척수·경로·그물 구간 결정의 판단 근거(생각 과정)")