"""지휘관 프롬프트 (개선판).

원본(MobRobGPT 3분할 골격)의 약점을 반영해 재작성:
- 기하 계산을 LLM 암산에서 덜어냄: 접근선/수직 단위벡터·중심좌표·per-cluster
  도달반경·lane 각도를 STATE로 내려주고, LLM은 "판단 + 준비된 벡터로 산술"만.
- 상충/미정의 제거: 반경 안에서 바깥 지향, WP0=현재 위치, WP0의 deploy_net 무시.
- situation에만 있던 제약(net_max_len)을 규칙으로 승격·강제.
- 단위(미터) 명시, 원칙 재번호(1~8) + PRIORITY 0 격상, few-shot 예시 1개 추가.

- SYSTEM_PROMPT: 역할 + 규칙 (고정 → 프롬프트 캐시 친화)
- build_user_content: 동적 전장 상태(JSON) + 선택적 자연어 지시
출력 형식은 스키마(format=)로 강제되므로 규칙은 '무엇을 결정하라'에 집중.
"""
import dataclasses
import json
import warnings
from typing import List, Optional

from .schema import BattlefieldState, ClusterDeployment, CommanderPlan


# SYSTEM_PROMPT가 가정하는 파생 기하 키(폴백 시 누락되면 품질이 조용히 무너짐 → 경고).
_REQUIRED_GEO_KEYS = ("center", "reachable_radius", "approach_dir", "perp_dir")


def _serialize_state(state: BattlefieldState) -> str:
    """BattlefieldState를 JSON 문자열로 직렬화 (구현 방식에 무관하게 동작).

    우선순위: to_prompt_json(파생 기하값 포함) > pydantic v2 > pydantic v1 >
    dataclass > __dict__ > str 폴백. to_prompt_json이 없어 폴백으로 내려가면
    파생 기하값이 빠지므로(SYSTEM_PROMPT가 이를 가정) 조용히 열화하는 대신 경고한다.
    """
    fn = getattr(state, "to_prompt_json", None)
    if callable(fn):
        return fn()

    # ── 폴백: 파생 기하값이 없을 수 있음 → 소리 나게 경고 ──
    warnings.warn(
        "BattlefieldState.to_prompt_json()이 없어 폴백 직렬화를 사용합니다. "
        "SYSTEM_PROMPT가 가정하는 파생 기하값"
        f"({', '.join(_REQUIRED_GEO_KEYS)})이 누락돼 배정 품질이 저하될 수 있습니다.",
        RuntimeWarning, stacklevel=2,
    )
    for attr in ("model_dump_json", "json"):
        fn = getattr(state, attr, None)
        if callable(fn):
            return fn()
    if dataclasses.is_dataclass(state):
        return json.dumps(dataclasses.asdict(state), ensure_ascii=False, default=str)
    if hasattr(state, "__dict__"):
        return json.dumps(vars(state), ensure_ascii=False, default=str)
    return str(state)


SYSTEM_PROMPT = """You are the tactical COMMANDER of a maritime defense mission.

[SITUATION & OBJECTIVE]
- A high-value mothership sits at the CENTER, and enemy clusters (id, count, bearing, etc.) advance from the edges.
- You command up to 3 friendly USVs (allies: id, pos, heading, nets_remaining, route, assigned_cluster, etc.) to intercept them using capture nets.
- Your output must be a single JSON object containing deployments and hold_ships.

[CORE TACTICAL PRINCIPLES]
1. CONTINUITY & STATE CONSISTENCY (★Priority #1): Maintain strict consistency in both target assignment (`assigned_cluster`) and motion state. Ships already cruising must continue advancing without unnecessary disruption. Conversely, ships that were already in HOLD or RESERVE should consistently remain paused unless a critical new threat demands their release, minimizing erratic start-stop behavior.
2. NET DEPLOYMENT LOCK-IN (NO MID-WAY INTERRUPTION): If a ship has already started laying a net, you must lock in its deployment. Do NOT reassign it to a different cluster or place it on HOLD until the current net wall is fully deployed and completed.
3. ALTERNATING HOLD RULE (NO CONSECUTIVE HOLDS): If a ship was placed in `hold_ships` during the previous decision cycle to avoid a collision, **it is strictly prohibited from being held again in the immediately following cycle**. It must be allowed to move so ships can take turns passing each other, preventing total stagnation.
4. ONE SHIP PER CLUSTER (NO DUPLICATION): Assigning multiple USVs to a single cluster is strictly prohibited. If enemy clusters approach from very similar bearings, a single ship's net wall can span the group; thus, it is perfectly fine if some clusters are left without a direct individual assignment.
5. MINIMUM FORCE WITH NO BREACH: Prefer using the fewest ships possible, but never allow an enemy to breach the defense line. Do not dispatch ships to sectors that are already net-covered (`net_covered:true`) or fully handled by a teammate's wall.
6. COLLISION AVOIDANCE & ANTI-STAGNATION (★NO ALL-SHIP HOLD): When a collision risk (overlapping routes) is detected, **keep the single ship closest to the enemy moving** and place the other conflicting ships into the `hold_ships` list. However, ensuring defense continuity means **you must never put all 3 alive ships on HOLD simultaneously**, which would paralyze the fleet. At least one ship must remain active.
7. MULTI-NET REDEPLOYMENT (MAX 3 DEPLOYMENTS): A friendly ship that has finished laying a net but has nets remaining (`nets_remaining > 0`) must be actively redeployed. If its current cluster needs more coverage, or a new threat appears, reassign the ship so it can deploy nets up to 3 times in total.
8. EFFICIENT SIDE-MATCH (nearest, least-turn, NON-CROSSING): assign each cluster to the ally that is CLOSEST and needs the LEAST turning. Every ally carries `to_clusters` = [{id, dist, turn}] (travel distance & turn to each cluster) and `bearing_from_center` (the direction the ally sits, in the SAME angle units as each cluster's `bearing`). Pick the ship↔cluster pairing that MINIMIZES total (dist + turn), and match each cluster to the ally whose `bearing_from_center` is nearest that cluster's `bearing` — the ally ALREADY on that side. A LEFT-side threat → the LEFT ally; REAR → rear; RIGHT → right. Lanes then fan out and NEVER cross. NEVER send a far / opposite-side ship across another ship's path (that is a wasteful crossing and a collision risk).
9. LEFT/RIGHT SIDE-ORDER (★keep each ship on its OWN side — a ship that cuts across the CENTER grazes the mothership and is SUNK; this also wastes travel). Decide each cluster's side from its `bearing` (0°=up/north, 90°=RIGHT/east, 180°=down/south, 270°=LEFT/west), equivalently from its `center.x` vs the mothership's x:
   • LEFT side (bearing 180–360°, i.e. west / center.x < mothership.x) → assign USVs in the fixed order **0 → 1 → 2** (take USV 0 first, then 1, then 2).
   • RIGHT side (bearing 0–180°, i.e. east / center.x > mothership.x) → assign USVs in the fixed order **2 → 1 → 0** (take USV 2 first, then 1, then 0).
   Left threats are thus handled by the low-id (left) ships and right threats by the high-id (right) ships, so NO ship ever crosses the mothership at the center.
   ★ STRENGTH BY POSITION — apply this order MOST STRICTLY for UPPER-corner threats: upper-LEFT (bearing ≈ 270–360°, strongest near 315°/NW) and upper-RIGHT (bearing ≈ 0–90°, strongest near 45°/NE). Those corner threats have the highest center-crossing / mothership-collision risk, so the fixed order is MANDATORY there. For low/rear threats (bearing near 180°) you may relax it only if `to_clusters` clearly favors another ship AND the path still does not pass near the center. NEVER send a right-order ship (2/1) to an upper-LEFT cluster, nor a left-order ship (0/1) to an upper-RIGHT cluster — that is exactly the dangerous center-crossing that sinks ships.

[ENEMY FORMATION PLAYBOOK — the command line carries "[ENEMY FORMATION: <name>]"; adapt to it]
Each ally carries ONLY the nets in `nets_remaining` (usually 1). A net, once laid, is spent. So HOW you spend nets across TIME is decisive, and it differs by formation:
- concentrated: ONE dense group from a single bearing. Assign ONE ship (its wall spans the group); add a 2nd only if count/spread is very large. Others stay in reserve.
- diversionary: feints from SEVERAL bearings at once. **In this formation, you must chase and track all detected attacks without exception. Putting ships on HOLD is strictly discouraged; all ships must move at full power to intercept every identified threat corridor.** Side-match one ship to each active cluster (nearest bearing_from_center) to establish an omnidirectional defense perimeter.
- wave: successive RANKS arrive over TIME from similar bearings, separated by gaps. ★THE KEY MISTAKE IS SPENDING ALL NETS ON THE FIRST RANK — then later ranks breach freely. STAGGER instead:
    · Meet the CURRENT front rank with just ONE ship deploying, pushing the intercept point out slightly so its wall finishes before the rank passes.
    · Keep the OTHER ships assigned but transit-only without deploying their net yet so they pre-position toward the ring (do not leave them unassigned as idle reserves, keep them moving).
    · On each re-plan, as the next rank closes in, flip one pre-positioned ship to deploy its net to meet it. Aim for one fresh net per rank.
    · Never let every ship deploy at once, and never leave a still-inbound rank with no net left to answer it.

[OUTPUT RULES]  (the JSON has rationale FIRST, then deployments, then hold_ships)
- ★rationale — ★★MUST BE WRITTEN IN KOREAN (반드시 한국어로 작성). English rationale is NOT allowed.★★
  Write this FIRST (think before you decide). 판단 근거를 한국어 2~4문장으로: 각 클러스터에 대해 어느 배가
  가장 가깝고·선회 적고·같은 쪽인지(원칙 8) 따진 뒤, 배정과 HOLD/예비 결정과 그 이유를 한국어로 서술.
  충돌 HOLD면 어느 배를 계속 움직였고(적에 가장 가까움) 어느 배를 멈췄는지(연속 HOLD 금지) 한국어로 밝힐 것.
- deployments: list, each element STRICTLY {cluster_id, ally_ids} ONLY. Leave ally_ids empty to let the system auto-assign by efficiency.
- hold_ships: list of ally IDs to pause in place this cycle (default []).
- 규칙: rationale 값만 한국어(문장). 나머지 JSON 키/구조/숫자는 그대로 유지.

Respond ONLY with the required JSON object. Do not add prose outside it."""


# ── FEW-SHOT 예시 ─────────────────────────────────────────────────────
# 규칙 문장으로는 잘 안 지켜지던 '경계 판단'만 시범으로 보여준다(설명이 아니라 시연).
#   Ex1  정상 배정   : 연속성 유지 + 1척/클러스터 + 효율 매칭(HOLD 없음 — 과잉 HOLD 방지 균형추)
#   Ex2  연속성/HOLD : 팀원 그물벽에 덮인 중복 배는 '재배정 아니라' 같은 클러스터에 둔 채 HOLD
#   Ex3  net_covered : 이미 그물로 차단된 클러스터엔 새로 안 깖(deploy_net=false) + 예비
# 매 호출 고정(캐시 프리픽스) → system 뒤, 실제 STATE 앞에 가짜 대화 턴으로 삽입한다.
# STATE JSON 은 schema.to_prompt_json 과 같은 키 구조(축약 realistic), Plan JSON 은
# CommanderPlan 스키마로 생성 → format=/response_format 강제와 100% 일치 보장.

def _fewshot_state(payload: dict) -> str:
    """예시 STATE dict → build_user_content 과 동일 형식의 user content 문자열."""
    return "BATTLEFIELD STATE:\n" + json.dumps(payload, ensure_ascii=False)


# 공통 상수(예시들 사이 반복 최소화)
_C = [3000.0, 3000.0]           # center
_CONST = {"world_size": 6000.0, "mothership_radius": 300.0, "net_max_len": 500.0,
          "ally_speed": 12.0, "enemy_speed": 24.0, "max_intercept_radius": 1500.0}


def _cl(cid, center, bearing, spread, count, appr, dist, adir, pdir, reach,
        gap, covered=False):
    return {"id": cid, "center": center, "bearing": bearing, "spread": spread,
            "count": count, "approach_speed": appr, "distance_to_center": dist,
            "approach_dir": adir, "perp_dir": pdir, "reachable_radius": reach,
            "net_covered": covered, "nearest_cluster_gap_deg": gap}


def _al(aid, pos, hdg, assigned, to_clusters, route, nets=2, alive=True,
        deploying=False, hits_net=False, covered_by_mate=False):
    lane = round((360.0 * aid) / 3.0, 2)
    return {"id": aid, "alive": alive, "pos": pos, "heading": hdg,
            "nets_remaining": nets, "assigned_cluster": assigned, "lane_angle": lane,
            "deploying": deploying, "route_hits_net": hits_net,
            "cluster_covered_by_teammate": covered_by_mate,
            "to_clusters": to_clusters, "route": route}


def _tc(*triples):
    return [{"id": i, "dist": d, "turn": t} for (i, d, t) in triples]


# ── Ex1: 정상 배정 — 세 클러스터가 방위상 잘 벌어짐, 배 3척이 이미 각자 담당 ──
_EX1_STATE = {
    "center": _C, **_CONST, "mothership_threat_level": 0.3,
    "enemy_clusters": [
        _cl(0, [4500.0, 3000.0], 90.0, 15.0, 4, 20.0, 1500.0, [-1.0, 0.0], [0.0, -1.0], 900.0, 120.0),
        _cl(1, [2550.0, 3780.0], 210.0, 12.0, 3, 18.0, 1500.0, [0.3, -0.95], [-0.95, -0.3], 900.0, 120.0),
        _cl(2, [2550.0, 2220.0], 330.0, 12.0, 3, 18.0, 1500.0, [0.3, 0.95], [0.95, -0.3], 900.0, 120.0),
    ],
    "allies": [
        _al(0, [3700.0, 3000.0], 90.0, 0, _tc((0, 200.0, 0.0), (1, 1390.0, 120.0), (2, 1390.0, 120.0)),
            [[3700.0, 3000.0], [3900.0, 3000.0]]),
        _al(1, [2650.0, 3600.0], 210.0, 1, _tc((0, 1200.0, 150.0), (1, 210.0, 10.0), (2, 1400.0, 90.0)),
            [[2650.0, 3600.0], [2550.0, 3780.0]]),
        _al(2, [2650.0, 2400.0], 330.0, 2, _tc((0, 1200.0, 150.0), (1, 1400.0, 90.0), (2, 210.0, 10.0)),
            [[2650.0, 2400.0], [2550.0, 2220.0]]),
    ],
    "command": None,
}
_EX1_PLAN = CommanderPlan(
    deployments=[
        ClusterDeployment(cluster_id=0, ally_ids=[0], deploy_net=True),
        ClusterDeployment(cluster_id=1, ally_ids=[1], deploy_net=True),
        ClusterDeployment(cluster_id=2, ally_ids=[2], deploy_net=True),
    ],
    hold_ships=[],
    rationale="세 클러스터가 방위상 충분히 벌어져(간격 120°) 각자 1척으로 차단합니다. "
              "각 배가 자기 담당 클러스터에 가장 가깝고 선회도 최소(to_clusters)라 그대로 유지, "
              "겹침·충돌 없어 HOLD 불필요합니다.",
)

# ── Ex2: 연속성/HOLD — c0·c1 근접(간격 10°), a1 그물벽이 c0까지 덮음 → a0 중복 ──
_EX2_STATE = {
    "center": _C, **_CONST, "mothership_threat_level": 0.5,
    "enemy_clusters": [
        _cl(0, [4500.0, 3000.0], 90.0, 15.0, 3, 18.0, 1500.0, [-1.0, 0.0], [0.0, -1.0], 900.0, 10.0),
        _cl(1, [4470.0, 2740.0], 100.0, 22.0, 5, 18.0, 1520.0, [-0.98, 0.17], [0.17, 0.98], 900.0, 10.0),
        _cl(2, [1500.0, 2000.0], 250.0, 12.0, 3, 18.0, 1800.0, [0.85, 0.53], [0.53, -0.85], 900.0, 150.0),
    ],
    "allies": [
        # a0: 담당 c0 이지만 a1(c1)의 넓은 그물벽에 c0 접근로가 이미 덮임 → 중복
        _al(0, [3850.0, 3000.0], 90.0, 0, _tc((0, 150.0, 0.0), (1, 320.0, 20.0), (2, 2400.0, 160.0)),
            [[3850.0, 3000.0], [3900.0, 3000.0]], covered_by_mate=True),
        _al(1, [3800.0, 2780.0], 100.0, 1, _tc((0, 330.0, 25.0), (1, 180.0, 5.0), (2, 2450.0, 155.0)),
            [[3800.0, 2780.0], [3960.0, 2650.0]]),
        _al(2, [2100.0, 2350.0], 250.0, 2, _tc((0, 2350.0, 160.0), (1, 2400.0, 150.0), (2, 210.0, 8.0)),
            [[2100.0, 2350.0], [1980.0, 2270.0]]),
    ],
    "command": None,
}
_EX2_PLAN = CommanderPlan(
    deployments=[
        ClusterDeployment(cluster_id=0, ally_ids=[0], deploy_net=True),
        ClusterDeployment(cluster_id=1, ally_ids=[1], deploy_net=True),
        ClusterDeployment(cluster_id=2, ally_ids=[2], deploy_net=True),
    ],
    hold_ships=[0],
    rationale="c0·c1이 방위상 근접(간격 10°)해 a1의 넓은 그물벽(spread 22°)이 c0 접근로까지 함께 "
              "차단합니다(a0의 cluster_covered_by_teammate=true). 따라서 a0는 중복이므로 다른 "
              "클러스터로 재배정하지 않고 담당 c0에 둔 채 HOLD해 연속성을 지킵니다. c2는 a2로 차단.",
)

# ── Ex3: net_covered — c0 는 이미 그물로 차단됨 → 새로 안 깖 + 그쪽 배는 HOLD ──
_EX3_STATE = {
    "center": _C, **_CONST, "mothership_threat_level": 0.4,
    "enemy_clusters": [
        _cl(0, [4500.0, 3000.0], 90.0, 14.0, 3, 18.0, 1500.0, [-1.0, 0.0], [0.0, -1.0], 900.0, 180.0, covered=True),
        _cl(1, [1500.0, 3000.0], 270.0, 16.0, 4, 20.0, 1500.0, [1.0, 0.0], [0.0, 1.0], 900.0, 180.0),
    ],
    "allies": [
        # a0: c0 로 이동 중이었으나 c0 가 net_covered → 그물 재투척 방지 위해 HOLD
        _al(0, [3800.0, 3000.0], 90.0, 0, _tc((0, 300.0, 0.0), (1, 2300.0, 175.0)),
            [[3800.0, 3000.0], [3900.0, 3000.0]]),
        _al(1, [2200.0, 3000.0], 270.0, 1, _tc((0, 2300.0, 175.0), (1, 300.0, 0.0)),
            [[2200.0, 3000.0], [2100.0, 3000.0]]),
        # a2: 미배정 예비(어느 deployment 에도 넣지 않음)
        _al(2, [3000.0, 3800.0], 0.0, None, _tc((0, 1620.0, 90.0), (1, 1620.0, 90.0)),
            [[3000.0, 3800.0]]),
    ],
    "command": None,
}
_EX3_PLAN = CommanderPlan(
    deployments=[
        ClusterDeployment(cluster_id=0, ally_ids=[0], deploy_net=False, net_legs=[]),
        ClusterDeployment(cluster_id=1, ally_ids=[1], deploy_net=True),
    ],
    hold_ships=[0],
    rationale="c0는 net_covered=true라 이미 그물로 차단돼 포획 예상 → 새 그물을 겹쳐 깔지 않도록 "
              "deploy_net=false로 두고, 그쪽으로 가던 a0는 재투척·낭비 방지 위해 HOLD합니다. "
              "c1은 a1이 차단하고, a2는 새 위협에 대비해 예비로 남깁니다.",
)


def _pair(state_payload: dict, plan: CommanderPlan) -> List[dict]:
    """(예시 STATE, 예시 Plan) → [user, assistant] 메시지 쌍. Plan 은 스키마 JSON 그대로."""
    return [
        {"role": "user", "content": _fewshot_state(state_payload)},
        {"role": "assistant", "content": plan.model_dump_json()},
    ]


# system 뒤·실제 STATE 앞에 삽입할 고정 few-shot 대화 턴(캐시 프리픽스).
FEWSHOT_MESSAGES: List[dict] = (
    _pair(_EX1_STATE, _EX1_PLAN)
    + _pair(_EX2_STATE, _EX2_PLAN)
    + _pair(_EX3_STATE, _EX3_PLAN)
)


def build_messages(state: BattlefieldState, instruction: Optional[str] = None,
                   fewshot: bool = False) -> List[dict]:
    """어댑터 공통 messages 조립: [system] + (few-shot 고정 턴) + [실제 STATE].

    few-shot 을 system 문자열에 박지 않고 별도 대화 턴으로 두는 이유:
    - 모델이 보는 입출력 형태가 실제와 100% 동일(STATE JSON→Plan JSON) — 설명이 아닌 시연.
    - system 이 고정이라 프롬프트 캐시 프리픽스가 살아남(예시도 매 호출 불변).
    fewshot=False 로 끄면 기존 2-메시지(zero-shot) 동작.
    """
    msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fewshot:
        msgs += FEWSHOT_MESSAGES
    msgs.append({"role": "user", "content": build_user_content(state, instruction)})
    return msgs


def build_user_content(state: BattlefieldState, instruction: Optional[str] = None) -> str:
    """동적 전장 상태를 사용자 메시지로 직렬화.

    자연어 지시는 state.command 한 곳으로 단일화한다(중복 방지). instruction 인자가
    주어지면 state.command를 그 값으로 덮어써서 직렬화하며, 별도 블록으로 다시 붙이지
    않는다 — 그러지 않으면 command 필드와 이중으로 들어가 토큰 낭비·혼동을 유발한다.

    직렬화는 _serialize_state가 처리. SYSTEM_PROMPT는 상태에 center, world_size,
    mothership_radius, net_max_len, 각 cluster의 approach_dir/perp_dir/reachable_radius,
    각 ally의 pos/lane_angle/nets_remaining이 있다고 가정한다(schema.to_prompt_json이 계산).
    """
    if instruction is not None:
        # command 필드로 승격해 단일 소스 유지. 원본 state는 불변(copy).
        copy_fn = getattr(state, "model_copy", None) or getattr(state, "copy")
        state = copy_fn(update={"command": instruction.strip()})
    return "BATTLEFIELD STATE:\n" + _serialize_state(state)