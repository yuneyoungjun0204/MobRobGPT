"""지휘관 프롬프트.

설계 원칙(A1 스키마 축소 이후 기준으로 재작성):
- LLM 이 실제로 조작할 수 있는 레버는 단 3개뿐이다 — deployments(어느 클러스터에 어느 배),
  hold_ships(누구를 세울지), rationale. WP/net_legs/deploy_net 은 스키마에서 제거됐으므로
  프롬프트는 그 3개로 표현 가능한 지시만 담는다(표현 불가능한 지시 = 죽은 토큰).
- 규칙 나열이 아니라 '결정 절차'를 준다. 7B 급 모델은 서로 ★ 붙은 규칙 9개보다
  우선순위 목록 + 순서가 정해진 단계를 훨씬 안정적으로 따른다.
- 출력 필드의 '실제 효과'를 계약으로 명시한다(sim_bridge.plan_to_assign 의 동작과 일치):
  hold_ships 는 배정을 덮어쓰고, deployments 에 올린 클러스터는 ally_ids 가 비어도
  시스템이 누군가를 채운다. 이 계약을 모르면 LLM 이 의도와 반대되는 계획을 낸다.
- few-shot 은 손으로 쓴 dict 가 아니라 실제 BattlefieldState → to_prompt_json() 으로
  만든다 → 키 집합·파생 기하 수치가 프로덕션과 100% 동일(스키마가 바뀌면 예시도 자동 추종).

구성:
- SYSTEM_PROMPT : 역할 + 출력계약 + 상태용어 + 우선순위 + 결정절차 (고정 → 프롬프트 캐시 친화)
- FEWSHOT_MESSAGES : 고정 예시 대화 3턴 (system 뒤 · 실제 STATE 앞)
- build_user_content : 동적 전장 상태(JSON) + 선택적 자연어 지시
출력 형식은 스키마(format=/response_format)로 강제되므로 규칙은 '무엇을 결정하라'에 집중.
"""
import dataclasses
import json
import math
import warnings
from typing import List, Optional

from .schema import (AllyShip, BattlefieldState, ClusterDeployment, CommanderPlan,
                     Constraints, EnemyCluster, Mothership, Point)


# SYSTEM_PROMPT 가 이름으로 지목해 쓰는 파생 키(폴백 직렬화 시 누락되면 품질이 조용히 무너짐).
_REQUIRED_GEO_KEYS = ("bearing_from_center", "to_clusters", "distance_to_center",
                      "nearest_cluster_gap_deg")


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
        "SYSTEM_PROMPT가 가정하는 파생값"
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

A high-value mothership sits at the CENTER. Enemy clusters advance toward it from the edges.
You command up to 3 friendly USVs (allies). Each ally carries capture nets; a laid net forms a
wall across one approach corridor and is spent once laid. Your ONLY job each cycle: decide
WHICH ally handles WHICH cluster, and WHICH allies stop this cycle. Routes, net geometry and
net timing are computed downstream — never describe them.

[OUTPUT CONTRACT — what each field actually does downstream]
- deployments: [{cluster_id, ally_ids}]. Nothing else is allowed in an element.
  · A cluster listed here WILL get a ship: if ally_ids is empty the system auto-picks the most
    efficient free ally. So a cluster you want to IGNORE must be OMITTED from the list entirely.
  · One ally id may appear in at most ONE deployment. Duplicates are dropped.
- hold_ships: [ally_id]. These allies freeze in place this cycle. HOLD OVERRIDES ASSIGNMENT:
  a ship in hold_ships loses its assignment even if you also listed it in ally_ids.
  · Therefore NEVER put the same ally in both ally_ids and hold_ships — pick one.
  · An ally that appears in neither is a RESERVE: it also stays put, but stays free for the
    next cycle. Use hold_ships only to stop a ship for a stated safety/duplication reason.
- rationale: Korean, written FIRST (see the last section).

[STATE GLOSSARY — the only fields you need]
Per cluster: `id`, `bearing` (deg from mothership, 0=north, 90=east, 180=south, 270=west),
  `count` (enemy ships), `spread` (angular width of the group, deg),
  `distance_to_center` (SMALLEST = the front rank, arrives first),
  `net_covered` (true = a net already blocks this corridor → OMIT this cluster),
  `nearest_cluster_gap_deg` (bearing gap to the closest other cluster; <= 20 means the two
   clusters share one corridor and ONE net wall can span both).
Per ally: `id`, `alive` (false = sunk → never assign, never hold),
  `nets_remaining` (0 = cannot block anything → deprioritize, keep as reserve),
  `assigned_cluster` (last cycle's target — the continuity hint),
  `deploying` (true = currently laying a net → LOCKED),
  `bearing_from_center` (where this ally sits, SAME angle convention as cluster `bearing`),
  `to_clusters` [{id, dist, turn}] (travel distance [m] and turn [deg] to each cluster —
   use these numbers directly, never estimate distances from coordinates),
  `route_hits_net` (true = its current route crosses an installed net → it will be sunk → HOLD),
  `cluster_covered_by_teammate` (true = a teammate's wall already blocks its corridor → redundant).
Everything else in the JSON (approach_dir, perp_dir, reachable_radius, lane_angle, net_max_len,
speeds, world_size) is for the downstream planner. Ignore it.

[PRECEDENCE — when rules conflict, the lower number wins]
1. SAFETY: never assign a dead ally; hold any ally with route_hits_net=true; never put ALL
   alive allies on hold at once (at least one must stay active).
2. LOCK-IN: an ally with deploying=true keeps its current cluster and is never held.
3. NO BREACH: every active cluster that is not net_covered and not already spanned by another
   ship's wall must have exactly one ship — as long as a suitable ally is free.
4. SIDE DISCIPLINE (no center-crossing): a ship that cuts across the CENTER grazes the
   mothership and is sunk. Decide a cluster's side from its `bearing`:
     · LEFT/west (bearing 180-360) → take allies in the fixed order 0 -> 1 -> 2
     · RIGHT/east (bearing 0-180)  → take allies in the fixed order 2 -> 1 -> 0
   This order is MANDATORY for upper-corner threats (upper-left ~270-360, worst near 315;
   upper-right ~0-90, worst near 45). For rear threats (bearing near 180) you may deviate
   when `to_clusters` clearly favours another ally.
5. CONTINUITY: keep each ally on its `assigned_cluster` unless rules 1-4 force a change.
   Do not swap targets for a marginal gain — thrashing wastes travel.
6. ECONOMY: among the allies still free, pick the one with the smallest (dist + turn) in
   `to_clusters` and whose `bearing_from_center` is nearest the cluster's `bearing`.
   Use the fewest ships; unused ships stay reserve so their nets answer the next threat.

[DECISION PROCEDURE — follow in order]
S1. Drop allies with alive=false. Drop (omit) clusters with net_covered=true.
S2. Lock every ally with deploying=true onto its assigned_cluster.
S3. For each remaining cluster, keep the ally that already had it (continuity) if that ally is
    free and does not violate rule 4.
S4. Fill the still-uncovered clusters using rules 4 then 6.
S5. Merge duplicates: if two clusters have nearest_cluster_gap_deg <= 20, keep ONE ship for the
    pair and OMIT the other cluster. Any ally with cluster_covered_by_teammate=true is
    redundant — remove it from deployments and put it in hold_ships.
S6. Safety holds: add every ally with route_hits_net=true to hold_ships. If two routes overlap,
    keep the ally CLOSER to its enemy moving and hold the other. Release a hold as soon as its
    cause (route_hits_net / cluster_covered_by_teammate / overlap) is gone — never hold a ship
    cycle after cycle for the same stale reason.
S7. Apply the formation playbook, then run the self-check.

[FORMATION PLAYBOOK — the command line may carry "[ENEMY FORMATION: <name>]"]
Nets are few (often 1 per ally) and spent on use, so HOW you spend them over TIME differs:
- concentrated: one dense group from one bearing. List ONE cluster with ONE ship; add a second
  ship only when count is large AND spread is wide. The rest stay reserve (list them nowhere).
- diversionary: simultaneous feints from several bearings. Cover EVERY active cluster with one
  ship each — an omnidirectional perimeter. hold_ships must stay EMPTY here unless rule 1
  forces a hold; never leave a corridor open in this formation.
- wave: successive ranks from similar bearings, separated in range. THE CLASSIC MISTAKE IS
  SPENDING EVERY NET ON THE FIRST RANK — later ranks then walk in free. Instead:
    · Answer only the FRONT rank (the cluster with the SMALLEST distance_to_center) with ONE ship.
    · Omit the further ranks from deployments and leave the other ships as reserve — they are
      your answer to the next rank, so do not spend them now.
    · On each replan the next rank becomes the front rank; commit exactly one more ship then.
    · Never let an inbound rank arrive with no ship left holding a net.

[SELF-CHECK before answering]
a. Every cluster_id in deployments exists and is not net_covered; each appears at most once.
b. No ally id appears twice, and no ally is in both ally_ids and hold_ships.
c. No dead ally anywhere; at least one alive ally is NOT held.
d. Every held ally has a stated reason (route_hits_net / covered by teammate / overlap).
e. No ship was sent across the center to the opposite side.

[RATIONALE — Korean only]
rationale 은 반드시 한국어로, JSON 의 맨 앞 필드로 먼저 쓴다(먼저 근거를 쓰고 배정을 정한다).
영어 rationale 은 허용되지 않는다. 2~4문장, 다음 형태를 따른다:
  "대형=<formation>. c<k>→a<i>(가장 가깝고 선회 적음/같은 쪽). c<m>은 <이유>로 제외.
   a<j>는 <이유>로 HOLD. a<n>은 다음 랭크 대비 예비."
한국어는 rationale 값에만 쓴다. 나머지 JSON 키·구조·숫자는 그대로 둔다.

Respond ONLY with the required JSON object. Do not add prose outside it."""


# ── FEW-SHOT 예시 ─────────────────────────────────────────────────────
# 규칙 문장으로는 잘 안 지켜지던 '경계 판단'만 시범으로 보여준다(설명이 아니라 시연).
#   Ex1  정상 배정   : 연속성 유지 + 1척/클러스터 + 좌우 규율(HOLD 없음 — 과잉 HOLD 방지 균형추)
#   Ex2  중복 제거   : 방위 근접(gap 10°) 두 클러스터는 1척이 커버 → 나머지 클러스터는 '목록에서 제외',
#                      중복 배는 deployments 에서 빼고 hold_ships 로만 세운다(배정+HOLD 동시 금지 시연)
#   Ex3  안전/절약   : net_covered 클러스터 제외 + route_hits_net 배 HOLD + nets_remaining=0 배는 예비
# 매 호출 고정(캐시 프리픽스) → system 뒤, 실제 STATE 앞에 가짜 대화 턴으로 삽입한다.
# ★ 예시 STATE 는 실제 BattlefieldState 를 만들어 to_prompt_json() 으로 뽑는다 →
#   키 집합·파생 기하 수치가 프로덕션 입력과 100% 동일(손으로 쓴 dict 의 기하 오류 원천 차단).
# ★ 예시 Plan 은 CommanderPlan 스키마로 생성 → format=/response_format 강제와 100% 일치.

_CX, _CY = 3000.0, 3000.0          # 예시 전장 중심(모선)
_CONS = Constraints(net_max_len=500.0, ally_speed=12.0, enemy_speed=24.0,
                    world_size=6000.0, max_intercept_radius=2000.0)


def _bearing(x: float, y: float) -> float:
    """모선 기준 방위[deg] — schema/rl_bridge 와 동일 규약(0=북, 시계방향)."""
    return round(math.degrees(math.atan2(x - _CX, y - _CY)) % 360.0, 2)


def _cl(cid: int, x: float, y: float, spread: float, count: int, covered: bool = False):
    """좌표에서 bearing 을 계산해 만드는 예시 클러스터(방위-좌표 불일치 원천 차단)."""
    return EnemyCluster(id=cid, center=Point(x=x, y=y), bearing=_bearing(x, y),
                        spread=spread, count=count, approach_speed=24.0,
                        net_covered=covered)


def _al(aid: int, x: float, y: float, hdg: float, assigned, nets: int = 1,
        alive: bool = True, deploying: bool = False, hits_net: bool = False,
        covered_by_mate: bool = False, route=((0.0, 0.0),)):
    return AllyShip(id=aid, pos=Point(x=x, y=y), heading=hdg, nets_remaining=nets,
                    alive=alive, assigned_cluster=assigned,
                    route=[Point(x=rx, y=ry) for rx, ry in route],
                    deploying=deploying, route_hits_net=hits_net,
                    cluster_covered_by_teammate=covered_by_mate)


def _state(clusters, allies, threat: float, command: Optional[str] = None) -> BattlefieldState:
    return BattlefieldState(
        mothership=Mothership(pos=Point(x=_CX, y=_CY), radius=300.0, threat_level=threat),
        enemy_clusters=clusters, allies=allies, constraints=_CONS, command=command)


# ── Ex1: 정상 배정 — 좌/후방/우 세 방향, 각 배가 이미 자기 쪽 담당 → 연속성 유지, HOLD 없음 ──
_EX1_STATE = _state(
    clusters=[
        _cl(0, 1701.0, 3750.0, 14.0, 4),      # bearing ≈ 300 (좌상)
        _cl(1, 3000.0, 1500.0, 12.0, 3),      # bearing = 180 (후방)
        _cl(2, 4299.0, 3750.0, 12.0, 3),      # bearing ≈ 60  (우상)
    ],
    allies=[
        _al(0, 2200.0, 3400.0, 300.0, 0, route=((2200.0, 3400.0), (2050.0, 3480.0))),
        _al(1, 3000.0, 2300.0, 180.0, 1, route=((3000.0, 2300.0), (3000.0, 2150.0))),
        _al(2, 3800.0, 3400.0, 60.0, 2, route=((3800.0, 3400.0), (3950.0, 3480.0))),
    ],
    threat=0.3, command="[ENEMY FORMATION: diversionary]")
_EX1_PLAN = CommanderPlan(
    rationale="대형=diversionary. 세 클러스터가 방위상 충분히 벌어져(간격 120°) 전방위 방어가 필요합니다. "
              "c0→a0(좌측 300°), c1→a1(후방 180°), c2→a2(우측 60°)로 각자 자기 쪽을 이어서 맡아 "
              "연속성과 좌우 규율을 함께 지킵니다. 경로 교차·중복이 없어 HOLD는 없습니다.",
    deployments=[
        ClusterDeployment(cluster_id=0, ally_ids=[0]),
        ClusterDeployment(cluster_id=1, ally_ids=[1]),
        ClusterDeployment(cluster_id=2, ally_ids=[2]),
    ],
    hold_ships=[],
)

# ── Ex2: 중복 제거 — c1·c2 가 우측에서 방위 10° 차 → a2 그물벽 하나로 함께 차단, a1 은 중복 ──
_EX2_STATE = _state(
    clusters=[
        _cl(0, 1524.0, 2739.0, 12.0, 3),      # bearing ≈ 260 (좌측)
        _cl(1, 4299.0, 3750.0, 16.0, 5),      # bearing ≈ 60  (우상)
        _cl(2, 4409.0, 3546.0, 20.0, 3),      # bearing ≈ 70  (우상, c1 과 10° 차)
    ],
    allies=[
        _al(0, 2350.0, 2880.0, 260.0, 0, route=((2350.0, 2880.0), (2200.0, 2855.0))),
        # a1: 담당 c2 이지만 a1 이 맡을 회랑이 a2(c1) 의 넓은 그물벽에 이미 덮임 → 중복
        _al(1, 3750.0, 3300.0, 70.0, 2, covered_by_mate=True,
            route=((3750.0, 3300.0), (3890.0, 3350.0))),
        _al(2, 3700.0, 3450.0, 60.0, 1, route=((3700.0, 3450.0), (3830.0, 3525.0))),
    ],
    threat=0.5, command="[ENEMY FORMATION: concentrated]")
_EX2_PLAN = CommanderPlan(
    rationale="대형=concentrated. c1·c2는 방위 간격이 10°라 한 회랑으로 취급되며 a2의 그물벽(spread 16°) "
              "하나가 둘을 함께 막습니다. 그래서 c2는 배정 목록에서 아예 제외하고, 중복이 된 a1은 "
              "deployments에서 빼고 hold_ships로만 세웁니다(cluster_covered_by_teammate=true). "
              "좌측 c0는 같은 쪽 배인 a0가 그대로 이어 맡습니다.",
    deployments=[
        ClusterDeployment(cluster_id=0, ally_ids=[0]),
        ClusterDeployment(cluster_id=1, ally_ids=[2]),
    ],
    hold_ships=[1],
)

# ── Ex3: 안전/절약 — net_covered 제외 + route_hits_net HOLD + 그물 소진 배는 예비 ──
_EX3_STATE = _state(
    clusters=[
        _cl(0, 4500.0, 3000.0, 14.0, 3, covered=True),   # bearing 90, 이미 그물로 차단
        _cl(1, 1500.0, 3000.0, 16.0, 4),                 # bearing 270 (좌측)
    ],
    allies=[
        _al(0, 2250.0, 3000.0, 270.0, 1, route=((2250.0, 3000.0), (2100.0, 3000.0))),
        # a1: 경로가 설치된 그물을 가로지름 → 그대로 두면 걸려서 격침 → 무조건 HOLD
        _al(1, 3600.0, 3200.0, 90.0, 0, hits_net=True,
            route=((3600.0, 3200.0), (3750.0, 3150.0))),
        # a2: 그물 소진(nets_remaining=0) → 차단 불가 → 배정하지 않고 예비
        _al(2, 3000.0, 3700.0, 0.0, None, nets=0, route=((3000.0, 3700.0),)),
    ],
    threat=0.4, command="[ENEMY FORMATION: wave]")
_EX3_PLAN = CommanderPlan(
    rationale="대형=wave. c0는 net_covered=true라 이미 그물로 막혀 있어 배정 목록에서 제외합니다. "
              "선두 랭크인 c1은 같은 좌측에 있고 선회가 가장 적은 a0가 맡습니다. "
              "a1은 route_hits_net=true라 그대로 두면 그물에 걸려 격침되므로 HOLD하고, "
              "a2는 nets_remaining=0이라 차단력이 없어 다음 랭크 대비 예비로 남깁니다.",
    deployments=[
        ClusterDeployment(cluster_id=1, ally_ids=[0]),
    ],
    hold_ships=[1],
)


def _pair(state: BattlefieldState, plan: CommanderPlan) -> List[dict]:
    """(예시 STATE, 예시 Plan) → [user, assistant] 메시지 쌍.

    user 는 build_user_content 와 **같은 함수**를 타므로 실제 입력과 한 글자도 다르지 않고,
    assistant 는 CommanderPlan 스키마 직렬화라 강제 출력 형식과 정확히 일치한다.
    """
    return [
        {"role": "user", "content": build_user_content(state)},
        {"role": "assistant", "content": plan.model_dump_json()},
    ]


def _build_fewshot() -> List[dict]:
    """예시 대화 턴 생성. 예시 조립이 실패해도 본 프롬프트는 살아야 하므로 방어적으로."""
    try:
        return (_pair(_EX1_STATE, _EX1_PLAN)
                + _pair(_EX2_STATE, _EX2_PLAN)
                + _pair(_EX3_STATE, _EX3_PLAN))
    except Exception as e:  # pragma: no cover - 스키마 변경 시 조용한 붕괴 방지
        warnings.warn(f"few-shot 예시 생성 실패({type(e).__name__}: {e}) → zero-shot 으로 동작합니다.",
                      RuntimeWarning, stacklevel=2)
        return []


# system 뒤·실제 STATE 앞에 삽입할 고정 few-shot 대화 턴(캐시 프리픽스).
# 실제 조립은 파일 끝(build_user_content 정의 이후)에서 한다 — _pair 가 그 함수를 쓰기 때문.
FEWSHOT_MESSAGES: List[dict] = []


def build_messages(state: BattlefieldState, instruction: Optional[str] = None,
                   fewshot: bool = True) -> List[dict]:
    """어댑터 공통 messages 조립: [system] + (few-shot 고정 턴) + [실제 STATE].

    few-shot 을 system 문자열에 박지 않고 별도 대화 턴으로 두는 이유:
    - 모델이 보는 입출력 형태가 실제와 100% 동일(STATE JSON→Plan JSON) — 설명이 아닌 시연.
    - system·예시가 모두 고정이라 프롬프트 캐시 프리픽스가 살아남(매 호출 재계산 없음).
    fewshot=False 로 끄면 2-메시지(zero-shot) 동작.
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

    직렬화는 _serialize_state가 처리. SYSTEM_PROMPT는 상태에 각 cluster 의 bearing/count/
    spread/distance_to_center/net_covered/nearest_cluster_gap_deg, 각 ally 의 alive/
    nets_remaining/assigned_cluster/deploying/bearing_from_center/to_clusters/
    route_hits_net/cluster_covered_by_teammate 가 있다고 가정한다(schema.to_prompt_json이 계산).
    """
    if instruction is not None:
        # command 필드로 승격해 단일 소스 유지. 원본 state는 불변(copy).
        copy_fn = getattr(state, "model_copy", None) or getattr(state, "copy")
        state = copy_fn(update={"command": instruction.strip()})
    return "BATTLEFIELD STATE:\n" + _serialize_state(state)


# build_user_content 가 정의된 뒤에야 예시를 실제 입력 형식으로 뽑을 수 있다.
FEWSHOT_MESSAGES[:] = _build_fewshot()
