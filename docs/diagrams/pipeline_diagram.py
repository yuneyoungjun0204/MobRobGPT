# -*- coding: utf-8 -*-
"""MobRobGPT 의사결정 파이프라인 — Diagram as Code (mingrammer/diagrams)

입력(센서/시뮬) → 전장상태 → LLM 지휘관(전략) → 래스터 관측
→ U-Net 점수맵 정책(경유점 생성) → 저수준 제어 → 액추에이터

렌더:
    python docs/diagrams/pipeline_diagram.py
출력:
    docs/diagrams/mobrobgpt_pipeline.{png,pdf}   (상세, 7단계 전체 — 부록/포스터용)
    docs/diagrams/mobrobgpt_overview.{png,pdf}   (요약, 발표 본문 슬라이드용)

★ PDF_TOO — 논문에는 반드시 PDF 를 실을 것. Graphviz 의 **PNG 래스터라이저가 한글
  글리프를 자리에 따라 잘못 고른다**: 같은 그림 안에서 "이동 WP + 그물벽" 은 정상인데
  "그물 도색" 이 "그불 도색", "저수준 제어" 가 "저수순 제어" 로 나온다. 글꼴을 바꿔도
  깨지는 글자만 달라진다(NanumGothic 에서는 "그룰"). SVG·PDF 출력의 텍스트는 정확하므로
  PNG 는 미리보기용으로만 쓴다.

수치 출처 — 창작 없음:
    boatattack_sim/models/u-net_map.pt (config)  … decision_period 25, arrive_radius 200,
        net_max_len 450, net_width 4, ally_speed 6.0, enemy_speed 9.0, cnn_grid_n 50,
        cnn_extent 6300, cnn_dup_r 200, cnn_min_valid 150, cnn_gate_r 3000, cnn_width 32,
        ally_max_turn 8.0, ally_turn_gain 0.6, ally_mother_radius 300, dt 1.0
    boatattack_sim/env/defense_env.py::_micro    … PD 추종 · 그물 도색 · ptr 전진
    boatattack_sim/model/cnn_actor.py            … UNetLite · q·key · greedy(K=2)
    commander/{prompts,sim_bridge,fallback,rl_bridge,ros2_sensor_bridge}.py
"""
import os

# Graphviz 는 winget 설치 직후 PATH 반영이 세션 재시작 전까지 안 될 수 있다.
_GV = r"C:\Program Files\Graphviz\bin"
if os.path.isdir(_GV) and _GV not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _GV

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.flowchart import (
    Action, Decision, Delay, Document, InputOutput, OffPageConnectorLeft,
    OffPageConnectorRight, PredefinedProcess, Preparation, StoredData,
)

OUT = os.path.dirname(os.path.abspath(__file__))

FONT = "Malgun Gothic"          # 한글 라벨
GRAPH = {
    "fontname": FONT, "fontsize": "30", "labelloc": "t",
    "bgcolor": "white", "pad": "0.6", "nodesep": "0.60", "ranksep": "1.10",
    "splines": "spline",
}
NODE = {"fontname": FONT, "fontsize": "11"}
EDGE = {"fontname": FONT, "fontsize": "10"}

# 계층별 색 — project_ppt.md 의 2계층 구분과 같은 의미
C_IN     = "#ECEFF1"   # 입력
C_STATE  = "#E3F2FD"   # 전장상태
C_LLM    = "#FFF3E0"   # 전략 계층 (지연이 큰 쪽)
C_OBS    = "#F1F8E9"   # 관측
C_POLICY = "#E8EAF6"   # 기동 계층 (학습 정책)
C_CTRL   = "#FCE4EC"   # 저수준 제어
C_OUT    = "#EEEEEE"   # 액추에이터

# 계층별 엣지 색
E_IN, E_ST, E_LLM, E_OBS, E_POL, E_CTL, E_OUT = (
    "#546E7A", "#1565C0", "#E65100", "#558B2F", "#283593", "#AD1457", "#455A64")
E_FB = "#90A4AE"       # 되먹임 — 흐름을 방해하지 않도록 옅게


def _cl(bg):
    return {"bgcolor": bg, "fontname": FONT, "fontsize": "15",
            "style": "rounded", "pencolor": "#607D8B", "penwidth": "1.4",
            "margin": "18"}


def _fb(label):
    """되먹임 엣지 — 랭킹에 참여하지 않고(constraint=false) 시각적으로 물러난다."""
    return Edge(color=E_FB, style="dashed", penwidth="1.0",
                constraint="false", fontcolor=E_FB, label=label)


# ════════════════════════════════════════════════════════════════════════
# 상세 파이프라인 (7단계)
# ════════════════════════════════════════════════════════════════════════
def build_detail():
    with Diagram(
        "자폭 USV 군집 방어 — LLM 지휘관 → 강화학습 경유점 생성 → 저수준 제어",
        filename=os.path.join(OUT, "mobrobgpt_pipeline"),
        outformat="png", show=False, direction="LR",
        graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE,
    ):
        # ── ① 입력 ─────────────────────────────────────────────────────
        with Cluster("①  입력\n센서(실기) / 시뮬", graph_attr=_cl(C_IN)):
            gps = InputOutput("GPS\n/ally_i/fix\n/enemy_i/fix\n/mothership/fix")
            imu = InputOutput("자세\n/ally_i/imu")
            geo = PredefinedProcess("GeoBridge\nGPS ↔ sim affine\n(t=0 1회 fit)")
            sim = PredefinedProcess("DefenseVecEnv\n시뮬 물리\ndt = 1 s")
            raw = StoredData("원시 상태\na_pos · a_hdg\ne_pos · center")
            back_in = OffPageConnectorLeft("⑦ 에서\n다음 스텝")
            [gps, imu] >> Edge(color=E_IN) >> geo >> Edge(color=E_IN) >> raw
            sim >> Edge(color=E_IN, style="dotted") >> raw
            back_in >> Edge(color=E_FB, style="dashed", penwidth="1.0") >> raw

        # ── ② 전장상태 ─────────────────────────────────────────────────
        with Cluster("②  전장상태 구성\nLLM 이 암산하지 않도록 기하량 선계산",
                     graph_attr=_cl(C_STATE)):
            clu = Action("cluster_by_gaps_vec\n방위 갭 분할 → ≤ 4 무리\n★ id 는 매번 재부여")
            pre = Action("geometry.intercepts\n요격점 · 거리 · 선회량")
            state = Document("BattlefieldState\nto_clusters[dist,turn]\n"
                             "net_covered · route_hits_net\nassigned_cluster/bearing")
            raw >> Edge(color=E_ST) >> [clu, pre]
            [clu, pre] >> Edge(color=E_ST) >> state

        # ── ③ 전략 계층 ────────────────────────────────────────────────
        # ★ "느린 주기"라고 쓰지 말 것 — 두 판단 계층은 같은 25 step 주기다.
        #   갈리는 것은 지연이다(14B 평균 16.6 s vs 순전파 1회).
        with Cluster("③  전략 계층 — LLM 지휘관\n25 step 주기 · 비동기 (14B 지연 평균 16.6 s)",
                     graph_attr=_cl(C_LLM)):
            prompt = Document("prompts.py\n전술 원칙 9개\n출력 스키마")
            llm = Delay("LLM 지휘관\nollama / openai\ntemperature = 0\n구조적 JSON")
            fb = Action("heuristic_plan\n(LLM 무응답 시)")
            plan = Document("CommanderPlan\nrationale\ndeployments[cluster,ships]\nhold_ships")
            asg = Action("plan_to_assign(llm)\nLLM 지정 그대로\n코드 보완 없음")
            out = InputOutput("_assign [P]\n배 → 무리 id\n−1 = HOLD")
            # prompt·fb 는 들어오는 엣지가 없어 rank 0 으로 밀린다 → ②와 상자가 겹친다.
            # 보이지 않는 엣지로 ② 뒤 대역에 고정한다.
            state >> Edge(style="invis") >> [prompt, fb]
            state >> Edge(color=E_LLM, penwidth="2.2", label=" 비동기 호출") >> llm
            prompt >> Edge(color=E_LLM, style="dashed") >> llm
            llm >> Edge(color=E_LLM, penwidth="2.2") >> plan
            fb >> Edge(color="#9E9E9E", style="dashed", label=" 폴백") >> plan
            plan >> Edge(color=E_LLM, penwidth="2.2") >> asg
            asg >> Edge(color=E_LLM, penwidth="2.2") >> out

        # ── ④ 관측 래스터화 ────────────────────────────────────────────
        with Cluster("④  관측 래스터화\n50 × 50 px (1 px = 252 m) · 15 ch",
                     graph_attr=_cl(C_OBS)):
            gmap = InputOutput("gmap [9,50,50]\n적 presence/v/threat\nally · net_installed\nannulus · land")
            smap = InputOutput("smap [P,3,50,50]\nself_marker\nself_intercept\nself_valid")
            own = InputOutput("own [P,8]\n위치 · sin/cos(hdg)\n잔여 그물 · 요격점")
            valid = Preparation("valid [P,50,50]\n행동 후보 마스크\nannulus ∩ ¬land\nr ≤ 3000 m · ±60°\n폴백 최소 150 px")
            # raw → gmap 은 ④를 가로지르는 먼 엣지다. 랭킹에 참여시키면 클러스터가
            # 서로 겹치므로 constraint 를 끈다(의미는 그대로 유지).
            raw >> Edge(color=E_OBS, style="dashed", constraint="false",
                        label=" 원시 상태") >> gmap
            out >> Edge(style="invis") >> gmap                     # gmap 랭크 고정
            out >> Edge(color=E_LLM, penwidth="2.2", label=" 배정 주입") >> [smap, own, valid]

        # ── ⑤ 기동 계층 ────────────────────────────────────────────────
        # 주기를 ③과 같이 명시한다. 한쪽에만 주기를 적으면 "다른 주기"로 읽힌다.
        with Cluster("⑤  기동 계층 — U-Net 점수맵 정책 (강화학습)\n"
                     "25 step 주기 · 동기 (배치 호출 1회) · u-net_map.pt",
                     graph_attr=_cl(C_POLICY)):
            unet = PredefinedProcess("UNetLite (w=32)\n+ CoordConv\n15 ch → key [32]/px")
            ctxn = Action("own_mlp(own) → ctx\nq_proj → q [32]")
            logit = Action("q · key → 2500 logits\nmasked softmax(valid)")
            pick = Action("greedy 자기회귀\nK = 2 픽셀\n2번째는 200 m 제외\n+ subpixel offset")
            dec = Action("flat_to_world\n픽셀 + offset\n→ 세계좌표")
            stop = Decision("정지 규칙\n_assign < 0 or\na_nets ≤ 0 ?")
            route = InputOutput("route [P,2,2]  ★생성된 경유점\n① 이동 WP  (net_mask 0)\n② 그물벽 끝점 (net_mask 1)")
            hold = Action("route ← 현재 위치\n완전 정지")
            [gmap, smap] >> Edge(color=E_POL) >> unet
            unet >> Edge(color=E_POL) >> logit
            own >> Edge(color=E_POL) >> ctxn >> Edge(color=E_POL) >> logit
            valid >> Edge(color=E_POL, style="dashed", label=" 마스크") >> logit
            logit >> Edge(color=E_POL, penwidth="2.2") >> pick
            pick >> Edge(color=E_POL) >> dec >> Edge(color=E_POL) >> stop
            stop >> Edge(color=E_POL, penwidth="2.2", label=" 아니오") >> route
            stop >> Edge(color="#B71C1C", style="dashed", label=" 예") >> hold

        # ── ⑥ 저수준 제어 ──────────────────────────────────────────────
        with Cluster("⑥  저수준 제어 — 매 스텝 (dt = 1 s)\n학습 정책과 분리된 결정론적 층",
                     graph_attr=_cl(C_CTRL)):
            keep = Action("mother_keepout\n모선(300 m) 척력\n→ steer_target")
            pd = Action("pd_follow (PD 조타)\n6.0 m/s\n최대선회 8°/step")
            arr = Decision("도달 판정\n|p − WP| ≤ 200 m ?")
            paint = Action("_paint 그물 도색\npaint_dist < 450 m\nnet_width = 4")
            ptr = Action("ptr 전진\n→ 다음 leg")
            route >> Edge(color=E_CTL, penwidth="2.2") >> keep
            keep >> Edge(color=E_CTL) >> pd >> Edge(color=E_CTL) >> arr
            arr >> Edge(color=E_CTL, label=" net_mask 1") >> paint
            arr >> Edge(color=E_CTL, label=" 도달") >> ptr

        # ── ⑦ 액추에이터 ───────────────────────────────────────────────
        with Cluster("⑦  액추에이터 · 출력", graph_attr=_cl(C_OUT)):
            pub = InputOutput("/ally_i/waypoints\nnav_msgs/Path\n→ 실기 USV")
            rend = InputOutput("DefenseVecEnv 갱신\n+ 점수맵 오버레이")
            back_out = OffPageConnectorRight("① 로\n다음 스텝 (매 tick)")
            ptr >> Edge(color=E_OUT) >> rend
            route >> Edge(color=E_OUT, style="dashed", constraint="false",
                          label=" ROS2 경로") >> pub
            ptr >> Edge(style="invis") >> pub                      # pub 을 ⑦ 대역으로
            rend >> Edge(color=E_FB, style="dashed", penwidth="1.0") >> back_out

        # ── 되먹임 ─────────────────────────────────────────────────────
        paint >> _fb("net_installed") >> gmap
        out >> _fb("배정 연속성") >> state


# ════════════════════════════════════════════════════════════════════════
# 요약 (발표 본문 슬라이드용)
# ════════════════════════════════════════════════════════════════════════
def build_overview():
    g = dict(GRAPH); g["fontsize"] = "26"; g["ranksep"] = "1.35"; g["nodesep"] = "0.8"
    n = dict(NODE); n["fontsize"] = "13"
    with Diagram(
        "판단 지연이 다른 2계층 — 같은 25 step 주기, 다른 응답 시간",
        filename=os.path.join(OUT, "mobrobgpt_overview"),
        outformat="png", show=False, direction="LR",
        graph_attr=g, node_attr=n, edge_attr=EDGE,
    ):
        src = InputOutput("입력\nGPS · IMU · 적 탐지\n(또는 시뮬 물리)")
        st = Document("전장상태\n무리 분할 + 기하 선계산")

        with Cluster("전략 계층 — 25 step 주기 · 비동기 (지연 평균 16.6 s)",
                     graph_attr=_cl(C_LLM)):
            llm = Delay("LLM 지휘관\n어느 배 → 어느 무리\nHOLD 판단")
            asg = InputOutput("_assign [P]\n−1 = HOLD")
            llm >> Edge(color=E_LLM, penwidth="2.4") >> asg

        with Cluster("기동 계층 — 25 step 주기 · 동기 (배치 호출 1회)",
                     graph_attr=_cl(C_POLICY)):
            obs = InputOutput("래스터 관측\n15 ch × 50 × 50")
            pol = PredefinedProcess("U-Net 점수맵 정책\n픽셀 2점 지목")
            wp = InputOutput("경유점 route\n이동 WP + 그물벽")
            obs >> Edge(color=E_POL, penwidth="2.4") >> pol
            pol >> Edge(color=E_POL, penwidth="2.4") >> wp

        # ★ 세 번째 시간척도를 명시한다. 두 판단 계층은 25 step 으로 같고, 제어만
        #   매 tick(1 s)이다 — 이걸 안 적으면 "계층마다 주기가 다르다"는 오해가 남는다.
        ctrl = Action("저수준 제어 — 매 tick (1 s)\nPD 조타 6.0 m/s\n그물 도색 450 m")
        act = InputOutput("액추에이터\n/ally_i/waypoints\n또는 시뮬 물리")

        src >> Edge(color=E_IN, penwidth="2.4") >> st
        st >> Edge(color=E_LLM, penwidth="2.4", label=" 비동기 호출") >> llm
        st >> Edge(color=E_OBS, style="dashed") >> obs
        asg >> Edge(color=E_LLM, penwidth="2.4", label=" 배정 주입") >> obs
        wp >> Edge(color=E_CTL, penwidth="2.4") >> ctrl
        ctrl >> Edge(color=E_OUT, penwidth="2.4") >> act
        act >> _fb("되먹임 (매 tick)") >> src


if __name__ == "__main__":
    build_detail()
    build_overview()
    print("OK ->", OUT)
