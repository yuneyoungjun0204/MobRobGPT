# -*- coding: utf-8 -*-
"""국방학술대회 양식은 그림·표의 제목과 표 안의 설명을 영어로 쓰게 한다(양식 3.2절 (1)).
용어는 논문초안/양식_참고/glossary_ko_en.md 에 맞춘다. 문자열 안의 $...$ 는 수식으로 조판된다.

CELL 은 표 칸의 한글 문자열 → 영문. 여기 없는 칸은 원문 그대로 두고 변환 로그에 남긴다.
"""

# ── 그림 제목(라벨 → 영문) ────────────────────────────────────────────────────
FIG_CAP = {
    "fig:architecture": "Two-layer architecture of the proposed system",
    "fig:formations": "Three attack formations: concentrated, diversionary and wave (left to right)",
    "fig:netconcept": "Capture concept of the stern-launched floating net",
    "fig:clustering": "Bearing sorting, clustering and cluster assignment",
    "fig:observation": "Example of the rasterized 15-channel observation (wave formation, north of Maemuldo, Tongyeong)",
    "fig:policy": "Structure of the score map policy: query-feature inner product and sub-pixel offset",
    "fig:validmask": "Validity mask and pixel pointing (actual output of the trained checkpoint)",
    "fig:grpocredit": "One update of group-relative optimization (illustrated with $G=4$, $H=4$)",
    "fig:convergence": "Training progress (grey: individual seeds, blue: seed mean): (a) policy entropy, "
                       "(b) approximate KL divergence, (c) parameter shift from the behaviour-cloning initialization",
    "fig:ue_env": "Qualitative verification in the Unreal Engine-based simulator",
    "fig:sites": "Six training sites (star: mothership, dashed circle: intercept annulus)",
    "fig:radar_maneuver": "Metric profiles: (a) overall, (b) concentrated, (c) diversionary, (d) wave",
    "fig:condbars": "Capture rate by formation and condition (error bars: 95 % CI)",
    "fig:forest": "Effect sizes $d_z$ for all metrics (light lines: corrected intervals)",
}

# ── 표 제목(라벨 → 영문) ──────────────────────────────────────────────────────
TAB_CAP = {
    "tab:scenario": "Engagement scenario and observation settings",
    "tab:metrics": "Ten evaluation metrics",
    "tab:channels": "Composition of the 15-channel grid observation",
    "tab:validmask_param": "Settings of the validity mask and pixel pointing",
    "tab:reward_coef": "Reward coefficients",
    "tab:simfidelity": "Physical and adjudication models of the simulator and their simplifications",
    "tab:hparams": "Training settings of the maneuver policy",
    "tab:conditions": "$2\\times2$ condition matrix (only two switches differ between conditions)",
    "tab:maneuver": "Stand-alone contribution of the maneuver layer (U-Net score map + GRPO)",
    "tab:algo": "Comparison of learning algorithms on the same pixel action space",
    "tab:main": "Capture rate by condition (mean, 95 % CI, difference from baseline)",
    "tab:planquality": "Commander plan quality and performance",
    "tab:interaction": "$2\\times2$ interaction decomposition (Gemini commander)",
    "tab:effectsize": "Effect sizes for all metrics (descending $|d_z|$)",
}

# ── 표 아래 주석(표 라벨 → 영문) ──────────────────────────────────────────────
NOTES = {
    "tab:maneuver": "$^{*}$ The confidence interval excludes 0. The overall row pools the three formations.",
    "tab:algo": "$^{*}$ The 95 % interval excludes 0. $^{\\dagger}$ Still excluded under Bonferroni correction (99.5 %). "
                "Capture rates of the individual MAPPO runs: 0.876, 0.838, 0.864. Three formations pooled (30 seeds per condition).",
    "tab:main": "$^{*}$ The confidence interval excludes 0. $d_z$: paired effect size. Rows sharing a commander give the "
                "maneuver-layer contribution; rows sharing a maneuver give the commander contribution. Qwen2.5: local (RTX 4070); "
                "Gemini: cloud API.",
    "tab:planquality": "Coverage: fraction of active clusters assigned. Churn: fraction of defenders changing assignment between "
                       "replans. Crossing: fraction of crossing intercept paths. Capture rate with heuristic maneuver; "
                       "fallback rate at most 0.007 for all models.",
    "tab:interaction": "Interaction $=$ (LLM+U-Net $-$ LLM+heuristic) $-$ (heuristic+U-Net $-$ baseline). "
                       "$^{*}$ significant interaction.",
    "tab:effectsize": "$\\uparrow$ higher is better, $\\downarrow$ lower is better. $^{*}$ 95 % CI excludes 0. "
                      "$^{\\dagger}$ Also excludes 0 under Bonferroni correction for 10 metrics.",
}

# ── 표 칸(한글 원문 → 영문) ───────────────────────────────────────────────────
CELL = {
    # tab:scenario
    "항목": "Item", "값": "Value",
    "적 USV 수 $M$": "Number of attacking USVs $M$",
    "방어정 수 $P$": "Number of defenders $P$",
    "적 무리 최대 수 $C$": "Maximum number of clusters $C$",
    "교전 해역 한 변": "Side length of the engagement area",
    "적 USV 속력 $v_e$": "Attacker speed $v_e$",
    "방어정 속력 $v_a$ (속력비 $v_e/v_a = 1.5$)": "Defender speed $v_a$ (speed ratio $v_e/v_a = 1.5$)",
    "결정 주기": "Decision period",
    "관측 격자 $n_{\\mathrm{g}} \\times n_{\\mathrm{g}}$ (모선 중심)": "Observation grid $n_{\\mathrm{g}} \\times n_{\\mathrm{g}}$ (centred on mothership)",
    "관측 픽셀 한 변 $\\ell$": "Observation pixel size $\\ell$",
    "방어정당 그물 (학습 / 평가)": "Nets per defender (training / evaluation)",
    "정책 파라미터 수": "Number of policy parameters",
    # tab:metrics
    "관점": "Aspect", "지표": "Metric", "방향": "Dir.", "정의": "Definition", "뜻": "Meaning",
    "방어 성과": "Defense outcome", "안전": "Safety", "자원 효율": "Resource efficiency",
    "기동 비용": "Maneuver cost", "교전 품질": "Engagement quality",
    "포획률": "Capture rate", "돌파 수": "Breaches", "충돌률": "Collision rate", "그물 걸림": "Net entanglements",
    "포획당 그물": "Nets per capture", "포획당 이동거리": "Distance per capture", "총 선회량": "Total turning",
    "포획 이격거리": "Capture distance", "평균 포획 시각": "Capture time", "전개 시 적 거리": "Enemy distance at deploy",
    "교전이 결판난 적선 중 포획된 비율": "Fraction of resolved attackers that were captured",
    "모선까지 도달한 적선 수": "Number of attackers that reached the mothership",
    "방어정 1척당 아군 상호·모선 충돌 건수": "Defender-defender and defender-mothership collisions per defender",
    "아군이 설치된 그물에 얽힌 횟수": "Times a defender was caught in a deployed net",
    "적 1척 포획에 소모한 그물 수": "Nets consumed per captured attacker",
    "적 1척 포획당 팀 총 이동거리": "Total team travel per captured attacker",
    "누적 방위 변화량(궤적 부드러움)": "Accumulated heading change (trajectory smoothness)",
    "포획 지점의 모선 이격(공간축 조기 제압)": "Distance of capture points from the mothership (early defeat in space)",
    "포획이 일어난 시각(시간축 조기 제압)": "Time at which captures occur (early defeat in time)",
    "그물 전개를 시작한 순간 최근접 적까지 거리": "Distance to the nearest attacker when net deployment starts",
    # tab:metrics (15쪽판의 압축 문구)
    "결판난 적선 중 포획 비율": "Captured / resolved",
    "모선에 도달한 적선 수": "Reached the mothership",
    "방어정당 아군·모선 충돌": "Collisions per defender",
    "아군 그물에 얽힌 횟수": "Own-net entanglements",
    "포획 1건당 그물 소모": "Nets used per capture",
    "포획 1건당 팀 이동거리": "Travel per capture",
    "누적 방위 변화(궤적 평활도)": "Total heading change",
    "포획 지점의 모선 거리": "Range from mothership",
    "포획이 일어난 시각": "Time of capture",
    "그물 전개 시 최근접 적 거리": "Nearest-attacker range",
    # tab:channels
    "#": "#", "묶음": "Group", "채널": "Channel", "범위": "Range",
    "전역": "Global", "방어정별": "Per defender", "좌표": "Coordinate",
    "적 밀도": "Enemy density", "적 진행 $x$": "Enemy heading $x$", "적 진행 $y$": "Enemy heading $y$",
    "적 위협도": "Enemy threat", "아군 위치": "Defender positions", "설치된 그물": "Deployed nets",
    "원점": "Origin", "요격 구역": "Intercept annulus", "육지": "Land",
    "자기 위치": "Own position", "배정 요격점": "Assigned intercept point", "유효 영역": "Valid region",
    "픽셀 내 생존 적선 수$/M$": "Surviving attackers in the pixel $/M$",
    "픽셀 내 적선 헤딩 단위벡터의 평균": "Mean heading unit vector of attackers in the pixel",
    "위와 같음": "Same as above",
    "모선 도달 임박도의 픽셀 내 최대": "Maximum imminence of reaching the mothership in the pixel",
    "픽셀 내 생존 방어정 수$/P$": "Surviving defenders in the pixel $/P$",
    "물리 격자의 그물 점유를 최대값 풀링": "Max-pooled net occupancy of the physics grid",
    "모선 반경 안의 픽셀": "Pixels within the mothership radius",
    "$r_{\\min} \\le \\lVert \\mathbf{x}_i - \\mathbf{m} \\rVert \\le r_{\\max}$ 인 픽셀":
        "Pixels with $r_{\\min} \\le \\lVert \\mathbf{x}_i - \\mathbf{m} \\rVert \\le r_{\\max}$",
    "고해상도 육지 마스크의 면적비": "Area fraction of the high-resolution land mask",
    "방어정 $p$ 가 놓인 픽셀": "Pixel containing defender $p$",
    "$\\mathbf{I}_p$ 가 놓인 픽셀 (미배정이면 전부 0)": "Pixel containing $\\mathbf{I}_p$ (all 0 if unassigned)",
    "유효 마스크 $\\mathcal{V}_p$, 식 {eq:validset}": "Validity mask $\\mathcal{V}_p$, Eq. {eq:validset}",
    # tab:validmask_param
    "구분": "Category", "기호 / 항목": "Symbol / item",
    "유효 픽셀 조건": "Valid pixel conditions", "픽셀 지목": "Pixel pointing", "실측": "Measured",
    "요격 환형 안·바깥 반경 $r_{\\min}$, $r_{\\max}$": "Inner and outer radii of the intercept annulus $r_{\\min}$, $r_{\\max}$",
    "요격점 반경 조건 $r_{\\mathrm{g}}$": "Range gate around the intercept point $r_{\\mathrm{g}}$",
    "무리 방위 조건 $\\vartheta_{\\mathrm{g}}$": "Bearing gate of the cluster $\\vartheta_{\\mathrm{g}}$",
    "유효 픽셀 수 하한": "Minimum number of valid pixels",
    "$150$ 칸": "$150$ cells",
    "방어정 1척당 경유점 수 $K$": "Waypoints per defender $K$",
    "두 점 간격 하한 $r_{\\mathrm{dup}}$": "Minimum spacing of the two points $r_{\\mathrm{dup}}$",
    "두 점 간격 상한": "Maximum spacing of the two points",
    "결정당 평균 유효 비율": "Mean valid fraction per decision",
    # tab:reward_coef
    "항": "Term", "기호": "Symbol", "의미": "Meaning",
    "이벤트": "Event", "퍼텐셜": "Potential", "조밀": "Dense",
    "포획 1건": "Per capture", "돌파 1건": "Per breach", "충돌·육지 충돌 1건": "Per collision or grounding",
    "그물 접촉 1건": "Per net contact", "창 안 종료 시 전멸": "Annihilation when the episode ends in the window",
    "창 안 종료 시 잔존 적 1척당": "Per surviving attacker when the episode ends in the window",
    "모선 근접도 합의 가중": "Weight of the summed proximity to the mothership",
    "근접도 정규화(교전 해역 반폭)": "Proximity normalization (half-width of the engagement area)",
    "할인": "Discount",
    "이동거리 비용": "Travel cost", "그물 소모 비용": "Net consumption cost", "선회량 비용": "Turning cost",
    "목표점 일관성 비용": "Target consistency cost", "모선·타 방어정 근접 비용": "Proximity cost to mothership and other defenders",
    "커버리지 보너스": "Coverage bonus", "배율판 보너스": "Ray-blocking bonus",
    "모선 안전 반경": "Mothership safety radius", "모선 근접 영향 반경": "Influence radius of mothership proximity",
    "타 방어정 근접 영향 반경": "Influence radius of proximity to other defenders",
    # tab:simfidelity
    "요소": "Element", "모델과 설정값": "Model and settings", "단순화": "Simplification",
    "시간": "Time", "방어정 운동": "Defender motion", "적 운동": "Attacker motion", "적 회피": "Attacker avoidance",
    "스폰": "Spawn", "그물벽": "Net wall", "포획·돌파": "Capture / breach", "충돌": "Collision",
    "공간 격자": "Spatial grid", "관측": "Observation",
    "$\\Delta t = 1\\ensuremath{\\,\\mathrm{s}}$; 에피소드 상한 2,000 step": "$\\Delta t = 1\\ensuremath{\\,\\mathrm{s}}$; episode limit 2,000 steps",
    "결정 주기 고정": "Fixed decision period",
    "정속 6.0 m/s; 최대 선회 $8^\\circ$/step; 경유점 도착 반경 200 m":
        "Constant 6.0 m/s; max. turn $8^\\circ$/step; waypoint arrival radius 200 m",
    "관성·조류·파랑 없는 질점 운동학": "Point-mass kinematics without inertia, current or waves",
    "정속 9.0 m/s; 모선 방위 추종 $+$ 사인 위빙; 최대 선회 $5^\\circ$/step":
        "Constant 9.0 m/s; mothership bearing tracking $+$ sinusoidal weaving; max. turn $5^\\circ$/step",
    "학습·통신 없는 규칙 기반": "Rule-based, no learning or communication",
    "전방 500 m 의 설치 그물·육지에 $\\pm32^\\circ$ 측면 조향":
        "$\\pm32^\\circ$ lateral steering for deployed nets or land within 500 m ahead",
    "배치 패턴을 학습하지 않는 지역적 반응": "Local reaction that does not learn deployment patterns",
    "집중·양동 반경 5,450 m; 파상 4,000 m 부터 1,000 m 간격 3단; 방위는 시드별 무작위":
        "Concentrated / diversionary at 5,450 m; wave in 3 ranks from 4,000 m at 1,000 m spacing; bearing random per seed",
    "도메인 랜덤화 없는 고정 구조": "Fixed structure without domain randomization",
    "두 픽셀을 잇는 선분, 최대 450 m; 이동하며 전개": "Segment joining two pixels, max. 450 m; deployed while moving",
    "설치 후 고정, 표류·유실 없음": "Fixed after deployment, no drift or loss",
    "포획 $=$ 설치 그물 칸 진입; 돌파 $=$ 모선 260 m 진입":
        "Capture $=$ entering a net cell; breach $=$ entering 260 m of the mothership",
    "얽힘 지연·부분 손상 없음": "No entanglement delay or partial damage",
    "방어정 상호 115 m, 모선 300 m, 육지, 설치 그물 진입; 해당 방어정 비활성화":
        "Defenders within 115 m, mothership within 300 m, land, or entering a deployed net; the defender is disabled",
    "접촉 즉시 완전 상실": "Total loss on contact",
    "물리·충돌·그물 격자 $200\\times200$, 63 m/칸": "Physics, collision and net grid $200\\times200$, 63 m/cell",
    "관측·행동 격자는 이보다 4배 거침": "Observation and action grids are 4 times coarser",
    "전 선박의 위치·헤딩·잔여 그물 완전 관측": "Full observation of position, heading and remaining nets of all vessels",
    "센서 잡음·지연·탐지 실패 없음": "No sensor noise, latency or missed detection",
    "OpenStreetMap 해안선 래스터; 학습 6곳(Fig. {fig:sites}), 평가 통영 매물도 서측":
        "OpenStreetMap coastline raster; 6 training sites (Fig. {fig:sites}), evaluation west of Maemuldo, Tongyeong",
    "수심·조석 없음": "No bathymetry or tides",
    # tab:hparams
    "환경": "Environment", "최적화": "Optimization", "BC 초기화": "BC warm-up", "평가·채택": "Evaluation / adoption",
    "병렬 전장 수 $N$": "Parallel worlds $N$",
    "학습 포메이션": "Training formations", "집중·양동·파상 순환": "Cycling concentrated, diversionary, wave",
    "척당 그물 (학습 / 평가)": "Nets per defender (training / evaluation)",
    "학습 해역": "Training sites", "실해역 6곳 순환": "Cycling 6 real coastal sites",
    "그룹 크기 $G$ (휴리스틱 포함)": "Group size $G$ (incl. heuristic)",
    "롤아웃 창 $H$": "Rollout window $H$", "180 step": "180 steps",
    "이득 정규화": "Advantage normalization", "휴리스틱 기준·그룹 표준편차": "Heuristic-anchored, group standard deviation",
    "분산 임계 $\\epsilon_{\\mathrm{var}}$": "Variance threshold $\\epsilon_{\\mathrm{var}}$",
    "크레딧 배분": "Credit assignment", "결합(joint)": "Joint",
    "옵티마이저": "Optimizer", "총 업데이트 $T$": "Total updates $T$",
    "학습률 $\\eta_0 \\to \\eta_{\\min}$, 코사인": "Learning rate $\\eta_0 \\to \\eta_{\\min}$, cosine",
    "엔트로피 계수 $\\lambda_0 \\to \\kappa\\lambda_0$, 선형": "Entropy coefficient $\\lambda_0 \\to \\kappa\\lambda_0$, linear",
    "기울기 노름 클립 $g_{\\max}$": "Gradient norm clip $g_{\\max}$",
    "갱신 수 / 학습률": "Updates / learning rate",
    "라벨 가우시안 $\\sigma_{\\mathrm{BC}}$": "Label Gaussian $\\sigma_{\\mathrm{BC}}$",
    "목표": "Target", "기하 휴리스틱(§{sec:heuristic})": "Geometric heuristic (Sec. {sec:heuristic})",
    "greedy 평가 주기 (채택 런 / 재현 런)": "Greedy evaluation interval (adopted / replication runs)",
    "200 / 100 업데이트": "200 / 100 updates",
    "채택 기준": "Adoption criterion", "평가 최고점 (600, 0.805)": "Best evaluation (600, 0.805)",
    "재현 런 시드": "Replication run seeds",
    # tab:conditions
    "조건": "Condition", "배정 $\\times$ 기동": "Assignment $\\times$ maneuver", "무엇을 측정하는가": "What it measures",
    "휴리스틱 $\\times$ 휴리스틱": "Heuristic $\\times$ heuristic", "기준선(baseline)": "Baseline",
    "휴리스틱 $\\times$ U-Net": "Heuristic $\\times$ U-Net", "기동 계층의 순수 기여(LLM 미사용)": "Pure contribution of the maneuver layer (no LLM)",
    "LLM $\\times$ 휴리스틱": "LLM $\\times$ heuristic", "전략 계층의 순수 기여": "Pure contribution of the strategic layer",
    "제안 체계(상호작용 포함)": "Proposed system (incl. interaction)",
    # tab:maneuver
    "포메이션": "Formation", "휴리스틱 기동": "Heuristic maneuver", "U-Net 기동": "U-Net maneuver",
    "집중": "Concentrated", "양동": "Diversionary", "파상": "Wave", "전체": "Overall",
    # tab:algo
    "휴리스틱": "Heuristic", "GRPO(채택)": "GRPO (adopted)",
    "포획률 $\\uparrow$": "Capture rate $\\uparrow$", "돌파 수 $\\downarrow$": "Breaches $\\downarrow$",
    "충돌률(척당) $\\downarrow$": "Collision rate (per defender) $\\downarrow$",
    "그물 걸림 $\\downarrow$": "Net entanglements $\\downarrow$", "포획당 그물 $\\downarrow$": "Nets per capture $\\downarrow$",
    "포획당 이동거리 (m) $\\downarrow$": "Distance per capture (m) $\\downarrow$",
    "총 선회량 (rad) $\\downarrow$": "Total turning (rad) $\\downarrow$",
    "포획 이격거리 (m) $\\uparrow$": "Capture distance (m) $\\uparrow$",
    "평균 포획 시각 (step) $\\downarrow$": "Capture time (step) $\\downarrow$",
    "전개 시 적 거리 (m) $\\downarrow$": "Enemy distance at deploy (m) $\\downarrow$",
    # tab:main
    "지휘관(배정)": "Commander (assignment)", "기동": "Maneuver", "포획률 [95 % CI]": "Capture rate [95 % CI]",
    "휴리스틱 (baseline)": "Heuristic (baseline)", "U-Net 점수맵": "U-Net score map",
    # tab:planquality
    "지휘관": "Commander", "커버리지 $\\uparrow$": "Coverage $\\uparrow$", "교차 $\\downarrow$": "Crossing $\\downarrow$",
    # tab:interaction
    "총효과": "Total effect", "상호작용 [95 % CI]": "Interaction [95 % CI]",
    "충돌률(척당)": "Collision rate (per defender)", "포획당 이동거리 (m)": "Distance per capture (m)",
    "총 선회량 (rad)": "Total turning (rad)", "포획 이격거리 (m)": "Capture distance (m)",
    "평균 포획 시각 (step)": "Capture time (step)", "전개 시 적 거리 (m)": "Enemy distance at deploy (m)",
    # tab:effectsize
    "제안": "Proposed",
    # 15쪽판(물리 단위 없음): tab:scenario 의 속력비 행, ell 단위 거리 지표
    "속력비 $v_e/v_a$ (적 USV / 방어정)": "Speed ratio $v_e/v_a$ (attacker / defender)",
    "포획당 이동거리 ($\\ell$)": "Distance per capture ($\\ell$)",
    "포획 이격거리 ($\\ell$)": "Capture distance ($\\ell$)",
    "전개 시 적 거리 ($\\ell$)": "Enemy distance at deploy ($\\ell$)",
}

# ── 15쪽판(USV_swarm_defense_국방발표_15p) 전용 덮어쓰기 ────────────────────────
#  돌파 수를 지표에서 뺀 판(지표 9개, Bonferroni 99.4 %). 라벨이 전체판과 같아 여기서만 바꾼다.
TAB_CAP_15P = {"tab:metrics": "Nine evaluation metrics"}
NOTES_15P = {
    "tab:effectsize": "$\\uparrow$ higher is better, $\\downarrow$ lower is better. $^{*}$ 95 % CI excludes 0. "
                      "$^{\\dagger}$ Also excludes 0 under Bonferroni correction for 9 metrics.",
}
