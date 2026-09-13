# -*- coding: utf-8 -*-
"""논문에 쓰는 지표 이름의 **단일 출처**.

왜 따로 뺐나
    같은 지표를 표는 "그물 걸림", 그림은 "그물 접촉" 이라고 불렀다. 표는
    tools/make_paper_tables.py 의 METRICS 를, 그림은 plots.METRIC_KO 를 쓰기 때문이다.
    독자가 표와 그림을 오갈 때 같은 양인지 확인해야 하는 것은 조판 결함이다.
    이 모듈을 양쪽이 함께 읽어 어휘를 하나로 묶는다.

plots.METRIC_KO 는 건드리지 않는다 — 탐색용 도판 수십 장이 그 이름을 쓰고 있고,
그쪽은 화면에서 빨리 읽히는 이름이 낫다. 논문에 실리는 것만 여기를 따른다.

무거운 의존이 없어야 한다(matplotlib 등). tools/ 의 표 생성기가 그림 없이 임포트한다.
"""
from __future__ import annotations

#: (지표 키, 한국어 이름, 낮을수록 좋은가, 소수 자릿수)
#  순서는 논문 표의 행 순서이자 forest 의 지표군 안 순서다.
METRICS = (
    ("capture_rate",         "포획률",                False, 3),
    ("breaches",             "돌파 수",               True,  2),
    ("collision_rate",       "충돌률(척당)",          True,  3),
    ("net_touches",          "그물 걸림",             True,  2),
    ("nets_per_capture",     "포획당 그물",           True,  3),
    ("traveled_per_capture", "포획당 이동거리 (m)",   True,  0),
    ("turn_sum_rad",         "총 선회량 (rad)",       True,  0),
    ("cap_dist_mean",        "포획 이격거리 (m)",     False, 0),
    ("cap_time_mean",        "평균 포획 시각 (step)", True,  0),
    ("net_deploy_edist",     "전개 시 적 거리 (m)",   True,  0),
)

#: 지표 키 → 논문 표기.
METRIC_KO = {k: ko for k, ko, _lb, _nd in METRICS}

#: 지표 키 → 낮을수록 좋은가.
LOWER_BETTER = frozenset(k for k, _ko, lb, _nd in METRICS if lb)


def label(metric: str, *, unit: bool = True) -> str:
    """논문 표기. `unit=False` 면 괄호 안 단위를 뗀다(그림 축이 좁을 때).

    단위를 떼도 이름 자체는 표와 같게 유지된다 --- "포획당 이동거리 (m)" 와
    "포획당 이동거리" 는 독자가 같은 양으로 읽지만, "포획 1척당 이동" 은 아니다.
    """
    ko = METRIC_KO.get(metric, metric)
    if not unit:
        ko = ko.split(" (")[0]
    return ko
