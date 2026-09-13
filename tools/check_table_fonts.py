# -*- coding: utf-8 -*-
"""PDF 에서 표 본문의 실제 글자 크기를 잰다.

\\resizebox 가 발동하면 소스의 \\footnotesize(10pt) 가 6.95pt 로 찍힌다 --- 소스만
봐서는 절대 알 수 없다. els-audit 이 이 방식으로 축소를 잡아냈으므로 같은 방식으로
검증한다.

기준: 캡션은 \\footnotesize(elsarticle 의 \\@makecaption) 이므로, 표 본문 글자가
      캡션보다 작으면 축소가 걸린 것이다.
"""
import os
import sys

import fitz

os.chdir(r"C:\Users\ANSL\orca\workspaces\MobRobGPT\U-net\논문초안")
PDF = sys.argv[1] if len(sys.argv) > 1 else "build/USV_swarm_defense_EAAI_submission.pdf"

#: (표를 알아볼 캡션 조각, 표 안에 있는 고유 문자열)
TARGETS = [
    ("조건별 포획률", "지휘관(배정)"),
    ("지휘관 계획 품질과 성능", "커버리지"),
    ("본 논문은 같은 문자를", "학습의 그룹 후보 수"),
    ("전 지표 효과크기", "총 선회량"),
    ("교전 시나리오와 관측 설정", "관측 격자 반폭"),
]

d = fitz.open(PDF)
print(f"{PDF}  ({d.page_count}쪽)\n")
print(f"{'표':<24}{'캡션 pt':>9}{'본문 pt':>9}{'배율':>8}  판정")
fail = 0
for cap_key, body_key in TARGETS:
    cap_sz = body_sz = None
    for pno in range(d.page_count):
        blocks = d[pno].get_text("dict")["blocks"]
        spans = [sp for b in blocks if b.get("type") == 0
                 for l in b["lines"] for sp in l["spans"]]
        txt = "".join(sp["text"] for sp in spans).replace(" ", "")
        if cap_key.replace(" ", "") not in txt:
            continue
        for sp in spans:
            t = sp["text"].replace(" ", "")
            if cap_sz is None and cap_key.replace(" ", "")[:8] in t:
                cap_sz = round(sp["size"], 2)
            if body_sz is None and body_key.replace(" ", "")[:6] in t:
                body_sz = round(sp["size"], 2)
        if cap_sz and body_sz:
            break
    if cap_sz is None or body_sz is None:
        print(f"{cap_key:<24}{'?':>9}{'?':>9}  못 찾음")
        continue
    ratio = body_sz / cap_sz
    okk = ratio > 0.97
    fail += (not okk)
    # ★ 배율을 항상 찍는다. 통과/실패만 내면 0.98 같은 미세 축소가 "ok" 뒤에 숨는다.
    #   눈에 안 보이는 정도라도 값은 보여야 판단할 수 있다.
    print(f"{cap_key:<24}{cap_sz:>9.2f}{body_sz:>9.2f}{ratio:>8.3f}  "
          f"{'ok' if okk else '!! 축소'}")
sys.exit(1 if fail else 0)
