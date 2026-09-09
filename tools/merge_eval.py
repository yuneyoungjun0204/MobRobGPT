"""병렬 수집분(results/eval + results/eval_gemini)을 하나로 합친다.

왜 별도 폴더인가
    두 드라이버가 out 디렉터리를 나눠 쓰도록 했다 — 같은 episodes.csv 에 두
    프로세스가 append 하면 컬럼이 어긋난다(과거 실제 발생). 그림을 그리려면
    합쳐야 하는데, 원본 두 개는 건드리지 않는다: LOCAL 드라이버가 아직 자기
    CSV 에 append 중일 수 있다. 그래서 results/eval_merged/ 로 따로 낸다.

    run_eval.py --from-csv 는 **CSV 와 같은 폴더의 llm_metrics.csv** 를 읽으므로
    (run_eval.py:268) 둘을 같은 폴더에 나란히 놓아야 LLM 품질 지표가 그림에 실린다.

중복 처리
    휴리스틱 조건(heur_*)은 양쪽 실행에 모두 들어 있다. 같은 (condition,
    formation, seed) 가 두 번 들어가면 시드 짝짓기가 깨지므로 첫 번째만 남긴다.

★ 폴백 검역
    폴백률이 높은 로그는 'LLM 성능'이 아니라 휴리스틱 출력이다(gpt-4o-mini 가
    크레딧 소진으로 폴백률 1.000 을 기록해 전량 폐기된 전례가 있다). 여기서도
    임계를 넘는 모델은 병합에서 제외하고 이유를 찍는다 — 검역 폴더에 옮겨두는
    것에만 의존하지 않는다.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIRS = [os.path.join(REPO, "results", "eval"),
            os.path.join(REPO, "results", "eval_gemini")]
DST_DIR = os.path.join(REPO, "results", "eval_merged")

KEY = ["condition", "formation", "seed"]
#: 이 비율 이상 폴백한 모델은 LLM 이 아니라 휴리스틱을 돌린 것이다.
FALLBACK_MAX = 0.2


def merge_episodes() -> pd.DataFrame | None:
    frames = []
    for d in SRC_DIRS:
        p = os.path.join(d, "episodes.csv")
        if not os.path.exists(p):
            print(f"[merge] 없음, 건너뜀: {os.path.relpath(p, REPO)}")
            continue
        f = pd.read_csv(p)
        print(f"[merge] {os.path.relpath(p, REPO)}  {len(f)} 행")
        frames.append(f)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    before = len(d)
    d = d.drop_duplicates(subset=KEY, keep="first")
    print(f"[merge] 중복 제거 {before} → {len(d)} (휴리스틱 조건이 양쪽에 있어 정상)")
    return d


def merge_llm() -> pd.DataFrame | None:
    """양쪽 폴더의 llm_calls_*.jsonl 을 하나의 llm_metrics 프레임으로."""
    from commander.llm_metrics import LLMRecorder
    frames = []
    for d in SRC_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "llm_calls_*.jsonl"))):
            f = LLMRecorder.load_jsonl(p)
            if f.empty:
                continue
            rate = float(f["fallback"].mean())
            tag = os.path.basename(p)[len("llm_calls_"):-len(".jsonl")]
            if rate > FALLBACK_MAX:
                print(f"[merge] ⚠ 제외: {tag} 폴백률 {rate:.1%} > {FALLBACK_MAX:.0%} "
                      f"— LLM 출력이 아니라 휴리스틱이다")
                continue
            print(f"[merge] LLM {tag:26s} {len(f):5d} 호출  폴백 {rate:.4f}")
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    sys.path.insert(0, REPO)
    d = merge_episodes()
    if d is None:
        print("[merge] 합칠 것이 없다"); return 1

    cov = d.groupby(["condition", "formation"])["seed"].nunique().unstack(fill_value=0)
    print("\n[커버리지]"); print(cov.to_string())
    short = [(c, f, int(v)) for c, r in cov.iterrows() for f, v in r.items() if v < 30]
    if short:
        print("\n[!] 30 시드 미만 — 집계에서 제외될 수 있다:")
        for c, f, v in short:
            print(f"      {c:34s} {f:14s} {v}/30")

    os.makedirs(DST_DIR, exist_ok=True)
    pe = os.path.join(DST_DIR, "episodes.csv")
    d.to_csv(pe, index=False, encoding="utf-8-sig")
    print(f"\n[merge] 에피소드 → {os.path.relpath(pe, REPO)}  ({len(d)} 행)")

    print()
    ldf = merge_llm()
    if ldf is not None:
        pl = os.path.join(DST_DIR, "llm_metrics.csv")
        ldf.to_csv(pl, index=False, encoding="utf-8-sig")
        print(f"[merge] LLM 계측 → {os.path.relpath(pl, REPO)}  ({len(ldf)} 행, "
              f"백엔드 {sorted(ldf['backend'].unique())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
