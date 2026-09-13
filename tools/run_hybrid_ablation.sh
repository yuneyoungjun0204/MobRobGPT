#!/usr/bin/env bash
# 배정권한(assign_mode) ablation 드라이버 — hybrid 조건을 모델별로 순차 수집한다.
#
# 왜 별도 드라이버인가
#   14b 는 에피소드당 ~350초라 전량이 17시간을 넘는다. 대화형 백그라운드로는 못 버티므로
#   분리 프로세스로 띄운다. run_eval.py 가 에피소드마다 체크포인트를 남기고 resume 하므로
#   중간에 죽어도 다시 띄우면 이어간다.
#
# 순서 = 정보량/시간 비. 싼 것부터 둬서 일찍 결론이 서게 한다.
#   ① qwen2.5:7b  (~3.3h)  가장 극적으로 실패한 모델 — hybrid 가 구제하는지가 핵심 질문
#   ② gemini      (~1.7h)  커버리지 0.881 이라 hybrid 로 12% 를 메우면 어떻게 되는지
#   ③ qwen2.5:14b (~17.5h) 임계 근처 모델. 가장 비싸므로 마지막
#
# ★★ --figdir 를 반드시 $OUT 아래로 둔다. 논문_그래프 를 가리키면 각 단계가 끝날 때마다
#   run_eval.py 가 **hybrid 원자료 기준으로 논문 도판 전체를 덮어쓴다.** 실제로 그렇게
#   해서 논문의 표(eval_merged)와 그림(eval_hybrid)이 서로 다른 데이터를 가리킨 사고가
#   있었다(2026-09-07). 논문 도판은 tools/make_paper_tables.py 와 짝을 이루어
#   results/eval_merged 에서만 생성한다.
#
# ★ 출력 디렉터리를 results/eval_merged 와 분리한다 — 같은 episodes.csv 에 두 프로세스가
#   append 하면 열이 어긋난다(과거 실제 발생). 병합은 tools/merge_eval.py 가 따로 한다.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

# .env 주입 (gemini_commander 는 os.environ 만 읽는다)
if [ -f .env ]; then
  while IFS= read -r line; do
    line="${line#$'\xef\xbb\xbf'}"                       # BOM 제거
    case "$line" in ''|'#'*) continue;; esac
    export "${line%%=*}"="${line#*=}"
  done < .env
fi

OUT=results/eval_hybrid
COMMON="--assign-mode hybrid --seeds 30 --seed-start 0 --replan-every 1 --out $OUT --figdir $OUT/figs"

run_stage () {                                            # $1=라벨 $2...=추가인자
  local label="$1"; shift
  echo "=================================================================="
  echo "[$(date '+%F %T')] ▶ $label 시작"
  echo "=================================================================="
  python run_eval.py $COMMON "$@" 2>&1
  echo "[$(date '+%F %T')] ◀ $label 종료 (exit=$?)"
}

run_stage "① qwen2.5:7b + hybrid"  --backends ollama --model qwen2.5:7b
run_stage "② gemini + hybrid"      --backends gemini --model gemini-3.5-flash-lite
run_stage "③ qwen2.5:14b + hybrid" --backends ollama --model qwen2.5:14b

echo "[$(date '+%F %T')] ★ 전체 완료 — 원자료: $OUT/episodes.csv"
