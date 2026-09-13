#!/usr/bin/env bash
# EAAI 투고본을 여러 판으로 굽는다.
#
#   ① submission  \documentclass[preprint,12pt]        실제로 제출하는 형태(단단)
#   ② journal     \documentclass[final,5p,times,twocolumn]  EAAI 지면에 실린 모습
#   ③ authoryear  ①과 같되 저자-연도 인용   ← EAAI 지정 스타일 확인용
#        ⚠ 이 판은 **모양 확인용**이다. 저자-연도로 실제 투고하려면
#          python tools/sort_refs.py --order alpha 로 참고문헌을 다시 정렬해야 한다
#          (수동 thebibliography 는 물리적 순서가 곧 번호다).
#
# 알려진 상태: 네 판 모두 에러 0 / Overfull 0 / undefined 0.
#   한때 review 판에만 Overfull vbox 2건(각 32.2pt)이 있었는데, 원인은 심사본에만
#   켜 두었던 쪽 채우기 해제 명령이었다. 그 줄을 빼자 사라졌다.
#
# 왜 스크립트로 두나: 클래스 옵션을 손으로 바꿔 가며 굽다 보면 어느 판을 봤는지
# 잃어버린다. 판마다 별도 PDF 로 떨어뜨리고 원본 .tex 는 건드리지 않는다.
#
# 사용: bash tools/build_paper.sh            (전부)
#       bash tools/build_paper.sh submission (하나만)
set -u

cd "$(dirname "$0")/../논문초안" || exit 1
SRC=USV_swarm_defense_EAAI.tex
BUILD=build
mkdir -p "$BUILD"

# 이전 실행이나 다른 세션의 잔류물을 치운다. 쥐고 있는 프로세스가 있으면
# 지워지지 않으므로 조용히 넘어가지 말고 알린다(그 상태로 구우면 옛 로그를 읽는다).
if ls _build_* >/dev/null 2>&1; then
  rm -f _build_* 2>/dev/null
  if ls _build_* >/dev/null 2>&1; then
    echo "[!!] 잔류 파일을 지우지 못했다 --- 다른 빌드가 돌고 있는가?"
    ls _build_*
  fi
fi

# 판 이름 → 클래스 옵션
opts_for() {
  case "$1" in
    submission) echo "preprint,12pt" ;;
    journal)    echo "final,5p,times,twocolumn" ;;
    review)     echo "review,12pt" ;;
    authoryear) echo "preprint,authoryear,12pt" ;;
    *) return 1 ;;
  esac
}

build_one() {
  local name="$1" opt
  opt="$(opts_for "$name")" || { echo "[!!] 모르는 판: $name"; return 1; }
  local out="$BUILD/${SRC%.tex}_$name.tex"

  # 클래스 옵션만 바꾼 사본을 만든다. 원본은 손대지 않는다.
  #  ★ 사본을 build/ 에 두면 \input 상대경로가 깨지므로 논문초안/ 에 임시로 둔다.
  local tmp="_build_$name.tex"
  sed "s|^\\\\documentclass\\[[^]]*\\]{elsarticle}|\\\\documentclass[$opt]{elsarticle}|" "$SRC" > "$tmp"

  # review 판은 심사용이라 행번호를 켠다(Elsevier 템플릿이 안내하는 방식).
  if [ "$name" = "review" ]; then
    sed -i 's|^% \\usepackage{lineno}|\\usepackage{lineno}|; s|^% \\linenumbers|\\linenumbers|' "$tmp"
  fi

  # 4회 — 3회로는 review 판의 float 배치가 수렴하지 않아 쪽수가 66/67 로
  # 흔들렸다. 투고 직전 쪽수를 확정하려면 안정된 값이 필요하다.
  for _ in 1 2 3 4; do
    xelatex -interaction=nonstopmode -halt-on-error "$tmp" > /dev/null 2>&1
  done

  local log="${tmp%.tex}.log"
  local err ovf und
  # ★ grep -c 는 0건일 때 exit 1 이다. `|| echo "?"` 를 붙이면 "0" 과 "?" 가 함께 찍힌다.
  err=$(grep -c '^!' "$log" 2>/dev/null; true)
  ovf=$(grep -c 'Overfull' "$log" 2>/dev/null; true)
  und=$(grep -ci 'undefined' "$log" 2>/dev/null; true)
  if [ -f "${tmp%.tex}.pdf" ]; then
    mv "${tmp%.tex}.pdf" "$BUILD/${SRC%.tex}_$name.pdf"
    local pages
    pages=$(python -c "import fitz,sys;print(fitz.open(sys.argv[1]).page_count)" \
            "$BUILD/${SRC%.tex}_$name.pdf" 2>/dev/null || echo "?")
    echo "[ok] $name  ($opt)  ${pages}쪽 · 에러 $err · Overfull $ovf · undefined $und"
  else
    echo "[!!] $name  ($opt)  PDF 생성 실패 — $log 확인"
  fi
  # KEEP=1 이면 중간 산출물을 남긴다 — 경고를 추적할 때 쓴다.
  if [ "${KEEP:-0}" = "1" ]; then
    echo "     로그: 논문초안/${tmp%.tex}.log"
  else
    rm -f "$tmp" "${tmp%.tex}."{aux,log,out,toc,spl}
  fi
}

if [ $# -gt 0 ]; then
  build_one "$1"
else
  for n in submission review journal authoryear; do build_one "$n"; done
fi
