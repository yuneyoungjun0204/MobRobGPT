#!/bin/sh
# 평가 드라이버가 끝날 때까지 대기한다. 폴링 간격 5분.
# 끝나면 커버리지 요약을 찍고 종료 → 호출자가 통지받는다.
PID="$1"
while tasklist //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; do
  sleep 300
done
echo "=== 드라이버 $PID 종료 ==="
tail -20 results/eval_all2.log
echo "=== 에러 ==="
tail -20 results/eval_all2.err
