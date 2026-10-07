#!/usr/bin/env bash
# Render both lyric timelines section-wise in Resolve and make the MP4s (v2.2 sound) + checks.
# Waits until the render queue is empty, nothing renders, the API answers and >= 14 GB RAM are free.
# Usage from the repo root: bash lyrics/render_all.sh   (logs in work/lyrics/)
# start only when the render queue is empty, nothing renders, the API answers and >= 14 GB are free
set -u
cd "$(dirname "$0")/.."
L=work/lyrics
streak=0
for i in $(seq 1 720); do
  ok=$(timeout 60 .venv/bin/python - <<'PY' 2>/dev/null
from lyrics import resolve_io as rio
try:
    r = rio.connect(); p = rio.project(r)
    q = p.GetRenderJobList()
    print("yes" if (q == [] or q == {}) and not p.IsRenderingInProgress() and r.GetCurrentPage() else "no")
except Exception:
    print("no")
PY
)
  avail=$(free -m | awk '/^Speicher|^Mem/ {print $7}')
  # the API sometimes answers "empty queue" once while Resolve is still stuck: require 3 in a row
  if [ "$ok" = "yes" ] && [ "$avail" -ge 14000 ]; then streak=$((${streak:-0} + 1)); else streak=0; fi
  if [ "$streak" -ge 3 ]; then break; fi
  [ $((i % 15)) -eq 0 ] && echo "$(date +%T) waiting: api=$ok avail=${avail}MB"
  sleep 20
done
echo "$(date +%T) queue clear, start 16x9 chunks"
.venv/bin/python -m lyrics.render chunks 16x9 > $L/chunks_16x9.log 2>&1 || { echo "16x9 chunks failed"; tail -5 $L/chunks_16x9.log; exit 1; }
echo "$(date +%T) 16x9 master ready"
( .venv/bin/python -m lyrics.render mp4 16x9 > $L/mp4_16x9.log 2>&1 && .venv/bin/python -m lyrics.render check 16x9 > $L/check_16x9.log 2>&1; echo "$(date +%T) 16x9 mp4+check exit $?" ) &
ENC=$!
.venv/bin/python -m lyrics.build place 9x16 background overlay Intro "Drop 1" "B Refrain 1" "B Strophe 1" "B Refrain 2" "Post-Chorus" Breakdown "Finaler Drop" Outro > $L/place_9x16.log 2>&1 || { echo "9x16 place failed"; wait $ENC; exit 1; }
.venv/bin/python -m lyrics.render chunks 9x16 > $L/chunks_9x16.log 2>&1 || { echo "9x16 chunks failed"; tail -5 $L/chunks_9x16.log; wait $ENC; exit 1; }
echo "$(date +%T) 9x16 master ready"
wait $ENC
.venv/bin/python -m lyrics.render mp4 9x16 > $L/mp4_9x16.log 2>&1 && .venv/bin/python -m lyrics.render check 9x16 > $L/check_9x16.log 2>&1
echo "$(date +%T) chain4 done exit $?"
