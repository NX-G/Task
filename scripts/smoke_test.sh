#!/usr/bin/env bash
# End-to-end smoke test: upload a drawing, wait for the job, save the outputs.
set -euo pipefail
FILE="${1:-src/sample_data/3D Drawing Example.pdf}"
API="${API_URL:-http://localhost:8000}"
OUT="${OUT_DIR:-data/smoke}"
AUTH=()
[[ -n "${API_KEY:-}" ]] && AUTH=(-H "X-API-Key: ${API_KEY}")
mkdir -p "$OUT"

JOB=$(curl -fsS ${AUTH[@]+"${AUTH[@]}"} -F "file=@${FILE}" -F "image_id=IMG_001" "$API/v1/jobs" | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
echo "job: $JOB"
for _ in $(seq 1 180); do
  STATUS=$(curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$API/v1/jobs/$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]] && break
  sleep 2
done
echo "status: $STATUS"
curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$API/v1/jobs/$JOB" | python3 -m json.tool
[[ "$STATUS" == "succeeded" ]] || exit 1
curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$API/v1/jobs/$JOB/result" -o "$OUT/result.json"
curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$API/v1/jobs/$JOB/visualization?page=0" -o "$OUT/visualization.png"
python3 - "$OUT/result.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for p in r["pages"]:
    print("page", p["page"], "counts", p["counts"], "merge", p["merge_report"], "drift", p["drift"]["status"], p["drift"].get("flags"))
print("models", r["model_versions"], "timings", r["timings"])
EOF
echo "outputs in $OUT/"
