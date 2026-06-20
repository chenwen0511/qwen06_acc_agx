#!/bin/bash
# 按 prompts.json 中全部 key 依次跑 HF + Edge-LLM 对比
# 默认 WARMUP=2 RUNS=5（全量 10/30 约 30–60+ 分钟，过长）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPTS_FILE="${PROMPTS_FILE:-$ROOT/prompts.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"
SKIP_HF="${SKIP_HF:-0}"
SKIP_EDGELLM="${SKIP_EDGELLM:-0}"

if [ ! -f "$PROMPTS_FILE" ]; then
  echo "[ERROR] 未找到 $PROMPTS_FILE" >&2
  exit 1
fi

mapfile -t PROMPT_KEYS < <(
  python3 - "$PROMPTS_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in data["prompts"]:
    print(key)
PY
)

echo ">>> prompts.json keys: ${PROMPT_KEYS[*]}"
echo ">>> max_new_tokens=$MAX_NEW_TOKENS warmup/runs=$WARMUP/$RUNS"
echo "========================================"

for key in "${PROMPT_KEYS[@]}"; do
  echo ""
  echo ">>> benchmark prompt key: $key"
  PROMPT_KEY="$key" \
  RESULTS_DIR="$ROOT/results/$key" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  WARMUP="$WARMUP" \
  RUNS="$RUNS" \
  SKIP_HF="$SKIP_HF" \
  SKIP_EDGELLM="$SKIP_EDGELLM" \
  MONITOR_GPU=0 \
  bash "$ROOT/acc/run_compare.sh"
done

python3 "$ROOT/acc/summarize_prompts.py" \
  --prompts-file "$PROMPTS_FILE" \
  --results-root "$ROOT/results" \
  --output-json "$ROOT/results/summary_all.json"

echo ""
echo "全部 prompt 测完。汇总: results/summary_all.json"
