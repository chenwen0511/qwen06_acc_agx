#!/bin/bash
# 一键运行 HF + TensorRT Edge-LLM benchmark，并生成 summary.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$ROOT/acc/common.sh"

PROMPT_KEY="${PROMPT_KEY:-short}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP="${WARMUP:-10}"
RUNS="${RUNS:-30}"
SKIP_HF="${SKIP_HF:-0}"
SKIP_EDGELLM="${SKIP_EDGELLM:-0}"
MONITOR_GPU="${MONITOR_GPU:-1}"

mkdir -p "$RESULTS_DIR"

PROMPT="$(python3 - "$ROOT/prompts.json" "$PROMPT_KEY" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["prompts"][sys.argv[2]])
PY
)"

echo "Prompt key      : $PROMPT_KEY"
echo "max_new_tokens  : $MAX_NEW_TOKENS"
echo "warmup/runs     : $WARMUP/$RUNS"
echo "----------------------------------------"

GPU_LOG="$RESULTS_DIR/gpu.log"
if [ "$MONITOR_GPU" = "1" ] && command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 100 > "$GPU_LOG" &
  TEGRA_PID=$!
  trap 'kill $TEGRA_PID 2>/dev/null || true' EXIT
fi

if [ "$SKIP_HF" != "1" ]; then
  echo ">>> Phase 1: Transformers"
  export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
  python infer_hf.py \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --output-json "$RESULTS_DIR/hf.json" \
    2>&1 | tee "$RESULTS_DIR/hf.log"
fi

if [ "$SKIP_EDGELLM" != "1" ]; then
  echo ">>> Phase 2: TensorRT Edge-LLM"
  PROMPT="$PROMPT" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  WARMUP="$WARMUP" \
  RUNS="$RUNS" \
  bash "$ROOT/acc/infer_edgellm.sh" \
    2>&1 | tee "$RESULTS_DIR/edgellm.log"
fi

python3 "$ROOT/acc/summarize_results.py" \
  --hf-json "$RESULTS_DIR/hf.json" \
  --edgellm-json "$RESULTS_DIR/edgellm.json" \
  --output-json "$RESULTS_DIR/summary.json" \
  | tee "$RESULTS_DIR/summary.log"

if [ "${TEGRA_PID:-}" != "" ]; then
  kill "$TEGRA_PID" 2>/dev/null || true
fi

echo "完成。结果目录: $RESULTS_DIR"
