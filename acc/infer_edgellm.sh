#!/bin/bash
# TensorRT Edge-LLM benchmark（venv + C++ runtime）
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

PROMPT="${PROMPT:-}"
PROMPT_KEY="${PROMPT_KEY:-short}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP="${WARMUP:-10}"
RUNS="${RUNS:-30}"
OUTPUT_JSON="${OUTPUT_JSON:-$RESULTS_DIR/edgellm.json}"

require_edgellm_runtime

if [ -z "$PROMPT" ]; then
  PROMPT="$(load_default_prompt)"
fi

if [ ! -d "$ENGINE_DIR" ] || [ -z "$(ls -A "$ENGINE_DIR" 2>/dev/null || true)" ]; then
  echo "[ERROR] 引擎不存在，请先运行:" >&2
  echo "  bash acc/export_onnx.sh" >&2
  echo "  bash acc/build_engine.sh" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
setup_edgellm_env

python3 "$ACC_DIR/benchmark_edgellm.py" \
  --engine-dir "$ENGINE_DIR" \
  --tokenizer-dir "$ENGINE_DIR" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --warmup "$WARMUP" \
  --runs "$RUNS" \
  --output-json "$OUTPUT_JSON" \
  --llm-inference "$LLM_INFERENCE" \
  --llm-bench "$LLM_BENCH"

echo "TensorRT Edge-LLM benchmark 完成: $OUTPUT_JSON"
