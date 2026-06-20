#!/bin/bash
# 单次 Edge-LLM 推理（新 AGX 最小用法）
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_runtime

PROMPT_TEXT="$(load_prompt)"
mkdir -p "$OUTPUT_DIR"

INPUT_JSON="$OUTPUT_DIR/input.json"
OUTPUT_JSON="$OUTPUT_DIR/output.json"
build_input_json "$INPUT_JSON" "$PROMPT_TEXT"

echo "Engine : $ENGINE_DIR"
echo "Prompt : ${PROMPT_TEXT:0:80}$([ "${#PROMPT_TEXT}" -gt 80 ] && echo ...)"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "----------------------------------------"

"$LLM_INFERENCE" \
  --engineDir "$ENGINE_DIR" \
  --inputFile "$INPUT_JSON" \
  --outputFile "$OUTPUT_JSON"

echo ""
print_response "$OUTPUT_JSON"
echo ""
echo "完整 JSON: $OUTPUT_JSON"
