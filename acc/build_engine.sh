#!/bin/bash
# 在 Orin 上构建 TensorRT Edge-LLM 引擎
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_edgellm_runtime

ONNX_LLM_DIR="$ONNX_DIR"
if [ -d "$ONNX_DIR/llm" ]; then
  ONNX_LLM_DIR="$ONNX_DIR/llm"
fi

if [ ! -d "$ONNX_LLM_DIR" ]; then
  echo "[ERROR] ONNX 不存在，请先导出 ONNX：" >&2
  echo "  x86 GPU 主机: bash acc/export_onnx_host.sh --docker" >&2
  echo "  拷回 Orin 后: bash acc/build_engine.sh" >&2
  exit 1
fi

MAX_INPUT_LEN="${MAX_INPUT_LEN:-2048}"
MAX_KV_CACHE="${MAX_KV_CACHE:-3072}"

echo "ONNX : $ONNX_LLM_DIR"
echo "Engine: $ENGINE_DIR"
echo "maxInputLen=$MAX_INPUT_LEN maxKVCache=$MAX_KV_CACHE"
echo "----------------------------------------"

mkdir -p "$ENGINE_DIR"

"$LLM_BUILD" \
  --onnxDir "$ONNX_LLM_DIR" \
  --engineDir "$ENGINE_DIR" \
  --maxBatchSize 1 \
  --maxInputLen "$MAX_INPUT_LEN" \
  --maxKVCacheCapacity "$MAX_KV_CACHE"

echo "引擎已生成: $ENGINE_DIR"
ls -la "$ENGINE_DIR"
