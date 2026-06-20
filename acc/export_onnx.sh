#!/bin/bash
# Orin 侧 ONNX 检查：Edge-LLM 导出需在 x86 GPU 主机完成（见 export_onnx_host.sh）
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if onnx_export_ready; then
  echo "ONNX 已存在，跳过导出:"
  if [ -d "$ONNX_DIR/llm" ]; then
    ls -la "$ONNX_DIR/llm"
  else
    ls -la "$ONNX_DIR"
  fi
  exit 0
fi

require_export_tools
require_torch_onnx_export

echo "模型目录 : $MODEL_DIR"
echo "ONNX 输出: $ONNX_DIR"
echo "----------------------------------------"

mkdir -p "$WORKSPACE"
rm -rf "$ONNX_DIR"

cd "$EDGELLM_SRC"
tensorrt-edgellm-export "$MODEL_DIR" "$ONNX_DIR"

if [ -d "$ONNX_DIR/llm" ]; then
  echo "ONNX 已导出: $ONNX_DIR/llm"
else
  echo "ONNX 已导出: $ONNX_DIR"
fi
ls -la "$ONNX_DIR"
