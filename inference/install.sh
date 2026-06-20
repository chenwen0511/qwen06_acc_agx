#!/bin/bash
# 在新 AGX 上解压推理包（或从首块 AGX 已 unpack 的 runtime/ 目录）
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

TARBALL="${TARBALL:-$INFERENCE_ROOT/artifacts/qwen06_edgellm_orin.tar.gz}"

if [ -f "$TARBALL" ]; then
  echo ">>> 解压 $TARBALL -> $RUNTIME_DIR"
  rm -rf "$RUNTIME_DIR"
  mkdir -p "$RUNTIME_DIR"
  tar -xzf "$TARBALL" -C "$RUNTIME_DIR"
elif [ -f "$RUNTIME_DIR/engine/llm.engine" ]; then
  echo ">>> 已存在 $RUNTIME_DIR/engine/llm.engine，跳过解压"
else
  echo "[ERROR] 未找到 $TARBALL，也未找到 $RUNTIME_DIR/engine/" >&2
  echo "" >&2
  echo "方案 A：从首块 AGX 拷贝 inference/artifacts/qwen06_edgellm_orin.tar.gz" >&2
  echo "方案 B：在新板 clone 工程后 bash setup_edgellm.sh && bash acc/build_engine.sh（需 ONNX）" >&2
  exit 1
fi

chmod +x "$LLM_INFERENCE" 2>/dev/null || true
require_runtime

echo ""
echo "安装完成。试跑："
echo "  bash inference/run.sh"
echo "  PROMPT='你好' bash inference/run.sh"
