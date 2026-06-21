#!/bin/bash
# 在 x86 Linux + NVIDIA GPU 上导出 Edge-LLM ONNX（Orin 无法导出）
#
# 用法（x86 主机）:
#   bash acc/setup_export_host.sh --conda
#   USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
#
# 或独立 venv:
#   bash acc/setup_export_host.sh
#   bash acc/export_onnx_host.sh
#
# 导出完成后拷到 Orin:
#   scp -r acc/workspace/onnx/ admin@<orin-ip>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ACC_ROOT/acc/common.sh"

run_export() {
  require_export_tools
  require_torch_onnx_export

  echo "模型目录 : $MODEL_DIR"
  echo "ONNX 输出: $ONNX_DIR"
  echo "PyTorch  : $(python3 -c 'import torch; print(torch.__version__)')"
  echo "----------------------------------------"

  if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 模型目录不存在: $MODEL_DIR" >&2
    exit 1
  fi

  mkdir -p "$WORKSPACE"
  rm -rf "$ONNX_DIR"

  cd "$EDGELLM_SRC"
  tensorrt-edgellm-export "$MODEL_DIR" "$ONNX_DIR"

  if [ -d "$ONNX_DIR/llm" ]; then
    echo "ONNX 已导出: $ONNX_DIR/llm"
    ls -la "$ONNX_DIR/llm"
  else
    echo "ONNX 已导出: $ONNX_DIR"
    ls -la "$ONNX_DIR"
  fi

  echo ""
  echo "下一步（在 Orin 上）:"
  echo "  bash acc/build_engine.sh"
  echo "  bash acc/infer_edgellm.sh"
}

case "${1:-}" in
  --help|-h)
    sed -n '2,15p' "$0"
    ;;
  *)
    run_export
    ;;
esac
