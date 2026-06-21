#!/bin/bash
# 打包 Edge-LLM ONNX 目录，供新板 onekey_deploy / fetch_edgellm_onnx 使用
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ACC_ROOT/acc/common.sh"

OUT="${EDGELLM_ONNX_TARBALL:-$ACC_DIR/artifacts/edgellm_onnx.tar.gz}"

if ! onnx_export_ready; then
  echo "[ERROR] 未找到 $ONNX_DIR，请先 export 或 scp 到位" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$WORKSPACE" onnx

echo "已打包: $OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo ""
echo "新板一键部署:"
echo "  scp $OUT admin@<新板>:~/stephen/01-code/qwen06_acc_agx/acc/artifacts/"
echo "  bash onekey_deploy.sh"
