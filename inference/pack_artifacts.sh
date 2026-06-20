#!/bin/bash
# 在「已 build 引擎」的首块 AGX 上执行，打包推理所需产物
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ENGINE_SRC="${ENGINE_SRC:-$PROJECT_ROOT/acc/workspace/engine}"
EDGELLM_SRC="${EDGELLM_SRC:-$PROJECT_ROOT/third_party/TensorRT-Edge-LLM}"
LLM_BIN="${LLM_BIN:-$EDGELLM_SRC/build/examples/llm/llm_inference}"
PLUGIN_SO="${PLUGIN_SO:-$EDGELLM_SRC/build/libNvInfer_edgellm_plugin.so}"

ARTIFACTS_DIR="$INFERENCE_ROOT/artifacts"
STAGING="$ARTIFACTS_DIR/staging"
TARBALL="$ARTIFACTS_DIR/qwen06_edgellm_orin.tar.gz"

for f in "$ENGINE_SRC/llm.engine" "$LLM_BIN" "$PLUGIN_SO"; do
  if [ ! -e "$f" ]; then
    echo "[ERROR] 缺少: $f" >&2
    echo "请先在首块 AGX 完成: bash setup_edgellm.sh && bash acc/build_engine.sh" >&2
    exit 1
  fi
done

rm -rf "$STAGING"
mkdir -p "$STAGING"/{engine,bin,lib}
cp -a "$ENGINE_SRC"/. "$STAGING/engine/"
cp "$LLM_BIN" "$STAGING/bin/llm_inference"
cp "$PLUGIN_SO" "$STAGING/lib/libNvInfer_edgellm_plugin.so"

mkdir -p "$ARTIFACTS_DIR"
tar -czf "$TARBALL" -C "$STAGING" .

echo "已打包: $TARBALL"
echo ""
echo "拷到新 AGX 后："
echo "  scp $TARBALL admin@<new-agx>:~/stephen/01-code/qwen06_acc_agx/inference/artifacts/"
echo "  ssh admin@<new-agx> 'cd ~/stephen/01-code/qwen06_acc_agx && bash inference/install.sh && bash inference/run.sh'"
