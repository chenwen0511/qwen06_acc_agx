#!/bin/bash
# 编译 Edge-LLM pybind，启用真 token 流式（可选，首块 AGX 执行一次）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EDGELLM="$ROOT/third_party/TensorRT-Edge-LLM"

if [ ! -d "$EDGELLM/.git" ]; then
  echo "[ERROR] 请先 bash setup_edgellm.sh" >&2
  exit 1
fi

if [ ! -d "$ROOT/venv" ]; then
  echo "[ERROR] 请先 bash setup_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$EDGELLM:${PYTHONPATH:-}"
export EDGELLM_PLUGIN_PATH="$EDGELLM/build/libNvInfer_edgellm_plugin.so"

pip install pybind11

cd "$EDGELLM"
AARCH64_BUILD=1 TRT_PACKAGE_DIR=/usr python3 experimental/server/setup_pybind.py build_ext --inplace

python3 - <<'PY'
from experimental.server import LLM
print("pybind OK:", LLM)
PY

echo ""
echo "真流式已就绪。启动："
echo "  BACKEND=pybind bash inference/serve.sh"
