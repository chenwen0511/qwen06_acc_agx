#!/bin/bash
# TensorRT Edge-LLM 公共变量（Orin venv + C++ runtime）
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACC_DIR="$ACC_ROOT/acc"

MODEL_DIR="${QWEN_MODEL_DIR:-/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct}"
EDGELLM_SRC="${EDGELLM_SRC:-$ACC_ROOT/third_party/TensorRT-Edge-LLM}"
WORKSPACE="${WORKSPACE:-$ACC_DIR/workspace}"
ONNX_DIR="${ONNX_DIR:-$WORKSPACE/onnx}"
ENGINE_DIR="${ENGINE_DIR:-$WORKSPACE/engine}"
RESULTS_DIR="${RESULTS_DIR:-$ACC_ROOT/results}"

LLM_BUILD="${LLM_BUILD:-$EDGELLM_SRC/build/examples/llm/llm_build}"
LLM_INFERENCE="${LLM_INFERENCE:-$EDGELLM_SRC/build/examples/llm/llm_inference}"
LLM_BENCH="${LLM_BENCH:-$EDGELLM_SRC/build/examples/llm/llm_bench}"
EDGELLM_PLUGIN="${EDGELLM_PLUGIN:-$EDGELLM_SRC/build/libNvInfer_edgellm_plugin.so}"

activate_venv() {
  # shellcheck disable=SC1091
  source "$ACC_ROOT/venv/bin/activate"
  export LD_LIBRARY_PATH="$ACC_ROOT/lib:${LD_LIBRARY_PATH:-}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
}

setup_export_env() {
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export PYTHONPATH="$EDGELLM_SRC:${PYTHONPATH:-}"

  if [ "${USE_CURRENT_ENV:-0}" = "1" ]; then
    return 0
  fi
  if [ -f "$ACC_ROOT/venv-export/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ACC_ROOT/venv-export/bin/activate"
    return 0
  fi
  if [ -f "$ACC_ROOT/venv/bin/activate" ]; then
    activate_venv
  fi
}

setup_edgellm_env() {
  setup_export_env
  if [ -f "$EDGELLM_PLUGIN" ]; then
    export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
  fi
}

require_edgellm_runtime() {
  setup_edgellm_env
  for bin in "$LLM_BUILD" "$LLM_INFERENCE"; do
    if [ ! -x "$bin" ]; then
      echo "[ERROR] 未找到 $bin，请先运行: bash setup_edgellm.sh" >&2
      return 1
    fi
  done
}

onnx_export_ready() {
  if [ -d "$ONNX_DIR/llm" ] && [ -f "$ONNX_DIR/llm/model.onnx" ]; then
    return 0
  fi
  if [ -f "$ONNX_DIR/model.onnx" ]; then
    return 0
  fi
  return 1
}

require_export_tools() {
  setup_export_env
  if ! command -v tensorrt-edgellm-export >/dev/null 2>&1; then
    echo "[ERROR] 未找到 tensorrt-edgellm-export。" >&2
    if [ -f "$ACC_ROOT/venv-export/bin/activate" ] || [ "${USE_CURRENT_ENV:-0}" = "1" ]; then
      echo "  请先运行: bash acc/setup_export_host.sh" >&2
    elif [ -f "$ACC_ROOT/venv/bin/activate" ]; then
      echo "  Orin: bash setup_edgellm.sh" >&2
      echo "  x86:  bash acc/setup_export_host.sh" >&2
    else
      echo "  x86:  bash acc/setup_export_host.sh" >&2
      echo "  Orin: bash setup_edgellm.sh" >&2
    fi
    return 1
  fi
}

require_torch_onnx_export() {
  setup_export_env
  python3 - <<'PY'
import inspect
import sys

import torch

sig = inspect.signature(torch.onnx.export)
if "dynamic_shapes" not in sig.parameters:
    print(
        f"[ERROR] 当前 PyTorch {torch.__version__} 不支持 Edge-LLM 的 dynamo ONNX 导出 "
        f"（缺少 dynamic_shapes 参数，需要 torch>=2.12）。\n"
        f"Jetson Orin 的 JetPack wheel 最高仅 PyTorch 2.5，无法在 Orin 上导出。\n"
        f"请在 x86 GPU 主机执行: bash acc/export_onnx_host.sh\n"
        f"导出完成后将 acc/workspace/onnx/ 拷回 Orin，再运行 bash acc/build_engine.sh",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

load_default_prompt() {
  python3 - "$ACC_ROOT/prompts.json" "${PROMPT_KEY:-short}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["prompts"][sys.argv[2]])
PY
}
