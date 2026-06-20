#!/bin/bash
# 安装 TensorRT Edge-LLM：Python 导出工具 + Orin C++ runtime（无 Docker）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EDGELLM_SRC="${EDGELLM_SRC:-$ROOT/third_party/TensorRT-Edge-LLM}"
EDGELLM_TAG="${EDGELLM_TAG:-main}"
# 国内网络可设：export GITHUB_MIRROR=https://ghproxy.net/https://github.com
GITHUB_MIRROR="${GITHUB_MIRROR:-}"

github_url() {
  local path="$1"
  if [ -n "$GITHUB_MIRROR" ]; then
    echo "${GITHUB_MIRROR}/${path}"
  else
    echo "https://github.com/${path}"
  fi
}

submodule_is_ready() {
  local rel_path="$1"
  local dir="$EDGELLM_SRC/$rel_path"
  [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]
}

all_submodules_ready() {
  local subpath
  while IFS= read -r subpath; do
    [ -z "$subpath" ] && continue
    if ! submodule_is_ready "$subpath"; then
      return 1
    fi
  done < <(git -C "$EDGELLM_SRC" config --file .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}')
  return 0
}

update_submodules_if_needed() {
  if [ "${SKIP_SUBMODULE_UPDATE:-0}" = "1" ]; then
    echo ">>> 跳过 submodule（SKIP_SUBMODULE_UPDATE=1）"
    return 0
  fi
  if all_submodules_ready; then
    echo ">>> submodule 已存在且非空，跳过下载"
    return 0
  fi

  if [ -n "$GITHUB_MIRROR" ]; then
    echo ">>> submodule 走镜像: $GITHUB_MIRROR"
    git -C "$EDGELLM_SRC" config url."${GITHUB_MIRROR}/".insteadOf "https://github.com/"
  fi
  git -C "$EDGELLM_SRC" submodule sync --recursive

  local subpath
  while IFS= read -r subpath; do
    [ -z "$subpath" ] && continue
    if submodule_is_ready "$subpath"; then
      echo ">>> submodule 已就绪，跳过: $subpath"
      continue
    fi
    echo ">>> 拉取 submodule: $subpath"
    git -C "$EDGELLM_SRC" submodule update --init --depth 1 "$subpath"
  done < <(git -C "$EDGELLM_SRC" config --file .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}')
}

clone_or_update_edgellm() {
  local repo_url
  repo_url="$(github_url "NVIDIA/TensorRT-Edge-LLM.git")"
  mkdir -p "$(dirname "$EDGELLM_SRC")"
  if [ ! -d "$EDGELLM_SRC/.git" ]; then
    git clone --depth 1 --branch "$EDGELLM_TAG" "$repo_url" "$EDGELLM_SRC"
  elif [ "${FORCE_UPDATE:-0}" = "1" ]; then
    echo ">>> 更新主仓库（FORCE_UPDATE=1）"
    git -C "$EDGELLM_SRC" fetch --depth 1 origin "$EDGELLM_TAG" || true
    git -C "$EDGELLM_SRC" checkout "$EDGELLM_TAG"
  else
    echo ">>> 主仓库已存在，跳过 clone/fetch"
  fi

  update_submodules_if_needed
}

if [ ! -d "venv" ]; then
  echo "[ERROR] 请先运行 bash setup_env.sh" >&2
  exit 1
fi

echo ">>> 安装 C++ 构建依赖（需要 sudo）"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y cmake build-essential git \
    libnvinfer-headers-dev libnvinfer-dev libnvonnxparsers-dev || true
fi

echo ">>> 克隆 TensorRT-Edge-LLM"
clone_or_update_edgellm

prepare_edgellm_requirements() {
  local req_filtered="/tmp/edgellm-reqs.txt"
  # Jetson venv 使用 PyTorch 3.10 + Jetson wheel；Edge-LLM 上游 requirements 面向 py3.11+
  grep -v -E '^(torch|numpy|transformers)([=<>].*)?$' requirements.txt > "$req_filtered"
  echo ">>> Edge-LLM Python 依赖（已剔除 torch/numpy/transformers，Orin 沿用项目 requirements）"
  cat "$req_filtered"
  pip install -r "$req_filtered"
  pip install 'transformers>=4.45.0,<5.0'
  python3 - <<'PY'
import numpy
print(f"numpy {numpy.__version__} (Jetson venv)")
PY
}

echo ">>> 安装 Python 导出工具到项目 venv"
# shellcheck disable=SC1091
source venv/bin/activate
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$EDGELLM_SRC:${PYTHONPATH:-}"

cd "$EDGELLM_SRC"
pip install --no-deps .
prepare_edgellm_requirements

echo ">>> 编译 C++ runtime（jetson-orin）"
rm -rf build
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin
make -j"$(nproc)"

cd "$ROOT"
export EDGELLM_PLUGIN_PATH="$EDGELLM_SRC/build/libNvInfer_edgellm_plugin.so"

"$EDGELLM_SRC/build/examples/llm/llm_build" --help >/dev/null
tensorrt-edgellm-export --help >/dev/null 2>&1 || tensorrt-edgellm-export-llm --help >/dev/null

cat <<EOF

TensorRT Edge-LLM 安装完成。

  export EDGELLM_PLUGIN_PATH=$EDGELLM_SRC/build/libNvInfer_edgellm_plugin.so
  export PYTHONPATH=$EDGELLM_SRC:\$PYTHONPATH
  source venv/bin/activate

下一步（Orin 上，需先把 x86 导出的 ONNX 拷到 acc/workspace/onnx/）：
  bash acc/export_onnx.sh    # 仅检查 ONNX 是否已就位
  bash acc/build_engine.sh
  bash acc/infer_edgellm.sh

ONNX 导出需在 x86 GPU 主机完成：
  bash acc/export_onnx_host.sh --docker

EOF
