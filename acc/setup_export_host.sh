#!/bin/bash
# x86 GPU 主机：安装 Edge-LLM 导出环境（不依赖 Docker）
#
# 用法:
#   bash acc/setup_export_host.sh           # 创建 venv-export/（推荐，隔离依赖）
#   bash acc/setup_export_host.sh --conda   # 安装到当前 conda/venv
#   bash acc/setup_export_host.sh --docker-toolkit  # 仅配置 Docker GPU
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EDGELLM_SRC="${EDGELLM_SRC:-$ROOT/third_party/TensorRT-Edge-LLM}"
EDGELLM_TAG="${EDGELLM_TAG:-main}"
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"
# PyTorch 2.12 不在 cu124 索引；默认依次尝试 cu130 / cu126
TORCH_CUDA="${TORCH_CUDA:-}"

install_torch() {
  local -a indices=()
  if [ -n "$TORCH_CUDA" ]; then
    indices=("$TORCH_CUDA")
  else
    indices=(cu130 cu126)
  fi

  local idx
  for idx in "${indices[@]}"; do
    echo ">>> 尝试 torch==${TORCH_VERSION} (${idx}, py$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'))"
    if pip install "torch==${TORCH_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${idx}"; then
      echo ">>> 已安装 torch ${TORCH_VERSION} from ${idx}"
      return 0
    fi
    echo "[WARN] ${idx} 无匹配 wheel，尝试下一个索引..."
  done

  cat >&2 <<EOF
[ERROR] 无法安装 torch==${TORCH_VERSION}

PyTorch 2.12 不在 cu124 索引（最高仅 2.6）。请指定:
  TORCH_CUDA=cu126 bash acc/setup_export_host.sh --conda
  TORCH_CUDA=cu130 bash acc/setup_export_host.sh --conda

当前 Python: $(python3 --version)
EOF
  exit 1
}

install_docker_toolkit() {
  echo ">>> 安装 nvidia-container-toolkit"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] 未找到 nvidia-smi" >&2
    exit 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  else
    echo "[ERROR] 请手动安装 nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
    exit 1
  fi

  sudo nvidia-ctk runtime configure --runtime=docker
  if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || true
  fi
  sudo systemctl restart docker

  echo ">>> 验证 Docker GPU"
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvcr.io/nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L
}

ensure_edgellm_src() {
  if [ ! -d "$EDGELLM_SRC/.git" ]; then
    echo ">>> clone TensorRT-Edge-LLM ($EDGELLM_TAG)"
    mkdir -p "$(dirname "$EDGELLM_SRC")"
    git clone --depth 1 --branch "$EDGELLM_TAG" \
      https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$EDGELLM_SRC"
    git -C "$EDGELLM_SRC" submodule update --init --recursive
  fi
}

install_edgellm_python() {
  ensure_edgellm_src
  local req_filtered="/tmp/edgellm-export-reqs.txt"
  grep -v -E '^torch([=<>].*)?$' "$EDGELLM_SRC/requirements.txt" > "$req_filtered"

  pip install --upgrade pip
  install_torch

  echo ">>> 安装 Edge-LLM Python 包"
  cd "$EDGELLM_SRC"
  pip install --no-deps .
  pip install -r "$req_filtered"

  export PYTHONPATH="$EDGELLM_SRC:${PYTHONPATH:-}"
  python3 - <<'PY'
import inspect
import torch

sig = inspect.signature(torch.onnx.export)
assert "dynamic_shapes" in sig.parameters, "torch.onnx.export 仍缺少 dynamic_shapes"
print(f"PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
  tensorrt-edgellm-export --help >/dev/null
  echo ""
  echo "导出环境就绪。运行:"
  echo "  export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct"
  echo "  USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh"
}

setup_venv_export() {
  if [ ! -d "venv-export" ]; then
    python3 -m venv venv-export
  fi
  # shellcheck disable=SC1091
  source venv-export/bin/activate
  install_edgellm_python
}

setup_conda_export() {
  echo ">>> 安装到当前 Python: $(which python3)"
  install_edgellm_python
  echo ""
  echo "使用当前环境导出:"
  echo "  USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh"
}

case "${1:-}" in
  --conda)
    setup_conda_export
    ;;
  --docker-toolkit)
    install_docker_toolkit
    ;;
  --help|-h)
    sed -n '2,8p' "$0"
    ;;
  *)
    setup_venv_export
    ;;
esac
