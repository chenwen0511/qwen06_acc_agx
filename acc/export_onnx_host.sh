#!/bin/bash
# 在 x86 Linux + NVIDIA GPU 上导出 Edge-LLM ONNX（Orin 无法导出）
#
# 用法（x86 主机）:
#   export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
#   bash acc/export_onnx_host.sh              # 本机 venv（需 torch>=2.12）
#   bash acc/export_onnx_host.sh --docker     # 推荐：NVIDIA PyTorch 容器
#
# Docker GPU 模式（可选）:
#   export DOCKER_GPU_MODE=runtime   # 绕过 --gpus CDI 问题
#   export DOCKER_GPU_MODE=cdi       # --device=nvidia.com/gpu=all
#
# 导出完成后拷到 Orin:
#   scp -r acc/workspace/onnx/ admin@<orin-ip>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ACC_ROOT/acc/common.sh"

DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
EDGELLM_TAG="${EDGELLM_TAG:-main}"
DOCKER_GPU_MODE="${DOCKER_GPU_MODE:-auto}"

CONTAINER_CMD='
set -euo pipefail
install_edgellm_in_container() {
  local edgellm_src="/workspace/third_party/TensorRT-Edge-LLM"
  if [ ! -d "$edgellm_src/.git" ]; then
    git clone --depth 1 --branch "${EDGELLM_TAG:-main}" \
      https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$edgellm_src"
    git -C "$edgellm_src" submodule update --init --recursive
  fi
  cd "$edgellm_src"
  python3 -m venv --system-site-packages /tmp/edgellm-venv
  source /tmp/edgellm-venv/bin/activate
  pip install --upgrade pip
  pip install --no-deps .
  grep -v -E "^(torch|numpy)([=<>].*)?$" requirements.txt > /tmp/edgellm-reqs.txt
  pip install -r /tmp/edgellm-reqs.txt
  export PYTHONPATH="$edgellm_src:${PYTHONPATH:-}"
}
install_edgellm_in_container
source /tmp/edgellm-venv/bin/activate
export PYTHONPATH="/workspace/third_party/TensorRT-Edge-LLM:${PYTHONPATH:-}"
bash /workspace/acc/export_onnx_host.sh --in-container
'

print_docker_gpu_fix() {
  cat <<'EOF'
[ERROR] Docker 无法挂载 GPU（CDI / nvidia-container-toolkit 未配置）。

修复方式（任选其一）:

1) 使用 legacy runtime（通常最快生效）:
   export DOCKER_GPU_MODE=runtime
   bash acc/export_onnx_host.sh --docker

2) 生成 CDI spec 并重启 Docker:
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   sudo nvidia-ctk runtime configure --runtime=docker --cdi.enabled
   sudo systemctl restart docker
   nvidia-ctk cdi list | grep nvidia.com/gpu
   bash acc/export_onnx_host.sh --docker

3) 不用 Docker，在本机 venv/conda 安装 torch>=2.12 后直接导出:
   pip install torch==2.12.0  # 或容器同版本
   bash acc/export_onnx_host.sh

验证 Docker GPU:
   docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
     nvcr.io/nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
EOF
}

docker_gpu_preflight() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] 未找到 nvidia-smi，请先安装 NVIDIA 驱动" >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi 无法访问 GPU" >&2
    exit 1
  fi
}

docker_run_with_gpu() {
  local mode="$1"
  shift
  local -a gpu_args=("$@")
  local -a common_args=(
    --rm
    -v "$ACC_ROOT:/workspace"
    -v "$model_parent:/models:ro"
    -w /workspace
    -e "QWEN_MODEL_DIR=/models/$model_name"
    -e "EDGELLM_TAG=$EDGELLM_TAG"
  )

  echo ">>> 尝试 Docker GPU 模式: $mode"
  docker run "${gpu_args[@]}" "${common_args[@]}" "$DOCKER_IMAGE" bash -lc "$CONTAINER_CMD"
}

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

run_docker_export() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 未找到 docker，请安装或使用本机 venv + torch>=2.12" >&2
    exit 1
  fi

  docker_gpu_preflight

  local model_parent model_name
  model_parent="$(cd "$(dirname "$MODEL_DIR")" && pwd)"
  model_name="$(basename "$MODEL_DIR")"

  echo ">>> Docker 镜像: $DOCKER_IMAGE"
  echo ">>> 挂载项目  : $ACC_ROOT -> /workspace"
  echo ">>> 挂载模型  : $model_parent -> /models (只读)"
  echo ">>> 容器内模型: /models/$model_name"
  echo ">>> GPU 模式  : $DOCKER_GPU_MODE"
  echo ""

  local -a modes=()
  case "$DOCKER_GPU_MODE" in
    auto)
      modes=(gpus runtime cdi)
      ;;
    gpus|runtime|cdi|none)
      modes=("$DOCKER_GPU_MODE")
      ;;
    *)
      echo "[ERROR] 未知 DOCKER_GPU_MODE=$DOCKER_GPU_MODE（可选: auto/gpus/runtime/cdi/none）" >&2
      exit 1
      ;;
  esac

  local mode err_log="/tmp/qwen06_docker_gpu.err"
  for mode in "${modes[@]}"; do
    case "$mode" in
      gpus)
        if docker_run_with_gpu gpus --gpus all -it 2>"$err_log"; then
          return 0
        fi
        ;;
      runtime)
        if docker run --rm --runtime=nvidia \
          -e NVIDIA_VISIBLE_DEVICES=all \
          -v "$ACC_ROOT:/workspace" \
          -v "$model_parent:/models:ro" \
          -w /workspace \
          -e "QWEN_MODEL_DIR=/models/$model_name" \
          -e "EDGELLM_TAG=$EDGELLM_TAG" \
          -it "$DOCKER_IMAGE" bash -lc "$CONTAINER_CMD" 2>"$err_log"; then
          return 0
        fi
        ;;
      cdi)
        if docker run --rm \
          --device=nvidia.com/gpu=all \
          -v "$ACC_ROOT:/workspace" \
          -v "$model_parent:/models:ro" \
          -w /workspace \
          -e "QWEN_MODEL_DIR=/models/$model_name" \
          -e "EDGELLM_TAG=$EDGELLM_TAG" \
          -it "$DOCKER_IMAGE" bash -lc "$CONTAINER_CMD" 2>"$err_log"; then
          return 0
        fi
        ;;
      none)
        echo "[WARN] DOCKER_GPU_MODE=none，容器内无 GPU（导出可能极慢或失败）"
        if docker run --rm -it \
          -v "$ACC_ROOT:/workspace" \
          -v "$model_parent:/models:ro" \
          -w /workspace \
          -e "QWEN_MODEL_DIR=/models/$model_name" \
          -e "EDGELLM_TAG=$EDGELLM_TAG" \
          "$DOCKER_IMAGE" bash -lc "$CONTAINER_CMD"; then
          return 0
        fi
        ;;
    esac
    echo "[WARN] 模式 $mode 失败: $(head -1 "$err_log" 2>/dev/null || echo unknown)"
  done

  print_docker_gpu_fix
  exit 1
}

case "${1:-}" in
  --docker)
    run_docker_export
    ;;
  --in-container)
    run_export
    ;;
  --help|-h)
    sed -n '2,18p' "$0"
    ;;
  *)
    run_export
    ;;
esac
