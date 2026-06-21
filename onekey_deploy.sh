#!/bin/bash
# Qwen2.5 Edge-LLM · Jetson AGX Orin 一键部署
#
# 流程: 环境准备 → ONNX 获取 → Engine 构建 → Web 服务上线
#
# 用法:
#   bash onekey_deploy.sh
#
# 常用环境变量:
#   SKIP_ENV=1          跳过 setup_env / setup_edgellm
#   SKIP_ONNX=1         跳过 ONNX 获取（需已有 acc/workspace/onnx/）
#   SKIP_ENGINE=1       跳过 llm_build
#   SKIP_SMOKE=1        跳过试跑 inference/run.sh
#   SKIP_SERVE=1        仅构建，不启动 Web
#   SERVE_BACKGROUND=1  后台启动（默认前台）
#   HOST=0.0.0.0 PORT=7860 BACKEND=subprocess
#
#   ONNX 来源（Edge-LLM 格式，三选一）:
#     acc/artifacts/edgellm_onnx.tar.gz   # 推荐，首块 bash acc/pack_edgellm_onnx.sh
#     EDGELLM_ONNX_URL=https://...        # 远程 tarball
#     ONNX_SRC=user@host:.../onnx         # scp 目录
#
# 前提: Jetson AGX Orin, JetPack 6.x, sudo 可用（setup_edgellm 装依赖）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
BACKEND="${BACKEND:-subprocess}"
SERVE_BACKGROUND="${SERVE_BACKGROUND:-0}"

ACC_DIR="$ROOT/acc"
WORKSPACE="${WORKSPACE:-$ACC_DIR/workspace}"
ONNX_DIR="${ONNX_DIR:-$WORKSPACE/onnx}"
ENGINE_DIR="${ENGINE_DIR:-$WORKSPACE/engine}"
EDGELLM_SRC="${EDGELLM_SRC:-$ROOT/third_party/TensorRT-Edge-LLM}"
LLM_BUILD="${LLM_BUILD:-$EDGELLM_SRC/build/examples/llm/llm_build}"
LLM_INFERENCE="${LLM_INFERENCE:-$EDGELLM_SRC/build/examples/llm/llm_inference}"
EDGELLM_PLUGIN="${EDGELLM_PLUGIN:-$EDGELLM_SRC/build/libNvInfer_edgellm_plugin.so}"

log() {
  echo ""
  echo "========================================"
  echo ">>> [$1] $2"
  echo "========================================"
}

need_sudo_hint() {
  echo "若 apt 失败，请手动: sudo apt-get install -y cmake build-essential libnvinfer-dev"
}

# ---------- 1. 环境准备 ----------
phase_env() {
  log "1/4" "环境准备"

  if [ "${SKIP_ENV:-0}" = "1" ]; then
    echo "SKIP_ENV=1，跳过"
    return 0
  fi

  if [ ! -d "$ROOT/venv" ]; then
    echo ">>> setup_env.sh（Transformers venv + lib）"
    bash "$ROOT/setup_env.sh"
  else
    echo ">>> venv 已存在，跳过 setup_env.sh"
  fi

  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
  export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"

  if [ ! -x "$LLM_BUILD" ]; then
    echo ">>> setup_edgellm.sh（clone + C++ runtime，首次约 10–30 min）"
    need_sudo_hint
    bash "$ROOT/setup_edgellm.sh"
  else
    echo ">>> Edge-LLM C++ runtime 已编译，跳过 setup_edgellm.sh"
  fi

  export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
  export PYTHONPATH="$EDGELLM_SRC:${PYTHONPATH:-}"

  echo ">>> Python Web 依赖"
  pip install -q -r "$ROOT/inference/requirements-web.txt"

  echo "环境 OK"
}

# ---------- 2. ONNX 获取 ----------
phase_onnx() {
  log "2/4" "ONNX 获取（Edge-LLM 格式）"

  if [ "${SKIP_ONNX:-0}" = "1" ]; then
    echo "SKIP_ONNX=1，跳过"
    return 0
  fi

  bash "$ROOT/acc/fetch_edgellm_onnx.sh"
}

# ---------- 3. Engine 构建 ----------
phase_engine() {
  log "3/4" "Engine 构建"

  if [ "${SKIP_ENGINE:-0}" = "1" ]; then
    echo "SKIP_ENGINE=1，跳过"
    return 0
  fi

  if [ -f "$ENGINE_DIR/llm.engine" ]; then
    echo ">>> llm.engine 已存在，跳过 build"
    ls -lh "$ENGINE_DIR/llm.engine"
    return 0
  fi

  bash "$ROOT/acc/build_engine.sh"
}

# ---------- 4. 部署上线 ----------
install_inference_runtime() {
  local rt="$ROOT/inference/runtime"
  mkdir -p "$rt/engine" "$rt/bin" "$rt/lib"

  echo ">>> 同步 inference/runtime/"
  rm -rf "$rt/engine"/*
  cp -a "$ENGINE_DIR"/. "$rt/engine/"
  cp "$LLM_INFERENCE" "$rt/bin/llm_inference"
  cp "$EDGELLM_PLUGIN" "$rt/lib/libNvInfer_edgellm_plugin.so"
  chmod +x "$rt/bin/llm_inference"
}

phase_deploy() {
  log "4/4" "部署上线"

  if [ ! -f "$ENGINE_DIR/llm.engine" ]; then
    echo "[ERROR] 缺少 $ENGINE_DIR/llm.engine" >&2
    exit 1
  fi
  if [ ! -x "$LLM_INFERENCE" ]; then
    echo "[ERROR] 缺少 $LLM_INFERENCE，请运行 setup_edgellm.sh" >&2
    exit 1
  fi

  install_inference_runtime

  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo ">>> 试跑 inference/run.sh"
    bash "$ROOT/inference/run.sh"
  fi

  if [ "${SKIP_SERVE:-0}" = "1" ]; then
    echo "SKIP_SERVE=1，未启动 Web 服务"
    echo "手动启动: bash inference/serve.sh"
    return 0
  fi

  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}' || echo "<本机IP>")"

  echo ""
  echo ">>> 启动 Chat Web 服务"
  echo "    本地: http://127.0.0.1:${PORT}/"
  echo "    局域网: http://${ip}:${PORT}/"
  echo "    API: curl http://127.0.0.1:${PORT}/api/health"
  echo ""

  export HOST PORT BACKEND
  if [ "$SERVE_BACKGROUND" = "1" ]; then
    nohup bash "$ROOT/inference/serve.sh" > "$ROOT/inference/output/serve.log" 2>&1 &
    echo ">>> 后台 PID $!，日志: inference/output/serve.log"
    sleep 2
    curl -sf "http://127.0.0.1:${PORT}/api/health" | python3 -m json.tool || true
  else
    exec bash "$ROOT/inference/serve.sh"
  fi
}

# ---------- main ----------
main() {
  echo "Qwen2.5 Edge-LLM 一键部署 @ $(uname -m) $(date '+%F %T')"
  echo "工程: $ROOT"

  phase_env
  phase_onnx
  phase_engine
  phase_deploy

  if [ "$SERVE_BACKGROUND" = "1" ] && [ "${SKIP_SERVE:-0}" != "1" ]; then
    echo ""
    echo "部署完成。"
  fi
}

main "$@"
