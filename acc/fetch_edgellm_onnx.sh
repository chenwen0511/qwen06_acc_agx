#!/bin/bash
# 获取 Edge-LLM 格式 ONNX（acc/workspace/onnx/llm/），供 build_engine 使用
#
# 优先级:
#   1) 已存在 acc/workspace/onnx/llm/model.onnx → 跳过
#   2) 本地 tarball: acc/artifacts/edgellm_onnx.tar.gz 或 EDGELLM_ONNX_TARBALL
#   3) 远程下载: EDGELLM_ONNX_URL
#   4) 本地/远程目录: ONNX_SRC（本地路径或 user@host:/path）
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ACC_ROOT/acc/common.sh"

TARBALL="${EDGELLM_ONNX_TARBALL:-$ACC_DIR/artifacts/edgellm_onnx.tar.gz}"
ONNX_SRC="${ONNX_SRC:-}"

onnx_ready() {
  onnx_export_ready
}

extract_tarball() {
  local tar_path="$1"
  if [ ! -f "$tar_path" ]; then
    return 1
  fi
  echo ">>> 解压 Edge-LLM ONNX: $tar_path"
  rm -rf "$ONNX_DIR"
  mkdir -p "$WORKSPACE"
  tar -xzf "$tar_path" -C "$WORKSPACE"
  if [ -d "$WORKSPACE/onnx/llm" ] || [ -f "$WORKSPACE/onnx/model.onnx" ]; then
    return 0
  fi
  if [ -d "$WORKSPACE/llm" ]; then
    mkdir -p "$ONNX_DIR"
    mv "$WORKSPACE/llm" "$ONNX_DIR/llm"
    return 0
  fi
  echo "[ERROR] tarball 内未找到 onnx/llm 或 model.onnx" >&2
  return 1
}

fetch_from_url() {
  local url="$1"
  local tmp
  tmp="$(mktemp /tmp/edgellm_onnx.XXXXXX.tar.gz)"
  echo ">>> 下载 Edge-LLM ONNX: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp" "$url"
  else
    echo "[ERROR] 需要 curl 或 wget" >&2
    return 1
  fi
  extract_tarball "$tmp"
  rm -f "$tmp"
}

copy_from_src() {
  local src="$1"
  echo ">>> 拷贝 ONNX: $src -> $ONNX_DIR"
  rm -rf "$ONNX_DIR"
  mkdir -p "$WORKSPACE"
  if [[ "$src" == *:* ]]; then
    scp -r "$src" "$WORKSPACE/onnx/"
  elif [ -d "$src" ]; then
    cp -a "$src" "$WORKSPACE/onnx"
  else
    echo "[ERROR] ONNX_SRC 无效: $src" >&2
    return 1
  fi
}

main() {
  if onnx_ready; then
    echo ">>> Edge-LLM ONNX 已就绪，跳过下载"
    ls -la "$ONNX_DIR/llm" 2>/dev/null || ls -la "$ONNX_DIR"
    return 0
  fi

  if [ -f "$TARBALL" ]; then
    extract_tarball "$TARBALL"
  elif [ -n "${EDGELLM_ONNX_URL:-}" ]; then
    fetch_from_url "$EDGELLM_ONNX_URL"
  elif [ -n "$ONNX_SRC" ]; then
    copy_from_src "$ONNX_SRC"
  else
    cat >&2 <<EOF
[ERROR] 未找到 Edge-LLM 格式 ONNX。

请任选其一:
  1) 放置 tarball: acc/artifacts/edgellm_onnx.tar.gz
     （首块机器: bash acc/pack_edgellm_onnx.sh）
  2) 设置 URL:  EDGELLM_ONNX_URL=https://.../edgellm_onnx.tar.gz bash acc/fetch_edgellm_onnx.sh
  3) 拷贝目录:  ONNX_SRC=admin@<ip>:~/.../acc/workspace/onnx bash acc/fetch_edgellm_onnx.sh
  4) x86 export: bash acc/setup_export_host.sh && bash acc/export_onnx_host.sh
EOF
    return 1
  fi

  if ! onnx_ready; then
    echo "[ERROR] 获取后仍缺少 Edge-LLM ONNX（需 llm/model.onnx 或 model.onnx）" >&2
    return 1
  fi

  echo ">>> Edge-LLM ONNX 就绪: $ONNX_DIR"
  ls -la "$ONNX_DIR/llm" 2>/dev/null || ls -la "$ONNX_DIR"
}

main "$@"
