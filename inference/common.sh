#!/bin/bash
# 第二块 AGX 推理公共路径（独立于 acc/benchmark）
set -euo pipefail

INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$INFERENCE_ROOT/.." && pwd)"

RUNTIME_DIR="${RUNTIME_DIR:-$INFERENCE_ROOT/runtime}"
ENGINE_DIR="${ENGINE_DIR:-$RUNTIME_DIR/engine}"
LLM_INFERENCE="${LLM_INFERENCE:-$RUNTIME_DIR/bin/llm_inference}"
EDGELLM_PLUGIN="${EDGELLM_PLUGIN:-$RUNTIME_DIR/lib/libNvInfer_edgellm_plugin.so}"
PROMPTS_FILE="${PROMPTS_FILE:-$PROJECT_ROOT/prompts.json}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROMPT_KEY="${PROMPT_KEY:-short}"
OUTPUT_DIR="${OUTPUT_DIR:-$INFERENCE_ROOT/output}"

setup_runtime_env() {
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -f "$EDGELLM_PLUGIN" ]; then
    export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
  fi
  if [ -d "$PROJECT_ROOT/lib" ]; then
    export LD_LIBRARY_PATH="$PROJECT_ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi
}

require_runtime() {
  setup_runtime_env
  local missing=0
  for f in "$LLM_INFERENCE" "$EDGELLM_PLUGIN" "$ENGINE_DIR/llm.engine"; do
    if [ ! -e "$f" ]; then
      echo "[ERROR] 缺少: $f" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "请先在新板上运行: bash inference/install.sh" >&2
    return 1
  fi
}

load_prompt() {
  if [ -n "${PROMPT:-}" ]; then
    echo "$PROMPT"
    return 0
  fi
  python3 - "$PROMPTS_FILE" "$PROMPT_KEY" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["prompts"][sys.argv[2]])
PY
}

build_input_json() {
  local out_path="$1"
  local prompt="$2"
  python3 - "$out_path" "$prompt" "$MAX_NEW_TOKENS" <<'PY'
import json, sys
path, prompt, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = {
    "batch_size": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_generate_length": max_new,
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "requests": [{"messages": [{"role": "user", "content": prompt}]}],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
}

print_response() {
  python3 - "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for i, resp in enumerate(data.get("responses", [])):
    text = resp.get("output_text") or resp.get("text") or ""
    reason = resp.get("finish_reason", "")
    print(f"--- response {i} ({reason}) ---")
    print(text)
PY
}
