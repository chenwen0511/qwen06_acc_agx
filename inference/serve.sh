#!/bin/bash
# 启动流式 Chat Web 服务
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d "$ROOT/venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
  echo ">>> 安装 Web 依赖"
  pip install -r inference/requirements-web.txt
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
BACKEND="${BACKEND:-auto}"

exec python3 inference/server.py --host "$HOST" --port "$PORT" --backend "$BACKEND"
