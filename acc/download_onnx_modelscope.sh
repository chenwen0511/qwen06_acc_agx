#!/bin/bash
# 从 ModelScope 下载 Qwen2.5-0.5B-Instruct 预转换 ONNX（Optimum / Transformers.js 格式）
#
# 模型页: https://modelscope.cn/models/onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA
#
# ⚠️  格式说明（必读）:
#   - 此为 HuggingFace Optimum 导出的 ONNX（onnx/model_*.onnx），供 ONNX Runtime / Web 使用
#   - 与 Edge-LLM 的 tensorrt-edgellm-export 产物（llm/model.onnx + model.onnx.data + embedding.safetensors）
#     计算图与目录结构不同，不能直接用于 acc/build_engine.sh
#
# 若目标是 Edge-LLM 加速，跳过 x86 export 的推荐做法:
#   scp -r admin@<已有Orin>:~/.../acc/workspace/onnx/ acc/workspace/
#
# 用法:
#   bash acc/download_onnx_modelscope.sh
#   ONNX_VARIANT=model_fp16 bash acc/download_onnx_modelscope.sh   # 默认，约 947MB
#   ONNX_VARIANT=model_q4f16 bash acc/download_onnx_modelscope.sh    # 更小量化版
set -euo pipefail

ACC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ACC_ROOT/acc/common.sh"

MODEL_ID="${MODEL_ID:-onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/onnx-modelscope}"
ONNX_VARIANT="${ONNX_VARIANT:-model_fp16}"

case "$ONNX_VARIANT" in
  model_fp16|model.onnx|model_q4f16|model_int8|model_q4|model_uint8|model_bnb4|model_quantized) ;;
  *)
    echo "[ERROR] 未知 ONNX_VARIANT=$ONNX_VARIANT" >&2
    echo "可选: model_fp16 model.onnx model_q4f16 model_int8 ..." >&2
    exit 1
    ;;
esac

if ! python3 -c "from modelscope.hub.snapshot_download import snapshot_download" 2>/dev/null; then
  echo "[ERROR] 未安装 modelscope，请执行: pip install modelscope" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo ">>> ModelScope 模型: $MODEL_ID"
echo ">>> 下载 ONNX 变体: onnx/${ONNX_VARIANT}.onnx"
echo ">>> 输出目录      : $OUT_DIR"
echo ""

python3 - "$MODEL_ID" "$OUT_DIR" "$ONNX_VARIANT" <<'PY'
import os
import shutil
import sys

from modelscope.hub.snapshot_download import snapshot_download

model_id, out_dir, variant = sys.argv[1:4]
patterns = [
    f"onnx/{variant}.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]

cache = snapshot_download(model_id, allow_patterns=patterns)
onnx_src = os.path.join(cache, "onnx", f"{variant}.onnx")
if not os.path.isfile(onnx_src):
    raise SystemExit(f"[ERROR] 未找到 {onnx_src}")

os.makedirs(out_dir, exist_ok=True)
onnx_dst = os.path.join(out_dir, f"{variant}.onnx")
shutil.copy2(onnx_src, onnx_dst)

for name in patterns:
    if name.startswith("onnx/"):
        continue
    src = os.path.join(cache, name)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(out_dir, name))

print(f"ONNX 已保存: {onnx_dst} ({os.path.getsize(onnx_dst)/1024/1024:.1f} MiB)")
print(f"Tokenizer 等: {out_dir}")
PY

cat <<EOF

下载完成。

【Edge-LLM 加速路径】本 ONNX 不能替代 acc/workspace/onnx/llm/，请任选其一:
  1) 从已 export 的机器拷贝 Edge-LLM ONNX:
       scp -r admin@<orin-ip>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/onnx/ \\
           $ACC_ROOT/acc/workspace/
       bash acc/build_engine.sh

  2) 在 x86 GPU 上自行 export:
       bash acc/setup_export_host.sh --conda
       USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh

【ONNX Runtime 等其它框架】可使用本次下载:
  $OUT_DIR/${ONNX_VARIANT}.onnx

模型页: https://modelscope.cn/models/onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA
EOF
