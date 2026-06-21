# 第二块 AGX Orin 推理（最小步骤）

在**已跑通的首块 AGX** 上打包引擎与 C++ runtime，拷到**新 AGX** 即可推理，**无需 Python venv、无需重新 export ONNX**（两块板均为 AGX Orin + 同 JetPack 时通常可直接用）。

**一键部署（环境 + ONNX + Engine + Web）**：在工程根目录执行 `bash onekey_deploy.sh`（需 `acc/artifacts/edgellm_onnx.tar.gz` 或 `ONNX_SRC`）。

## 前提

| 项 | 要求 |
|----|------|
| 硬件 | 两块均为 **Jetson AGX Orin**（同算力、同 JetPack 更稳） |
| 首块 AGX | 已完成 `acc/build_engine.sh`，存在 `acc/workspace/engine/llm.engine` |
| 新 AGX | JetPack 6.x、系统 TensorRT 10.3、CUDA 12.6 |

若 JetPack / TensorRT 版本不一致，请在新板上保留 ONNX，重新 `bash acc/build_engine.sh`，勿直接复用引擎。

---

## 步骤 1：首块 AGX — 打包

```bash
cd ~/stephen/01-code/qwen06_acc_agx
bash inference/pack_artifacts.sh
```

产出：`inference/artifacts/qwen06_edgellm_orin.tar.gz`（约 1.4 GB，含 engine + `llm_inference` + plugin）

---

## 步骤 2：拷到新 AGX

```bash
# 在首块或 PC 上
scp inference/artifacts/qwen06_edgellm_orin.tar.gz \
    admin@<新板IP>:~/stephen/01-code/qwen06_acc_agx/inference/artifacts/

# 新板只需 inference/ 目录 + 工程根目录少量文件（见下「最简目录」）
# 或整仓 scp -r qwen06_acc_agx admin@<新板IP>:~/stephen/01-code/
```

**最简目录**（新板若不想 clone 全仓）：

```
qwen06_acc_agx/
├── inference/
│   ├── install.sh
│   ├── run.sh
│   ├── infer.py              # Python 推理入口（等价 run.sh）
│   ├── common.sh
│   └── artifacts/qwen06_edgellm_orin.tar.gz
└── prompts.json          # 可选，run.sh 用 PROMPT_KEY 时需要
```

---

## 步骤 3：新 AGX — 安装并推理

```bash
cd ~/stephen/01-code/qwen06_acc_agx
bash inference/install.sh
bash inference/run.sh
```

自定义 prompt：

```bash
PROMPT='用一句话介绍 Jetson。' bash inference/run.sh
# 或
PROMPT_KEY=short bash inference/run.sh   # 读 ../prompts.json
MAX_NEW_TOKENS=256 bash inference/run.sh
```

输出：`inference/output/output.json`

### 流式 Web 界面

```bash
bash inference/install.sh          # 首次
bash inference/serve.sh            # 默认 http://0.0.0.0:7860/
# 局域网访问：http://<板子IP>:7860/
```

依赖：`pip install -r inference/requirements-web.txt`（`serve.sh` 会自动安装）

| backend | 说明 |
|---------|------|
| `auto`（默认） | 有 pybind 则真流式，否则 subprocess 完成后按 token 回放 |
| `subprocess` | 调用 `llm_inference`，适合第二块 AGX（无需 venv） |
| `pybind` | 真 token 流式，需首块执行 `bash inference/setup_pybind.sh` |
| `hf` | Transformers 真流式，需 `setup_env.sh` + 模型权重 |

真流式（可选，首块 AGX 一次）：

```bash
bash inference/setup_pybind.sh
BACKEND=pybind bash inference/serve.sh
```

### curl 调用（需先 `bash inference/serve.sh`）

默认地址 `http://127.0.0.1:7860`，远程板子请把 host 换成 `<板子IP>`。

**健康检查**

```bash
curl -s http://127.0.0.1:7860/api/health | python3 -m json.tool
```

**非流式：一次返回完整 JSON**

```bash
curl -s http://127.0.0.1:7860/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "用一句话介绍 NVIDIA Jetson AGX Orin。",
    "max_new_tokens": 128,
    "temperature": 0,
    "backend": "subprocess"
  }' | python3 -m json.tool
```

响应示例：

```json
{
  "text": "……模型回复……",
  "finish_reason": "stop",
  "metrics": {
    "wall_ms": 1542.0,
    "ttft_ms": 39.0,
    "tokens_per_sec": 33.7
  },
  "backend": "subprocess"
}
```

**流式：SSE（Server-Sent Events）**

```bash
curl -N http://127.0.0.1:7860/api/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "prompt": "用一句话介绍 Jetson。",
    "max_new_tokens": 128,
    "backend": "auto"
  }'
```

SSE 事件类型（每行 `data: {...}`）：

| type | 含义 |
|------|------|
| `status` | 开始推理 |
| `metrics` | TTFT / tok/s / wall 时间 |
| `token` | 增量文本片段 |
| `done` | 完整回复 |
| `error` | 错误信息 |

只提取最终文本（需 `jq`）：

```bash
curl -N -s http://127.0.0.1:7860/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"你好","max_new_tokens":64,"backend":"subprocess"}' \
  | while IFS= read -r line; do
      case "$line" in
        data:\ *) echo "${line#data: }" ;;
      esac
    done | jq -r 'select(.type=="done") | .text'
```

指定 backend / 更长输出：

```bash
curl -s http://127.0.0.1:7860/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"解释 Edge-LLM","max_new_tokens":256,"backend":"pybind"}'
```

Python 单次推理（等价 `run.sh`）：

```bash
python3 inference/infer.py
python3 inference/infer.py --prompt '用一句话介绍 Jetson。'
python3 inference/infer.py --prompt-key short --max-new-tokens 256
python3 inference/infer.py --dump-profile   # 额外输出 TTFT / decode 吞吐
python3 inference/infer.py --input-file inference/input.example.json
```

---

## 备选：新板自行 build 引擎（无 tarball）

新板 clone 全工程，把首块的 `acc/workspace/onnx/` 拷过来后：

```bash
bash setup_edgellm.sh          # 编译 C++ runtime（约 10–30 min）
bash acc/build_engine.sh
bash inference/run.sh
```

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `pack_artifacts.sh` | 首块 AGX 打包 |
| `install.sh` | 新板解压到 `inference/runtime/` |
| `run.sh` | 单次 `llm_inference`（bash） |
| `infer.py` | 单次 `llm_inference`（Python） |
| `serve.sh` | 流式 Chat Web 服务 |
| `server.py` | FastAPI + SSE 后端 |
| `engine.py` | 推理/流式引擎封装 |
| `setup_pybind.sh` | 编译 pybind 真流式（可选） |
| `web/index.html` | Chat 界面 |
| `input.example.json` | 手写 input 参考 |
| `runtime/` | 解压后目录（勿提交 git） |
| `artifacts/` | tarball（勿提交 git） |
