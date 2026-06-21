# Qwen2.5-0.5B · Jetson AGX Orin 推理加速

在 **NVIDIA Jetson AGX Orin** 上，将 **Qwen2.5-0.5B-Instruct** 从 Hugging Face **Transformers** 基线迁移到 **TensorRT Edge-LLM** FP16 引擎，完成 **导出 → 构建 → 推理 → 性能对比 → 多板部署** 的全流程工程化脚本。

默认模型路径（可通过 `QWEN_MODEL_DIR` 覆盖）：

```
/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct
```

---

## 项目目标

| 目标 | 说明 |
|------|------|
| **加速验证** | 量化 Edge-LLM 相对 Transformers 的 TTFT、吞吐、延迟 |
| **流程可复现** | 一键脚本覆盖 x86 导出 ONNX、Orin build 引擎、双后端 benchmark |
| **边缘部署** | 提供 tarball 打包、第二块 AGX 最小推理、Web/API 在线服务 |

实测（short prompt，AGX Orin）：Edge-LLM 相对 HF 约 **2.4× E2E 吞吐**、TTFT **快 ~2×**，输出 token 数与文本一致。详见 [实测结论](#实测结论)。

---

## 方案原理

### 为什么不用 Orin 直接导出 ONNX？

Edge-LLM 的 ONNX 导出依赖 PyTorch **≥ 2.12**（`torch.export` + `dynamic_shapes`）。Jetson Orin JetPack 官方 wheel 最高 **PyTorch 2.5**，无法在 Orin 上完成 dynamo 导出。

因此采用 **双机流水线**（无 x86 GPU 时，可 **scp 已有 `acc/workspace/onnx/`** 或 `acc/artifacts/edgellm_onnx.tar.gz` 跳过 export，见 [步骤 2](#步骤-2准备-onnx)）：

```
┌─────────────────────────────────────────────────────────────────┐
│  x86 GPU 主机（或 Jetson Thor）                                    │
│  tensorrt-edgellm-export → FP16 ONNX → acc/workspace/onnx/       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ scp
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Jetson AGX Orin                                                 │
│  llm_build → TensorRT FP16 引擎 → acc/workspace/engine/          │
│  llm_inference / llm_bench → C++ 推理                            │
└─────────────────────────────────────────────────────────────────┘
```

### Transformers vs Edge-LLM

| 维度 | Transformers（基线） | TensorRT Edge-LLM（加速） |
|------|---------------------|----------------------------|
| 运行时 | PyTorch + Python | TensorRT + C++ `llm_inference` |
| 精度 | FP16（eager） | FP16 引擎（本工程默认） |
| 显存 | ~974 MB（含 PyTorch） | 引擎 ~1 GB 级，无 PyTorch 开销 |
| 适用 | 开发调试、输出对齐 | 边缘部署、低延迟高吞吐 |

Qwen2.5 属于 Edge-LLM 官方支持的 Qwen2/Qwen2.5 系列，见 [Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)。

### 关键约束

1. **引擎必须在目标 Orin 上 build**，不可跨设备拷贝 `.engine`（同 JetPack 时可 tarball 打包二进制 + 引擎一起分发，见 `inference/`）。
2. **勿 pip install tensorrt**，使用 JetPack 系统 TensorRT 10.3。
3. **torch / torchvision CUDA 索引必须一致**（x86 导出环境），否则 `torchvision::nms` 报错。

---

## 本工程实现的功能

### Phase 1：Transformers 基线

| 脚本 | 功能 |
|------|------|
| `setup_env.sh` | Orin venv：JetPack PyTorch 2.5 + Transformers 4.x |
| `infer_hf.py` | 固定 prompt benchmark：TTFT、decode/E2E tok/s、peak GPU 显存 |
| `prompts.json` | short / medium / long 三条测试 prompt |

### Phase 2：Edge-LLM 加速流水线（`acc/`）

| 脚本 | 功能 |
|------|------|
| `acc/setup_export_host.sh` | x86 导出环境（torch 2.12 + torchvision 0.27，支持 `--conda` 或独立 `venv-export/`） |
| `acc/export_onnx_host.sh` | x86 GPU 本机导出 ONNX（Edge-LLM 格式） |
| `acc/fetch_edgellm_onnx.sh` | 获取 Edge-LLM ONNX（tar / URL / scp） |
| `acc/pack_edgellm_onnx.sh` | 打包 Edge-LLM ONNX 供新板使用 |
| `acc/export_onnx.sh` | Orin 侧检查 ONNX 是否就位 |
| `setup_edgellm.sh` | clone Edge-LLM、安装 Python 工具、Orin 编译 C++ runtime |
| `acc/build_engine.sh` | `llm_build` 构建 FP16 引擎 |
| `acc/infer_edgellm.sh` | Edge-LLM 单次 / benchmark 推理 |
| `acc/benchmark_edgellm.py` | 调用 `llm_inference --dumpProfile` + `llm_bench`，输出 JSON |
| `acc/run_compare.sh` | HF + Edge-LLM 同 prompt 一键对比 |
| `acc/run_prompts.sh` | 遍历 `prompts.json` 全部 key |
| `acc/summarize_results.py` | 单 prompt 汇总 `summary.json` |
| `acc/summarize_prompts.py` | 多 prompt 汇总 `summary_all.json` |

详细步骤与 build 日志见 [`acc/acc.md`](acc/acc.md)。

### Phase 3：推理部署（`inference/`）

| 脚本 | 功能 |
|------|------|
| `inference/pack_artifacts.sh` | 首块 AGX 打包 engine + `llm_inference` + plugin |
| `inference/install.sh` | 新 AGX 解压 tarball |
| `inference/run.sh` / `infer.py` | 命令行单次推理 |
| `inference/serve.sh` | 流式 Chat Web 服务（FastAPI + SSE） |
| `inference/server.py` | HTTP API：`/api/chat`、`/api/chat/stream` |
| `inference/setup_pybind.sh` | 可选：编译 pybind 实现真 token 流式 |

部署与 curl 示例见 [`inference/README.md`](inference/README.md)。

---

## 环境要求

### Jetson AGX Orin（推理 / build 引擎）

| 组件 | 版本 |
|------|------|
| JetPack | R36.4（6.2.x） |
| CUDA | 12.6 |
| Python | 3.10 |
| TensorRT | 系统 **10.3.0** |
| Edge-LLM | `third_party/TensorRT-Edge-LLM`（`setup_edgellm.sh` 编译） |

### x86 GPU 主机（ONNX 导出）

| 组件 | 版本 |
|------|------|
| PyTorch | **2.12.0**（`cu130` 或 `cu126`） |
| torchvision | **0.27.0**（与 torch **同一 CUDA 索引**） |
| transformers | 5.9.0（导出环境） |

国内网络 clone 建议：

```bash
export GITHUB_MIRROR=https://ghproxy.net/https://github.com
```

---

## 完整使用流程

### 步骤 0：克隆工程

```bash
git clone <repo-url> qwen06_acc_agx
cd qwen06_acc_agx
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct   # 可选
```

### 步骤 1：Orin — 安装 Transformers 环境

```bash
bash setup_env.sh
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

验证基线：

```bash
python infer_hf.py --warmup 2 --runs 5 --max-new-tokens 128
```

### 步骤 2：准备 ONNX

Edge-LLM 构建引擎需要目录 `acc/workspace/onnx/llm/`（含 `model.onnx`、`model.onnx.data`、`embedding.safetensors` 等）。

#### 方式 A：跳过 x86 export — 拷贝或 tarball（推荐）

```bash
# scp 目录
scp -r admin@<已有机器IP>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/onnx/ acc/workspace/

# 或 tarball（首块: bash acc/pack_edgellm_onnx.sh）
scp acc/artifacts/edgellm_onnx.tar.gz admin@<新板>:~/.../acc/artifacts/
bash acc/fetch_edgellm_onnx.sh
```

#### 方式 B：x86 本机 export

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
# 或: bash acc/setup_export_host.sh && bash acc/export_onnx_host.sh
scp -r acc/workspace/onnx/ admin@<orin-ip>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/
```

### 步骤 3：Orin — 安装 Edge-LLM + 构建引擎

```bash
bash setup_edgellm.sh          # 首次约 10–30 min
bash acc/export_onnx.sh         # 检查 ONNX（已存在则跳过）
bash acc/build_engine.sh        # 产出 acc/workspace/engine/
```

引擎目录主要文件：`llm.engine`（~1 GB）、`embedding.safetensors`、`tokenizer.json` 等。

### 步骤 4：Orin — 推理与性能对比

单后端：

```bash
bash acc/infer_edgellm.sh
python infer_hf.py --warmup 10 --runs 30
```

双后端一键对比：

```bash
bash acc/run_compare.sh
# 快速冒烟
WARMUP=2 RUNS=5 bash acc/run_compare.sh
# 多 prompt
WARMUP=2 RUNS=5 bash acc/run_prompts.sh
```

结果目录 `results/`：

| 文件 | 内容 |
|------|------|
| `hf.json` / `edgellm.json` | 原始 benchmark 数据 |
| `summary.json` | 单 prompt 对比汇总 |
| `summary_all.json` | 多 prompt 汇总（需跑完 `run_prompts.sh`） |

### 步骤 5：部署到其他 AGX / 在线服务

#### 一键部署（推荐）

在 **Jetson AGX Orin** 上，准备好 Edge-LLM ONNX tarball 后：

```bash
# 首块机器打包 ONNX（一次性）
bash acc/pack_edgellm_onnx.sh
scp acc/artifacts/edgellm_onnx.tar.gz admin@<新板>:~/stephen/01-code/qwen06_acc_agx/acc/artifacts/

# 新板一键：环境 → ONNX → Engine → Web 上线
bash onekey_deploy.sh
# 后台服务: SERVE_BACKGROUND=1 bash onekey_deploy.sh
```

`onekey_deploy.sh` 四步：

| 步骤 | 内容 |
|------|------|
| 1 环境 | `setup_env.sh` + `setup_edgellm.sh` + Web 依赖 |
| 2 ONNX | `acc/fetch_edgellm_onnx.sh`（本地 tar / URL / scp） |
| 3 Engine | `acc/build_engine.sh` |
| 4 上线 | 同步 `inference/runtime/` + 试跑 + `inference/serve.sh` |

ONNX 未打包时也可：

```bash
ONNX_SRC=admin@<首块IP>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/onnx bash onekey_deploy.sh
# 或
EDGELLM_ONNX_URL=https://.../edgellm_onnx.tar.gz bash onekey_deploy.sh
```

#### 手动打包部署

```bash
bash inference/pack_artifacts.sh
scp inference/artifacts/qwen06_edgellm_orin.tar.gz admin@<新板IP>:~/.../inference/artifacts/
# 新板
bash inference/install.sh && bash inference/run.sh
```

**Web 界面 + HTTP API**

```bash
bash inference/serve.sh
# 浏览器: http://<板子IP>:7860/
# API 文档见 inference/README.md
```

---

## 实测结论

测试配置：Qwen2.5-0.5B-Instruct FP16 引擎，`prompts.json` → **short**，`max_new_tokens=128`，warmup/runs **10/30**。

| 指标 | Transformers | TensorRT Edge-LLM | Edge-LLM 相对 HF |
|------|--------------|-------------------|------------------|
| TTFT | **75.2 ms** | **39.0 ms** | **1.93× 更快** |
| Decode 吞吐 | **14.4 tokens/s** | **34.6 tokens/s** | **2.40×** |
| E2E 吞吐 | **14.1 tokens/s** | **33.7 tokens/s** | **2.39×** |
| 总延迟 | 3680 ms | 1542 ms | **2.39× 更快** |
| Peak GPU 显存 | **974 MB** | — | — |
| 输出 tokens | 52 | 52 | 一致 |

数据来源：`results/hf.json`、`results/edgellm.json`。

> **计时说明**：Edge-LLM 指标来自 `llm_inference --dumpProfile`（引擎内计时）；HF 为进程内计时。日常测试建议 `WARMUP=2 RUNS=5`，全量 10/30 跑三条 prompt 约 30–60+ 分钟。

---

## 目录结构

```
qwen06_acc_agx/
├── README.md                 # 本文档
├── onekey_deploy.sh          # AGX 一键：环境 → ONNX → Engine → Web
├── setup_env.sh              # Orin Transformers venv
├── setup_edgellm.sh          # Edge-LLM clone + C++ 编译
├── infer_hf.py               # HF benchmark
├── prompts.json              # 测试 prompt
├── requirements.txt
├── third_party/TensorRT-Edge-LLM/
├── acc/                      # 加速流水线
│   ├── setup_export_host.sh
│   ├── export_onnx.sh
│   ├── export_onnx_host.sh
│   ├── fetch_edgellm_onnx.sh
│   ├── pack_edgellm_onnx.sh
│   ├── build_engine.sh
│   ├── infer_edgellm.sh
│   ├── benchmark_edgellm.py
│   ├── run_compare.sh
│   ├── run_prompts.sh
│   ├── summarize_results.py
│   ├── summarize_prompts.py
│   ├── workspace/            # onnx + engine（勿提交）
│   └── acc.md                # Edge-LLM 详细文档
├── inference/                # 多板部署 + Web/API
│   ├── README.md
│   ├── pack_artifacts.sh
│   ├── install.sh
│   ├── run.sh / infer.py
│   ├── serve.sh / server.py
│   └── web/
└── results/                  # benchmark 输出
```

---

## 常见问题

### `TypeError: export() got an unexpected keyword argument 'dynamic_shapes'`

Orin PyTorch 过旧。请在 **x86 GPU** 运行 `acc/export_onnx_host.sh`，再将 ONNX scp 到 Orin。

### `operator torchvision::nms does not exist`

torch 与 torchvision CUDA 索引不匹配：

```bash
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
```

### `llm_build: No such file`

`setup_edgellm.sh` C++ 编译失败，检查 `libnvinfer-dev` 等依赖。

### 引擎跨设备不可用

引擎必须在**运行它的 Orin** 上 build。JetPack 不一致时需保留 ONNX 在新板重新 `acc/build_engine.sh`。

### `Version tag does not match`（TensorRT）

venv 内误装 pip 版 tensorrt，卸载后使用系统 TensorRT 10.3。

更多问题见 [`acc/acc.md`](acc/acc.md)。

---

## 参考

- [TensorRT Edge-LLM Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
- [TensorRT Edge-LLM Jetson 教程](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- [Edge-LLM Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)
