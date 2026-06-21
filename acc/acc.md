# Qwen2.5-0.5B TensorRT Edge-LLM 推理加速（Orin AGX）

将 Qwen2.5-0.5B-Instruct 从 Hugging Face Transformers 替换为 **TensorRT Edge-LLM** FP16 引擎，在 Orin AGX 上验证 LLM 推理加速效果。

---

## 方案概述

```
HF 权重 (safetensors)
        │
        ▼
┌───────────────────────────────┐
│ tensorrt-edgellm-export       │  x86 GPU 主机 / Jetson Thor（非 Orin）
│ acc/export_onnx_host.sh       │  FP16 ONNX → acc/workspace/onnx/
└───────────────┬───────────────┘
                │ scp 到 Orin
                ▼
┌───────────────────────────────┐
│ llm_build                     │  acc/workspace/engine/（必须在 Orin 上 build）
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ llm_inference + llm_bench     │  与 infer_hf.py 同 prompt / 参数
└───────────────────────────────┘
```

Qwen2.5 属于 Edge-LLM 官方支持的 Qwen2/Qwen2.5 系列，见 [Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)。

---

## 环境要求

| 组件 | 本机（Orin AGX） |
|------|------------------|
| JetPack | R36.4 / CUDA 12.6 |
| Python venv | `setup_env.sh` + `setup_edgellm.sh` |
| TensorRT | 系统 **10.3.0**（勿 pip install tensorrt） |
| Edge-LLM | `third_party/TensorRT-Edge-LLM`（Orin cmake 编译） |
| 模型 | `/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct` |

Orin 支持的精度：**FP16**、**INT4 AWQ**（0.5B + 64GB 默认用 FP16 即可）。

---

## 步骤 1：安装

```bash
cd ~/stephen/01-code/qwen06_acc_agx
bash setup_env.sh
bash setup_edgellm.sh

export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH=$PWD/third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export PYTHONPATH=$PWD/third_party/TensorRT-Edge-LLM:$PYTHONPATH
source venv/bin/activate
```

---

## 步骤 2：准备 ONNX

### 2A. 跳过 x86 export（推荐，已有 Edge-LLM ONNX 时）

```bash
scp -r admin@<已有Orin>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/onnx/ \
    ~/stephen/01-code/qwen06_acc_agx/acc/workspace/
bash acc/export_onnx.sh   # 检查，已存在则跳过
```

### 2B. x86 GPU 本机 export

Orin 上 JetPack PyTorch 2.5 不支持 Edge-LLM 的 dynamo 导出 API。在 **x86 Linux + NVIDIA GPU** 上执行：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
```

或使用独立 venv：`bash acc/setup_export_host.sh && bash acc/export_onnx_host.sh`

拷回 Orin：

```bash
scp -r acc/workspace/onnx/ admin@<orin-ip>:~/stephen/01-code/qwen06_acc_agx/acc/workspace/
```

Orin 上若 ONNX 已存在，`bash acc/export_onnx.sh` 会直接跳过。

### 2C. ModelScope 下载预转换 ONNX（Optimum 格式）

模型页：[Qwen2.5-0.5B-Instruct-ONNX-MHA](https://modelscope.cn/models/onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA)

```bash
pip install modelscope
bash acc/download_onnx_modelscope.sh
```

产出：`acc/workspace/onnx-modelscope/model_fp16.onnx` 及 tokenizer 文件。

**注意**：此为 HuggingFace Optimum / Transformers.js 格式，**不能**直接用于 `acc/build_engine.sh`。Edge-LLM 需要 `onnx/llm/` 下的 `model.onnx` + `model.onnx.data` + `embedding.safetensors` 等 sidecar。无 x86 时请用 **2A** 拷贝 Edge-LLM ONNX，勿与 ModelScope Optimum ONNX 混淆。

---

## 步骤 3：构建引擎

```bash
bash acc/build_engine.sh
```

产出：`acc/workspace/engine/`

### 实测日志（Orin AGX，2026-06-20）

命令：`bash acc/build_engine.sh`（ONNX 来源：x86 主机 `export_onnx_host.sh` 导出后 scp 至 `acc/workspace/onnx/`）

| 项 | 值 |
|----|-----|
| 构建耗时 | **87.9 s** |
| 权重显存 | 1076146176 B（≈ **1.00 GiB**） |
| TRT 分配器峰值 | GPU **1026 MiB**，CPU 0 MiB |
| 构建+序列化峰值 | CPU **4901 MiB** |
| 网络 I/O | 29 inputs / 25 outputs |
| Activation Memory | prefill **66063360 B**，decode **1585152 B** |
| 插件 | AttentionPlugin ×24（DLA fallback → GPU） |

产出目录 `acc/workspace/engine/`：

| 文件 | 大小 |
|------|------|
| `llm.engine` | 1,089,994,884 B（≈ **1.01 GiB**） |
| `embedding.safetensors` | 272,269,400 B（≈ **260 MiB**） |
| `tokenizer.json` | 7,031,645 B |
| `tokenizer_config.json` | 7,305 B |
| `processed_chat_template.json` | 546 B |
| `config.json` | 727 B |

关键日志摘录：

```
[21:30:27.874] [INFO] [TensorRT] Total Weights Memory: 1076146176
[21:30:27.880] [INFO] [TensorRT] Engine generation completed in 87.9465 seconds.
[21:30:27.882] [INFO] [TensorRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 1026 MiB
[21:30:28.249] [INFO] [TensorRT] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 4901 MiB
[21:30:29.433] [INFO] [builderUtils.cpp:328:buildAndSerializeEngine] Engine saved to .../acc/workspace/engine/llm.engine
[21:30:30.110] [INFO] [llm_build.cpp:246:main] LLM engine built successfully.
```

---

## 步骤 4：Benchmark

```bash
bash acc/infer_edgellm.sh
# 或
WARMUP=10 RUNS=30 MAX_NEW_TOKENS=128 bash acc/infer_edgellm.sh
```

---

## 步骤 5：与 Transformers 对比

```bash
bash acc/run_compare.sh
```

仅重跑 Edge-LLM：

```bash
SKIP_HF=1 bash acc/run_compare.sh
```

---

## 加速效果对比（实测模板）

| 指标 | Transformers | TensorRT Edge-LLM | 对比 |
|------|--------------|-------------------|------|
| TTFT avg | _待填_ ms | _待填_ ms | _待填_ |
| E2E tokens/s avg | _待填_ | _待填_ | _待填_× |
| Peak GPU mem | _待填_ MB | N/A | |

开发阶段 Transformers 短测约 **14.5 E2E tokens/s**，TTFT **~73 ms**。完整结果以 `bash acc/run_compare.sh` 为准。

---

## 常见问题

### `operator torchvision::nms does not exist`

torch 与 torchvision CUDA 索引不匹配。重新安装导出环境：

```bash
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
```

### `TypeError: export() got an unexpected keyword argument 'dynamic_shapes'`

Orin venv 的 PyTorch 2.5 过旧。请在 **x86 GPU 主机** 运行 `bash acc/export_onnx_host.sh`，再将 `acc/workspace/onnx/` 拷回 Orin。

### `tensorrt-edgellm-export: command not found`

重新运行 `bash setup_edgellm.sh`，并确认 `source venv/bin/activate`。

### `llm_build: No such file`

C++ runtime 未编译成功，检查 `setup_edgellm.sh` 输出，确认安装了 `libnvinfer-dev` 等依赖。

### 引擎与 ONNX 不匹配

引擎必须在 **运行它的同一台 Orin** 上 build，不可跨设备拷贝 `.plan`。

### `Version tag does not match`

venv 内误装 pip 版 tensorrt，卸载后使用系统 TensorRT 10.3（同 sense_voice 项目做法）。

---

## 文件说明

```
acc/
├── common.sh
├── export_onnx.sh
├── export_onnx_host.sh
├── download_onnx_modelscope.sh
├── build_engine.sh
├── infer_edgellm.sh
├── benchmark_edgellm.py
├── run_compare.sh
├── summarize_results.py
├── workspace/          # onnx + engine（勿提交）
└── acc.md
setup_edgellm.sh        # clone + Python 工具 + C++ 编译
```

## 参考

- [TensorRT Edge-LLM Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
- [Jetson AI Lab 教程](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
