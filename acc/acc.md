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

## 步骤 2：导出 ONNX（x86 GPU 主机）

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

---

## 步骤 3：构建引擎

```bash
bash acc/build_engine.sh
```

产出：`acc/workspace/engine/`

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

### Docker `unknown or invalid runtime name: nvidia`

未安装 nvidia-container-toolkit。推荐改走本机导出：

```bash
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
```

或一次性修复 Docker：`bash acc/setup_export_host.sh --docker-toolkit`

### Docker `failed to discover GPU vendor from CDI`

主机 `nvidia-smi` 正常但 Docker 报 CDI 错误时，优先试 legacy runtime：

```bash
export DOCKER_GPU_MODE=runtime
bash acc/export_onnx_host.sh --docker
```

或修复 CDI：

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker --cdi.enabled
sudo systemctl restart docker
```

### `TypeError: export() got an unexpected keyword argument 'dynamic_shapes'`

Orin venv 的 PyTorch 2.5 过旧。请在 x86 GPU 主机运行 `bash acc/export_onnx_host.sh --docker`，再将 `acc/workspace/onnx/` 拷回 Orin。

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
