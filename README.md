# Qwen2.5-0.5B Orin 推理加速验证

在 NVIDIA Jetson AGX Orin（JetPack R36.4 / CUDA 12.6）上，对比 **Transformers** 与 **TensorRT Edge-LLM** 推理 Qwen2.5-0.5B-Instruct 的性能。全部在 **venv + 原生 C++ runtime** 中完成，无 Docker。

## 模型路径

```
/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct
```

可通过环境变量 `QWEN_MODEL_DIR` 覆盖。

## 环境要求

| 组件 | 版本 |
|------|------|
| JetPack | R36.4（6.2.x） |
| CUDA | 12.6 |
| Python | 3.10 |
| TensorRT | 系统 10.3.0（勿 pip install tensorrt） |
| Edge-LLM | [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)（源码 clone + Orin 编译） |

## 快速开始

### 1. 安装 Transformers 环境

```bash
cd ~/stephen/01-code/qwen06_acc_agx
bash setup_env.sh
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

### 2. Phase 1：Transformers 基线

```bash
python infer_hf.py --warmup 10 --runs 30 --max-new-tokens 128
```

### 3. 安装 TensorRT Edge-LLM

```bash
# 国内网络建议加镜像
export GITHUB_MIRROR=https://ghproxy.net/https://github.com
bash setup_edgellm.sh
```

克隆 `third_party/TensorRT-Edge-LLM`，安装 Python 导出工具，并在 Orin 上编译 C++ runtime（首次约 10–30 分钟）。

### 4. Phase 2：导出 ONNX（x86）→ Orin 构建引擎 → benchmark

Edge-LLM 的 ONNX 导出需要 **PyTorch ≥ 2.12**（dynamo + `dynamic_shapes`），Jetson Orin 的 JetPack wheel 最高仅 2.5，**无法在 Orin 上导出**。官方流程：在 x86 GPU 主机（或 Jetson Thor）导出，再把 ONNX 拷到 Orin。

**在 x86 GPU 主机上**（推荐 Docker）：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
bash acc/export_onnx_host.sh --docker
# 将 acc/workspace/onnx/ scp 到 Orin 同路径
```

**在 Orin 上**（构建引擎 + 推理，ONNX 到位后）：

```bash
bash acc/build_engine.sh     # llm_build
bash acc/infer_edgellm.sh    # llm_inference + llm_bench
```

### 5. 一键对比

```bash
bash acc/run_compare.sh
```

结果输出到 `results/`：

- `hf.json` / `hf.log`
- `edgellm.json` / `edgellm.log`
- `summary.json`

## 目录结构

```
qwen06_acc_agx/
├── setup_env.sh
├── setup_edgellm.sh
├── infer_hf.py
├── prompts.json
├── third_party/TensorRT-Edge-LLM/   # setup 时 clone
├── acc/
│   ├── export_onnx.sh         # Orin 侧检查（若 ONNX 已存在则跳过）
│   ├── export_onnx_host.sh    # x86 GPU 导出（--docker 推荐）
│   ├── build_engine.sh
│   ├── infer_edgellm.sh
│   ├── benchmark_edgellm.py
│   ├── run_compare.sh
│   ├── summarize_results.py
│   ├── workspace/                   # onnx + engine 产物
│   └── acc.md
└── results/
```

## 参考

- [TensorRT Edge-LLM Jetson 教程](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- [sense_voice_agx](../sense_voice_agx) — Orin 原生 TensorRT 加速参考
