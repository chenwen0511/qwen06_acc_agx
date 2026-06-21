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

### x86 GPU 主机（ONNX 导出）

| 组件 | 版本 |
|------|------|
| PyTorch | **2.12.0**（`cu130` 或 `cu126`） |
| torchvision | **0.27.0**（须与 torch **同一 CUDA 索引**） |
| transformers | 5.9.0 |

`bash acc/setup_export_host.sh` 会自动安装匹配的 `torch` + `torchvision`。勿混用不同 CUDA 索引（如 `torch 2.12+cu130` + `torchvision 0.25+cu128`），否则导出收尾会因 `torchvision::nms` 报错失败。

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

**在 x86 GPU 主机上**（Docker 未配置时推荐本机导出）：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
bash acc/setup_export_host.sh --conda   # 安装 torch 2.12 + torchvision 0.27（同 CUDA 索引）
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
# 将 acc/workspace/onnx/ scp 到 Orin 同路径
```

独立 venv（与 sam3 等 conda 环境隔离，推荐）：

```bash
bash acc/setup_export_host.sh            # 创建 venv-export/
bash acc/export_onnx_host.sh
```

Docker 导出（需先 `bash acc/setup_export_host.sh --docker-toolkit`）：

```bash
bash acc/export_onnx_host.sh --docker
```

**在 Orin 上**（构建引擎 + 推理，ONNX 到位后）：

```bash
bash acc/build_engine.sh     # llm_build
bash acc/infer_edgellm.sh    # llm_inference + llm_bench
```

### 5. 一键对比

单条 prompt（默认 `prompts.json` 的 `short`）：

```bash
bash acc/run_compare.sh
# 或指定 key：PROMPT_KEY=medium bash acc/run_compare.sh
```

快速冒烟（推荐，约 1–2 分钟/prompt）：

```bash
WARMUP=2 RUNS=5 bash acc/run_compare.sh
WARMUP=2 RUNS=5 bash acc/run_prompts.sh   # 遍历 short / medium / long
```

全量 benchmark（正式对比，约 5–8 分钟/prompt）：

```bash
WARMUP=10 RUNS=30 bash acc/run_compare.sh
```

结果输出到 `results/`（多 prompt 时在 `results/<key>/`）：

- `hf.json` / `edgellm.json` / `summary.json`
- 全 prompt 汇总：`results/summary_all.json`（需跑完 `run_prompts.sh`）

### 6. 第二块 AGX 推理（最小步骤）

见 [`inference/README.md`](inference/README.md)。概要：

```bash
# 首块 AGX
bash inference/pack_artifacts.sh
scp inference/artifacts/qwen06_edgellm_orin.tar.gz admin@<新板IP>:~/.../inference/artifacts/

# 新 AGX
bash inference/install.sh && bash inference/run.sh
```

---

## 实测对比结论（AGX Orin）

### 测试配置

| 项 | 值 |
|----|-----|
| 模型 | Qwen2.5-0.5B-Instruct（FP16 引擎） |
| Prompt 来源 | `prompts.json` → **short**（一句话介绍 Orin） |
| max_new_tokens | 128 |
| warmup / runs | **10 / 30**（下文数据）；日常建议 **2 / 5** |
| HF | PyTorch 2.5 + Transformers 4.x，`attn_implementation=eager` |
| Edge-LLM | x86 导出 ONNX → Orin `llm_build`，指标来自 `llm_inference --dumpProfile` |

`prompts.json` 另含 **medium**、**long**（更长输入，单次 HF 推理更慢）。全量 `WARMUP=10 RUNS=30` 跑三条 × 双后端约 **30–60+ 分钟**，实测仅 **short** 完整跑完；medium 在 HF 阶段被中断。后续请用 `WARMUP=2 RUNS=5 bash acc/run_prompts.sh` 做快速覆盖。

### short prompt 性能（30 次平均）

数据来源：`results/hf.json`、`results/edgellm.json`（与 `results/short/` 复测一致，加速比 ~**2.3–2.4×**）。

| 指标 | Transformers | TensorRT Edge-LLM | Edge-LLM 相对 HF |
|------|--------------|-------------------|------------------|
| TTFT | **75.2 ms** | **39.0 ms** | **1.93× 更快** |
| Decode 吞吐 | **14.4 tokens/s** | **34.6 tokens/s** | **2.40×** |
| E2E 吞吐 | **14.1 tokens/s** | **33.7 tokens/s** | **2.39×** |
| 总延迟 | 3680 ms | 1542 ms | **2.39× 更快** |
| Peak GPU 显存 | **974 MB** | —（C++ 未统计） | — |
| 输出 tokens | 52 | 52 | 一致 |

### 结论

1. **加速效果**：short prompt 下 Edge-LLM 相对 HF 约 **2.4× 吞吐**、TTFT 约 **快 2×**，与 Qwen2.5-0.5B + Orin FP16 的预期收益一致。
2. **输出一致**：两侧均生成 52 token，文本相同（见下），FP16 引擎在该任务上与 HF 对齐。
3. **部署**：HF 固定占用 ~974 MB 显存（含 PyTorch）；Edge-LLM 为 C++ 推理路径，适合边缘部署。
4. **prompt 长度**：medium/long 输入更长，prefill 与总耗时显著增加；不建议默认 10/30 全量扫三条，用 `WARMUP=2 RUNS=5` 或单 key 测试即可。
5. **方法差异**：Edge-LLM 为 profile 内计时（不含每次子进程冷启动 ~4 s wall）；HF 为进程内计时。对比吞吐以 profile 指标为准。

样例输出（short，两侧相同）：

> NVIDIA Jetson AGX Orin 是一款专为 AI 和机器学习应用设计的高性能计算平台，集成了最新的 NVIDIA Jetson 平台架构，提供强大的计算能力和灵活的扩展性，适用于各种 AI 和机器学习项目。

### prompts.json 覆盖状态

| Key | 说明 | 全量 10/30 | 备注 |
|-----|------|------------|------|
| `short` | 一句话介绍 Orin | ✅ 已完成 | 见上表 |
| `medium` | Transformers vs TRT-LLM 差异 | ⏸ 中断 | 输入更长，HF 单轮 ~数分钟级 |
| `long` | 嵌入式部署短文 | ⏸ 未跑 | 同上 |

快速补测：

```bash
WARMUP=2 RUNS=5 PROMPT_KEY=medium bash acc/run_compare.sh
WARMUP=2 RUNS=5 bash acc/run_prompts.sh && python3 acc/summarize_prompts.py
```

---
- `hf.json` / `hf.log`
- `edgellm.json` / `edgellm.log`
- `summary.json`

## 常见问题（导出）

### `operator torchvision::nms does not exist`

`torch` 与 `torchvision` CUDA 索引或版本不匹配。修复：

```bash
pip install torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
# 或
bash acc/setup_export_host.sh --conda
```

详见 [acc/acc.md](acc/acc.md) 常见问题章节。

## 目录结构

```
qwen06_acc_agx/
├── setup_env.sh
├── setup_edgellm.sh
├── infer_hf.py
├── prompts.json
├── third_party/TensorRT-Edge-LLM/   # setup 时 clone
├── acc/
│   ├── setup_export_host.sh   # x86 导出环境（torch 2.12 + torchvision 0.27）
│   ├── export_onnx.sh         # Orin 侧检查（若 ONNX 已存在则跳过）
│   ├── export_onnx_host.sh    # x86 GPU 导出（--docker 推荐）
│   ├── build_engine.sh
│   ├── infer_edgellm.sh
│   ├── benchmark_edgellm.py
│   ├── run_compare.sh
│   ├── run_prompts.sh           # 遍历 prompts.json（默认 2/5 快速）
│   ├── summarize_results.py
│   ├── summarize_prompts.py
│   ├── workspace/                   # onnx + engine 产物
│   └── acc.md
└── results/
```

## 参考

- [TensorRT Edge-LLM Jetson 教程](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- [sense_voice_agx](../sense_voice_agx) — Orin 原生 TensorRT 加速参考
