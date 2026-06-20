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

**在 x86 GPU 主机上**（Docker 未配置时推荐本机导出）：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
bash acc/setup_export_host.sh --conda
USE_CURRENT_ENV=1 bash acc/export_onnx_host.sh
# 将 acc/workspace/onnx/ scp 到 Orin 同路径
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

```bash
bash acc/run_compare.sh
```

结果输出到 `results/`：

- `hf.json` / `hf.log`
- `edgellm.json` / `edgellm.log`
- `summary.json`

---

## 实测对比结论（AGX Orin）

**测试条件（两侧一致）**

| 项 | 值 |
|----|-----|
| 模型 | Qwen2.5-0.5B-Instruct（FP16） |
| Prompt | `用一句话介绍 NVIDIA Jetson AGX Orin。` |
| max_new_tokens | 128 |
| warmup / runs | 10 / 30 |
| 实际输出 | 52 tokens（两侧相同，见下方样例） |
| HF | PyTorch 2.5 + Transformers 4.x，`attn_implementation=eager` |
| Edge-LLM | TensorRT FP16 引擎（x86 导出 ONNX → Orin `llm_build`） |

**性能对比（30 次平均）**

| 指标 | Transformers | TensorRT Edge-LLM | Edge-LLM 相对 HF |
|------|--------------|-------------------|------------------|
| TTFT | **75.2 ms** | **39.0 ms** | **1.93× 更快**（约 −48%） |
| Decode 吞吐 | **14.4 tokens/s** | **34.6 tokens/s** | **2.40×** |
| E2E 吞吐 | **14.1 tokens/s** | **33.7 tokens/s** | **2.39×** |
| 总延迟 | 3680 ms | 1542 ms | **2.39× 更快** |
| Peak GPU 显存 | **974 MB** | —（C++ runtime 未统计） | — |
| 输出 tokens | 52 | 52 | 一致 |

**结论**

1. **吞吐**：在相同 prompt 与生成上限下，Edge-LLM E2E 约 **2.4×** 于 Transformers（33.7 vs 14.1 tokens/s），decode 阶段约 **2.4×**。
2. **首 token 延迟**：TTFT 从 ~75 ms 降至 ~39 ms，约 **快 1.9×**。
3. **输出一致性**：两侧 `sample_output` 文本相同（52 token 中文简介），说明 FP16 引擎在该任务上结果与 HF 基线一致。
4. **适用场景**：0.5B 小模型在 Orin 上 Edge-LLM 收益明显；HF 仍占 ~974 MB 显存（含 PyTorch 运行时），Edge-LLM 为纯 C++ 推理路径，部署侧更轻。
5. **方法说明**：Edge-LLM 指标来自 `llm_inference --dumpProfile`（非 wall-clock 冷启动）；HF 为 Python 进程内计时。完整 raw 数据见 `results/hf.json`、`results/edgellm.json`。

样例输出（两侧相同）：

> NVIDIA Jetson AGX Orin 是一款专为 AI 和机器学习应用设计的高性能计算平台，集成了最新的 NVIDIA Jetson 平台架构，提供强大的计算能力和灵活的扩展性，适用于各种 AI 和机器学习项目。

---

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
