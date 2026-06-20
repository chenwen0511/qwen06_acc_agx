# 第二块 AGX Orin 推理（最小步骤）

在**已跑通的首块 AGX** 上打包引擎与 C++ runtime，拷到**新 AGX** 即可推理，**无需 Python venv、无需重新 export ONNX**（两块板均为 AGX Orin + 同 JetPack 时通常可直接用）。

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
| `run.sh` | 单次 `llm_inference` |
| `input.example.json` | 手写 input 参考 |
| `runtime/` | 解压后目录（勿提交 git） |
| `artifacts/` | tarball（勿提交 git） |
