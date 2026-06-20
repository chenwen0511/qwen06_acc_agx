#!/usr/bin/env python3
"""Qwen2.5-0.5B Transformers baseline benchmark (NVIDIA Orin AGX)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPTS_FILE = os.path.join(_PROJECT_ROOT, "prompts.json")
DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


@dataclass
class RunMetrics:
    ttft_ms: float
    decode_ms: float
    total_ms: float
    output_tokens: int
    decode_tokens_per_sec: float
    e2e_tokens_per_sec: float
    peak_gpu_mem_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B HF benchmark")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("QWEN_MODEL_DIR", DEFAULT_MODEL_DIR),
        help="本地 HF 模型目录",
    )
    parser.add_argument(
        "--prompts-file",
        default=DEFAULT_PROMPTS_FILE,
        help="prompt 配置文件（JSON）",
    )
    parser.add_argument(
        "--prompt-key",
        default="short",
        help="prompts.json 中的 key，或直接配合 --prompt 使用",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="直接指定 prompt 文本（优先级高于 --prompt-key）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("QWEN_DEVICE", "cuda:0"),
        help="推理设备",
    )
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--runs", type=int, default=30, help="计时次数")
    parser.add_argument(
        "--output-json",
        default=os.path.join(DEFAULT_RESULTS_DIR, "hf.json"),
        help="JSON 结果输出路径",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="仅单次推理，跳过 benchmark",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    with open(args.prompts_file, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    if args.prompt_key not in prompts:
        raise KeyError(f"prompt key 不存在: {args.prompt_key}")
    return prompts[args.prompt_key]


def sync_device(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: str) -> float:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def build_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    return inputs.to(device)


def greedy_generate_timed(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, RunMetrics]:
    import torch

    reset_peak_memory(device)
    inputs = build_inputs(tokenizer, prompt, device)
    input_ids = inputs.input_ids
    attention_mask = getattr(inputs, "attention_mask", None)

    sync_device(device)
    total_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        sync_device(device)
        ttft_end = time.perf_counter()

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token.item()]

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

        sync_device(device)
        decode_end = time.perf_counter()

    total_end = decode_end
    ttft_ms = (ttft_end - total_start) * 1000
    decode_ms = (decode_end - decode_start) * 1000
    total_ms = (total_end - total_start) * 1000
    output_tokens = len(generated_ids)
    decode_tps = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0
    e2e_tps = output_tokens / (total_ms / 1000) if total_ms > 0 else 0.0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics = RunMetrics(
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        output_tokens=output_tokens,
        decode_tokens_per_sec=decode_tps,
        e2e_tokens_per_sec=e2e_tps,
        peak_gpu_mem_mb=peak_memory_mb(device),
    )
    return text, metrics


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    result = {
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 2:
        result["stdev"] = statistics.stdev(values)
    else:
        result["stdev"] = 0.0
    return result


def print_run_metrics(run_idx: int, total: int, metrics: RunMetrics) -> None:
    print(
        f"  run {run_idx}/{total}: "
        f"ttft={metrics.ttft_ms:.2f} ms, "
        f"decode={metrics.decode_ms:.2f} ms, "
        f"tokens={metrics.output_tokens}, "
        f"decode_tps={metrics.decode_tokens_per_sec:.2f}, "
        f"e2e_tps={metrics.e2e_tokens_per_sec:.2f}"
    )


def print_summary(title: str, runs: list[RunMetrics]) -> None:
    print(f"\n=== {title} ===")
    for field, label in [
        ("ttft_ms", "TTFT (ms)"),
        ("decode_ms", "Decode (ms)"),
        ("total_ms", "Total (ms)"),
        ("decode_tokens_per_sec", "Decode tokens/s"),
        ("e2e_tokens_per_sec", "E2E tokens/s"),
        ("peak_gpu_mem_mb", "Peak GPU mem (MB)"),
    ]:
        stats = summarize([getattr(r, field) for r in runs])
        print(
            f"{label}: avg={stats['avg']:.2f}, p50={stats['p50']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}, stdev={stats['stdev']:.2f}"
        )


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {path}")


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[ERROR] 模型目录不存在: {args.model_dir}", file=sys.stderr)
        return 1

    prompt = load_prompt(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用", file=sys.stderr)
        return 1

    print(f"模型目录 : {args.model_dir}")
    print(f"设备     : {device}")
    print(f"Prompt   : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    if not args.once:
        print(f"预热次数 : {args.warmup}")
        print(f"推理次数 : {args.runs}")
    print("-" * 50)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    load_elapsed = time.perf_counter() - load_start
    print(f"模型加载耗时: {load_elapsed:.2f}s")

    if args.once:
        text, metrics = greedy_generate_timed(
            model, tokenizer, prompt, device, args.max_new_tokens
        )
        print_run_metrics(1, 1, metrics)
        print("\n生成结果:")
        print(text)
        return 0

    print(f"\n预热 {args.warmup} 次...")
    for i in range(args.warmup):
        greedy_generate_timed(model, tokenizer, prompt, device, args.max_new_tokens)
        print(f"  warmup {i + 1}/{args.warmup} done")

    print(f"\n批量推理 {args.runs} 次（计时）...")
    runs: list[RunMetrics] = []
    last_text = ""
    for i in range(args.runs):
        last_text, metrics = greedy_generate_timed(
            model, tokenizer, prompt, device, args.max_new_tokens
        )
        runs.append(metrics)
        print_run_metrics(i + 1, args.runs, metrics)

    print_summary("Transformers Benchmark", runs)
    print("\n最后一次生成结果:")
    print(last_text)

    payload = {
        "backend": "transformers",
        "model_dir": args.model_dir,
        "device": device,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "load_time_s": load_elapsed,
        "metrics": {
            "ttft_ms": summarize([r.ttft_ms for r in runs]),
            "decode_ms": summarize([r.decode_ms for r in runs]),
            "total_ms": summarize([r.total_ms for r in runs]),
            "decode_tokens_per_sec": summarize(
                [r.decode_tokens_per_sec for r in runs]
            ),
            "e2e_tokens_per_sec": summarize([r.e2e_tokens_per_sec for r in runs]),
            "peak_gpu_mem_mb": summarize([r.peak_gpu_mem_mb for r in runs]),
            "output_tokens": summarize([float(r.output_tokens) for r in runs]),
        },
        "raw_runs": [asdict(r) for r in runs],
        "sample_output": last_text,
    }
    save_json(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
