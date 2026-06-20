#!/usr/bin/env python3
"""TensorRT Edge-LLM benchmark - 调用 llm_inference / llm_bench 并输出 JSON。"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )


@dataclass
class RunMetrics:
    ttft_ms: float
    decode_ms: float
    total_ms: float
    output_tokens: int
    decode_tokens_per_sec: float
    e2e_tokens_per_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT Edge-LLM benchmark")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--llm-inference", default=os.environ.get("LLM_INFERENCE", ""))
    parser.add_argument("--llm-bench", default=os.environ.get("LLM_BENCH", ""))
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    result = {
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }
    return result


def count_output_tokens(tokenizer_dir: str, text: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_response_text(output_path: str) -> str:
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        responses = data.get("responses", [])
        if responses and isinstance(responses[0], dict):
            return responses[0].get("text", "")
        if "text" in data:
            return str(data["text"])
    return json.dumps(data, ensure_ascii=False)


def parse_bench_ms(output: str, mode: str) -> float | None:
    patterns = [
        rf"{mode}.*?([0-9]+(?:\.[0-9]+)?)\s*ms",
        rf"{mode} latency.*?([0-9]+(?:\.[0-9]+)?)",
        r"latency\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        r"([0-9]+(?:\.[0-9]+)?)\s*ms",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def run_llm_bench(
    llm_bench: str,
    engine_dir: str,
    mode: str,
    input_len: int,
    output_len: int,
) -> tuple[float | None, str]:
    if not llm_bench or not os.path.isfile(llm_bench):
        return None, ""
    cmd = [
        llm_bench,
        "--engineDir",
        engine_dir,
        "--mode",
        mode,
        "--inputLen",
        str(input_len),
        "--outputLen",
        str(output_len),
        "--batchSize",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    return parse_bench_ms(output, mode), output


def build_input_json(path: str, prompt: str, max_new_tokens: int) -> None:
    payload = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": max_new_tokens,
        "requests": [{"messages": [{"role": "user", "content": prompt}]}],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_inference_once(
    llm_inference: str,
    engine_dir: str,
    input_path: str,
    output_path: str,
) -> tuple[float, str]:
    if os.path.exists(output_path):
        os.remove(output_path)
    start = time.perf_counter()
    proc = subprocess.run(
        [
            llm_inference,
            "--engineDir",
            engine_dir,
            "--inputFile",
            input_path,
            "--outputFile",
            output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        raise RuntimeError(
            f"llm_inference failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    text = extract_response_text(output_path)
    return elapsed_ms, text


def estimate_metrics(
    total_ms: float,
    prompt: str,
    output_text: str,
    tokenizer_dir: str,
    bench_prefill_ms: float | None,
    bench_decode_ms: float | None,
) -> RunMetrics:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = prompt
    input_tokens = len(tokenizer.encode(prompt_text))
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    output_tokens = max(output_tokens, 1)

    if bench_prefill_ms is not None and bench_decode_ms is not None:
        ttft_ms = bench_prefill_ms
        decode_ms = bench_decode_ms
    else:
        denom = max(input_tokens + output_tokens, 1)
        ttft_ms = total_ms * input_tokens / denom
        decode_ms = total_ms - ttft_ms

    decode_tps = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0
    e2e_tps = output_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
    return RunMetrics(
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        output_tokens=output_tokens,
        decode_tokens_per_sec=decode_tps,
        e2e_tokens_per_sec=e2e_tps,
    )


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    llm_inference = args.llm_inference or os.environ.get(
        "LLM_INFERENCE",
        os.path.join(
            _PROJECT_ROOT,
            "third_party/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
        ),
    )
    llm_bench = args.llm_bench or os.environ.get(
        "LLM_BENCH",
        os.path.join(
            _PROJECT_ROOT,
            "third_party/TensorRT-Edge-LLM/build/examples/llm/llm_bench",
        ),
    )

    if not os.path.isfile(llm_inference):
        print(f"[ERROR] 未找到 llm_inference: {llm_inference}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.engine_dir):
        print(f"[ERROR] engine 目录不存在: {args.engine_dir}", file=sys.stderr)
        return 1

    results_dir = os.path.dirname(args.output_json) or "results"
    input_json = os.path.join(results_dir, "edgellm_input.json")
    build_input_json(input_json, args.prompt, args.max_new_tokens)

    print(f"Engine dir : {args.engine_dir}")
    print(f"Prompt     : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"warmup/runs: {args.warmup}/{args.runs}")
    print("-" * 50)

    from transformers import AutoTokenizer

    _ = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    prompt_tokens = len(
        _.encode(
            _.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if hasattr(_, "apply_chat_template")
            else args.prompt
        )
    )

    prefill_ms, prefill_log = run_llm_bench(
        llm_bench, args.engine_dir, "prefill", prompt_tokens, args.max_new_tokens
    )
    decode_ms, decode_log = run_llm_bench(
        llm_bench, args.engine_dir, "generation", prompt_tokens, args.max_new_tokens
    )
    if prefill_log:
        print("llm_bench prefill:\n" + prefill_log.strip()[-500:])
    if decode_log:
        print("llm_bench generation:\n" + decode_log.strip()[-500:])

    print(f"\n预热 {args.warmup} 次...")
    for i in range(args.warmup):
        out_path = os.path.join(results_dir, f"edgellm_warmup_{i}.json")
        run_inference_once(llm_inference, args.engine_dir, input_json, out_path)
        print(f"  warmup {i + 1}/{args.warmup} done")

    print(f"\n批量推理 {args.runs} 次（计时）...")
    runs: list[RunMetrics] = []
    last_text = ""
    for i in range(args.runs):
        out_path = os.path.join(results_dir, f"edgellm_run_{i}.json")
        total_ms, last_text = run_inference_once(
            llm_inference, args.engine_dir, input_json, out_path
        )
        metrics = estimate_metrics(
            total_ms,
            args.prompt,
            last_text,
            args.tokenizer_dir,
            prefill_ms,
            decode_ms,
        )
        runs.append(metrics)
        print(
            f"  run {i + 1}/{args.runs}: "
            f"total={metrics.total_ms:.2f} ms, "
            f"tokens={metrics.output_tokens}, "
            f"e2e_tps={metrics.e2e_tokens_per_sec:.2f}"
        )

    payload = {
        "backend": "tensorrt-edge-llm",
        "engine_dir": args.engine_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "llm_bench_prefill_ms": prefill_ms,
        "llm_bench_generation_ms": decode_ms,
        "metrics": {
            "ttft_ms": summarize([r.ttft_ms for r in runs]),
            "decode_ms": summarize([r.decode_ms for r in runs]),
            "total_ms": summarize([r.total_ms for r in runs]),
            "decode_tokens_per_sec": summarize(
                [r.decode_tokens_per_sec for r in runs]
            ),
            "e2e_tokens_per_sec": summarize([r.e2e_tokens_per_sec for r in runs]),
            "output_tokens": summarize([float(r.output_tokens) for r in runs]),
        },
        "raw_runs": [asdict(r) for r in runs],
        "sample_output": last_text,
    }
    save_json(args.output_json, payload)
    print(f"\n结果已写入: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
