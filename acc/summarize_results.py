#!/usr/bin/env python3
"""合并 HF 与 TensorRT Edge-LLM benchmark JSON，生成 summary.json。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def backend_summary(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    return {
        "backend": data.get("backend"),
        "load_time_s": data.get("load_time_s"),
        "ttft_ms_avg": metrics.get("ttft_ms", {}).get("avg"),
        "decode_tokens_per_sec_avg": metrics.get("decode_tokens_per_sec", {}).get(
            "avg"
        ),
        "e2e_tokens_per_sec_avg": metrics.get("e2e_tokens_per_sec", {}).get("avg"),
        "peak_gpu_mem_mb_avg": metrics.get("peak_gpu_mem_mb", {}).get("avg"),
        "output_tokens_avg": metrics.get("output_tokens", {}).get("avg"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-json", default="results/hf.json")
    parser.add_argument("--edgellm-json", default="results/edgellm.json")
    parser.add_argument("--output-json", default="results/summary.json")
    args = parser.parse_args()

    hf = load_json(args.hf_json)
    edge = load_json(args.edgellm_json)

    summary: dict[str, Any] = {
        "hf": backend_summary(hf) if hf else None,
        "tensorrt_edge_llm": backend_summary(edge) if edge else None,
        "speedup": {},
    }

    if hf and edge:
        hf_e2e = summary["hf"]["e2e_tokens_per_sec_avg"] or 0.0
        edge_e2e = summary["tensorrt_edge_llm"]["e2e_tokens_per_sec_avg"] or 0.0
        hf_ttft = summary["hf"]["ttft_ms_avg"] or 0.0
        edge_ttft = summary["tensorrt_edge_llm"]["ttft_ms_avg"] or 0.0
        summary["speedup"] = {
            "e2e_tokens_per_sec": ratio(edge_e2e, hf_e2e),
            "decode_tokens_per_sec": ratio(
                summary["tensorrt_edge_llm"]["decode_tokens_per_sec_avg"] or 0.0,
                summary["hf"]["decode_tokens_per_sec_avg"] or 0.0,
            ),
            "ttft_ms_inverse": ratio(hf_ttft, edge_ttft),
        }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary 已写入: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
