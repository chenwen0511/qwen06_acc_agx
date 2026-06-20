#!/usr/bin/env python3
"""汇总 prompts.json 各 key 的 HF / Edge-LLM benchmark 结果。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from summarize_results import backend_summary, load_json, ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", default="prompts.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-json", default="results/summary_all.json")
    args = parser.parse_args()

    with open(args.prompts_file, encoding="utf-8") as f:
        prompts_data = json.load(f)
    prompt_keys = list(prompts_data.get("prompts", {}).keys())

    per_prompt: dict[str, Any] = {}
    for key in prompt_keys:
        result_dir = os.path.join(args.results_root, key)
        hf = load_json(os.path.join(result_dir, "hf.json"))
        edge = load_json(os.path.join(result_dir, "edgellm.json"))
        entry: dict[str, Any] = {
            "prompt": prompts_data["prompts"][key],
            "hf": backend_summary(hf) if hf else None,
            "tensorrt_edge_llm": backend_summary(edge) if edge else None,
            "speedup": {},
        }
        if hf and edge:
            hf_e2e = entry["hf"]["e2e_tokens_per_sec_avg"] or 0.0
            edge_e2e = entry["tensorrt_edge_llm"]["e2e_tokens_per_sec_avg"] or 0.0
            hf_ttft = entry["hf"]["ttft_ms_avg"] or 0.0
            edge_ttft = entry["tensorrt_edge_llm"]["ttft_ms_avg"] or 0.0
            entry["speedup"] = {
                "e2e_tokens_per_sec": ratio(edge_e2e, hf_e2e),
                "decode_tokens_per_sec": ratio(
                    entry["tensorrt_edge_llm"]["decode_tokens_per_sec_avg"] or 0.0,
                    entry["hf"]["decode_tokens_per_sec_avg"] or 0.0,
                ),
                "ttft_ms_inverse": ratio(hf_ttft, edge_ttft),
            }
        per_prompt[key] = entry

    payload = {
        "prompts_file": args.prompts_file,
        "prompt_keys": prompt_keys,
        "per_prompt": per_prompt,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSummary 已写入: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
