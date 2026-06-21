#!/usr/bin/env python3
"""第二块 AGX 最小 Edge-LLM 推理（调用 llm_inference，无需 venv）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(INFERENCE_ROOT))

from engine import (  # noqa: E402
    build_input_json,
    default_paths,
    extract_response_text,
    load_prompt,
    require_runtime,
    setup_env,
)


def parse_args(defaults) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge-LLM 单次推理（inference/runtime）")
    parser.add_argument("--engine-dir", type=Path, default=defaults.engine_dir)
    parser.add_argument("--llm-inference", type=Path, default=defaults.llm_inference)
    parser.add_argument("--plugin", type=Path, default=defaults.plugin)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--prompt", help="直接指定 user prompt")
    parser.add_argument("--prompt-key", default="short", help="从 prompts.json 读取")
    parser.add_argument("--prompts-file", type=Path, default=defaults.prompts_file)
    parser.add_argument("--input-file", type=Path, help="直接使用已有 input JSON")
    parser.add_argument("--output-json", type=Path, help="输出 JSON 路径，默认 output/output.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--dump-profile", action="store_true", help="输出 profile 计时")
    return parser.parse_args()


def main() -> int:
    defaults = default_paths()
    args = parse_args(defaults)
    paths = defaults
    paths.engine_dir = args.engine_dir
    paths.llm_inference = args.llm_inference
    paths.plugin = args.plugin
    paths.output_dir = args.output_dir
    paths.prompts_file = args.prompts_file

    setup_env(args.plugin)
    require_runtime(paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "output.json")
    profile_json = args.output_dir / "profile.json" if args.dump_profile else None

    if args.input_file:
        prompt_preview = f"<input-file: {args.input_file}>"
        input_json = args.input_file
    elif args.prompt:
        input_json = args.output_dir / "input.json"
        build_input_json(
            input_json,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
        )
        prompt_preview = args.prompt
    else:
        prompt = load_prompt(args.prompts_file, args.prompt_key)
        input_json = args.output_dir / "input.json"
        build_input_json(
            input_json,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
        )
        prompt_preview = prompt

    for path in (output_json, profile_json):
        if path and path.is_file():
            path.unlink()

    cmd = [
        str(paths.llm_inference),
        "--engineDir",
        str(paths.engine_dir),
        "--inputFile",
        str(input_json),
        "--outputFile",
        str(output_json),
    ]
    if args.dump_profile:
        cmd.append("--dumpProfile")
        if profile_json:
            cmd.extend(["--profileOutputFile", str(profile_json)])

    print(f"Engine : {paths.engine_dir}")
    preview = prompt_preview[:80] + ("..." if len(prompt_preview) > 80 else "")
    print(f"Prompt : {preview}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("-" * 40)

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, file=sys.stderr, end="")
        return 1

    profile: dict = {}
    if profile_json and profile_json.is_file():
        profile = json.loads(profile_json.read_text(encoding="utf-8"))

    text = extract_response_text(output_json)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    responses = data.get("responses", [])
    finish_reason = responses[0].get("finish_reason", "") if responses else ""

    print(f"--- response ({finish_reason}) ---")
    print(text)
    print()
    if args.dump_profile and profile:
        prefill = profile.get("prefill", {})
        generation = profile.get("generation", {})
        if prefill:
            print(f"TTFT: {prefill.get('average_time_per_run_ms', 0):.2f} ms")
        if generation:
            print(
                "Decode: "
                f"{generation.get('tokens_per_second', 0):.2f} tokens/s, "
                f"tokens={generation.get('generated_tokens', 0)}"
            )
    print(f"wall: {wall_ms:.0f} ms")
    print(f"完整 JSON: {output_json}")
    if profile_json and profile_json.is_file():
        print(f"Profile : {profile_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
