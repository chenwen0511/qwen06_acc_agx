"""Edge-LLM 推理引擎：路径解析、批式推理、流式生成（pybind / 模拟 / HF）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

INFERENCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_ROOT.parent


@dataclass
class RuntimePaths:
    engine_dir: Path
    llm_inference: Path
    plugin: Path
    output_dir: Path
    prompts_file: Path
    tokenizer_dir: Path


def default_paths() -> RuntimePaths:
    runtime = INFERENCE_ROOT / "runtime"
    acc_engine = PROJECT_ROOT / "acc" / "workspace" / "engine"
    edgellm_root = PROJECT_ROOT / "third_party" / "TensorRT-Edge-LLM"
    edgellm_bin = edgellm_root / "build" / "examples" / "llm" / "llm_inference"
    edgellm_plugin = edgellm_root / "build" / "libNvInfer_edgellm_plugin.so"

    engine_dir = (
        runtime / "engine"
        if (runtime / "engine" / "llm.engine").is_file()
        else acc_engine
    )
    llm_inference = runtime / "bin" / "llm_inference"
    if not llm_inference.is_file():
        llm_inference = edgellm_bin

    plugin = runtime / "lib" / "libNvInfer_edgellm_plugin.so"
    if not plugin.is_file():
        plugin = edgellm_plugin

    return RuntimePaths(
        engine_dir=engine_dir,
        llm_inference=llm_inference,
        plugin=plugin,
        output_dir=INFERENCE_ROOT / "output",
        prompts_file=PROJECT_ROOT / "prompts.json",
        tokenizer_dir=engine_dir,
    )


def setup_env(plugin: Path) -> None:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    if plugin.is_file():
        os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin)
    lib_dir = PROJECT_ROOT / "lib"
    if lib_dir.is_dir():
        os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (
            f":{os.environ['LD_LIBRARY_PATH']}" if os.environ.get("LD_LIBRARY_PATH") else ""
        )


def require_runtime(paths: RuntimePaths) -> None:
    missing = []
    for label, path in (
        ("llm_inference", paths.llm_inference),
        ("plugin", paths.plugin),
        ("llm.engine", paths.engine_dir / "llm.engine"),
    ):
        if not path.is_file():
            missing.append(f"{label}: {path}")
    if missing:
        lines = "\n  ".join(missing)
        raise RuntimeError(f"缺少运行时文件:\n  {lines}\n请先: bash inference/install.sh")


def load_prompt(prompts_file: Path, prompt_key: str) -> str:
    data = json.loads(prompts_file.read_text(encoding="utf-8"))
    try:
        return str(data["prompts"][prompt_key])
    except KeyError as exc:
        keys = ", ".join(sorted(data.get("prompts", {})))
        raise KeyError(f"prompts.json 无 key '{prompt_key}'，可选: {keys}") from exc


def build_input_json(
    path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
) -> None:
    payload = {
        "batch_size": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_generate_length": max_new_tokens,
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "requests": [{"messages": [{"role": "user", "content": prompt}]}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_response_text(output_path: Path) -> str:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    responses = data.get("responses", [])
    if responses and isinstance(responses[0], dict):
        resp = responses[0]
        for key in ("output_text", "text"):
            value = resp.get(key)
            if value:
                return str(value)
    if isinstance(data, dict) and "text" in data:
        return str(data["text"])
    return ""


def run_edgellm_batch(
    paths: RuntimePaths,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    dump_profile: bool = False,
    work_dir: Path | None = None,
) -> tuple[str, float, dict[str, Any]]:
    work = work_dir or paths.output_dir
    work.mkdir(parents=True, exist_ok=True)
    input_json = work / "input.json"
    output_json = work / "output.json"
    profile_json = work / "profile.json" if dump_profile else None

    build_input_json(
        input_json, prompt, max_new_tokens, temperature, top_p, top_k
    )
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
    if dump_profile:
        cmd.append("--dumpProfile")
        if profile_json:
            cmd.extend(["--profileOutputFile", str(profile_json)])

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"llm_inference 失败 (exit={proc.returncode}):\n{detail}")

    profile: dict[str, Any] = {}
    if profile_json and profile_json.is_file():
        profile = json.loads(profile_json.read_text(encoding="utf-8"))
    return extract_response_text(output_json), elapsed_ms, profile


def _load_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "subprocess 流式回放需要 transformers，请 pip install transformers "
            "或使用 --backend pybind / hf"
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


def stream_text_chars(text: str, chars_per_sec: float = 40.0) -> Iterator[str]:
    delay = 1.0 / max(chars_per_sec, 1.0)
    for ch in text:
        yield ch
        time.sleep(delay)


def stream_text_tokens(
    text: str,
    tokenizer_dir: Path,
    tokens_per_sec: float = 34.0,
) -> Iterator[str]:
    """将完整文本按 token 切块输出，用于 subprocess 后模拟流式展示。"""
    try:
        tokenizer = _load_tokenizer(tokenizer_dir)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        delay = 1.0 / max(tokens_per_sec, 1.0)
        for token_id in token_ids:
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            if piece:
                yield piece
            time.sleep(delay)
    except ImportError:
        yield from stream_text_chars(text, chars_per_sec=max(tokens_per_sec, 1.0) * 1.2)


def stream_edgellm_subprocess(
    paths: RuntimePaths,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
) -> Generator[dict[str, Any], None, None]:
    """subprocess 推理完成后按 token 节奏回放（第二块 AGX 默认可用）。"""
    yield {"type": "status", "message": "Edge-LLM 推理中…"}
    text, wall_ms, profile = run_edgellm_batch(
        paths,
        prompt,
        max_new_tokens,
        temperature=temperature,
        dump_profile=True,
    )
    tps = 34.0
    generation = profile.get("generation") or {}
    if generation.get("tokens_per_second"):
        tps = float(generation["tokens_per_second"])

    yield {
        "type": "metrics",
        "wall_ms": wall_ms,
        "ttft_ms": (profile.get("prefill") or {}).get("average_time_per_run_ms"),
        "tokens_per_sec": tps,
    }

    for piece in stream_text_tokens(text, paths.tokenizer_dir, tps):
        yield {"type": "token", "text": piece}

    yield {"type": "done", "text": text, "finish_reason": "stop"}


def _try_load_pybind_llm(engine_dir: Path):
    edgellm_root = PROJECT_ROOT / "third_party" / "TensorRT-Edge-LLM"
    if not edgellm_root.is_dir():
        return None
    root = str(edgellm_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from experimental.server import LLM

        return LLM(engine_dir=str(engine_dir))
    except Exception:
        return None


def stream_edgellm_pybind(
    paths: RuntimePaths,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
) -> Generator[dict[str, Any], None, None]:
    """真流式：需已编译 experimental pybind（setup_pybind）。"""
    llm = _try_load_pybind_llm(paths.engine_dir)
    if llm is None:
        raise RuntimeError(
            "pybind 未就绪。请在首块 AGX 执行 inference/setup_pybind.sh，"
            "或使用 --backend subprocess"
        )

    from experimental.server import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        top_k=1,
        max_tokens=max_new_tokens,
    )
    messages = [{"role": "user", "content": prompt}]
    full_parts: list[str] = []

    yield {"type": "status", "message": "Edge-LLM 流式生成中…"}
    for delta in llm.generate_stream(messages, params):
        if delta.text:
            full_parts.append(delta.text)
            yield {"type": "token", "text": delta.text}
        if delta.finished:
            yield {
                "type": "done",
                "text": "".join(full_parts),
                "finish_reason": delta.finish_reason or "stop",
            }
            return

    yield {
        "type": "done",
        "text": "".join(full_parts),
        "finish_reason": "stop",
    }


def stream_hf(
    prompt: str,
    max_new_tokens: int,
    *,
    model_dir: str | None = None,
    device: str = "cuda:0",
    temperature: float = 0.0,
) -> Generator[dict[str, Any], None, None]:
    """Transformers 真流式（需 venv + 模型权重）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    model_path = model_dir or os.environ.get(
        "QWEN_MODEL_DIR", "/home/admin/stephen/02-weight/Qwen2.5-0.5B-Instruct"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = prompt

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        use_cache=True,
    )
    if temperature <= 0:
        gen_kwargs.pop("temperature", None)
        gen_kwargs["do_sample"] = False

    yield {"type": "status", "message": "Transformers 流式生成中…"}

    error_holder: list[BaseException] = []

    def _generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except BaseException as exc:
            error_holder.append(exc)

    worker = threading.Thread(target=_generate, daemon=True)
    worker.start()

    full_parts: list[str] = []
    for piece in streamer:
        if not piece:
            continue
        full_parts.append(piece)
        yield {"type": "token", "text": piece}

    worker.join(timeout=120.0)
    if error_holder:
        raise error_holder[0]

    yield {
        "type": "done",
        "text": "".join(full_parts),
        "finish_reason": "stop",
    }


def choose_stream_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if _try_load_pybind_llm(default_paths().engine_dir) is not None:
        return "pybind"
    return "subprocess"


def stream_chat(
    prompt: str,
    max_new_tokens: int,
    *,
    backend: str = "auto",
    paths: RuntimePaths | None = None,
    temperature: float = 0.0,
    model_dir: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    runtime = paths or default_paths()
    setup_env(runtime.plugin)
    require_runtime(runtime)

    mode = choose_stream_backend(backend)
    if mode == "pybind":
        yield from stream_edgellm_pybind(
            runtime, prompt, max_new_tokens, temperature=temperature
        )
    elif mode == "subprocess":
        yield from stream_edgellm_subprocess(
            runtime, prompt, max_new_tokens, temperature=temperature
        )
    elif mode == "hf":
        yield from stream_hf(
            prompt, max_new_tokens, model_dir=model_dir, temperature=temperature
        )
    else:
        raise ValueError(f"未知 backend: {backend}")
