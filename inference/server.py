#!/usr/bin/env python3
"""Edge-LLM 流式 Chat Web 服务（SSE + 简易界面）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parent
WEB_DIR = INFERENCE_ROOT / "web"
sys.path.insert(0, str(INFERENCE_ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:
    print(
        "[ERROR] 缺少 Web 依赖，请执行:\n"
        "  pip install -r inference/requirements-web.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from engine import default_paths, setup_env, stream_chat


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    backend: str = Field(default="auto")


def create_app(backend: str, host: str, port: int) -> FastAPI:
    paths = default_paths()
    setup_env(paths.plugin)

    app = FastAPI(title="Qwen Edge-LLM Chat", version="1.0.0")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/api/health")
    def health() -> dict:
        ok = (paths.engine_dir / "llm.engine").is_file() and paths.llm_inference.is_file()
        return {
            "status": "ok" if ok else "missing_runtime",
            "engine_dir": str(paths.engine_dir),
            "backend_default": backend,
            "url": f"http://{host}:{port}/",
        }

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        mode = body.backend if body.backend != "auto" else backend

        def event_generator():
            try:
                for event in stream_chat(
                    body.prompt.strip(),
                    body.max_new_tokens,
                    backend=mode,
                    paths=paths,
                    temperature=body.temperature,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                err = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat")
    def chat_once(body: ChatRequest) -> dict:
        mode = body.backend if body.backend != "auto" else backend
        text = ""
        finish_reason = "stop"
        metrics: dict = {}
        try:
            for event in stream_chat(
                body.prompt.strip(),
                body.max_new_tokens,
                backend=mode,
                paths=paths,
                temperature=body.temperature,
            ):
                if event.get("type") == "token":
                    text += event.get("text", "")
                elif event.get("type") == "done":
                    text = event.get("text") or text
                    finish_reason = event.get("finish_reason") or finish_reason
                elif event.get("type") == "metrics":
                    metrics = event
                elif event.get("type") == "error":
                    raise HTTPException(status_code=500, detail=event.get("message"))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "text": text,
            "finish_reason": finish_reason,
            "metrics": metrics,
            "backend": mode,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen Edge-LLM 流式 Web 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--backend",
        choices=("auto", "subprocess", "pybind", "hf"),
        default="auto",
        help="auto=有 pybind 则真流式，否则 subprocess 模拟流式",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not WEB_DIR.joinpath("index.html").is_file():
        print(f"[ERROR] 缺少 {WEB_DIR / 'index.html'}", file=sys.stderr)
        return 1

    import uvicorn

    app = create_app(args.backend, args.host, args.port)
    print(f"Chat UI: http://{args.host}:{args.port}/")
    print(f"Backend : {args.backend}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
