import base64
import io
import json
import os
import tempfile
import time
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware


def create_app(title: str = "OpenAI-Compatible Audio API") -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def register_hostnames_entry(app: FastAPI, env_key: str, default_port: int) -> None:
    @app.on_event("startup")
    def _write_mapping() -> None:
        try:
            port = int(os.getenv("PORT", str(default_port)))
            host = os.getenv("HOST", "127.0.0.1")
            hostname = os.popen("hostname").read().strip() or host
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            hostnames_path = os.path.join(root_dir, 'hostnames.txt')
            new_line = f"{env_key}=http://{hostname}:{port}/v1\n"
            lines = []
            if os.path.exists(hostnames_path):
                with open(hostnames_path, 'r') as f:
                    lines = f.readlines()
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{env_key}="):
                    lines[i] = new_line
                    found = True
                    break
            if not found:
                lines.append(new_line)
            with open(hostnames_path, 'w') as f:
                f.writelines(lines)
        except Exception:
            # Best-effort; do not block startup
            pass


def _now_ts() -> int:
    return int(time.time())


def _make_completion_id() -> str:
    return f"chatcmpl_{int(time.time() * 1000)}"


def parse_messages_for_text_and_audio(messages: list) -> Tuple[str, Optional[str], str]:
    """
    Extract the first text prompt and first audio (base64) from messages.
    Returns: (text_prompt, audio_base64_or_none, audio_format)
    """
    text_prompt = ""
    audio_b64: Optional[str] = None
    audio_format = "wav"
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for c in content:
                ctype = c.get("type")
                if ctype == "text" and not text_prompt:
                    text_prompt = c.get("text", "")
                if ctype == "input_audio" and audio_b64 is None:
                    ia = c.get("input_audio", {})
                    data = ia.get("data")
                    fmt = ia.get("format", "wav")
                    if isinstance(data, str):
                        audio_b64 = data
                        audio_format = fmt
        elif isinstance(content, str) and not text_prompt:
            text_prompt = content
    return text_prompt, audio_b64, audio_format


def save_base64_audio_to_temp(audio_b64: str, audio_format: str = "wav") -> str:
    raw = base64.b64decode(audio_b64)
    suffix = f".{audio_format}" if audio_format and not audio_format.startswith(".") else (audio_format or ".wav")
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(raw)
    return path


def sse_format(data: Dict) -> bytes:
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")


def completion_response(
    *,
    text: str,
    model: str,
    stream: bool,
    chunk_size: int = 64,
) -> Union[JSONResponse, StreamingResponse]:
    created = _now_ts()
    cid = _make_completion_id()

    if not stream:
        body = {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(content=body)

    def gen() -> Iterable[bytes]:
        # stream chunks with OpenAI-like delta
        sent_any = False
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
            }
            sent_any = True
            yield sse_format(chunk)
        # final chunk
        final = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        if sent_any:
            yield sse_format(final)
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def mount_chat_completions(
    app: FastAPI,
    handler: Callable[[Dict], Tuple[str, Optional[str]]],
):
    """
    Mount a /v1/chat/completions endpoint on app.

    handler(request_json) -> (final_text, model_name)
    """

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        payload = await req.json()
        stream = bool(payload.get("stream", False))
        try:
            text, model_name = handler(payload)
            model_name = model_name or payload.get("model", "local-model")
            return completion_response(text=text, model=model_name, stream=stream)
        except Exception as e:
            err = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                }
            }
            return JSONResponse(status_code=500, content=err)


