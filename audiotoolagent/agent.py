"""Runtime primitives for orchestrating AudioToolAgent pipelines."""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from qwen_agent.llm.schema import Message
from sentence_transformers import SentenceTransformer

from .config import (
    initialize_agent,
    initialize_tools,
    load_config,
    select_orchestrators,
    setup_logging,
)

logger = logging.getLogger(__name__)

_sentence_model: SentenceTransformer | None = None


def _sentence_encoder() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sentence_model


def clean_final_answer(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    cleaned: List[str] = []
    for line in lines:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _find_most_similar_option(text: str, options: List[str]) -> Optional[str]:
    encoder = _sentence_encoder()
    text_embedding = encoder.encode([text])
    option_embeddings = encoder.encode(options)
    similarities = np.dot(text_embedding, option_embeddings.T)[0]
    best_idx = int(np.argmax(similarities))
    return options[best_idx]


def extract_selected_option(final_content: str, options: List[str]) -> Optional[str]:
    if not options or not final_content:
        return None

    if "<answer>" in final_content and "</answer>" in final_content:
        answer = final_content.split("<answer>")[1].split("</answer>")[0]
    else:
        answer = final_content.split("[assistant]")[-1]

    answer = answer.strip()

    if len(answer) == 1:
        for option in options:
            if answer.lower() == option[0].lower():
                return option

    for option in options:
        if answer.lower() == option.lower():
            return option
        if option.lower() in answer.lower() or answer.lower() in option.lower():
            return option

    return _find_most_similar_option(answer, options)


class AudioToolAgent:
    """Orchestrates tool-using audio reasoning runs."""

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.tools = initialize_tools(self.config)
        self._orchestrators = select_orchestrators(self.config)
        if not self._orchestrators:
            raise ValueError("Configuration must declare at least one orchestrator section")

        temp_cfg = dict(self.config)
        temp_cfg["orchestrator"] = self._orchestrators[0]
        self.assistant = initialize_agent(temp_cfg, self.tools)
        self.stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        logger.info("Initialised AudioToolAgent with %d tools", len(self.tools))

    def _set_stream_callbacks(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        for tool in self.tools:
            if hasattr(tool, "set_stream_callback"):
                tool.set_stream_callback(callback)
            elif hasattr(tool, "stream_callback"):
                tool.stream_callback = callback  # type: ignore[attr-defined]

    def process(
        self,
        *,
        audio_path: str,
        question: str,
        options: Optional[List[str]] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run a single audio question through the agent pipeline."""
        self._set_stream_callbacks(stream_callback)

        prompt_lines = [f"Audio file: {audio_path}", f"Question: {question}"]
        if options:
            prompt_lines.append("Options: " + ", ".join(options))
            prompt_lines.append("Please answer with the letter or exact option and wrap the final answer in <answer> tags.")
        else:
            prompt_lines.append("Provide a concise answer wrapped in <answer> tags.")
        query = "\n".join(prompt_lines)

        start_time = time.time()
        last_exc: Optional[Exception] = None

        for orch_cfg in random.sample(self._orchestrators, len(self._orchestrators)):
            temp_cfg = dict(self.config)
            temp_cfg["orchestrator"] = orch_cfg
            try:
                self.assistant = initialize_agent(temp_cfg, self.tools)
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                continue

            accumulated_text = ""
            messages = [Message("user", query)]
            try:
                for response in self.assistant.run(messages, stream=True):
                    accumulated_text = self._update_stream(accumulated_text, response, stream_callback)

                cleaned = clean_final_answer(accumulated_text)
                return {
                    "answer": cleaned,
                    "selected_option": extract_selected_option(cleaned, options) if options else None,
                    "total_duration": time.time() - start_time,
                }
            except Exception as exc:  # pragma: no cover - resilience for flaky APIs
                last_exc = exc
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("All orchestrators failed to produce a response")

    # Streaming helpers -------------------------------------------------
    def _update_stream(
        self,
        accumulated: str,
        response: Any,
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> str:
        messages = response if isinstance(response, list) else [response]
        current_text = self._build_text(messages)
        if len(current_text) > len(accumulated):
            delta = current_text[len(accumulated) :]
            if callback and delta:
                callback({"type": "message_delta", "delta": delta})
            for msg in messages:
                if hasattr(msg, "function_call") and msg.function_call:
                    self._handle_function_call(msg.function_call, callback)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        self._handle_tool_call(tool_call, callback)
            return current_text
        return accumulated

    def _build_text(self, messages: List[Any]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = getattr(msg, "role", "assistant")
            if getattr(msg, "reasoning_content", None):
                parts.append(f"\n[{role}]\nReasoning: {msg.reasoning_content}")

            if getattr(msg, "function_call", None):
                func_call = msg.function_call
                func_name = func_call.get("name", "Unknown")
                args = func_call.get("arguments", {})
                if isinstance(args, str):
                    trimmed = args.strip()
                    if trimmed.startswith("{") and trimmed.endswith("}") and len(trimmed) > 2:
                        parts.append(f"\n\n[{role}]\nFunction Call: {func_name}\nArguments: {args}\n")
                elif args:
                    parts.append(f"\n\n[{role}]\nFunction Call: {func_name}\nArguments: {json.dumps(args)}\n")

            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                has_structured = any(
                    getattr(msg, attr, None)
                    for attr in ("reasoning_content", "function_call", "tool_calls")
                )
                if not has_structured:
                    parts.append(f"\n[{role}]\nFinal Response: {content}")
        return "".join(parts)

    def _handle_function_call(
        self,
        func_call: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if not callback or not isinstance(func_call, dict):
            return
        tool_name = func_call.get("name", "Unknown")
        args = func_call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:  # pragma: no cover - best effort only
                return
        callback({"type": "tool_start", "tool": tool_name, "params": args})

    def _handle_tool_call(
        self,
        tool_call: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if not callback or not isinstance(tool_call, dict):
            return
        tool_name = tool_call.get("type") or tool_call.get("function", {}).get("name", "Unknown")
        args = tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:  # pragma: no cover - best effort only
                return
        callback({"type": "tool_start", "tool": tool_name, "params": args})


def console_stream_callback(update: Dict[str, Any]) -> None:
    """Human-friendly console streaming callback used by CLI scripts."""
    if update["type"] == "tool_start":
        tool_name = update.get("tool", "Tool")
        print(f"\n🛠️  {tool_name} Starting...")
        params = json.dumps(update.get("params", {}), indent=2)
        print(f"   Parameters: {params}")
        print("   ", end="", flush=True)
    elif update["type"] == "tool_progress":
        print(update.get("content", ""), end="", flush=True)
    elif update["type"] == "tool_end":
        tool_name = update.get("tool", "Tool")
        print(f"\n✓  {tool_name} Complete\n")
    elif update["type"] == "message_delta":
        if not hasattr(console_stream_callback, "message_started"):
            console_stream_callback.message_started = True  # type: ignore[attr-defined]
            print("\n📝 Complete Message Flow:\n", end="", flush=True)
        print(update["delta"], end="", flush=True)
