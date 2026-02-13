"""Configuration helpers for AudioToolAgent."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from qwen_agent.agents import Assistant

from .tools import (
    AudioFlamingo3Tool,
    AudioFlamingoAPITool,
    DeSTA25Tool,
    Gemini3ProAudioTool,
    GeminiAudioTool,
    GPT4oAudioTool,
    GraniteSpeechTool,
    Qwen25OmniTool,
    Qwen3InstructTool,
    VoxtralAPITool,
    WhisperTool,
)

logger = logging.getLogger(__name__)


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logging(config: Dict[str, Any]) -> None:
    """Initialise logging using optional config overrides."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)


def _tool_registry() -> Dict[str, type]:
    """Return the mapping of tool type identifiers to classes."""
    return {
        "audioflamingo3": AudioFlamingo3Tool,
        "audioflamingo_api": AudioFlamingoAPITool,
        "desta25": DeSTA25Tool,
        "gemini": GeminiAudioTool,
        "gemini3_pro": Gemini3ProAudioTool,
        "gpt4o": GPT4oAudioTool,
        "granite_speech": GraniteSpeechTool,
        "qwen2_5omni": Qwen25OmniTool,
        "qwen3_instruct": Qwen3InstructTool,
        "voxtral_api": VoxtralAPITool,
        "whisper": WhisperTool,
    }


def initialize_tools(config: Dict[str, Any]) -> List[Any]:
    """Instantiate tool objects declared in the configuration."""
    tools: List[Any] = []
    tool_configs = config.get("tools", [])
    registry = _tool_registry()

    for entry in tool_configs:
        if not entry.get("enabled", True):
            continue
        tool_type = entry.get("type")
        if tool_type not in registry:
            logger.warning("Skipping unknown tool type '%s'", tool_type)
            continue
        tool_cls = registry[tool_type]
        params = entry.get("params", {})
        tool = tool_cls(**params)
        tools.append(tool)
        logger.info("Initialised tool: %s", tool_type)

    return tools


def build_llm_config(orchestrator_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate orchestrator configuration into a qwen-agent LLM spec."""
    llm_type = orchestrator_cfg.get("llm_type", "vllm")

    base: Dict[str, Any] = {
        "model": orchestrator_cfg.get("llm_model", "Qwen/Qwen3-32B-AWQ"),
        "max_tokens": orchestrator_cfg.get("max_tokens", 16384),
    }

    if "temperature" in orchestrator_cfg or llm_type == "chutes":
        base["temperature"] = orchestrator_cfg.get("temperature", 0.7)

    if llm_type == "vllm":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "http://127.0.0.1:8000/v1"),
                "api_key": orchestrator_cfg.get("api_key", "dummy"),
            }
        )
    elif llm_type == "openai":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://api.openai.com/v1"),
                "api_key": orchestrator_cfg.get("api_key")
                or os.getenv(orchestrator_cfg.get("api_key_env", "OPENAI_API_KEY"), ""),
            }
        )
    elif llm_type == "google":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                "api_key": orchestrator_cfg.get("api_key")
                or os.getenv(orchestrator_cfg.get("api_key_env", "GOOGLE_API_KEY"), ""),
            }
        )
    elif llm_type == "custom":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url"),
                "api_key": orchestrator_cfg.get("api_key", ""),
            }
        )
    elif llm_type == "anthropic":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://api.anthropic.com/v1"),
                "api_key": orchestrator_cfg.get("api_key")
                or os.getenv(orchestrator_cfg.get("api_key_env", "ANTHROPIC_API_KEY"), ""),
            }
        )
    elif llm_type == "openrouter":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://openrouter.ai/api/v1"),
                "api_key": orchestrator_cfg.get("api_key")
                or os.getenv(orchestrator_cfg.get("api_key_env", "OPENROUTER_API_KEY"), ""),
            }
        )
    elif llm_type == "mistral":
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://api.mistral.ai/v1"),
                "api_key": orchestrator_cfg.get("api_key")
                or os.getenv(orchestrator_cfg.get("api_key_env", "MISTRAL_API_KEY"), ""),
            }
        )
    elif llm_type == "chutes":
        if "api_key" in orchestrator_cfg and orchestrator_cfg.get("api_key"):
            chosen_key = orchestrator_cfg["api_key"]
        else:
            candidates = [
                os.getenv("CHUTES_API_KEY"),
                os.getenv("CHUTES_API_KEY2"),
                os.getenv("CHUTES_API_KEY3"),
                os.getenv("CHUTES_API_KEY4"),
            ]
            candidates = [k for k in candidates if k]
            chosen_key = random.choice(candidates) if candidates else ""
        base.update(
            {
                "model_server": orchestrator_cfg.get("llm_url", "https://llm.chutes.ai/v1"),
                "api_key": chosen_key,
            }
        )
    else:
        raise ValueError(f"Unsupported llm_type '{llm_type}'")

    extra_params = orchestrator_cfg.get("llm_extra_params", {})
    base.update({k: v for k, v in extra_params.items() if k != "generate_cfg"})

    max_retries = orchestrator_cfg.get("max_retries", 6)
    generate_cfg: Dict[str, Any] = {"max_retries": max_retries}
    # Only add seed for APIs that support it (Google/Mistral APIs don't)
    if llm_type not in ("google", "mistral"):
        seed = orchestrator_cfg.get("seed", 42)
        generate_cfg["seed"] = seed
    if "generate_cfg" in extra_params:
        generate_cfg.update(extra_params["generate_cfg"])
    base["generate_cfg"] = generate_cfg

    return base


def initialize_agent(config: Dict[str, Any], tools: List[Any]) -> Assistant:
    """Create the qwen-agent Assistant instance."""
    orchestrator_cfg = config.get("orchestrator", {})
    llm_cfg = build_llm_config(orchestrator_cfg)

    return Assistant(
        llm=llm_cfg,
        system_message=orchestrator_cfg.get(
            "system_prompt",
            (
                "You are an expert audio analyst with access to specialist tools. "
                "Reason carefully before answering and place your final answer between "
                "<answer> and </answer> tags."
            ),
        ),
        function_list=tools,
        name=orchestrator_cfg.get("name", "AudioToolAgent"),
        description=orchestrator_cfg.get("description", "Agent for audio understanding"),
    )


def select_orchestrators(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the orchestrator section into a list (with fallbacks).

    Each entry may contain a nested ``fallback`` key.  When present the entry
    is expanded into ``[primary, fallback]`` so the agent can try the primary
    first and fall back automatically.
    """
    raw: List[Dict[str, Any]]
    if "orchestrators" in config and isinstance(config["orchestrators"], list):
        raw = [entry or {} for entry in config["orchestrators"] if isinstance(entry, dict)]
    elif isinstance(config.get("orchestrator"), list):
        raw = [entry or {} for entry in config["orchestrator"] if isinstance(entry, dict)]
    else:
        raw = [config.get("orchestrator", {})]

    expanded: List[Dict[str, Any]] = []
    for entry in raw:
        fallback = entry.pop("fallback", None)
        expanded.append(entry)
        if isinstance(fallback, dict):
            expanded.append(fallback)
    return expanded
