"""Configuration helpers for AudioToolAgent."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from qwen_agent.agents import Assistant

from .tools import (
    AudioFlamingo3Tool,
    AudioFlamingoAPITool,
    DeSTA25Tool,
    GeminiAudioTool,
    GPT4oAudioTool,
    GraniteSpeechTool,
    Qwen25OmniTool,
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
        "gpt4o": GPT4oAudioTool,
        "granite_speech": GraniteSpeechTool,
        "qwen2_5omni": Qwen25OmniTool,
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

    if "temperature" in orchestrator_cfg:
        base["temperature"] = orchestrator_cfg["temperature"]

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
    else:
        raise ValueError(f"Unsupported llm_type '{llm_type}'")

    extra_params = orchestrator_cfg.get("llm_extra_params", {})
    base.update({k: v for k, v in extra_params.items() if k != "generate_cfg"})

    max_retries = orchestrator_cfg.get("max_retries", 6)
    seed = orchestrator_cfg.get("seed", 42)
    generate_cfg = {"max_retries": max_retries, "seed": seed}
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
    """Expand the orchestrator section into a list (with fallbacks)."""
    if "orchestrators" in config and isinstance(config["orchestrators"], list):
        return [entry or {} for entry in config["orchestrators"] if isinstance(entry, dict)]
    if isinstance(config.get("orchestrator"), list):
        return [entry or {} for entry in config["orchestrator"] if isinstance(entry, dict)]
    return [config.get("orchestrator", {})]
