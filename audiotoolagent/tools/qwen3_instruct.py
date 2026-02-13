"""
Qwen3 Instruct Tool (via local vLLM OpenAI server)
"""
import base64
import logging
import os
import random
from typing import Any, Callable, Dict, Optional

from openai import OpenAI
from qwen_agent.tools.base import register_tool

from .audio_utils import cleanup_temp_file, validate_audio_file
from .common import AudioAnalysisModelTool, get_server_urls

logger = logging.getLogger(__name__)


@register_tool('qwen3_instruct')
class Qwen3InstructTool(AudioAnalysisModelTool):
    """Qwen3 Instruct for audio understanding and question answering via OpenAI-compatible local server"""

    description = 'Analyze audio or answer questions about audio using Qwen/Qwen3-Omni-30B-A3B-Instruct served by vLLM.'

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 4096,
        dtype: str = "auto",
        stream_callback: Optional[Callable] = None,
        gpu_device: Optional[int] = None,
        **kwargs: Any,
    ):
        additional_args = {"max_num_seqs": 4}
        server_urls = get_server_urls("QWEN3_INSTRUCT_SERVER", "http://0.0.0.0:4014/v1")
        super().__init__(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            stream_callback=stream_callback,
            additional_vllm_args=additional_args,
            gpu_device=gpu_device,
            server_url=server_urls[0]
        )
        self._server_urls = server_urls
        self._clients = [
            OpenAI(base_url=url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
            for url in server_urls
        ]
        self.client = self._clients[0]
        if len(server_urls) > 1:
            logger.info(f"Qwen3 Instruct load-balancing across {len(server_urls)} servers: {server_urls}")

    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        if not validate_audio_file(params['audio_path']):
            return f"Invalid or missing audio file: {params['audio_path']}"
        task = params.get('task', 'asr')
        if task not in ('asr', 'analysis'):
            return 'Invalid task. Must be "asr" or "analysis"'
        if task == 'analysis' and not params.get('prompt'):
            return 'Missing required parameter: prompt (for analysis)'
        return None

    def _extract_audio_if_needed(self, params: Dict[str, Any]) -> tuple[str, bool]:
        return params['audio_path'], False

    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audio_path, is_temp = self._extract_audio_if_needed(params)
        task = params.get('task', 'asr')
        prompt = params.get('prompt', 'Describe this audio.') if task == 'asr' else params.get('prompt', '')

        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": "Running Qwen3 Instruct...",
                "full_content": "Running Qwen3 Instruct...",
            })

        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        user_text = prompt if task == 'analysis' else "Please describe and caption the following audio."
        client = random.choice(self._clients)
        chat_completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.2 if task == 'asr' else 0.7,
        )
        response_text = (chat_completion.choices[0].message.content or "").strip()

        if self.stream_callback and response_text:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": response_text,
                "full_content": response_text,
            })
        cleanup_temp_file(audio_path, is_temp)
        return {"result": response_text, "message": f"Qwen3 Instruct {task} complete"}
