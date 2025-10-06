"""
DeSTA2.5 Audio Analysis Tool (via local vLLM OpenAI server)
"""
import logging
import os
import base64
from typing import Dict, Any, Optional, Callable

from qwen_agent.tools.base import register_tool
from openai import OpenAI

from .common import AudioAnalysisModelTool, get_server_url
from .audio_utils import cleanup_temp_file, validate_audio_file

logger = logging.getLogger(__name__)


@register_tool('desta25')
class DeSTA25Tool(AudioAnalysisModelTool):
    """DeSTA2.5 for audio analysis using local vLLM server"""

    description = 'Analyze audio content using DeSTA2.5-Audio-Llama model via OpenAI-compatible endpoint.'

    def __init__(self,
                 model_name: str = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
                 gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 4096,
                 dtype: str = "auto",
                 stream_callback: Optional[Callable] = None,
                 gpu_device: Optional[int] = None, **kwargs):
        additional_args = {"max_num_seqs": 2}
        super().__init__(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            stream_callback=stream_callback,
            additional_vllm_args=additional_args,
            gpu_device=gpu_device,
            server_url=get_server_url("DESTA25_SERVER", "http://0.0.0.0:4004/v1")
        )
        self.client = OpenAI(base_url=self.server_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        if not validate_audio_file(params['audio_path']):
            return f"Invalid or missing audio file: {params['audio_path']}"
        return None

    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audio_path, is_temp = self._extract_audio_if_needed(params)
        question = params['prompt']

        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": "Processing audio with DeSTA2.5...",
                "full_content": "Processing audio with DeSTA2.5..."
            })

        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        response_text = (chat_completion.choices[0].message.content or "").strip()

        if self.stream_callback and response_text:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": response_text,
                "full_content": response_text
            })

        cleanup_temp_file(audio_path, is_temp)
        return {'result': response_text, 'message': 'Audio analysis complete with DeSTA2.5'}
