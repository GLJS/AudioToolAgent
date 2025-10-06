"""
Qwen2.5-Omni Audio Analysis Tool
Uses offline VLLM for audio understanding with Qwen2.5-Omni model
"""
import logging
import os
import base64
from typing import Dict, Any, Optional, Callable

from qwen_agent.tools.base import register_tool
from openai import OpenAI

from .common import AudioAnalysisModelTool
from .audio_utils import cleanup_temp_file
from .common import get_server_url

logger = logging.getLogger(__name__)


@register_tool('qwen2_5omni')
class Qwen25OmniTool(AudioAnalysisModelTool):
    """Qwen2.5-Omni for audio analysis using offline VLLM"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-Omni-7B",
                 gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 28672,
                 dtype: str = "auto",
                 stream_callback: Optional[Callable] = None,
                 gpu_device: Optional[int] = None):
        """Initialize Qwen2.5-Omni tool"""
        
        # Additional VLLM args for audio support
        additional_args = {
            "max_num_seqs": 5,
            # limit_mm_per_prompt removed
        }
        
        super().__init__(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            stream_callback=stream_callback,
            additional_vllm_args=additional_args,
            gpu_device=gpu_device,
            server_url=get_server_url("QWEN25_OMNI_SERVER", "http://0.0.0.0:4002/v1")
        )
        # OpenAI-compatible HTTP client for local vLLM server
        self.client = OpenAI(base_url=self.server_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
    
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio analysis request with Qwen2.5-Omni"""
        # Extract audio segment if needed
        audio_path, is_temp = self._extract_audio_if_needed(params)
        
        # Get prompt
        question = params['prompt']
        
        # Use pre-initialized OpenAI-compatible client
        
        # Stream progress if callback available
        if self.stream_callback:
                self.stream_callback({
                    "type": "tool_progress",
                    "tool": self.__class__.__name__,
                    "content": "Processing audio with Qwen2.5-Omni...",
                    "full_content": "Processing audio with Qwen2.5-Omni..."
                })
            
        # Read audio and encode as base64
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_base64, "format": "wav"},
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.7,
        )
        response_text = (chat_completion.choices[0].message.content or "").strip()
        
        # Stream the complete response
        if self.stream_callback:
                self.stream_callback({
                    "type": "tool_progress",
                    "tool": self.__class__.__name__,
                    "content": response_text,
                    "full_content": response_text
                })
            
        logger.info("Successfully analyzed audio with Qwen2.5-Omni")
        
        result = {
            'result': response_text,
            'message': 'Audio analysis complete with Qwen2.5-Omni'
        }
        
        # Clean up temporary file if created
        cleanup_temp_file(audio_path, is_temp)
        
        return result