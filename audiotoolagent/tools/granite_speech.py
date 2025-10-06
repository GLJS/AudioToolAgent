"""
Granite Speech Transcription Tool
Uses offline VLLM for speech-to-text with IBM's Granite speech model
"""
import logging
import os
import base64
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from qwen_agent.tools.base import register_tool
from openai import OpenAI

from .common import AudioTranscriptionModelTool
from .audio_utils import cleanup_temp_file
from .common import get_server_url

logger = logging.getLogger(__name__)


@register_tool('granite_speech')
class GraniteSpeechTool(AudioTranscriptionModelTool):
    """Granite Speech for audio transcription using offline VLLM"""
    
    def __init__(self, 
                 model_name: str = "ibm-granite/granite-speech-3.3-8b",
                 gpu_memory_utilization: float = 0.35,
                 max_model_len: int = 32768,
                 dtype: str = "auto",
                 stream_callback: Optional[Callable] = None,
                 gpu_device: Optional[int] = None):
        """Initialize Granite Speech tool"""
        
        # Additional VLLM args for Granite (used by server startup script)
        additional_args = {
            "trust_remote_code": True,
            "max_num_seqs": 2,
            "enable_lora": True,
            "max_lora_rank": 64,
        }
        
        super().__init__(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            stream_callback=stream_callback,
            additional_vllm_args=additional_args,
            gpu_device=gpu_device,
            server_url=get_server_url("GRANITE_SPEECH_SERVER", "http://127.0.0.1:4000/v1")
        )
        # OpenAI-compatible HTTP client for local vLLM server
        self.client = OpenAI(base_url=self.server_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
    
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process transcription request with Granite Speech"""
        # Extract audio segment if needed
        audio_path, is_temp = self._extract_audio_if_needed(params)

        # Format prompt for Granite Speech
        # The model has an audio-specific LoRA; server is launched with it
        date_string = datetime.now().strftime("%B %d, %Y")
        system_prompt = (
            f"Knowledge Cutoff Date: April 2024.\n"
            f"Today's Date: {date_string}.\n"
            f"You are Granite, developed by IBM. Transcribe the user's audio accurately."
        )

        # Stream progress if callback available
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": "Transcribing audio with Granite Speech...",
                "full_content": "Transcribing audio with Granite Speech..."
            })

        # Read audio and encode as base64
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio."},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_base64, "format": "wav"},
                        },
                    ],
                },
            ],
            max_tokens=28672,
            temperature=0.0,
        )
        transcription = (chat_completion.choices[0].message.content or "").strip()
        
        # Stream the transcription
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": transcription,
                "full_content": transcription
            })
        
        logger.info(f"Successfully transcribed audio with Granite Speech: {len(transcription.split())} words")
        
        # Clean up temporary file if created
        cleanup_temp_file(audio_path, is_temp)
        
        return {
            'text': transcription,
            'message': f'Transcription complete: {len(transcription.split())} words'
        }
            
