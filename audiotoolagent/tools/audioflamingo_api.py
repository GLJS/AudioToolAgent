"""
AudioFlamingo API Tool
Simple API-based wrapper that calls the AudioFlamingo FastAPI server.
Opens the local audio file, converts it to base64, and sends it to the
OpenAI-compatible chat completions endpoint.
"""
import json
import logging
import base64
import os
from typing import Dict, Any, Optional, Callable

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.base import BaseTool
from openai import OpenAI

from .common import get_server_url
from .audio_utils import cleanup_temp_file, validate_audio_file, resolve_audio_path


logger = logging.getLogger(__name__)


@register_tool('audioflamingo_api')
class AudioFlamingoAPITool(BaseTool):
    """AudioFlamingo API tool that calls a FastAPI server endpoint."""

    description = 'Analyze audio or transcribe speech using the AudioFlamingo FastAPI server.'
    parameters = [
        {
            'name': 'audio_path',
            'type': 'string',
            'description': 'Path to the audio file to process',
            'required': True
        },
        {
            'name': 'prompt',
            'type': 'string',
            'description': 'Prompt or question for the model',
            'required': True
        },
    ]

    def __init__(self, 
                 model_name: str = "nvidia/audio-flamingo-3",
                 stream_callback: Optional[Callable] = None,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.stream_callback = stream_callback
        self.server_url = get_server_url("AF3_SERVER", "http://0.0.0.0:4010/v1")
        self.client = OpenAI(base_url=self.server_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        print(f"Initializing {self.__class__.__name__} with server_url={self.server_url}")
        logger.info(f"Initialized {self.__class__.__name__} with server_url={self.server_url}")

    def call(self, params: Any, **kwargs) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                return json.dumps({'error': 'Invalid JSON params'})

        # Resolve short filename via env roots before validation
        if isinstance(params, dict) and 'audio_path' in params:
            try:
                params['audio_path'] = resolve_audio_path(params['audio_path'])
            except Exception:
                pass

        validation_error = self._validate_params(params)
        if validation_error:
            return json.dumps({'error': validation_error})

        if self.stream_callback:
            self.stream_callback({
                'type': 'tool_start',
                'tool': self.__class__.__name__,
                'params': params,
            })

        try:
            result = self._process_request(params)
        except Exception as e:
            logger.exception("AudioFlamingo API call failed")
            result = {'error': str(e)}

        if self.stream_callback:
            self.stream_callback({
                'type': 'tool_end',
                'tool': self.__class__.__name__,
                'result': result,
            })

        return json.dumps(result, ensure_ascii=False)

    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        audio_path = params['audio_path']
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        return None

    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = params['audio_path']
        prompt = params['prompt']

        if self.stream_callback:
            self.stream_callback({
                'type': 'tool_progress',
                'tool': self.__class__.__name__,
                'content': 'Processing audio with AudioFlamingo...',
                'full_content': 'Processing audio with AudioFlamingo...'
            })

        # Convert audio to base64 (optionally segment first)
        seg_path = audio_path
        is_temp = False
        with open(seg_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}},
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        response_text = (chat_completion.choices[0].message.content or "").strip()

        if self.stream_callback and response_text:
            self.stream_callback({
                'type': 'tool_progress',
                'tool': self.__class__.__name__,
                'content': response_text,
                'full_content': response_text
            })

        cleanup_temp_file(seg_path, is_temp)
        return {'result': response_text, 'message': 'AudioFlamingo API call complete'}


