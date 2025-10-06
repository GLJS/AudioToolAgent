"""
Voxtral API Tool
Transcribes audio using the Voxtral model via the Mistral API.
"""
import base64
import json
import logging
import os
from typing import Any, Dict, Optional

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.base import BaseTool
from mistralai import Mistral

from .audio_utils import cleanup_temp_file, validate_audio_file, resolve_audio_path


logger = logging.getLogger(__name__)


@register_tool('voxtral_api')
class VoxtralAPITool(BaseTool):
    """Transcribe audio using the Mistral Voxtral model via API."""

    parameters = [
        {
            'name': 'audio_path',
            'type': 'string',
            'description': 'Path to the audio file to transcribe',
            'required': True
        },
        {
            'name': 'prompt',
            'type': 'string',
            'description': 'Prompt for the transcription',
            'required': False
        }
    ]

    def __init__(self, model_name: str = "voxtral-small-latest", default_max_tokens: int = 4096, stream_callback: Optional[Any] = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.stream_callback = stream_callback
        self.client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
        logger.info(f"Initialized {self.__class__.__name__} with model={model_name}")

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
            logger.exception("Mistral Voxtral API call failed")
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
        if not validate_audio_file(params['audio_path']):
            return f"Invalid or missing audio file: {params['audio_path']}"
        return None

    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = params['audio_path']
        model_name = params.get('model_name', self.model_name)
        max_tokens = int(params.get('max_tokens', self.default_max_tokens))
        prompt = params.get('prompt', 'Transcribe the audio verbatim.')

        if self.stream_callback:
            self.stream_callback({
                'type': 'tool_progress',
                'tool': self.__class__.__name__,
                'content': f'Transcribing via Mistral Voxtral ({model_name})...',
                'full_content': f'Transcribing via Mistral Voxtral ({model_name})...'
            })

        if 'MISTRAL_API_KEY' not in os.environ:
            raise RuntimeError('MISTRAL_API_KEY environment variable is not set')

        # No segment extraction; use original file
        seg_path = audio_path
        is_temp = False

        with open(seg_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        chat_response = self.client.chat.complete(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio_base64},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens
        )


        # The SDK returns a string in message.content
        text = chat_response.choices[0].message.content
        cleanup_temp_file(seg_path, is_temp)
        return {
            'text': text,
            'message': f'Transcription complete ({len(text.split())} words)'
        }


