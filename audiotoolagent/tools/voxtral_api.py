"""
Voxtral API Tool
Transcribes audio using the Voxtral model via OpenRouter API.
"""
import base64
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.base import BaseTool
from openai import OpenAI

from .audio_utils import cleanup_temp_file, validate_audio_file, resolve_audio_path


logger = logging.getLogger(__name__)


@register_tool('voxtral_api')
class VoxtralAPITool(BaseTool):
    """Transcribe audio using the Mistral Voxtral model via OpenRouter."""

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

    def __init__(self, model_name: str = "mistralai/voxtral-small-24b-2507", default_max_tokens: int = 4096, stream_callback: Optional[Any] = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.stream_callback = stream_callback
        self.client = OpenAI(
            api_key=os.environ['OPENROUTER_API_KEY'],
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info(f"Initialized {self.__class__.__name__} with model={model_name} via OpenRouter")

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
            logger.exception("Voxtral API call failed")
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
                'content': f'Transcribing via Voxtral ({model_name}) on OpenRouter...',
                'full_content': f'Transcribing via Voxtral ({model_name}) on OpenRouter...'
            })

        if 'OPENROUTER_API_KEY' not in os.environ:
            raise RuntimeError('OPENROUTER_API_KEY environment variable is not set')

        seg_path = audio_path
        is_temp = False

        with open(seg_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Retry loop with exponential backoff
        chat_response = None
        for attempt in range(10):
            try:
                chat_response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"Voxtral API attempt {attempt+1}/10 failed: {e} (retry in {wait_time}s)")
                if attempt < 9:
                    time.sleep(wait_time)
                else:
                    raise

        text = chat_response.choices[0].message.content
        cleanup_temp_file(seg_path, is_temp)
        return {
            'text': text,
            'message': f'Transcription complete ({len(text.split())} words)'
        }
