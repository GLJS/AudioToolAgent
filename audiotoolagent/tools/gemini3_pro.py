"""
Google Gemini 3 Pro Audio Tool
Uses Google's Gemini 3 Pro API with native audio support
"""
import logging
import os
from typing import Any, Callable, Dict, Optional

from google import genai
from google.genai import types
from qwen_agent.tools.base import register_tool

from .audio_utils import cleanup_temp_file, validate_audio_file
from .common import ExternalAPITool

logger = logging.getLogger(__name__)


@register_tool('gemini3_pro')
class Gemini3ProAudioTool(ExternalAPITool):
    """Google Gemini 3 Pro for audio transcription and analysis"""

    description = 'Process audio using Google Gemini 3 Pro. Can transcribe speech with timestamps or analyze audio content based on your prompt.'
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
            'description': 'Prompt for the model (e.g., "Transcribe this audio with timestamps" or "What sounds are in this audio?")',
            'required': True
        },
        {
            'name': 'task_type',
            'type': 'string',
            'description': 'Type of task: "transcription" or "analysis"',
            'required': False
        },
    ]

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-pro-preview", stream_callback: Optional[Callable] = None, **kwargs):
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")

        super().__init__(api_key=api_key, stream_callback=stream_callback)
        self.client = genai.Client(api_key=self.api_key)
        self.model = model_name
        logger.info(f"Initialized Gemini 3 Pro {model_name} for audio processing")

    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        audio_path, is_temp = self._extract_audio_if_needed(params)
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        return None

    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params['prompt']
        task_type = params.get('task_type', 'analysis')
        audio_path, is_temp = self._extract_audio_if_needed(params)

        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": f"Processing audio with Gemini 3 Pro {self.model} ({task_type})...",
                "full_content": f"Processing audio with Gemini 3 Pro {self.model} ({task_type})..."
            })

        if task_type == "transcription" or "transcrib" in prompt.lower():
            if "timestamp" not in prompt.lower():
                prompt = prompt + " Please include timestamps for each segment of speech."

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(mime_type="audio/x-wav", data=audio_data),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            max_output_tokens=4096
        )

        full_text = ""
        if self.stream_callback:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=generate_content_config,
            ):
                if chunk.text:
                    full_text += chunk.text
                    self.stream_callback({
                        "type": "tool_progress",
                        "tool": self.__class__.__name__,
                        "content": chunk.text,
                        "full_content": full_text
                    })
            response_text = full_text
        else:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=generate_content_config,
            )
            try:
                response_text = response.text
            except Exception as e:
                logger.error(f"Gemini 3 Pro Error generating content: {e}")
                logger.error(f"Gemini 3 Pro Response: {response}")
                raise Exception(f"Error generating content: {e}")

        logger.info(f"Successfully processed audio with Gemini 3 Pro {self.model}")

        if task_type == "transcription":
            result = {'text': response_text, 'message': f'Transcription complete: {len(response_text.split())} words'}
        else:
            result = {'result': response_text, 'message': f'Audio analysis complete with Gemini 3 Pro {self.model}'}

        cleanup_temp_file(audio_path, is_temp)
        return result
