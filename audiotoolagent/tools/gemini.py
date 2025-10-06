"""
Google Gemini 2.5 Audio Tool
Uses Google's Gemini API with native audio support
"""
import os
import base64
from typing import Dict, Any, Optional, Callable
from qwen_agent.tools.base import register_tool
from .common import ExternalAPITool
from .audio_utils import cleanup_temp_file, validate_audio_file
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@register_tool('gemini')
class GeminiAudioTool(ExternalAPITool):
    """Google Gemini for audio transcription and analysis"""
    
    description = 'Process audio using Google Gemini 2.5 Flash. Can transcribe speech with timestamps or analyze audio content based on your prompt.'
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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", stream_callback: Optional[Callable] = None, **kwargs):
        """Initialize Gemini tool"""
        # Get API key from env if not provided
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
        
        super().__init__(api_key=api_key, stream_callback=stream_callback)
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        self.model = model_name
        
        logger.info(f"Initialized Gemini {model_name} for audio processing")
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters"""
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        
        # Extract audio segment if needed
        audio_path = params['audio_path']
        audio_path, is_temp = self._extract_audio_if_needed(params)
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        
        return None
    
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio with Gemini API"""
        audio_path = params['audio_path']
        prompt = params['prompt']
        task_type = params.get('task_type', 'analysis')
        # Optionally segment based on gating
        audio_path, is_temp = self._extract_audio_if_needed(params)
        
        # Read audio file
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Stream progress if callback available
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": f"Processing audio with Gemini {self.model} ({task_type})...",
                "full_content": f"Processing audio with Gemini {self.model} ({task_type})..."
            })
        
        # For transcription tasks, enhance the prompt to request timestamps
        if task_type == "transcription" or "transcrib" in prompt.lower():
            if "timestamp" not in prompt.lower():
                prompt = prompt + " Please include timestamps for each segment of speech."
        
        # Create content with audio
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="audio/x-wav",
                        data=audio_data,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024,
            ),
            max_output_tokens=4096
        )
        
        # Generate response with streaming if callback available
        full_text = ""
        if self.stream_callback:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
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
            # Non-streaming response
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )
            try:
                response_text = response.text
            except Exception as e:
                logger.error(f"Gemini Error generating content: {e}")
                logger.error(f"Gemini Response: {response}")
                raise Exception(f"Error generating content: {e}")
        
        logger.info(f"Successfully processed audio with Gemini {self.model}")
        
        # Format response based on task type
        if task_type == "transcription":
            result = {
                'text': response_text,
                'message': f'Transcription complete: {len(response_text.split())} words'
            }
        else:
            result = {
                'result': response_text,
                'message': f'Audio analysis complete with Gemini {self.model}'
            }
        
        # Clean up temporary file if created
        cleanup_temp_file(audio_path, is_temp)
        return result