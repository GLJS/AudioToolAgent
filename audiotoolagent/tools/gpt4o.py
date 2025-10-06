"""
OpenAI GPT-4o Audio Tool
Uses OpenAI's GPT-4o API for both audio transcription and analysis
"""
import os
import base64
from typing import Dict, Any, Optional, Callable
from qwen_agent.tools.base import register_tool
from .common import ExternalAPITool
from .audio_utils import cleanup_temp_file, validate_audio_file
import logging

logger = logging.getLogger(__name__)


@register_tool('gpt4o')
class GPT4oAudioTool(ExternalAPITool):
    """OpenAI GPT-4o for audio transcription and analysis"""
    
    description = 'Process audio using OpenAI GPT-4o. Can transcribe speech or analyze audio content based on your prompt.'
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
            'description': 'Prompt for the model (e.g., "Transcribe this audio" or "What is happening in this audio?")',
            'required': True
        },
        {
            'name': 'task_type',
            'type': 'string',
            'description': 'Type of task: "transcription" or "analysis"',
            'required': False
        },
    ]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-audio-preview", stream_callback: Optional[Callable] = None, **kwargs):
        """Initialize GPT-4o tool"""
        # Get API key from env if not provided
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        super().__init__(api_key=api_key, stream_callback=stream_callback)
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        logger.info(f"Initialized GPT-4o {self.model} for audio processing")
    
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
        """Process audio with GPT-4o API"""
        audio_path = params['audio_path']
        prompt = params['prompt']
        task_type = params.get('task_type', 'analysis')
        # No segment extraction
        audio_path, is_temp = self._extract_audio_if_needed(params)
        
        # For pure transcription, use Whisper API
        if task_type == "transcription":
            if self.stream_callback:
                    self.stream_callback({
                        "type": "tool_progress",
                        "tool": self.__class__.__name__,
                        "content": "Transcribing audio with Whisper...",
                        "full_content": "Transcribing audio with Whisper..."
                    })
                
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model.replace("audio-preview", "transcribe"),
                    file=audio_file
                )
            
            response_text = transcript.text
            
            if self.stream_callback:
                    self.stream_callback({
                        "type": "tool_progress",
                        "tool": self.__class__.__name__,
                        "content": response_text,
                        "full_content": response_text
                    })
                
            logger.info("Successfully transcribed audio with Whisper")
            
            result = {
                'text': response_text,
                'message': f'Transcription complete: {len(response_text.split())} words'
            }
        
        # For analysis or complex tasks, use GPT-4o with audio
        else:
            # Read audio file and encode to base64
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Stream progress if callback available
            if self.stream_callback:
                    self.stream_callback({
                        "type": "tool_progress",
                        "tool": self.__class__.__name__,
                        "content": f"Processing audio with GPT-4o {self.model} ({task_type})...",
                        "full_content": f"Processing audio with GPT-4o {self.model} ({task_type})..."
                    })
                
            # Create messages with audio
            messages = [
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing and understanding audio content."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ]
                
            # Generate response with streaming if callback available
            if self.stream_callback:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=4096
                )
                
                full_text = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        full_text += delta
                        self.stream_callback({
                            "type": "tool_progress",
                            "tool": self.__class__.__name__,
                            "content": delta,
                            "full_content": full_text
                        })
                
                response_text = full_text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                response_text = response.choices[0].message.content
            
            logger.info("Successfully processed audio with GPT-4o")
            
            result = {
                'result': response_text,
                'message': 'Audio analysis complete with GPT-4o'
            }
        # Clean up temporary file if created
        cleanup_temp_file(audio_path, is_temp)
        return result