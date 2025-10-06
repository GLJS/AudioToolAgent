"""
AudioFlamingo3 Analysis Tool
Uses VILA/llava framework for audio understanding with AudioFlamingo3 model
"""
import json
import logging
import os
from typing import Dict, Any, Optional, Callable, Union

from qwen_agent.tools.base import register_tool, BaseTool
from huggingface_hub import snapshot_download
import torch

from .audio_utils import cleanup_temp_file, validate_audio_file, resolve_audio_path

logger = logging.getLogger(__name__)


@register_tool('audioflamingo3')
class AudioFlamingo3Tool(BaseTool):
    """AudioFlamingo3 for audio analysis using VILA/llava framework"""
    
    description = 'Analyze audio content and generate captions using AudioFlamingo3 multimodal model. Supports audio understanding and captioning tasks.'
    parameters = [
        {
            'name': 'audio_path',
            'type': 'string',
            'description': 'Path to the audio file to analyze',
            'required': True
        },
        {
            'name': 'prompt',
            'type': 'string',
            'description': 'Custom prompt or question for the analysis',
            'required': True
        },
    ]
    
    def __init__(self, 
                 model_name: str = "nvidia/audio-flamingo-3",
                 conv_mode: str = "auto",
                 stream_callback: Optional[Callable] = None,
                 gpu_device: Optional[int] = None,
                 **kwargs):
        """
        Initialize AudioFlamingo3 tool
        
        Args:
            model_name: HuggingFace model name/path
            conv_mode: Conversation mode for llava
            stream_callback: Optional streaming callback
            gpu_device: GPU device ID to use (0, 1, etc.)
        """
        super().__init__()
        self.model_name = model_name
        self.conv_mode = conv_mode
        self.stream_callback = stream_callback
        self.gpu_device = gpu_device
        
        # Model state management
        self.model = None
        self.model_path = None
        self.clib = None
        
        gpu_info = f", gpu_device={gpu_device}" if gpu_device is not None else ""
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}{gpu_info}")
    
    def _initialize_model(self):
        """Initialize model on first use"""
        if self.model is None:
            # Store original CUDA_VISIBLE_DEVICES
            original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # Set CUDA_VISIBLE_DEVICES if gpu_device is specified
            if self.gpu_device is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device)
                logger.info(f"Loading model {self.model_name} on GPU {self.gpu_device}...")
            else:
                logger.info(f"Loading model {self.model_name}...")
            
            try:
                # Import llava modules (lazy import to avoid issues if not installed)
                import llava
                from llava import conversation as clib
                from llava.media import Sound
                
                self.clib = clib
                self.Sound = Sound
                
                # Download model if needed
                self.model_path = snapshot_download(self.model_name)
                
                # Check for stage35 model (thinking mode)
                model_think_path = os.path.join(self.model_path, 'stage35')
                if os.path.exists(model_think_path):
                    logger.info(f"Found stage35 thinking model at {model_think_path}")
                
                # Load the main model
                self.model = llava.load(self.model_path)
                self.model = self.model.to("cuda")
                
                # Set conversation mode
                clib.default_conversation = clib.conv_templates[self.conv_mode].copy()
                
                gpu_info = f" on GPU {self.gpu_device}" if self.gpu_device is not None else ""
                logger.info(f"Model {self.model_name} loaded successfully{gpu_info}")
                
            except ImportError as e:
                logger.error(f"Failed to import llava modules: {e}")
                raise RuntimeError("llava module not installed. Please install it to use AudioFlamingo3Tool")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_devices is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Main entry point for tool calls
        
        Args:
            params: Tool parameters as JSON string or dict
            
        Returns:
            JSON string with results
        """
        # Parse parameters if string
        if isinstance(params, str):
            params = json.loads(params)
        
        # Resolve short filename via env roots before validation
        if isinstance(params, dict) and 'audio_path' in params:
            try:
                params['audio_path'] = resolve_audio_path(params['audio_path'])
            except Exception:
                pass

        # Validate required parameters
        validation_error = self._validate_params(params)
        if validation_error:
            return json.dumps({'error': validation_error})
        
        # Stream tool start notification
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_start",
                "tool": self.__class__.__name__,
                "params": params
            })
        
        try:
            # Initialize model if not already loaded
            self._initialize_model()
            
            # Process the request
            result = self._process_request(params)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            result = {'error': str(e)}
        
        # Stream tool end notification
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_end",
                "tool": self.__class__.__name__,
                "result": result
            })
        
        return json.dumps(result, ensure_ascii=False)
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate input parameters"""
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        
        audio_path = params['audio_path']
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        
        # Check if file has supported extension
        supported_extensions = ['.wav', '.mp3', '.flac']
        if not any(audio_path.lower().endswith(ext) for ext in supported_extensions):
            return f'Unsupported audio format. Supported formats: {", ".join(supported_extensions)}'
        
        return None
    
    def _extract_audio_if_needed(self, params: Dict[str, Any]) -> tuple[str, bool]:
        """No-op: always return original audio path (segmenting disabled)."""
        audio_path = params['audio_path']
        return audio_path, False
    
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio analysis request with AudioFlamingo3"""
        # Extract audio segment if needed
        audio_path, is_temp = self._extract_audio_if_needed(params)
        question = params['prompt']
        
        # Stream progress if callback available
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_progress",
                "tool": self.__class__.__name__,
                "content": "Processing audio with AudioFlamingo3...",
                "full_content": "Processing audio with AudioFlamingo3..."
            })
        
        try:
            # Create Sound object for audio
            audio_media = self.Sound(audio_path)
            
            # Prepare multi-modal prompt
            prompt = [audio_media, question]
            
            # Generate response
            response_text = self.model.generate_content(prompt)
            
            # Stream the complete response
            if self.stream_callback:
                self.stream_callback({
                    "type": "tool_progress",
                    "tool": self.__class__.__name__,
                    "content": response_text,
                    "full_content": response_text
                })
            
            logger.info("Successfully analyzed audio with AudioFlamingo3")
            
            result = {
                'result': response_text,
                'message': 'Audio analysis complete with AudioFlamingo3'
            }
            
        except Exception as e:
            logger.error(f"Error during audio processing: {e}")
            result = {
                'error': str(e),
                'message': 'Failed to analyze audio with AudioFlamingo3'
            }
        finally:
            # Clean up temporary file if created
            cleanup_temp_file(audio_path, is_temp)
        
        return result
    
    def set_stream_callback(self, callback: Optional[Callable] = None):
        """Set or update the streaming callback"""
        self.stream_callback = callback