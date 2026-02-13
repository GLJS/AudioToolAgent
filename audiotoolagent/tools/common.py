"""
Common base classes and utilities for audio tools
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Callable

from dotenv import load_dotenv
from qwen_agent.tools.base import BaseTool
from vllm import LLM

from .audio_utils import validate_audio_file, resolve_audio_path

load_dotenv()

logger = logging.getLogger(__name__)

# Get temp directory from environment
TMP_DIR = os.getenv('TMP_DIR', '/tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def _read_hostnames_map(hostnames_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        with open(hostnames_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    mapping[key.strip()] = value.strip()
                else:
                    mapping[line] = ''
    except Exception:
        pass
    return mapping


def get_server_url(env_var: str, default_url: str) -> str:
    """Resolve server URL from env or hostnames.txt (workspace root)."""
    url = os.getenv(env_var)
    if url:
        return url
    # hostnames.txt lives at repo root relative to this file
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    hostnames_path = os.path.join(root_dir, 'hostnames.txt')
    mapping = _read_hostnames_map(hostnames_path)
    url_from_file = mapping.get(env_var)
    return url_from_file if url_from_file else default_url


def get_server_urls(env_var: str, default_url: str) -> list:
    """Resolve server URL(s) from env or hostnames.txt. Supports comma-separated URLs for load balancing."""
    url = os.getenv(env_var)
    if url:
        return [u.strip() for u in url.split(',') if u.strip()]
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    hostnames_path = os.path.join(root_dir, 'hostnames.txt')
    mapping = _read_hostnames_map(hostnames_path)
    url_from_file = mapping.get(env_var)
    if url_from_file:
        return [u.strip() for u in url_from_file.split(',') if u.strip()]
    return [default_url]


class OfflineAudioModelTool(BaseTool, ABC):
    """Base class for offline VLLM audio model tools"""
    
    def __init__(self, 
                 model_name: str,
                 gpu_memory_utilization: float = 0.3,
                 max_model_len: int = 16384,
                 dtype: str = "auto",
                 stream_callback: Optional[Callable] = None,
                 additional_vllm_args: Optional[Dict[str, Any]] = None,
                 gpu_device: Optional[int] = None,
                 server_url: Optional[str] = None):
        """
        Initialize offline VLLM model tool
        
        Args:
            model_name: HuggingFace model name/path
            gpu_memory_utilization: GPU memory to use (0-1)
            max_model_len: Maximum model sequence length
            dtype: Model dtype (auto, float16, bfloat16)
            stream_callback: Optional streaming callback
            additional_vllm_args: Additional arguments for VLLM initialization
            gpu_device: GPU device ID to use (0, 1, etc.) - if None, uses CUDA_VISIBLE_DEVICES or default
        """
        super().__init__()
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.stream_callback = stream_callback
        self.additional_vllm_args = additional_vllm_args or {}
        self.gpu_device = gpu_device
        # If provided, use external OpenAI-compatible server instead of local LLM
        self.server_url: Optional[str] = server_url
        
        # Model state management
        self.llm = None
        
        gpu_info = f", gpu_device={gpu_device}" if gpu_device is not None else ""
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}, gpu_memory_utilization={gpu_memory_utilization}{gpu_info}")
    
    def _sleep(self, level: int = 1):
        """Put the model to sleep"""
        if self.llm is not None:
            self.llm.sleep(level=1)
        
    def _wake(self):
        """Wake the model up"""
        if self.llm is not None:
            self.llm.wake_up()
        
    
    def _initialize_model(self):
        """Initialize model on first use"""
        if self.llm is None:
            # Store original CUDA_VISIBLE_DEVICES
            original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # Set CUDA_VISIBLE_DEVICES if gpu_device is specified
            if self.gpu_device is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device)
                logger.info(f"Loading model {self.model_name} on GPU {self.gpu_device} with gpu_memory_utilization={self.gpu_memory_utilization}...")
            else:
                logger.info(f"Loading model {self.model_name} with gpu_memory_utilization={self.gpu_memory_utilization}...")
            
            # Base VLLM arguments
            vllm_args = {
                "model": self.model_name,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "dtype": self.dtype,
                "trust_remote_code": True,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
            }
            
            # Add model-specific arguments
            vllm_args.update(self.additional_vllm_args)
            
            # Initialize VLLM
            self.llm = LLM(**vllm_args)
            gpu_info = f" on GPU {self.gpu_device}" if self.gpu_device is not None else ""
            logger.info(f"Model {self.model_name} loaded successfully{gpu_info}")
            
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
        
        # Resolve audio_path if provided (supports short filenames via env roots)
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

        # self._wake()
        
        # Initialize model if not using external server
        if not getattr(self, 'server_url', None):
            self._initialize_model()

        
        # Process the request
        result = self._process_request(params)
        
        # Stream tool end notification
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_end",
                "tool": self.__class__.__name__,
                "result": result
            })
        
        # self._sleep(level=1)
        
        return json.dumps(result, ensure_ascii=False)
    
    @abstractmethod
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate input parameters
        
        Returns:
            Error message if validation fails, None if valid
        """
        pass
    
    @abstractmethod
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the actual request with the loaded model
        
        Returns:
            Dictionary with results
        """
        pass
    
    def set_stream_callback(self, callback: Optional[Callable] = None):
        """Set or update the streaming callback"""
        self.stream_callback = callback


class AudioTranscriptionModelTool(OfflineAudioModelTool):
    """Base class for audio transcription models"""
    
    description = 'Transcribe speech from audio files.'
    parameters = [
        {
            'name': 'audio_path',
            'type': 'string',
            'description': 'Path to the audio file to transcribe',
            'required': True
        },
    ]
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate transcription parameters"""
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        
        audio_path = params['audio_path']
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        
        return None
    
    def _extract_audio_if_needed(self, params: Dict[str, Any]) -> tuple[str, bool]:
        """No-op: always return original audio path (segmenting disabled)."""
        audio_path = params['audio_path']
        return audio_path, False


class AudioAnalysisModelTool(OfflineAudioModelTool):
    """Base class for audio analysis models"""
    
    description = 'Analyze audio content with a custom prompt. Can generate captions, answer questions, or perform any audio analysis task.'
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
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate analysis parameters"""
        if 'audio_path' not in params:
            return 'Missing required parameter: audio_path'
        
        if 'prompt' not in params:
            return 'Missing required parameter: prompt'
        
        audio_path = params['audio_path']
        if not validate_audio_file(audio_path):
            return f'Invalid or missing audio file: {audio_path}'
        
        return None
    
    def _extract_audio_if_needed(self, params: Dict[str, Any]) -> tuple[str, bool]:
        """No-op: always return original audio path (segmenting disabled)."""
        audio_path = params['audio_path']
        return audio_path, False


class ExternalAPITool(BaseTool, ABC):
    """Base class for external API-based audio tools (Google Gemini, OpenAI GPT-4o)"""
    
    def __init__(self, api_key: str, stream_callback: Optional[Callable] = None):
        """
        Initialize API-based tool
        
        Args:
            api_key: API key for the service
            stream_callback: Optional streaming callback
        """
        super().__init__()
        self.api_key = api_key
        self.stream_callback = stream_callback
        logger.info(f"Initialized {self.__class__.__name__} with API")
    
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
        
        # Resolve audio_path if provided (supports short filenames via env roots)
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
        
        # Process the request
        result = self._process_request(params)
        
        # Stream tool end notification
        if self.stream_callback:
            self.stream_callback({
                "type": "tool_end",
                "tool": self.__class__.__name__,
                "result": result
            })
        
        return json.dumps(result, ensure_ascii=False)
    
    def _extract_audio_if_needed(self, params: Dict[str, Any]) -> tuple[str, bool]:
        """No-op: always return original audio path (segmenting disabled)."""
        audio_path = params['audio_path']
        return audio_path, False
    
    @abstractmethod
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate input parameters"""
        pass
    
    @abstractmethod
    def _process_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the actual request"""
        pass
    
    def set_stream_callback(self, callback: Optional[Callable] = None):
        """Set or update the streaming callback"""
        self.stream_callback = callback