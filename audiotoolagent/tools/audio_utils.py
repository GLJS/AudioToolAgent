"""
Audio utility functions for processing and segment extraction
"""
import os
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get temp directory from environment
TMP_DIR = os.getenv('TMP_DIR', '/tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def get_audio_roots() -> List[str]:
    """
    Return a list of candidate root directories for audio files, derived from
    environment variables in priority order.
    """
    roots: List[str] = []
    for env_name in ("AUDIO_ROOT", "AUDIO_DATA_ROOT", "MMAR_AUDIO_ROOT"):
        root = os.getenv(env_name)
        if root:
            roots.append(root)
    return roots


def resolve_audio_path(path_or_name: str) -> str:
    """
    Resolve a possibly-short audio file name to a full path using env roots.

    Behavior:
    - If input is an absolute existing path, return as-is.
    - Otherwise, try joining with AUDIO_ROOT, then AUDIO_DATA_ROOT, then
      MMAR_AUDIO_ROOT. Return the first existing path.
    - If none exist, return the input unchanged.
    """
    candidate = str(path_or_name)
    try:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        for root in get_audio_roots():
            joined = os.path.join(root, candidate)
            if os.path.exists(joined):
                return joined
    except Exception:
        # Fall through to return the original
        pass
    return candidate

def cleanup_temp_file(file_path: str, is_temporary: bool):
    """
    Clean up temporary file if needed.
    
    Args:
        file_path: Path to the file
        is_temporary: Whether the file is temporary and should be deleted
    """
    if is_temporary and os.path.exists(file_path):
        os.unlink(file_path)
        logger.debug(f"Cleaned up temporary file: {file_path}")


def ensure_mono(input_path: str) -> Tuple[str, bool]:
    """
    Ensure the audio file is mono. If stereo/multi-channel, downmix to mono and
    write to a temporary file alongside the original.

    Returns (path, is_temporary). If conversion is not needed, returns the
    original path with is_temporary=False.
    """
    try:
        info = sf.info(input_path)
        if getattr(info, "channels", 1) == 1:
            return input_path, False
    except Exception:
        # If we cannot read info, fallback to ffmpeg conversion to mono
        pass

    # Convert to mono using soundfile if possible; fall back to ffmpeg
    try:
        data, samplerate = sf.read(input_path, always_2d=True)
        # data shape: (num_frames, num_channels)
        if data.shape[1] == 1:
            return input_path, False
        mono = np.mean(data, axis=1)
        suffix = Path(input_path).suffix or ".wav"
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="audio_mono_", dir=TMP_DIR)
        os.close(temp_fd)
        sf.write(temp_path, mono, samplerate)
        return temp_path, True
    except Exception as e:
        logger.warning(f"soundfile mono conversion failed, trying ffmpeg: {e}")

    # Fallback: ffmpeg downmix
    suffix = Path(input_path).suffix
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="audio_mono_", dir=TMP_DIR)
    os.close(temp_fd)
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ac', '1',  # force mono
        '-y', temp_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return temp_path, True
    else:
        logger.error(f"Failed to convert to mono: {result.stderr}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return input_path, False

def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that the audio file exists and is readable.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        True if file is valid, False otherwise
    """
    # Resolve short filenames via env roots as a safety net
    try:
        resolved = resolve_audio_path(audio_path)
    except Exception:
        resolved = audio_path
    if not os.path.exists(resolved):
        logger.error(f"Audio file not found: {audio_path}")
        return False
        
    # Try to read audio info
    sf.info(resolved)
    return True