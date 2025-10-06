import logging
import os
from typing import Dict, Optional, Tuple

import torch
import llava
from huggingface_hub import snapshot_download

from audiotoolagent.apis.common import (
    create_app,
    mount_chat_completions, 
    parse_messages_for_text_and_audio,
    save_base64_audio_to_temp,
    register_hostnames_entry
)

logger = logging.getLogger(__name__)

app = create_app("Audio Flamingo 3 OpenAI-Compatible API")
register_hostnames_entry(app, "AF3_SERVER", default_port=4010)

_model = None
_generation_config = None

def _load_model(model_name: str):
    global _model, _generation_config
    if _model is not None:
        return
    
    logger.info("Loading Audio Flamingo 3 model...")
    MODEL_BASE = snapshot_download(repo_id="nvidia/audio-flamingo-3")
    
    _model = llava.load(MODEL_BASE, model_base=None)
    _model = _model.to("cuda")
    _generation_config = _model.default_generation_config
    _generation_config.max_new_tokens = 2048
    logger.info("Model loaded successfully")


def _process(payload: Dict) -> Tuple[str, Optional[str]]:
    torch.manual_seed(42)
    model_name = payload.get("model", "nvidia/audio-flamingo-3")
    messages = payload.get("messages", [])
    _load_model(model_name)
    
    prompt, audio_b64, audio_fmt = parse_messages_for_text_and_audio(messages)
    if not audio_b64:
        raise ValueError("input_audio not found in messages")
    
    audio_path = save_base64_audio_to_temp(audio_b64, audio_fmt)
    sound = llava.Sound(audio_path)
    full_prompt = f"<sound>\n{prompt or 'Describe the audio.'}"
    
    response = _model.generate_content([sound, full_prompt], generation_config=_generation_config)
    return response, model_name

mount_chat_completions(app, _process)


@app.on_event("startup")
def load_model_on_startup():
    _load_model("nvidia/audio-flamingo-3")
