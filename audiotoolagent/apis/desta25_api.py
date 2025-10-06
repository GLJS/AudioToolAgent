import logging
from typing import Dict, Optional, Tuple

import torch
from desta import DeSTA25AudioModel
from desta.models.modeling_desta25 import GenerationOutput

from .common import create_app, mount_chat_completions, parse_messages_for_text_and_audio, save_base64_audio_to_temp, register_hostnames_entry

logger = logging.getLogger(__name__)

app = create_app("DeSTA2.5 OpenAI-Compatible API")
register_hostnames_entry(app, "DESTA25_SERVER", default_port=4004)

_model = None
_device = None


def _load_model(model_name: str):
    global _model, _device
    if _model is not None:
        return
    _device = "cuda"
    _model = DeSTA25AudioModel.from_pretrained(model_name)
    _model.to(_device)


def _process(payload: Dict) -> Tuple[str, Optional[str]]:
    torch.manual_seed(42)
    model_name = payload.get("model", "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    messages = payload.get("messages", [])
    _load_model(model_name)

    prompt, audio_b64, audio_fmt = parse_messages_for_text_and_audio(messages)
    if not audio_b64:
        raise ValueError("input_audio not found in messages")

    audio_path = save_base64_audio_to_temp(audio_b64, audio_fmt)

    desta_messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {
            "role": "user",
            "content": f"<|AUDIO|>\n{prompt or 'Describe the audio.'}",
            "audios": [{"audio": audio_path, "text": None}],
        },
    ]

    outputs = _model.generate(
        messages=desta_messages,
        do_sample=False,
        top_p=1.0,
        temperature=1.0,
        max_new_tokens=512,
    )
    if isinstance(outputs, list) or isinstance(outputs, GenerationOutput):
        text = outputs.text[0].strip()
    else:
        text = outputs.text.strip()
    return text, model_name


mount_chat_completions(app, _process)


@app.on_event("startup")
def load_model_on_startup():
    _load_model("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")


