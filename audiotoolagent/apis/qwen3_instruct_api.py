import logging
import os
from typing import Dict, Optional, Tuple

import torch
import soundfile as sf

# Monkey-patch transformers bug: Qwen3OmniMoeTalkerCodePredictorConfig missing use_sliding_window
try:
    from transformers.models.qwen3_omni_moe import configuration_qwen3_omni_moe as _qom_cfg
    _OrigCodePredCfg = _qom_cfg.Qwen3OmniMoeTalkerCodePredictorConfig
    _orig_cp_init = _OrigCodePredCfg.__init__

    def _patched_cp_init(self, *args, use_sliding_window=False, **kwargs):
        self.use_sliding_window = use_sliding_window
        _orig_cp_init(self, *args, **kwargs)

    _OrigCodePredCfg.__init__ = _patched_cp_init
except Exception:
    pass

from .common import create_app, mount_chat_completions, parse_messages_for_text_and_audio, save_base64_audio_to_temp, register_hostnames_entry

logger = logging.getLogger(__name__)

app = create_app("Qwen3 Instruct OpenAI-Compatible API")
register_hostnames_entry(app, "QWEN3_INSTRUCT_SERVER", default_port=4014)

_model = None
_processor = None


def _load_model(model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"):
    global _model, _processor
    if _model is not None:
        return

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    logger.info(f"Loading Qwen3 Instruct model from {model_name}...")

    _processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)

    _model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    _model.disable_talker()
    _model.eval()
    logger.info("Qwen3 Instruct model loaded successfully (talker disabled, text-only mode)")


def _process(payload: Dict) -> Tuple[str, Optional[str]]:
    torch.manual_seed(42)
    model_name = payload.get("model", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    messages = payload.get("messages", [])
    _load_model(model_name)

    prompt, audio_b64, audio_fmt = parse_messages_for_text_and_audio(messages)
    if not audio_b64:
        raise ValueError("input_audio not found in messages")

    audio_path = save_base64_audio_to_temp(audio_b64, audio_fmt)

    try:
        from qwen_omni_utils import process_mm_info

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        if prompt:
            conversation[0]["content"].append({"type": "text", "text": prompt})

        text = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
        inputs = _processor(text=text, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(_model.device).to(_model.dtype)

        with torch.no_grad():
            outputs = _model.generate(**inputs, max_new_tokens=512, do_sample=False)

        if isinstance(outputs, tuple):
            text_outputs = outputs[0]
        else:
            text_outputs = outputs

        input_len = inputs["input_ids"].shape[1]
        generated = text_outputs[0][input_len:]
        response_text = _processor.decode(generated, skip_special_tokens=True).strip()

    except ImportError as e:
        logger.warning(f"qwen_omni_utils not available: {e}, using fallback")

        audio_data, sr = sf.read(audio_path)
        text_input = prompt or "Describe this audio in detail."

        inputs = _processor(
            text=text_input,
            audio=audio_data,
            sampling_rate=sr,
            return_tensors="pt",
        ).to(_model.device)

        with torch.no_grad():
            generated_ids = _model.generate(**inputs, max_new_tokens=512)

        response_text = _processor.decode(generated_ids[0], skip_special_tokens=True)

    logger.info(f"Response text: {response_text}")

    try:
        os.unlink(audio_path)
    except Exception:
        pass

    return response_text, model_name


mount_chat_completions(app, _process)


@app.on_event("startup")
def load_model_on_startup():
    _load_model("Qwen/Qwen3-Omni-30B-A3B-Instruct")
