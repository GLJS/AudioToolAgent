"""Local Whisper transcription tool built on top of faster-whisper."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from qwen_agent.tools.base import BaseTool, register_tool

from .audio_utils import cleanup_temp_file, validate_audio_file

logger = logging.getLogger(__name__)


@register_tool("whisper")
class WhisperTool(BaseTool):
    """Transcribe audio offline using the `faster-whisper` implementation."""

    description = "Transcribe speech using an offline Whisper model (faster-whisper)."
    parameters = [
        {
            "name": "audio_path",
            "type": "string",
            "description": "Path to the audio file that should be transcribed",
            "required": True,
        },
        {
            "name": "beam_size",
            "type": "integer",
            "description": "Beam size used during decoding (default: 5)",
            "required": False,
        },
    ]

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
        **_: Any,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.default_beam_size = beam_size
        self._model = None
        logger.info(
            "Initialised WhisperTool with model=%s device=%s compute_type=%s",
            model_name,
            device,
            compute_type,
        )

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise RuntimeError(
                "faster-whisper is required for WhisperTool. Install it with `pip install faster-whisper`."
            ) from exc

        logger.info(
            "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    # ------------------------------------------------------------------
    def call(self, params: Any, **kwargs: Any) -> str:  # noqa: ARG002
        if isinstance(params, str):
            raise TypeError("WhisperTool expects dictionary parameters")

        audio_path = params.get("audio_path") if isinstance(params, dict) else None
        if not audio_path:
            return self._error("Missing required parameter: audio_path")
        if not validate_audio_file(audio_path):
            return self._error(f"Invalid or missing audio file: {audio_path}")

        beam_size = int(params.get("beam_size", self.default_beam_size)) if isinstance(params, dict) else self.default_beam_size

        self._load_model()
        assert self._model is not None

        try:
            segments, info = self._model.transcribe(audio_path, beam_size=beam_size)
            transcript_parts = [seg.text.strip() for seg in segments]
            transcript = " ".join(part for part in transcript_parts if part)
            logger.info(
                "Transcribed %s (duration=%.2fs, language=%s)",
                audio_path,
                getattr(info, "duration", 0.0),
                getattr(info, "language", "unknown"),
            )
            cleanup_temp_file(audio_path, False)
            return self._success(transcript)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Whisper transcription failed")
            return self._error(str(exc))

    # ------------------------------------------------------------------
    def _success(self, transcript: str) -> str:
        return self._json_response({
            "text": transcript,
            "message": f"Transcription complete: {len(transcript.split())} words",
        })

    def _error(self, message: str) -> str:
        return self._json_response({"error": message})

    @staticmethod
    def _json_response(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)
