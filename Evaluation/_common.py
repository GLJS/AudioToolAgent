"""Shared helpers for benchmark evaluation scripts."""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from audiotoolagent.agent import AudioToolAgent, console_stream_callback

logger = logging.getLogger(__name__)


@dataclass
class PreparedSample:
    sample_id: str
    audio_path: str
    question: str
    options: List[str]
    answer: Optional[str]
    metadata: Optional[dict] = None


PrepareFn = Callable[[dict, str], PreparedSample]


def run_benchmark(
    *,
    config_path: str,
    raw_samples: Iterable[dict],
    prepare_fn: PrepareFn,
    limit: Optional[int] = None,
    stream: bool = True,
    output_path: Optional[str] = None,
) -> dict:
    """Evaluate a stream of raw dataset entries against the agent."""
    agent = AudioToolAgent(config_path)
    stream_cb = console_stream_callback if stream else None

    totals = {
        "count": 0,
        "correct": 0,
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="audio_eval_") as work_dir:
        for idx, raw in enumerate(raw_samples):
            if limit is not None and idx >= limit:
                break

            prepared = prepare_fn(raw, work_dir)
            logger.info("Processing sample %s", prepared.sample_id)

            result = agent.process(
                audio_path=prepared.audio_path,
                question=prepared.question,
                options=prepared.options or None,
                stream_callback=stream_cb,
            )

            predicted = result.get("selected_option") or result.get("answer", "")
            is_correct = bool(prepared.answer) and predicted and prepared.answer.strip() == predicted.strip()

            totals["count"] += 1
            totals["correct"] += int(is_correct)

            result_payload = {
                "id": prepared.sample_id,
                "question": prepared.question,
                "options": prepared.options,
                "ground_truth": prepared.answer,
                "prediction": predicted,
                "raw_answer": result.get("answer"),
                "metadata": prepared.metadata or {},
                "correct": is_correct,
            }
            totals["results"].append(result_payload)

    accuracy = (totals["correct"] / totals["count"]) if totals["count"] else 0.0
    summary = {
        "config": config_path,
        "samples": totals["count"],
        "correct": totals["correct"],
        "accuracy": accuracy,
    }

    logger.info("Evaluation complete: %s", summary)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": summary,
            "results": totals["results"],
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved detailed results to %s", out_path)

    return summary
