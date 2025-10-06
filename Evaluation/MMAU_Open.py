#!/usr/bin/env python3
"""Evaluate AudioToolAgent (open configuration) on MMAU."""
from __future__ import annotations

import argparse
import io
from typing import Iterable

import soundfile as sf
from datasets import Audio, load_dataset

from Evaluation._common import PreparedSample, run_benchmark

REPO_ID = "gijs/mmau-processed"


def _prepare_mmau(sample: dict, work_dir: str) -> PreparedSample:
    sample_id = str(sample.get("id", "unknown"))

    audio_bytes = sample["audio"]["bytes"]
    audio_io = io.BytesIO(audio_bytes)
    audio_array, sampling_rate = sf.read(audio_io)

    audio_path = f"{work_dir}/mmau_{sample_id}.wav"
    sf.write(audio_path, audio_array, sampling_rate)

    choices = sample.get("choices", "")
    options = [line.strip() for line in choices.splitlines() if line.strip() and not line.lower().startswith("choices")]
    answer = sample.get("answer")

    metadata = {
        "task": sample.get("task"),
        "domain": sample.get("domain"),
        "difficulty": sample.get("difficulty"),
    }

    return PreparedSample(
        sample_id=sample_id,
        audio_path=audio_path,
        question=sample.get("question_only", sample.get("question", "")),
        options=options,
        answer=answer,
        metadata=metadata,
    )


def _iter_dataset(split: str) -> Iterable[dict]:
    dataset = load_dataset(REPO_ID, split=split, streaming=False)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for item in dataset:
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMAU benchmark with the open configuration")
    parser.add_argument("--config", default="configs/open_deepseek.yaml", help="Path to open configuration file")
    parser.add_argument("--split", default="test_mini", help="Dataset split to evaluate (default: test_mini)")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--output", help="Optional path to save detailed results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _iter_dataset(args.split)
    run_benchmark(
        config_path=args.config,
        raw_samples=samples,
        prepare_fn=_prepare_mmau,
        limit=args.limit,
        stream=not args.no_stream,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
