#!/usr/bin/env python3
"""Evaluate AudioToolAgent (closed configuration) on MMAU-Pro."""
from __future__ import annotations

import argparse
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from Evaluation._common import PreparedSample, run_benchmark

REPO_ID = "gamma-lab-umd/MMAU-Pro"


def _resolve_audio(paths) -> str:
    if isinstance(paths, str):
        candidates = [paths]
    else:
        candidates = list(paths or [])
    if not candidates:
        raise ValueError("Sample does not contain an audio_path entry")
    return hf_hub_download(repo_id=REPO_ID, filename=candidates[0].lstrip("./"))


def _prepare_mmau_pro(sample: dict, work_dir: str) -> PreparedSample:  # noqa: ARG001
    sample_id = str(sample.get("id", "unknown"))
    audio_path = _resolve_audio(sample.get("audio_path"))
    options = list(sample.get("choices", []))
    answer = sample.get("answer")

    metadata = {
        "category": sample.get("category"),
        "perceptual_skills": sample.get("perceptual_skills"),
        "reasoning_skills": sample.get("reasoning_skills"),
        "sub_category": sample.get("sub-cat"),
        "task": sample.get("task_identifier"),
    }

    return PreparedSample(
        sample_id=sample_id,
        audio_path=audio_path,
        question=sample.get("question", ""),
        options=options,
        answer=answer,
        metadata=metadata,
    )


def _iter_dataset(split: str) -> Iterable[dict]:
    dataset = load_dataset(REPO_ID, split=split, streaming=False)
    for item in dataset:
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMAU-Pro benchmark with closed configuration")
    parser.add_argument("--config", default="configs/audiotoolagent.yaml", help="Path to closed configuration file")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
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
        prepare_fn=_prepare_mmau_pro,
        limit=args.limit,
        stream=not args.no_stream,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
