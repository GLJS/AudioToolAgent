#!/usr/bin/env python3
"""Evaluate AudioToolAgent (closed configuration) on MMAR."""
from __future__ import annotations

import argparse
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from Evaluation._common import PreparedSample, run_benchmark

REPO_ID = "gijs/mmar"


def _prepare_mmar(sample: dict, work_dir: str) -> PreparedSample:  # noqa: ARG001 (work_dir unused)
    sample_id = str(sample.get("id", "unknown"))

    rel_path = sample.get("audio_path") or ""
    resolved_path = hf_hub_download(repo_id=REPO_ID, filename=rel_path.lstrip("./"))

    options = list(sample.get("choices", []))
    answer = sample.get("answer")

    metadata = {
        "modality": sample.get("modality"),
        "category": sample.get("category"),
        "sub_category": sample.get("sub-category"),
    }

    return PreparedSample(
        sample_id=sample_id,
        audio_path=resolved_path,
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
    parser = argparse.ArgumentParser(description="Run MMAR benchmark with closed configuration")
    parser.add_argument("--config", default="configs/audiotoolagent.yaml", help="Path to closed configuration file")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate (default: train)")
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
        prepare_fn=_prepare_mmar,
        limit=args.limit,
        stream=not args.no_stream,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
