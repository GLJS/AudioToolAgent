#!/usr/bin/env python3
"""Single-run convenience script for AudioToolAgent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from audiotoolagent.agent import AudioToolAgent, console_stream_callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AudioToolAgent pipeline for a single audio question",
    )
    parser.add_argument("--config", required=True, help="Path to the agent configuration YAML")
    parser.add_argument("--audio", required=True, help="Path to the input audio file")
    parser.add_argument("--question", required=True, help="Question to ask about the audio")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Optional multiple-choice options. If provided, the agent will select one option.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the JSON result (defaults to stdout only)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output in the terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    agent = AudioToolAgent(args.config)

    stream_cb = None if args.no_stream else console_stream_callback

    result = agent.process(
        audio_path=args.audio,
        question=args.question,
        options=args.options,
        stream_callback=stream_cb,
    )

    print("\n📋 Final Result:\n")
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
