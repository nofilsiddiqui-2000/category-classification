#!/usr/bin/env python3
"""CLI runner for Task 1: category super-classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from category_pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify product category names into super-categories."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input categories file (.csv or .xlsx).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to output file (.csv or .xlsx). If omitted, output is created next"
            " to the input as <name>_classified.<ext>."
        ),
    )
    parser.add_argument(
        "--category-column",
        default=None,
        help="Column that contains category names. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/task1",
        help="Directory for artifacts (metrics, model, unique predictions).",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM classification for ambiguous categories (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model name used when --enable-llm is set.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=30,
        help="Number of categories sent per LLM request.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY env var is used.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path for thresholds and random seed.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_classified{input_path.suffix}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    summary = run_pipeline(
        input_path=input_path,
        output_path=output_path,
        category_column=args.category_column,
        artifacts_dir=args.artifacts_dir,
        enable_llm=args.enable_llm,
        config_path=args.config,
        llm_model=args.llm_model,
        llm_batch_size=args.llm_batch_size,
        llm_api_key=args.api_key,
    )

    print("Task 1 pipeline completed.")
    print(f"Input: {summary.input_path}")
    print(f"Output: {summary.output_path}")
    print(f"Artifacts: {summary.artifacts_dir}")
    print(f"Rows: {summary.input_rows} | Unique categories: {summary.unique_categories}")
    print(f"LLM used: {summary.used_llm} | ML model trained: {summary.trained_ml_model}")
    print("Label counts:", json.dumps(summary.label_counts, indent=2))
    print("Source counts:", json.dumps(summary.source_counts, indent=2))
    if summary.model_metrics:
        print("Model metrics:", json.dumps(summary.model_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
