#!/usr/bin/env python
"""Command-line entry point for the Conidia2 analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from conidia_analysis.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Conidia2 summary tables and Supplementary Data S1."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the input master CSV files (default: current directory).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write output Excel files (default: ./outputs).",
    )
    p.add_argument(
        "--severe-cutoff",
        type=int,
        default=4,
        help="Score threshold for a severe infection event (default: 4).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        severe_cutoff=args.severe_cutoff,
    )


if __name__ == "__main__":
    main()
