#!/usr/bin/env python3
"""Run plant-level recovery and source-experiment analyses."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional directory containing the three source workbooks.",
    )
    parser.add_argument(
        "--recovered-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing recovered CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source = root / "src"

    if args.input_dir is not None:
        run(
            [
                sys.executable,
                str(source / "recover_data.py"),
                str(args.input_dir),
                str(args.recovered_dir),
            ]
        )

    run(
        [
            sys.executable,
            str(source / "analyze_data.py"),
            "--recovered-dir",
            str(args.recovered_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    print(f"Completed. Tables are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
