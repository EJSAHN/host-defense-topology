from __future__ import annotations

import argparse
from pathlib import Path

from conidia_analysis.extended_analysis import run_extended_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extended analysis tables: design audits, uncertainty intervals, diagnostics, and sensitivity checks."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing the three master CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("extended_outputs"), help="Output directory for extended analysis tables.")
    parser.add_argument("--severe-cutoff", type=int, default=4, help="Primary severe-disease cutoff; default: score >= 4.")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations for uncertainty intervals; use 500 for testing and 5000 for final outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap reproducibility.")
    args = parser.parse_args()

    run_extended_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        severe_cutoff=args.severe_cutoff,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
