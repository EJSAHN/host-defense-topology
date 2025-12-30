This repository contains a deterministic, command-line workflow to reproduce the
summary tables used in the Conidia project manuscript. The analysis follows the
definitions in the manuscript (severe score threshold, B = -ln(P), Bliss
independence for mixtures).

Inputs
------
Place the following input files in an input directory (default: current working directory):

  - dose_response_master.csv
  - mixture_master.csv
  - barrier_summary_master.csv

Run
---
conda env create -f environment.yml
conda activate conidia2
python run_analysis.py --input-dir . --output-dir outputs

Run the pipeline:

  python run_analysis.py --input-dir . --output-dir outputs

Outputs
-------
The script writes:

  - individual Excel files for each table
  - Supplementary_Data_S1.xlsx (multi-sheet workbook)
  - manifest.json (file hashes + software versions)

Notes
-----
This repository intentionally does not generate figures. A separate, local-only
script can be used for figure generation.
