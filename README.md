This repository contains a deterministic, command-line workflow to reproduce the
summary tables used in the sorghum–johnsongrass–Colletotrichum sublineola manuscript.

The analysis follows the definitions in the manuscript:
  - severe disease is defined as score >= 4
  - high-dose barrier summary is B_dose = -ln(P_max)
  - mixed-isolate outcomes are evaluated as deviation from Bliss independence

Inputs
------
Place the following input files in an input directory (default: current working directory):

  - dose_response_master.csv
  - mixture_master.csv
  - barrier_summary_master.csv

Run
---
conda env create -f environment.yml
conda activate host-response-analysis

Run the pipeline:

  python run_analysis.py --input-dir . --output-dir outputs

Outputs
-------
The script writes individual Excel files for each analysis table, a multi-sheet
Supplementary_Data_S1.xlsx workbook, and a manifest.json file with file hashes
and software versions.

The main output tables are:

  - dose_response_summary.xlsx
  - dose_response_fit_details.xlsx
  - high_dose_barrier_summary.xlsx
  - high_dose_barrier_details.xlsx
  - mixed_isolate_bliss_summary.xlsx
  - mixed_isolate_single_controls.xlsx
  - mixed_isolate_host_env_summary.xlsx
  - regrowth_barrier_changes.xlsx
  - structural_comparison_summary.xlsx
  - host_response_summary.xlsx
  - Supplementary_Data_S1.xlsx
  - manifest.json

Notes
-----
This repository intentionally does not generate figures. A separate, local-only
script can be used for figure generation.
