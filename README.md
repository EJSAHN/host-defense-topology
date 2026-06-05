# host-defense-topology

Reproducible analysis workflow for quantitative host defense thresholds and mixed-isolate disease outcomes in the sorghum–*Colletotrichum sublineola* pathosystem.

The repository contains curated master tables and table-generating analysis scripts. It intentionally does **not** include manuscript figure-generation code.

## Core definitions

- Severe disease is defined as disease score >= 4.
- Logistic dose-response fits estimate the inoculum threshold associated with severe disease and the steepness of the severe-disease response.
- The high-dose barrier summary is defined as `B_dose = -ln(P_max)`, where `P_max` is the observed severe-disease probability at the highest tested inoculum concentration.
- Mixed-isolate outcomes are summarized as deviation from Bliss independence: `delta_bliss = observed_p_severe - expected_p_severe`.
- Regrowth effects are summarized as `delta_B = B_round2 - B_round1`.

## Input files

The repository contains the curated input tables used by the analysis scripts:

- `dose_response_master.csv`
- `mixture_master.csv`
- `barrier_summary_master.csv`

The master tables retain source-file provenance columns and cleaned isolate / mixture fields for reproducible filtering, grouping, and audit checks.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate host-response-analysis
```

An existing environment with `pandas`, `numpy`, `statsmodels`, `scipy`, and `openpyxl` can also run the scripts.

## Run the primary analysis

```bash
python run_analysis.py --input-dir . --output-dir outputs
```

This writes the primary derived summary tables and a multi-sheet supplementary workbook:

- `outputs/Supplementary_Data_S1.xlsx`
- `outputs/dose_response_summary.xlsx`
- `outputs/dose_response_fit_details.xlsx`
- `outputs/high_dose_barrier_summary.xlsx`
- `outputs/high_dose_barrier_details.xlsx`
- `outputs/mixed_isolate_bliss_summary.xlsx`
- `outputs/mixed_isolate_single_controls.xlsx`
- `outputs/mixed_isolate_host_env_summary.xlsx`
- `outputs/regrowth_barrier_changes.xlsx`
- `outputs/structural_comparison_summary.xlsx`
- `outputs/host_response_summary.xlsx`
- `outputs/manifest.json`

## Run the extended table analysis

```bash
python run_extended_analysis.py --input-dir . --output-dir extended_outputs --n-bootstrap 5000 --seed 42
```

This writes uncertainty summaries, model diagnostics, cutoff-sensitivity analyses, and data-structure audits as tables only:

- `extended_outputs/tables/dose_response_fit_uncertainty.csv`
- `extended_outputs/tables/high_dose_barrier_uncertainty.csv`
- `extended_outputs/tables/mixed_isolate_bliss_uncertainty.csv`
- `extended_outputs/tables/regrowth_deltaB_uncertainty.csv`
- `extended_outputs/tables/cutoff_sensitivity_dose_response.csv`
- `extended_outputs/tables/cutoff_sensitivity_barrier.csv`
- `extended_outputs/tables/cutoff_sensitivity_bliss.csv`
- `extended_outputs/tables/continuous_score_trend_sensitivity.csv`
- `extended_outputs/tables/pooled_dose_model_sensitivity.csv`
- `extended_outputs/tables/host_backgrounds.csv`
- `extended_outputs/tables/isolate_inventory.csv`
- `extended_outputs/tables/mixed_isolate_composition_audit.csv`
- `extended_outputs/tables/mixed_isolate_single_controls.csv`
- `extended_outputs/Extended_Analysis_Tables.xlsx`
- `extended_outputs/extended_analysis_manifest.json`

## Repository structure

```text
.
├── conidia_analysis/
│   ├── pipeline.py
│   ├── extended_analysis.py
│   └── utils.py
├── dose_response_master.csv
├── mixture_master.csv
├── barrier_summary_master.csv
├── run_analysis.py
├── run_extended_analysis.py
├── environment.yml
├── DATA_DICTIONARY.md
└── README.md
```

## Notes

The master data use cleaned isolate and mixture fields to separate single-isolate and mixed-isolate records. In the BTx398 mixed-isolate block, the three-component mixture is represented as `AMP99+AMP155+AMP170`, consistent with the source records and component columns.

Generated output directories are not tracked in this clean repository package. Re-running the scripts from the master tables will regenerate the table outputs.
