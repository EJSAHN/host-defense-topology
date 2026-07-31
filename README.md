# Plant-level analysis of sorghum anthracnose dose and mixture assays

This repository contains the data-recovery and statistical workflow used to analyze ordinal disease scores from sorghum and johnsongrass excised-leaf assays with *Colletotrichum sublineola*.

The workflow preserves the source experiment and the plant as the independent experimental unit. Digit-level 1–5 scores recorded within each plant-by-treatment cell are treated as technical subsamples.

## Questions addressed

1. Within a source experiment and isolate, do ordinal disease scores increase across the ordered inoculum-dose series?
2. Within a source experiment, does a 1:1 isolate mixture differ consistently from the mean response of its dose-matched component isolates?

Published experiments form the primary analysis set. Two historical blocks with incomplete environmental metadata are analyzed separately and are used only for within-block comparisons.

## Repository contents

```text
src/
  recover_data.py
  analyze_data.py
data/
  recovered_ordinal_scores.csv
  recovered_plant_condition_means.csv
  experiment_design.csv
  recovery_validation.txt
  input_manifest.json
results/
  dose_trend_results.csv
  ordinal_gee_sensitivity.csv
  mixture_results.csv
  mixture_by_dose.csv
  analysis_summary.txt
  analysis_manifest.json
run_analysis.py
DATA_DICTIONARY.md
environment.yml
CITATION.cff
LICENSE
```

Figure-generation code is not included in this repository.

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate sorghum-dose-mixture
```

Run the analyses from the recovered data supplied in `data/`:

```bash
python run_analysis.py --recovered-dir data --output-dir results
```

### Optional source reconstruction

For users with access to the original source workbooks, the recovered tables can be reconstructed with:

```bash
python run_analysis.py --input-dir /path/to/source_workbooks --recovered-dir data --output-dir results
```

Use the following preferred filenames:

```text
source_workbook_A_experiments_1_2.xlsx
source_workbook_B_experiment_3_historical_block_1.xlsx
source_workbook_C_experiments_4_5_historical_block_2.xlsx
```

The recovery script also recognizes the original local filenames for backward compatibility. Each accepted workbook is verified against the SHA-256 hash of the source file used for the analysis.

## Statistical approach

- Exact conditional Page tests evaluate increasing dose trends within source experiment and isolate, using plant as the block.
- Holm adjustment is applied separately to the published primary family and the historical secondary family.
- A cumulative-logit ordinal generalized estimating equation, clustered by plant and weighted so each plant-treatment cell has equal total weight, is provided as a sensitivity analysis.
- Mixture effects are summarized as plant-level contrasts between the 1:1 mixture and the mean of the two dose-matched component isolates.
- One exact two-sided sign test is performed per source experiment, with Holm adjustment within the published and historical families.
- Contrasts with absolute values less than or equal to 1e-12 are treated as numerical ties before sign counting.

No pooled host effect is estimated because host, isolate, and source experiment are not fully crossed.

## Version

The analyses associated with the manuscript are preserved in GitHub release **v2.0.0**:

https://github.com/EJSAHN/host-defense-topology/releases/tag/v2.0.0

## Reproducibility

Input and output SHA-256 hashes are written to JSON manifests. The scripts stop when plant labels, treatment cells, dose matching, or score ranges fail validation.

