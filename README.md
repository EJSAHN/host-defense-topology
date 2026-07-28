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

To reconstruct the recovered data from the three source workbooks:

```bash
python run_analysis.py --input-dir /path/to/source_workbooks --recovered-dir data --output-dir results
```

The required workbook names are:

```text
Raw score2 (1).xlsx
Raw score2 (2).xlsx
Experiment 3 Leaf Assay-2.xlsx
```

## Statistical approach

- Exact conditional Page tests evaluate increasing dose trends within source experiment and isolate, using plant as the block.
- Holm adjustment is applied separately to the published primary family and the historical secondary family.
- A cumulative-logit ordinal generalized estimating equation, clustered by plant and weighted so each plant-treatment cell has equal total weight, is provided as a sensitivity analysis.
- Mixture effects are summarized as plant-level contrasts between the 1:1 mixture and the mean of the two dose-matched component isolates.
- One exact two-sided sign test is performed per source experiment, with Holm adjustment within the published and historical families.

No pooled host effect is estimated because host, isolate, and source experiment are not fully crossed.

## Reproducibility

Input and output SHA-256 hashes are written to JSON manifests. The scripts stop when plant labels, treatment cells, dose matching, or score ranges fail validation.

## Associated manuscript and archive

The workflow supports the manuscript *Plant-level ordinal analysis of inoculum-dose and mixed-isolate responses in sorghum anthracnose leaf assays*.

A citable archive is available through Zenodo: 
