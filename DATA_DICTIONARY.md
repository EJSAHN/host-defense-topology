# Data dictionary

## `dose_response_master.csv`

Curated observation-level data for single-isolate dose-response analyses.

Key columns:

- `host`: host genotype or accession.
- `replicate_id`: plant / replicate identifier when available.
- `tissue`: evaluated tissue, when available.
- `treatment_raw`: original treatment label from the source file.
- `isolate`: raw or standardized treatment label.
- `isolate_clean`: cleaned single-isolate identifier; blank for mixed-isolate rows.
- `concentration`: inoculum concentration in conidia mL-1.
- `score`: disease severity score.
- `is_mixture_clean`: whether the row represents a mixed-isolate treatment after cleaning.
- `isolate_1`, `isolate_2`, `isolate_3`: mixture components when applicable.
- `mixture_label`: standardized mixture label when applicable.
- `source_file`: source data file name.

## `mixture_master.csv`

Curated observation-level data for single- and mixed-isolate assays used for Bliss-independence summaries.

Key columns:

- `host`: host genotype or accession.
- `environment`: assay environment or context, for example `lab` or `GH`.
- `treatment_raw`: original treatment label.
- `isolate_clean`: cleaned single-isolate identifier for single-isolate controls.
- `is_mixture_clean`: whether the row is a mixed-isolate treatment.
- `isolate_1`, `isolate_2`, `isolate_3`: mixture components.
- `ratio_1`, `ratio_2`, `ratio_3`: component ratios when available.
- `mixture_label`: standardized mixture label.
- `concentration`: inoculum concentration in conidia mL-1; fixed-dose assays without explicit concentration are represented as 1e6.
- `score`: disease severity score.
- `source_file`: source data file name.

## `barrier_summary_master.csv`

Curated summary-level data for regrowth, structural, and related high-dose barrier summaries.

Key columns:

- `experiment`: analysis context.
- `host`: host genotype or accession.
- `isolate_clean`: cleaned isolate identifier.
- `round`: regrowth comparison round when applicable.
- `n`: number of underlying observations represented by the summary row.
- `p`: severe-disease probability.
- `B`: barrier summary, `-ln(p)`, when finite.
- `mean_score`: mean disease score.
- structural columns such as `thickness_mean` or `leaf_angle_mean` when applicable.
