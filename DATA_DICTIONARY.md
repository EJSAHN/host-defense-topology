# Data dictionary

## `data/recovered_ordinal_scores.csv`

One row per digit-level technical subscore.

| Field | Meaning |
|---|---|
| `experiment_id` | Stable source-experiment identifier |
| `experiment_label` | Reader-facing experiment label |
| `publication_status` | `published` or `historical` |
| `source_workbook` | Source workbook |
| `source_row`, `source_column` | Original workbook location |
| `host` | Host genotype or accession |
| `host_type` | Sorghum or johnsongrass |
| `plant_id` | Block-specific plant identifier |
| `treatment_class` | `single` or `mixture` |
| `isolate_1`, `isolate_2` | Component isolate identities |
| `mixture_ratio` | Mixture ratio when applicable |
| `total_dose_conidia_per_ml` | Applied single-isolate dose or total mixture dose |
| `component_dose_conidia_per_ml` | Dose of each component in the treatment |
| `subsample_id` | Technical subscore index within a plant-treatment cell |
| `ordinal_score` | Disease score from 1 to 5 |

## `data/recovered_plant_condition_means.csv`

One row per plant-by-treatment cell. `plant_mean_score` is the mean of the digit-level technical subscores in that cell.

## `data/experiment_design.csv`

Source experiment, publication status, assay scope, host, isolates, plant counts, dose series, mixture ratio, metadata limits, and allowed inference.

## `results/dose_trend_results.csv`

Plant-level exact Page trend results, descriptive effect summaries, and family-specific Holm-adjusted P values.

## `results/ordinal_gee_sensitivity.csv`

Cumulative-logit ordinal GEE sensitivity results, clustered by plant.

## `results/mixture_results.csv`

One plant-level mixture test per source experiment. The contrast is:

```text
mixture score - mean(dose-matched component scores)
```

The three dose-specific contrasts are averaged within plant before the exact sign test.

## `results/mixture_by_dose.csv`

Descriptive dose-specific mixture contrasts, component doses, sign counts, and score summaries.
# Data dictionary

## `data/recovered_ordinal_scores.csv`

One row per digit-level technical subscore.

| Field | Meaning |
|---|---|
| `experiment_id` | Stable source-experiment identifier |
| `experiment_label` | Reader-facing experiment label |
| `publication_status` | `published` or `historical` |
| `source_workbook` | Source workbook |
| `source_row`, `source_column` | Original workbook location |
| `host` | Host genotype or accession |
| `host_type` | Sorghum or johnsongrass |
| `plant_id` | Block-specific plant identifier |
| `treatment_class` | `single` or `mixture` |
| `isolate_1`, `isolate_2` | Component isolate identities |
| `mixture_ratio` | Mixture ratio when applicable |
| `total_dose_conidia_per_ml` | Applied single-isolate dose or total mixture dose |
| `component_dose_conidia_per_ml` | Dose of each component in the treatment |
| `subsample_id` | Technical subscore index within a plant-treatment cell |
| `ordinal_score` | Disease score from 1 to 5 |

## `data/recovered_plant_condition_means.csv`

One row per plant-by-treatment cell. `plant_mean_score` is the mean of the digit-level technical subscores in that cell.

## `data/experiment_design.csv`

Source experiment, publication status, assay scope, host, isolates, plant counts, dose series, mixture ratio, metadata limits, and allowed inference.

## `results/dose_trend_results.csv`

Plant-level exact Page trend results, descriptive effect summaries, and family-specific Holm-adjusted P values.

## `results/ordinal_gee_sensitivity.csv`

Cumulative-logit ordinal GEE sensitivity results, clustered by plant.

## `results/mixture_results.csv`

One plant-level mixture test per source experiment. The contrast is:

```text
mixture score - mean(dose-matched component scores)
```

The three dose-specific contrasts are averaged within plant before the exact sign test.

## `results/mixture_by_dose.csv`

Descriptive dose-specific mixture contrasts, component doses, sign counts, and score summaries.
