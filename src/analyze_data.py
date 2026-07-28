#!/usr/bin/env python3
"""Run source-experiment-specific dose and mixture analyses.

Primary inference is restricted to the five experiments whose assay details
were previously published. Two additional historical blocks are analyzed
separately because detailed environmental setpoints were not uniformly
recoverable.

Primary dose analysis
---------------------
An exact conditional Page test evaluates an increasing ordered trend across
six inoculum doses using the plant as the block. Holm adjustment is applied
within the ten published isolate-specific tests. The four historical tests
form a separate secondary family.

Sensitivity dose analysis
-------------------------
A cumulative-logit ordinal GEE is fitted to the individual 1-5 subscores,
clustered by plant. Each plant-treatment cell receives total weight one so
that technical subsample count does not change the biological weight of a
cell. Because each experiment contains only six or seven plants, GEE results
are treated as sensitivity analyses.

Primary mixture analysis
------------------------
For each plant and total mixture dose, the mixture score is compared with the
mean of the two component-isolate scores at one-half the total mixture dose.
The three dose-specific contrasts are averaged within plant, followed by one
exact two-sided sign test per source experiment. Holm adjustment is applied
within the five published experiments; the two historical experiments form a
separate secondary family.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def exact_page_test(matrix: np.ndarray) -> tuple[int, float, int]:
    """Exact one-sided Page test, conditioning on tied ranks within plant."""
    matrix = np.asarray(matrix, dtype=float)
    dose_scores = np.arange(1, matrix.shape[1] + 1)
    observed = 0
    row_distributions: list[Counter[int]] = []

    for row in matrix:
        doubled_ranks = np.rint(stats.rankdata(row, method="average") * 2).astype(int)
        observed += int(np.dot(dose_scores, doubled_ranks))
        distribution: Counter[int] = Counter()
        # All label permutations are counted. Repeated statistics arising from
        # tied ranks retain their correct multiplicity in the randomization null.
        for permutation in itertools.permutations(doubled_ranks.tolist()):
            statistic = int(np.dot(dose_scores, np.asarray(permutation)))
            distribution[statistic] += 1
        row_distributions.append(distribution)

    combined: Counter[int] = Counter({0: 1})
    for row_distribution in row_distributions:
        updated: Counter[int] = Counter()
        for left_statistic, left_count in combined.items():
            for right_statistic, right_count in row_distribution.items():
                updated[left_statistic + right_statistic] += left_count * right_count
        combined = updated

    total = sum(combined.values())
    upper_tail = sum(count for statistic, count in combined.items() if statistic >= observed)
    return observed, upper_tail / total, len(combined)


def sign_test(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    positive = int((array > 0).sum())
    negative = int((array < 0).sum())
    zero = int((array == 0).sum())
    nonzero = positive + negative
    p_value = stats.binomtest(positive, nonzero, 0.5).pvalue if nonzero else 1.0
    return {
        "plants_total": int(array.size),
        "plants_nonzero": nonzero,
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "exact_sign_p": float(p_value),
        "mean_contrast": float(array.mean()),
        "median_contrast": float(np.median(array)),
        "min_contrast": float(array.min()),
        "max_contrast": float(array.max()),
    }


def family_name(status: str) -> str:
    return "published_primary" if status == "published" else "historical_secondary"


def analyze(recovered_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_rows = read_csv(recovered_dir / "recovered_plant_condition_means.csv")
    score_rows = read_csv(recovered_dir / "recovered_ordinal_scores.csv")
    design_rows = read_csv(recovered_dir / "experiment_design.csv")
    design_by_experiment = {row["experiment_id"]: row for row in design_rows}

    for row in cell_rows:
        row["total_dose_conidia_per_ml"] = int(float(row["total_dose_conidia_per_ml"]))
        row["component_dose_conidia_per_ml"] = int(float(row["component_dose_conidia_per_ml"]))
        row["plant_mean_score"] = float(row["plant_mean_score"])
        row["technical_subsamples"] = int(row["technical_subsamples"])
        row["plant_number"] = int(row["plant_number"])

    for row in score_rows:
        row["total_dose_conidia_per_ml"] = int(float(row["total_dose_conidia_per_ml"]))
        row["component_dose_conidia_per_ml"] = int(float(row["component_dose_conidia_per_ml"]))
        row["ordinal_score"] = int(row["ordinal_score"])
        row["plant_number"] = int(row["plant_number"])

    # Dose analysis at the plant level.
    dose_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in cell_rows:
        if row["treatment_class"] == "single":
            dose_groups[(row["experiment_id"], row["host"], row["isolate_1"])].append(row)

    dose_results: list[dict] = []
    gee_results: list[dict] = []

    for (experiment_id, host, isolate), rows in sorted(dose_groups.items()):
        design = design_by_experiment[experiment_id]
        doses = sorted({row["total_dose_conidia_per_ml"] for row in rows})
        plants = sorted({row["plant_id"] for row in rows})
        lookup = {
            (row["plant_id"], row["total_dose_conidia_per_ml"]): row["plant_mean_score"]
            for row in rows
        }
        matrix = np.asarray([[lookup[(plant, dose)] for dose in doses] for plant in plants], dtype=float)
        if matrix.shape != (len(plants), 6):
            raise ValueError(f"Expected a complete plants x 6 dose matrix for {experiment_id}, {isolate}.")

        page_statistic, page_p, null_states = exact_page_test(matrix)
        friedman = stats.friedmanchisquare(*[matrix[:, column] for column in range(matrix.shape[1])])
        plant_rho = [
            stats.spearmanr(np.log10(doses), plant_scores).statistic
            for plant_scores in matrix
        ]

        dose_results.append(
            {
                "analysis_family": family_name(design["publication_status"]),
                "experiment_id": experiment_id,
                "experiment_label": design["experiment_label"],
                "publication_status": design["publication_status"],
                "host": host,
                "isolate": isolate,
                "independent_unit": "plant",
                "plants": len(plants),
                "dose_levels": len(doses),
                "minimum_dose_conidia_per_ml": min(doses),
                "maximum_dose_conidia_per_ml": max(doses),
                "page_statistic_times2": page_statistic,
                "page_p_exact_one_sided": page_p,
                "exact_null_states": null_states,
                "friedman_chi_square": float(friedman.statistic),
                "friedman_p": float(friedman.pvalue),
                "median_plant_spearman_rho": float(np.nanmedian(plant_rho)),
                "mean_score_lowest_dose": float(matrix[:, 0].mean()),
                "mean_score_highest_dose": float(matrix[:, -1].mean()),
                "mean_change_high_minus_low": float(np.mean(matrix[:, -1] - matrix[:, 0])),
            }
        )

        # Ordinal GEE sensitivity analysis.
        subset = [
            row
            for row in score_rows
            if row["experiment_id"] == experiment_id
            and row["treatment_class"] == "single"
            and row["isolate_1"] == isolate
        ]
        response = np.asarray([row["ordinal_score"] for row in subset], dtype=float)
        log_dose = np.log10([row["total_dose_conidia_per_ml"] for row in subset])
        centered_log_dose = log_dose - log_dose.mean()
        predictor = centered_log_dose[:, None]
        plant_map = {plant: index for index, plant in enumerate(sorted({row["plant_id"] for row in subset}))}
        clusters = np.asarray([plant_map[row["plant_id"]] for row in subset])
        # Each plant-treatment cell contributes total weight one.
        cell_size = Counter(row["condition_id"] + "|" + row["plant_id"] for row in subset)
        weights = np.asarray(
            [1.0 / cell_size[row["condition_id"] + "|" + row["plant_id"]] for row in subset],
            dtype=float,
        )
        gee_record: dict[str, object] = {
            "analysis_family": family_name(design["publication_status"]),
            "experiment_id": experiment_id,
            "experiment_label": design["experiment_label"],
            "publication_status": design["publication_status"],
            "host": host,
            "isolate": isolate,
            "clusters": len(plant_map),
            "technical_scores": len(subset),
            "working_correlation": "independence",
            "cell_equalizing_weights": True,
            "covariance": "bias_reduced",
        }
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = OrdinalGEE(
                    response,
                    predictor,
                    clusters,
                    cov_struct=Independence(),
                    weights=weights,
                ).fit(maxiter=1000, cov_type="bias_reduced")
            coefficient = float(result.params.iloc[-1])
            standard_error = float(result.bse.iloc[-1])
            finite = (
                math.isfinite(coefficient)
                and math.isfinite(standard_error)
                and standard_error > 0
            )
            z_value = coefficient / standard_error if finite else math.nan
            p_value = float(2 * stats.norm.sf(abs(z_value))) if finite else math.nan
            gee_record.update(
                {
                    "fit_success": bool(result.converged and finite),
                    "converged": bool(result.converged),
                    "log10_dose_coefficient": coefficient,
                    "bias_reduced_standard_error": standard_error,
                    "coefficient_lcl_95": coefficient - 1.959963984540054 * standard_error,
                    "coefficient_ucl_95": coefficient + 1.959963984540054 * standard_error,
                    "z": z_value,
                    "p": p_value,
                    "warning_count": len(captured),
                    "warnings": " | ".join(sorted({str(item.message) for item in captured})),
                }
            )
        except Exception as error:
            gee_record.update(
                {
                    "fit_success": False,
                    "converged": False,
                    "log10_dose_coefficient": math.nan,
                    "bias_reduced_standard_error": math.nan,
                    "coefficient_lcl_95": math.nan,
                    "coefficient_ucl_95": math.nan,
                    "z": math.nan,
                    "p": math.nan,
                    "warning_count": 1,
                    "warnings": repr(error),
                }
            )
        gee_results.append(gee_record)

    # Multiplicity adjustment is separate for published and historical families.
    for family in ("published_primary", "historical_secondary"):
        family_rows = [row for row in dose_results if row["analysis_family"] == family]
        adjusted = holm_adjust(row["page_p_exact_one_sided"] for row in family_rows)
        for row, p_adjusted in zip(family_rows, adjusted):
            row["page_p_holm_within_family"] = p_adjusted

        family_gee = [row for row in gee_results if row["analysis_family"] == family]
        valid_indices = [
            index for index, row in enumerate(family_gee) if math.isfinite(float(row["p"]))
        ]
        if valid_indices:
            adjusted_gee = holm_adjust(float(family_gee[index]["p"]) for index in valid_indices)
            for index, p_adjusted in zip(valid_indices, adjusted_gee):
                family_gee[index]["p_holm_within_family"] = p_adjusted
        for row in family_gee:
            row.setdefault("p_holm_within_family", math.nan)

    # Mixture analysis.
    mixture_results: list[dict] = []
    dose_specific_results: list[dict] = []

    for experiment_id, design in sorted(design_by_experiment.items()):
        block_rows = [row for row in cell_rows if row["experiment_id"] == experiment_id]
        single_isolates = sorted(
            {row["isolate_1"] for row in block_rows if row["treatment_class"] == "single"}
        )
        if len(single_isolates) != 2:
            raise ValueError(f"Expected exactly two component isolates in {experiment_id}.")
        isolate_a, isolate_b = single_isolates
        mixture_key = "+".join(sorted(single_isolates))
        plants = sorted({row["plant_id"] for row in block_rows})

        lookup: dict[tuple[str, str, str, int], float] = {}
        for row in block_rows:
            treatment_key = (
                row["isolate_1"]
                if row["treatment_class"] == "single"
                else "+".join(sorted([row["isolate_1"], row["isolate_2"]]))
            )
            lookup[
                (
                    row["plant_id"],
                    row["treatment_class"],
                    treatment_key,
                    row["total_dose_conidia_per_ml"],
                )
            ] = row["plant_mean_score"]

        mixture_doses = sorted(
            {
                row["total_dose_conidia_per_ml"]
                for row in block_rows
                if row["treatment_class"] == "mixture"
            }
        )
        plant_contrasts: dict[str, list[float]] = {plant: [] for plant in plants}

        for total_dose in mixture_doses:
            contrasts: list[float] = []
            mixture_values: list[float] = []
            control_averages: list[float] = []
            component_dose = total_dose // 2

            for plant in plants:
                mixture_score = lookup[(plant, "mixture", mixture_key, total_dose)]
                score_a = lookup[(plant, "single", isolate_a, component_dose)]
                score_b = lookup[(plant, "single", isolate_b, component_dose)]
                control_average = (score_a + score_b) / 2.0
                contrast = mixture_score - control_average
                contrasts.append(contrast)
                mixture_values.append(mixture_score)
                control_averages.append(control_average)
                plant_contrasts[plant].append(contrast)

            dose_record = {
                "analysis_family": family_name(design["publication_status"]),
                "experiment_id": experiment_id,
                "experiment_label": design["experiment_label"],
                "publication_status": design["publication_status"],
                "host": design["host"],
                "component_isolate_A": isolate_a,
                "component_isolate_B": isolate_b,
                "mixture_ratio": "1:1",
                "total_mixture_dose_conidia_per_ml": total_dose,
                "component_dose_each_conidia_per_ml": component_dose,
                "contrast_definition": "mixture minus mean of dose-matched component scores",
                "mean_mixture_score": float(np.mean(mixture_values)),
                "mean_component_control_average": float(np.mean(control_averages)),
            }
            dose_record.update(sign_test(contrasts))
            dose_specific_results.append(dose_record)

        plant_average_contrasts = [
            float(np.mean(plant_contrasts[plant])) for plant in plants
        ]
        mixture_record = {
            "analysis_family": family_name(design["publication_status"]),
            "experiment_id": experiment_id,
            "experiment_label": design["experiment_label"],
            "publication_status": design["publication_status"],
            "host": design["host"],
            "component_isolate_A": isolate_a,
            "component_isolate_B": isolate_b,
            "mixture_ratio": "1:1",
            "tested_total_mixture_doses_conidia_per_ml": ";".join(map(str, mixture_doses)),
            "component_dose_rule": "one-half of total mixture dose for each component",
            "contrast_definition": "plant mean across three doses of mixture minus mean of dose-matched components",
            "plant_average_contrasts": ";".join(f"{value:.8g}" for value in plant_average_contrasts),
        }
        mixture_record.update(sign_test(plant_average_contrasts))
        mixture_results.append(mixture_record)

    for family in ("published_primary", "historical_secondary"):
        family_rows = [row for row in mixture_results if row["analysis_family"] == family]
        adjusted = holm_adjust(row["exact_sign_p"] for row in family_rows)
        for row, p_adjusted in zip(family_rows, adjusted):
            row["p_holm_within_family"] = p_adjusted

    write_csv(output_dir / "dose_trend_results.csv", dose_results)
    write_csv(output_dir / "ordinal_gee_sensitivity.csv", gee_results)
    write_csv(output_dir / "mixture_results.csv", mixture_results)
    write_csv(output_dir / "mixture_by_dose.csv", dose_specific_results)

    summary_lines = [
        "Source-experiment analysis summary",
        "==================================",
        "",
        "Dose analyses",
        "-------------",
    ]
    for row in dose_results:
        summary_lines.append(
            f"{row['experiment_label']} | {row['host']} | {row['isolate']}: "
            f"exact Page P={row['page_p_exact_one_sided']:.6g}; "
            f"Holm P={row['page_p_holm_within_family']:.6g}; "
            f"median plant rho={row['median_plant_spearman_rho']:.3f}; "
            f"mean low/high={row['mean_score_lowest_dose']:.3f}/"
            f"{row['mean_score_highest_dose']:.3f}."
        )
    summary_lines.extend(["", "Mixture analyses", "----------------"])
    for row in mixture_results:
        summary_lines.append(
            f"{row['experiment_label']} | {row['host']} | "
            f"{row['component_isolate_A']} + {row['component_isolate_B']}: "
            f"mean contrast={row['mean_contrast']:.3f}; "
            f"exact sign P={row['exact_sign_p']:.6g}; "
            f"Holm P={row['p_holm_within_family']:.6g}."
        )
    (output_dir / "analysis_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "inputs": [
            {
                "filename": filename,
                "sha256": sha256(recovered_dir / filename),
            }
            for filename in (
                "recovered_ordinal_scores.csv",
                "recovered_plant_condition_means.csv",
                "experiment_design.csv",
            )
        ],
        "outputs": [],
    }
    for path in sorted(output_dir.glob("*.csv")):
        manifest["outputs"].append({"filename": path.name, "sha256": sha256(path)})
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.recovered_dir, args.output_dir)


if __name__ == "__main__":
    main()
