#!/usr/bin/env python3
"""Recover plant-level ordinal scores from the original excised-leaf workbooks.

The raw workbooks store one plant per row. Each treatment cell contains a
three-digit (occasionally four-digit) code whose digits are 1-5 ordinal
subscores. Matching plant numbers are repeated across the two single-isolate
sections and the mixture section within each source experiment.

Outputs
-------
recovered_ordinal_scores.csv
    One row per digit-level technical subscore.
recovered_plant_condition_means.csv
    One row per plant-by-treatment cell.
experiment_design.csv
    Reader-facing experiment provenance and design summary.
recovery_validation.txt
    Automated validation report.
input_manifest.json
    SHA-256 hashes of the three source workbooks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS, "r": REL_NS}

SINGLE_DOSES = (1_000_000, 500_000, 100_000, 50_000, 10_000, 5_000)
MIXTURE_TOTAL_DOSES = (1_000_000, 100_000, 10_000)


@dataclass(frozen=True)
class SectionSpec:
    header_row: int
    plant_rows: range
    treatment_class: str
    isolates: tuple[str, ...]
    doses: tuple[int, ...]


@dataclass(frozen=True)
class WorkbookSpec:
    public_name: str
    legacy_names: tuple[str, ...]
    expected_sha256: str


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_label: str
    source_workbook: str
    host: str
    host_type: str
    publication_status: str
    source_reference: str
    metadata_scope: str
    sections: tuple[SectionSpec, ...]


WORKBOOK_SPECS = (
    WorkbookSpec(
        "source_workbook_A_experiments_1_2.xlsx",
        ("Raw score2 (1).xlsx",),
        "74ccb911db053406833aeab2489aa9f8770d679e6e88cce035df6ae65bbbaf54",
    ),
    WorkbookSpec(
        "source_workbook_B_experiment_3_historical_block_1.xlsx",
        ("Raw score2 (2).xlsx",),
        "bfb571ec640d64a0c7ba4c1c82f48bb5e72fcd7387a3ff66cee909c8ff3ae05b",
    ),
    WorkbookSpec(
        "source_workbook_C_experiments_4_5_historical_block_2.xlsx",
        ("Experiment 3 Leaf Assay-2.xlsx",),
        "e08514e07e161dee2fc013eded8132dcb2c60182f91f1dbcfac0253e3e0f0fc0",
    ),
)


EXPERIMENTS = (
    ExperimentSpec(
        "E1_BTx398_AMP155_AMP170",
        "Experiment 1",
        "source_workbook_A_experiments_1_2.xlsx",
        "BTx398",
        "sorghum",
        "published",
        "Ahn et al. 2021",
        "Published assay conditions available.",
        (
            SectionSpec(2, range(3, 9), "single", ("AMP155",), SINGLE_DOSES),
            SectionSpec(9, range(10, 16), "single", ("AMP170",), SINGLE_DOSES),
            SectionSpec(16, range(17, 23), "mixture", ("AMP155", "AMP170"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "E2_RTx2536_AMP77_AMP207",
        "Experiment 2",
        "source_workbook_A_experiments_1_2.xlsx",
        "RTx2536",
        "sorghum",
        "published",
        "Ahn et al. 2021",
        "Published assay conditions available.",
        (
            SectionSpec(24, range(25, 31), "single", ("AMP77",), SINGLE_DOSES),
            SectionSpec(31, range(32, 38), "single", ("AMP207",), SINGLE_DOSES),
            SectionSpec(38, range(39, 45), "mixture", ("AMP77", "AMP207"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "E3_BTx398_AMP99_AMP170",
        "Experiment 3",
        "source_workbook_B_experiment_3_historical_block_1.xlsx",
        "BTx398",
        "sorghum",
        "published",
        "Ahn et al. 2021",
        "Published assay conditions available.",
        (
            SectionSpec(2, range(3, 10), "single", ("AMP99",), SINGLE_DOSES),
            SectionSpec(10, range(11, 18), "single", ("AMP170",), SINGLE_DOSES),
            SectionSpec(18, range(19, 26), "mixture", ("AMP99", "AMP170"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "H1_RTx2536_AMP20_AMP27",
        "Historical block 1",
        "source_workbook_B_experiment_3_historical_block_1.xlsx",
        "RTx2536",
        "sorghum",
        "historical",
        "Historical screening records",
        "Detailed environmental setpoints were not uniformly recoverable; analyses are restricted to within-block contrasts.",
        (
            SectionSpec(27, range(28, 34), "single", ("AMP20",), SINGLE_DOSES),
            SectionSpec(34, range(35, 41), "single", ("AMP27",), SINGLE_DOSES),
            SectionSpec(41, range(42, 48), "mixture", ("AMP20", "AMP27"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "H2_IS18760_AMP20_AMP27",
        "Historical block 2",
        "source_workbook_C_experiments_4_5_historical_block_2.xlsx",
        "IS18760",
        "sorghum",
        "historical",
        "Historical screening records",
        "Detailed environmental setpoints were not uniformly recoverable; analyses are restricted to within-block contrasts.",
        (
            SectionSpec(2, range(3, 9), "single", ("AMP20",), SINGLE_DOSES),
            SectionSpec(9, range(10, 16), "single", ("AMP27",), SINGLE_DOSES),
            SectionSpec(17, range(18, 24), "mixture", ("AMP20", "AMP27"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "E4_Theis_AMP99_AMP170",
        "Experiment 4",
        "source_workbook_C_experiments_4_5_historical_block_2.xlsx",
        "Theis",
        "sorghum",
        "published",
        "Ahn et al. 2021",
        "Published assay conditions available.",
        (
            SectionSpec(25, range(26, 32), "single", ("AMP99",), SINGLE_DOSES),
            SectionSpec(32, range(33, 39), "single", ("AMP170",), SINGLE_DOSES),
            SectionSpec(40, range(41, 47), "mixture", ("AMP99", "AMP170"), MIXTURE_TOTAL_DOSES),
        ),
    ),
    ExperimentSpec(
        "E5_SH1152_FSP35_FSP53",
        "Experiment 5",
        "source_workbook_C_experiments_4_5_historical_block_2.xlsx",
        "SH1152",
        "johnsongrass",
        "published",
        "Ahn et al. 2021",
        "Published assay conditions available.",
        (
            SectionSpec(48, range(49, 55), "single", ("FSP35",), SINGLE_DOSES),
            SectionSpec(55, range(56, 62), "single", ("FSP53",), SINGLE_DOSES),
            SectionSpec(63, range(64, 70), "mixture", ("FSP35", "FSP53"), MIXTURE_TOTAL_DOSES),
        ),
    ),
)


def column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if match is None:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def read_first_sheet(path: Path) -> dict[int, dict[int, str]]:
    """Read the first worksheet directly from OOXML without altering the file."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join((node.text or "") for node in item.iter(f"{{{MAIN_NS}}}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        first_sheet = workbook.find("m:sheets", NS)[0]
        relationship_id = first_sheet.attrib[f"{{{REL_NS}}}id"]
        target = rel_map[relationship_id]
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")

        worksheet = ET.fromstring(archive.read(str(Path(target))))
        rows: dict[int, dict[int, str]] = {}
        for row in worksheet.find("m:sheetData", NS).findall("m:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", NS):
                index = column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value = ""
                raw_value = cell.find("m:v", NS)
                if cell_type == "inlineStr":
                    inline = cell.find("m:is", NS)
                    if inline is not None:
                        value = "".join((node.text or "") for node in inline.iter(f"{{{MAIN_NS}}}t"))
                elif raw_value is not None:
                    raw = raw_value.text or ""
                    if cell_type == "s":
                        value = shared[int(raw)]
                    else:
                        value = raw
                values[index] = value
            rows[int(row.attrib.get("r", "0"))] = values
        return rows


def parse_ordinal_code(value: str) -> list[int]:
    number = int(round(float(str(value).strip())))
    scores = [int(char) for char in str(number)]
    if not scores or any(score < 1 or score > 5 for score in scores):
        raise ValueError(f"Cell value {value!r} cannot be parsed as 1-5 ordinal subscores.")
    return scores


def plant_number(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    if match is None:
        raise ValueError(f"Plant label lacks a numeric identifier: {label!r}")
    return int(match.group(1))


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


def resolve_source_workbooks(input_dir: Path) -> dict[str, Path]:
    """Resolve preferred public filenames with legacy-name fallback.

    Each accepted file is verified against the SHA-256 hash of the source
    workbook used for the analysis. Outputs always use the preferred public
    filename, so runs are identical whether the local files use preferred or
    legacy names.
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for spec in WORKBOOK_SPECS:
        candidate_names = (spec.public_name, *spec.legacy_names)
        existing = [input_dir / name for name in candidate_names if (input_dir / name).exists()]
        if not existing:
            missing.append(f"{spec.public_name} (accepted legacy name: {', '.join(spec.legacy_names)})")
            continue

        hashes = {path.name: sha256(path) for path in existing}
        invalid = {name: digest for name, digest in hashes.items() if digest != spec.expected_sha256}
        if invalid:
            details = ", ".join(f"{name}: {digest}" for name, digest in sorted(invalid.items()))
            raise ValueError(
                f"Unexpected content for {spec.public_name}. "
                f"Expected SHA-256 {spec.expected_sha256}; found {details}."
            )

        preferred_path = input_dir / spec.public_name
        resolved[spec.public_name] = preferred_path if preferred_path.exists() else existing[0]

    if missing:
        raise FileNotFoundError(
            "Missing required source workbooks:\n- " + "\n- ".join(missing)
        )
    return resolved


def recover(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_workbooks = resolve_source_workbooks(input_dir)
    sheet_cache: dict[str, dict[int, dict[int, str]]] = {}
    score_rows: list[dict] = []
    cell_rows: list[dict] = []
    design_rows: list[dict] = []
    validation_messages: list[str] = []

    for experiment in EXPERIMENTS:
        workbook_path = resolved_workbooks[experiment.source_workbook]
        if experiment.source_workbook not in sheet_cache:
            sheet_cache[experiment.source_workbook] = read_first_sheet(workbook_path)
        rows = sheet_cache[experiment.source_workbook]

        section_plant_sets: list[set[int]] = []
        before_scores = len(score_rows)
        before_cells = len(cell_rows)

        for section in experiment.sections:
            section_plants: set[int] = set()
            for source_row in section.plant_rows:
                row = rows.get(source_row, {})
                number = plant_number(row.get(0, ""))
                section_plants.add(number)
                plant_id = f"{experiment.experiment_id}_P{number:02d}"

                for column, dose in zip(range(1, 1 + len(section.doses)), section.doses):
                    raw_value = row.get(column, "")
                    if raw_value == "":
                        raise ValueError(
                            f"Missing treatment cell: {workbook_path.name}, "
                            f"row {source_row}, column {column + 1}."
                        )
                    scores = parse_ordinal_code(raw_value)
                    component_dose = dose if section.treatment_class == "single" else dose // len(section.isolates)
                    treatment_id = "+".join(section.isolates)
                    condition_id = f"{experiment.experiment_id}|{treatment_id}|{dose}"

                    cell_rows.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "experiment_label": experiment.experiment_label,
                            "publication_status": experiment.publication_status,
                            "source_workbook": experiment.source_workbook,
                            "source_row": source_row,
                            "source_column": column + 1,
                            "host": experiment.host,
                            "host_type": experiment.host_type,
                            "plant_id": plant_id,
                            "plant_number": number,
                            "treatment_class": section.treatment_class,
                            "isolate_1": section.isolates[0],
                            "isolate_2": section.isolates[1] if len(section.isolates) == 2 else "",
                            "mixture_ratio": "1:1" if section.treatment_class == "mixture" else "",
                            "total_dose_conidia_per_ml": dose,
                            "component_dose_conidia_per_ml": component_dose,
                            "condition_id": condition_id,
                            "technical_subsamples": len(scores),
                            "plant_mean_score": sum(scores) / len(scores),
                            "raw_code": str(int(round(float(raw_value)))),
                        }
                    )
                    for subsample_id, score in enumerate(scores, start=1):
                        score_rows.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "experiment_label": experiment.experiment_label,
                                "publication_status": experiment.publication_status,
                                "source_workbook": experiment.source_workbook,
                                "source_row": source_row,
                                "source_column": column + 1,
                                "host": experiment.host,
                                "host_type": experiment.host_type,
                                "plant_id": plant_id,
                                "plant_number": number,
                                "treatment_class": section.treatment_class,
                                "isolate_1": section.isolates[0],
                                "isolate_2": section.isolates[1] if len(section.isolates) == 2 else "",
                                "mixture_ratio": "1:1" if section.treatment_class == "mixture" else "",
                                "total_dose_conidia_per_ml": dose,
                                "component_dose_conidia_per_ml": component_dose,
                                "condition_id": condition_id,
                                "subsample_id": subsample_id,
                                "ordinal_score": score,
                                "raw_code": str(int(round(float(raw_value)))),
                            }
                        )
            section_plant_sets.append(section_plants)

        if len({tuple(sorted(values)) for values in section_plant_sets}) != 1:
            raise ValueError(f"Plant labels are not aligned across sections in {experiment.experiment_id}.")
        aligned_plants = sorted(section_plant_sets[0])

        design_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "experiment_label": experiment.experiment_label,
                "publication_status": experiment.publication_status,
                "source_reference_or_file": experiment.source_reference,
                "source_workbook": experiment.source_workbook,
                "host": experiment.host,
                "host_type": experiment.host_type,
                "component_isolates": "+".join(
                    sorted({isolate for section in experiment.sections for isolate in section.isolates})
                ),
                "assay": "excised-leaf spot inoculation",
                "independent_unit": "plant",
                "plant_linkage_basis": "matching plant numbers repeated across single-isolate and mixture sections",
                "technical_unit": "digit-level 1-5 ordinal subscore nested within plant-treatment cell",
                "plants": len(aligned_plants),
                "plant_condition_cells": len(cell_rows) - before_cells,
                "ordinal_scores": len(score_rows) - before_scores,
                "single_isolate_doses_conidia_per_ml": "5000;10000;50000;100000;500000;1000000",
                "mixture_total_doses_conidia_per_ml": "10000;100000;1000000",
                "mixture_component_doses_conidia_per_ml": "5000;50000;500000",
                "mixture_ratio": "1:1",
                "environmental_metadata": experiment.metadata_scope,
                "allowed_inference": "within-experiment dose trend and dose-matched mixture contrast",
                "cross_experiment_host_effect": "not estimated",
            }
        )
        validation_messages.append(
            f"{experiment.experiment_id}: {len(aligned_plants)} aligned plants; "
            f"{len(cell_rows) - before_cells} plant-condition cells; "
            f"{len(score_rows) - before_scores} ordinal subscores."
        )

    score_counts = Counter(row["ordinal_score"] for row in score_rows)
    technical_counts = Counter(row["technical_subsamples"] for row in cell_rows)
    if any(score not in range(1, 6) for score in score_counts):
        raise ValueError("Recovered scores outside the 1-5 scale.")
    if len({row["plant_id"] for row in cell_rows}) != sum(row["plants"] for row in design_rows):
        raise ValueError("Block-specific plant identifiers are not unique.")

    write_csv(output_dir / "recovered_ordinal_scores.csv", score_rows)
    write_csv(output_dir / "recovered_plant_condition_means.csv", cell_rows)
    write_csv(output_dir / "experiment_design.csv", design_rows)

    report = [
        "Recovery validation",
        "===================",
        f"Source experiments: {len(design_rows)}",
        f"Block-specific plants: {len({row['plant_id'] for row in cell_rows})}",
        f"Plant-condition cells: {len(cell_rows)}",
        f"Ordinal technical scores: {len(score_rows)}",
        f"Technical subsamples per cell: {dict(sorted(technical_counts.items()))}",
        f"Ordinal score counts: {dict(sorted(score_counts.items()))}",
        "",
        *validation_messages,
    ]
    (output_dir / "recovery_validation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "inputs": [
            {
                "filename": spec.public_name,
                "sha256": sha256(resolved_workbooks[spec.public_name]),
                "bytes": resolved_workbooks[spec.public_name].stat().st_size,
            }
            for spec in WORKBOOK_SPECS
        ]
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the three source workbooks under preferred or accepted legacy filenames.",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for recovered CSV files.")
    args = parser.parse_args()
    recover(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
