from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import read_csv_required, safe_mkdir, write_manifest


@dataclass(frozen=True)
class InputPaths:
    dose_response_master: Path
    mixture_master: Path
    barrier_summary_master: Path


def _require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_dose_response_tables(dose: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dose-response logistic fits by host x isolate.

    Model: logit(P(severe)) = intercept + alpha * log10(concentration)

    Returns
    -------
    fit_details : pd.DataFrame
        Includes both successful and failed fits with diagnostic fields.
    summary_success : pd.DataFrame
        Successful fits only, matching the compact table used for manuscript plotting.
    """
    required = ["host", "is_mixture_clean", "isolate_clean", "concentration", "score"]
    _require_columns(dose, required, "dose_response_master.csv")

    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)
    df["log10C"] = np.log10(df["concentration"].astype(float))

    rows: List[Dict] = []
    for (host, isolate), g in df.groupby(["host", "isolate_clean"], sort=True):
        y = g["severe"].astype(int)
        out: Dict = {"host": host, "isolate": isolate, "observations_used": int(len(g))}
        if y.nunique() < 2:
            out.update({"fit_success": False, "fit_note": "severe mono"})
            rows.append(out)
            continue
        if g["log10C"].nunique() < 2:
            out.update({"fit_success": False, "fit_note": "dose mono"})
            rows.append(out)
            continue

        X = sm.add_constant(g["log10C"])
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            intercept = float(res.params["const"])
            alpha = float(res.params["log10C"])
            alpha_se = float(res.bse["log10C"]) if "log10C" in res.bse else np.nan

            log10_C50 = -intercept / alpha if alpha != 0 else np.nan
            C50 = float(10 ** log10_C50) if np.isfinite(log10_C50) else np.nan

            out.update(
                {
                    "fit_success": True,
                    "fit_note": "OK",
                    "intercept": intercept,
                    "alpha": alpha,
                    "alpha_se": alpha_se,
                    "log10_C50": float(log10_C50),
                    "C50": C50,
                }
            )
        except Exception as e:
            out.update({"fit_success": False, "fit_note": str(e)})
        rows.append(out)

    fit_details = pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)
    summary_success = fit_details[fit_details["fit_success"] == True][["host", "isolate", "alpha", "log10_C50", "C50"]].copy()
    summary_success = summary_success.sort_values(["host", "isolate"]).reset_index(drop=True)
    return fit_details, summary_success


def compute_high_dose_barrier_tables(dose: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Barrier summary at the highest tested dose by host x isolate."""
    required = ["host", "is_mixture_clean", "isolate_clean", "concentration", "score"]
    _require_columns(dose, required, "dose_response_master.csv")

    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)

    def _at_max(g: pd.DataFrame) -> pd.Series:
        maxC = float(g["concentration"].max())
        gg = g[g["concentration"] == maxC]
        n = int(len(gg))
        k = int(gg["severe"].sum())
        p = float(gg["severe"].mean()) if n > 0 else np.nan
        B = float(-np.log(p)) if (np.isfinite(p) and p > 0) else np.inf
        return pd.Series(
            {
                "C_max": maxC,
                "observations_at_Cmax": n,
                "severe_events_at_Cmax": k,
                "p_severe_max": p,
                "B_dose": B,
            }
        )

    details = df.groupby(["host", "isolate_clean"], sort=True).apply(_at_max).reset_index()
    details = details.rename(columns={"isolate_clean": "isolate"})
    details = details.sort_values(["host", "isolate"]).reset_index(drop=True)

    summary = details[["host", "isolate", "B_dose"]].copy()
    return details, summary


def compute_mixed_isolate_bliss(mix: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bliss-independence deviation for mixed-isolate inoculations.

    Returns
    -------
    summary : pd.DataFrame
        Columns: host, environment, mixture_label, observed_p_severe,
        expected_p_severe, delta_bliss.
    single_controls : pd.DataFrame
        Single-isolate severe-disease probabilities used for Bliss expectations.
    """
    required = [
        "host",
        "environment",
        "score",
        "is_mixture_clean",
        "isolate_clean",
        "mixture_label",
        "isolate_1",
        "isolate_2",
        "isolate_3",
    ]
    _require_columns(mix, required, "mixture_master.csv")

    df = mix.copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)

    singles = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    single_controls = (
        singles.groupby(["host", "environment", "isolate_clean"], sort=True)
        .agg(observations_used=("severe", "size"), p_severe=("severe", "mean"))
        .reset_index()
        .rename(columns={"isolate_clean": "isolate"})
    )
    p_map = {(r.host, r.environment, r.isolate): float(r.p_severe) for r in single_controls.itertuples(index=False)}

    mix_rows = df[df["is_mixture_clean"] == True].copy()

    def _expected(row: pd.Series) -> float:
        host = row["host"]
        env = row["environment"]

        isolates: List[str] = []
        for col in ["isolate_1", "isolate_2", "isolate_3"]:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                isolates.append(val.strip())

        if not isolates and isinstance(row.get("mixture_label"), str):
            isolates = [x.strip() for x in row["mixture_label"].split("+") if x.strip()]

        if not isolates:
            return np.nan

        ps: List[float] = []
        for iso in isolates:
            key = (host, env, iso)
            if key not in p_map:
                return np.nan
            ps.append(p_map[key])

        prod = 1.0
        for p in ps:
            prod *= (1.0 - p)
        return float(1.0 - prod)

    mix_rows["expected_p_severe_row"] = mix_rows.apply(_expected, axis=1)
    mix_rows["observed_p_severe_row"] = mix_rows["severe"].astype(float)

    summary = (
        mix_rows.groupby(["host", "environment", "mixture_label"], sort=True)
        .agg(
            observed_p_severe=("observed_p_severe_row", "mean"),
            expected_p_severe=("expected_p_severe_row", "mean"),
            observations_used=("observed_p_severe_row", "size"),
        )
        .reset_index()
    )
    summary["delta_bliss"] = summary["observed_p_severe"] - summary["expected_p_severe"]
    summary = summary.sort_values(["host", "environment", "mixture_label"]).reset_index(drop=True)

    # Compact table matches the manuscript-oriented output; keep n in the detailed single-control table.
    compact = summary[["host", "environment", "mixture_label", "observed_p_severe", "expected_p_severe", "delta_bliss"]].copy()
    return compact, single_controls


def compute_mixed_isolate_host_env_summary(
    mix: pd.DataFrame, mixed_isolate_bliss: pd.DataFrame, severe_cutoff: int
) -> pd.DataFrame:
    """Host x environment summary for mixed-isolate experiments."""
    df = mix.copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)

    singles = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    p_iso = (
        singles.groupby(["host", "environment", "isolate_clean"], sort=True)
        .agg(p=("severe", "mean"))
        .reset_index()
    )

    host_env = (
        p_iso.groupby(["host", "environment"], sort=True)
        .agg(
            mean_single_isolate_p=("p", "mean"),
            max_single_isolate_p=("p", "max"),
            n_single_isolates=("isolate_clean", "nunique"),
        )
        .reset_index()
    )

    delta_env = (
        mixed_isolate_bliss.groupby(["host", "environment"], sort=True)
        .agg(
            mean_delta_bliss=("delta_bliss", "mean"),
            min_delta_bliss=("delta_bliss", "min"),
            max_delta_bliss=("delta_bliss", "max"),
            n_mixtures=("delta_bliss", "size"),
        )
        .reset_index()
    )

    out = host_env.merge(delta_env, on=["host", "environment"], how="outer")
    out = out[
        [
            "host",
            "environment",
            "mean_single_isolate_p",
            "max_single_isolate_p",
            "n_single_isolates",
            "mean_delta_bliss",
            "min_delta_bliss",
            "max_delta_bliss",
            "n_mixtures",
        ]
    ].sort_values(["host", "environment"]).reset_index(drop=True)
    return out


def compute_regrowth_barrier_changes(barrier: pd.DataFrame) -> pd.DataFrame:
    required = ["experiment", "host", "isolate_clean", "round", "B"]
    _require_columns(barrier, required, "barrier_summary_master.csv")

    reg = barrier[barrier["experiment"] == "regrowth"].copy()
    if reg.empty:
        return pd.DataFrame(columns=["host", "isolate", "B_round1", "B_round2", "delta_B"])

    reg["round"] = reg["round"].astype(int)
    piv = reg.pivot_table(index=["host", "isolate_clean"], columns="round", values="B", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"isolate_clean": "isolate", 1: "B_round1", 2: "B_round2"})
    piv["delta_B"] = piv.get("B_round2") - piv.get("B_round1")
    piv = piv[["host", "isolate", "B_round1", "B_round2", "delta_B"]].sort_values(["host", "isolate"]).reset_index(drop=True)
    return piv


def compute_structural_comparison_summary(barrier: pd.DataFrame) -> pd.DataFrame:
    required = ["experiment"]
    _require_columns(barrier, required, "barrier_summary_master.csv")
    out = barrier[barrier["experiment"] == "leaf_structure"].copy().reset_index(drop=True)
    out = out.rename(columns={"n": "observations_used", "p": "p_severe", "B": "barrier_summary"})
    return out


def compute_host_response_summary(
    dose_hosts: List[str],
    mixture_hosts: List[str],
    dose_response_summary: pd.DataFrame,
    high_dose_barrier_summary: pd.DataFrame,
    mixed_isolate_host_env_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Host-level synthesis table (alpha, B_dose, mean_delta_bliss).

    Host list matches the union of hosts present in the dose-response and mixture masters.
    """
    hosts = sorted(set(dose_hosts) | set(mixture_hosts))
    out = pd.DataFrame({"host": hosts})

    alpha = dose_response_summary.groupby("host", sort=True).agg(alpha=("alpha", "mean")).reset_index()
    out = out.merge(alpha, on="host", how="left")

    def _finite_mean(x: pd.Series) -> float:
        x2 = x.replace([np.inf, -np.inf], np.nan).dropna()
        if x2.empty:
            return np.nan
        return float(x2.mean())

    B_host = high_dose_barrier_summary.groupby("host", sort=True)["B_dose"].apply(_finite_mean).reset_index()
    B_host = B_host.rename(columns={"B_dose": "B_dose"})
    out = out.merge(B_host, on="host", how="left")

    lab = mixed_isolate_host_env_summary[
        mixed_isolate_host_env_summary["environment"] == "lab"
    ][["host", "mean_delta_bliss"]].copy()
    delta_map = dict(zip(lab["host"], lab["mean_delta_bliss"]))
    out["mean_delta_bliss"] = out["host"].map(delta_map).fillna(0.0)

    out = out[["host", "alpha", "B_dose", "mean_delta_bliss"]]
    return out


def _write_excel(path: Path, df: pd.DataFrame, sheet_name: str = "Sheet1") -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet_name)


def _write_supplementary(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            # Excel sheet names are limited to 31 chars.
            sheet = name[:31]
            df.to_excel(xw, index=False, sheet_name=sheet)


def run_pipeline(input_dir: Path, output_dir: Path, severe_cutoff: int = 4) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    safe_mkdir(output_dir)

    inputs = InputPaths(
        dose_response_master=input_dir / "dose_response_master.csv",
        mixture_master=input_dir / "mixture_master.csv",
        barrier_summary_master=input_dir / "barrier_summary_master.csv",
    )

    dose = read_csv_required(inputs.dose_response_master)
    mix = read_csv_required(inputs.mixture_master)
    barrier = read_csv_required(inputs.barrier_summary_master)

    dose_response_fit_details, dose_response_summary = compute_dose_response_tables(dose, severe_cutoff=severe_cutoff)
    high_dose_barrier_details, high_dose_barrier_summary = compute_high_dose_barrier_tables(dose, severe_cutoff=severe_cutoff)

    mixed_isolate_bliss, mixed_isolate_single_controls = compute_mixed_isolate_bliss(mix, severe_cutoff=severe_cutoff)
    mixed_isolate_host_env = compute_mixed_isolate_host_env_summary(
        mix, mixed_isolate_bliss=mixed_isolate_bliss, severe_cutoff=severe_cutoff
    )

    regrowth_barrier_changes = compute_regrowth_barrier_changes(barrier)
    structural_comparison_summary = compute_structural_comparison_summary(barrier)

    dose_hosts = sorted(dose["host"].dropna().unique().tolist()) if "host" in dose.columns else []
    mixture_hosts = sorted(mix["host"].dropna().unique().tolist()) if "host" in mix.columns else []
    host_response_summary = compute_host_response_summary(
        dose_hosts=dose_hosts,
        mixture_hosts=mixture_hosts,
        dose_response_summary=dose_response_summary,
        high_dose_barrier_summary=high_dose_barrier_summary,
        mixed_isolate_host_env_summary=mixed_isolate_host_env,
    )

    outputs: Dict[str, Path] = {}

    # Individual Excel files (compact + detailed where helpful)
    p = output_dir / "dose_response_summary.xlsx"
    _write_excel(p, dose_response_summary, sheet_name="dose_response_summary")
    outputs["dose_response_summary"] = p

    p = output_dir / "dose_response_fit_details.xlsx"
    _write_excel(p, dose_response_fit_details, sheet_name="dose_response_fit_details")
    outputs["dose_response_fit_details"] = p

    p = output_dir / "high_dose_barrier_summary.xlsx"
    _write_excel(p, high_dose_barrier_summary, sheet_name="high_dose_barrier_summary")
    outputs["high_dose_barrier_summary"] = p

    p = output_dir / "high_dose_barrier_details.xlsx"
    _write_excel(p, high_dose_barrier_details, sheet_name="high_dose_barrier_details")
    outputs["high_dose_barrier_details"] = p

    p = output_dir / "mixed_isolate_bliss_summary.xlsx"
    _write_excel(p, mixed_isolate_bliss, sheet_name="mixed_isolate_bliss_summary")
    outputs["mixed_isolate_bliss_summary"] = p

    p = output_dir / "mixed_isolate_single_controls.xlsx"
    _write_excel(p, mixed_isolate_single_controls, sheet_name="mixed_isolate_single_controls")
    outputs["mixed_isolate_single_controls"] = p

    p = output_dir / "mixed_isolate_host_env_summary.xlsx"
    _write_excel(p, mixed_isolate_host_env, sheet_name="mixed_isolate_host_env_summary")
    outputs["mixed_isolate_host_env_summary"] = p

    p = output_dir / "regrowth_barrier_changes.xlsx"
    _write_excel(p, regrowth_barrier_changes, sheet_name="regrowth_barrier_changes")
    outputs["regrowth_barrier_changes"] = p

    p = output_dir / "structural_comparison_summary.xlsx"
    _write_excel(p, structural_comparison_summary, sheet_name="structural_comparison_summary")
    outputs["structural_comparison_summary"] = p

    p = output_dir / "host_response_summary.xlsx"
    _write_excel(p, host_response_summary, sheet_name="host_response_summary")
    outputs["host_response_summary"] = p

    # Supplementary workbook
    supp_path = output_dir / "Supplementary_Data_S1.xlsx"
    _write_supplementary(
        supp_path,
        sheets={
            "dose_response_summary": dose_response_summary,
            "dose_response_fit_details": dose_response_fit_details,
            "high_dose_barrier_summary": high_dose_barrier_summary,
            "high_dose_barrier_details": high_dose_barrier_details,
            "mixed_isolate_bliss_summary": mixed_isolate_bliss,
            "mixed_isolate_single_controls": mixed_isolate_single_controls,
            "mixed_isolate_host_env_summary": mixed_isolate_host_env,
            "regrowth_barrier_changes": regrowth_barrier_changes,
            "structural_comparison_summary": structural_comparison_summary,
            "host_response_summary": host_response_summary,
        },
    )
    outputs["Supplementary_Data_S1"] = supp_path

    # Manifest
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        inputs={
            "dose_response_master.csv": inputs.dose_response_master,
            "mixture_master.csv": inputs.mixture_master,
            "barrier_summary_master.csv": inputs.barrier_summary_master,
        },
        outputs=outputs,
    )
