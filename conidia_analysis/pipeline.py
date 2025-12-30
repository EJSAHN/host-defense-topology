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


def compute_dose_c50_tables(dose: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dose-response logistic fits by host x isolate.

    Model: logit(P(severe)) = intercept + alpha * log10(concentration)

    Returns
    -------
    fits_all : pd.DataFrame
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
        out: Dict = {"host": host, "isolate": isolate, "n_obs": int(len(g))}
        if y.nunique() < 2:
            out.update({"success": False, "note": "severe mono"})
            rows.append(out)
            continue
        if g["log10C"].nunique() < 2:
            out.update({"success": False, "note": "dose mono"})
            rows.append(out)
            continue

        X = sm.add_constant(g["log10C"])
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            intercept = float(res.params["const"])
            alpha = float(res.params["log10C"])
            alpha_se = float(res.bse["log10C"]) if "log10C" in res.bse else np.nan

            log10C50 = -intercept / alpha if alpha != 0 else np.nan
            C50 = float(10 ** log10C50) if np.isfinite(log10C50) else np.nan

            out.update(
                {
                    "success": True,
                    "note": "OK",
                    "intercept": intercept,
                    "alpha": alpha,
                    "alpha_se": alpha_se,
                    "log10C50": float(log10C50),
                    "C50": C50,
                }
            )
        except Exception as e:
            out.update({"success": False, "note": str(e)})
        rows.append(out)

    fits_all = pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)
    summary_success = fits_all[fits_all["success"] == True][["host", "isolate", "alpha", "log10C50", "C50"]].copy()
    summary_success = summary_success.sort_values(["host", "isolate"]).reset_index(drop=True)
    return fits_all, summary_success


def compute_dose_B_tables(dose: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Barrier B at saturating dose (max concentration) by host x isolate."""
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
        return pd.Series({"C_max": maxC, "n_max": n, "k_max": k, "p_severe_max": p, "B_dose": B})

    details = df.groupby(["host", "isolate_clean"], sort=True).apply(_at_max).reset_index()
    details = details.rename(columns={"isolate_clean": "isolate"})
    details = details.sort_values(["host", "isolate"]).reset_index(drop=True)

    summary = details[["host", "isolate", "B_dose"]].copy()
    return details, summary


def compute_mixture_synergy_bliss(mix: pd.DataFrame, severe_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bliss independence synergy for mixed infections.

    Returns
    -------
    summary : pd.DataFrame
        Columns: host, environment, mixture_label, p_obs, p_exp, delta_bliss
    singles_table : pd.DataFrame
        Single-isolate probabilities used for expectation.
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
    singles_table = (
        singles.groupby(["host", "environment", "isolate_clean"], sort=True)
        .agg(n=("severe", "size"), p=("severe", "mean"))
        .reset_index()
    )
    p_map = {(r.host, r.environment, r.isolate_clean): float(r.p) for r in singles_table.itertuples(index=False)}

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

    mix_rows["p_exp_row"] = mix_rows.apply(_expected, axis=1)
    mix_rows["p_obs_row"] = mix_rows["severe"].astype(float)

    summary = (
        mix_rows.groupby(["host", "environment", "mixture_label"], sort=True)
        .agg(p_obs=("p_obs_row", "mean"), p_exp=("p_exp_row", "mean"), n=("p_obs_row", "size"))
        .reset_index()
    )
    summary["delta_bliss"] = summary["p_obs"] - summary["p_exp"]
    summary = summary.sort_values(["host", "environment", "mixture_label"]).reset_index(drop=True)

    # Compact table matches the manuscript-oriented output; keep n in the detailed table.
    compact = summary[["host", "environment", "mixture_label", "p_obs", "p_exp", "delta_bliss"]].copy()
    return compact, singles_table


def compute_mixture_host_env_summary(
    mix: pd.DataFrame, synergy: pd.DataFrame, severe_cutoff: int
) -> pd.DataFrame:
    """Host x environment summary for mixture experiments."""
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
            p_single_mean=("p", "mean"),
            p_single_max=("p", "max"),
            n_isolates=("isolate_clean", "nunique"),
        )
        .reset_index()
    )

    syn_env = (
        synergy.groupby(["host", "environment"], sort=True)
        .agg(
            Bliss_mean=("delta_bliss", "mean"),
            Bliss_min=("delta_bliss", "min"),
            Bliss_max=("delta_bliss", "max"),
            n_mixtures=("delta_bliss", "size"),
        )
        .reset_index()
    )

    out = host_env.merge(syn_env, on=["host", "environment"], how="outer")
    out = out[
        [
            "host",
            "environment",
            "p_single_mean",
            "p_single_max",
            "n_isolates",
            "Bliss_mean",
            "Bliss_min",
            "Bliss_max",
            "n_mixtures",
        ]
    ].sort_values(["host", "environment"]).reset_index(drop=True)
    return out


def compute_regrowth_summary(barrier: pd.DataFrame) -> pd.DataFrame:
    required = ["experiment", "host", "isolate_clean", "round", "B"]
    _require_columns(barrier, required, "barrier_summary_master.csv")

    reg = barrier[barrier["experiment"] == "regrowth"].copy()
    if reg.empty:
        return pd.DataFrame(columns=["host", "isolate", "1", "2", "Delta_B"])

    reg["round"] = reg["round"].astype(int)
    piv = reg.pivot_table(index=["host", "isolate_clean"], columns="round", values="B", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"isolate_clean": "isolate", 1: "1", 2: "2"})
    piv["Delta_B"] = piv.get("2") - piv.get("1")
    piv = piv[["host", "isolate", "1", "2", "Delta_B"]].sort_values(["host", "isolate"]).reset_index(drop=True)
    return piv


def compute_leaf_structure_summary(barrier: pd.DataFrame) -> pd.DataFrame:
    required = ["experiment"]
    _require_columns(barrier, required, "barrier_summary_master.csv")
    out = barrier[barrier["experiment"] == "leaf_structure"].copy().reset_index(drop=True)
    return out


def compute_host_evolution_summary(
    dose_hosts: List[str],
    mixture_hosts: List[str],
    dose_c50_summary: pd.DataFrame,
    dose_B_summary: pd.DataFrame,
    mixture_host_env_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Host-level synthesis table (alpha, B_dose, Bliss_mean).

    Host list matches the union of hosts present in the dose-response and mixture masters.
    """
    hosts = sorted(set(dose_hosts) | set(mixture_hosts))
    out = pd.DataFrame({"host": hosts})

    alpha = dose_c50_summary.groupby("host", sort=True).agg(alpha=("alpha", "mean")).reset_index()
    out = out.merge(alpha, on="host", how="left")

    def _finite_mean(x: pd.Series) -> float:
        x2 = x.replace([np.inf, -np.inf], np.nan).dropna()
        if x2.empty:
            return np.nan
        return float(x2.mean())

    B_host = dose_B_summary.groupby("host", sort=True)["B_dose"].apply(_finite_mean).reset_index()
    B_host = B_host.rename(columns={"B_dose": "B_dose"})
    out = out.merge(B_host, on="host", how="left")

    lab = mixture_host_env_summary[mixture_host_env_summary["environment"] == "lab"][["host", "Bliss_mean"]].copy()
    bliss_map = dict(zip(lab["host"], lab["Bliss_mean"]))
    out["Bliss_mean"] = out["host"].map(bliss_map).fillna(0.0)

    out = out[["host", "alpha", "B_dose", "Bliss_mean"]]
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

    dose_c50_fits_all, dose_c50_summary = compute_dose_c50_tables(dose, severe_cutoff=severe_cutoff)
    dose_B_details, dose_B_summary = compute_dose_B_tables(dose, severe_cutoff=severe_cutoff)

    mix_synergy, mix_singles = compute_mixture_synergy_bliss(mix, severe_cutoff=severe_cutoff)
    mix_host_env = compute_mixture_host_env_summary(mix, synergy=mix_synergy, severe_cutoff=severe_cutoff)

    regrowth = compute_regrowth_summary(barrier)
    leaf_structure = compute_leaf_structure_summary(barrier)

    dose_hosts = sorted(dose["host"].dropna().unique().tolist()) if "host" in dose.columns else []
    mixture_hosts = sorted(mix["host"].dropna().unique().tolist()) if "host" in mix.columns else []
    host_evolution = compute_host_evolution_summary(
        dose_hosts=dose_hosts,
        mixture_hosts=mixture_hosts,
        dose_c50_summary=dose_c50_summary,
        dose_B_summary=dose_B_summary,
        mixture_host_env_summary=mix_host_env,
    )

    outputs: Dict[str, Path] = {}

    # Individual Excel files (compact + detailed where helpful)
    p = output_dir / "dose_c50_summary.xlsx"
    _write_excel(p, dose_c50_summary, sheet_name="dose_c50_summary")
    outputs["dose_c50_summary"] = p

    p = output_dir / "dose_c50_fits_all.xlsx"
    _write_excel(p, dose_c50_fits_all, sheet_name="dose_c50_fits_all")
    outputs["dose_c50_fits_all"] = p

    p = output_dir / "dose_B_dose_summary.xlsx"
    _write_excel(p, dose_B_summary, sheet_name="dose_B_dose_summary")
    outputs["dose_B_dose_summary"] = p

    p = output_dir / "dose_B_dose_details.xlsx"
    _write_excel(p, dose_B_details, sheet_name="dose_B_dose_details")
    outputs["dose_B_dose_details"] = p

    p = output_dir / "mixture_synergy_bliss.xlsx"
    _write_excel(p, mix_synergy, sheet_name="mixture_synergy_bliss")
    outputs["mixture_synergy_bliss"] = p

    p = output_dir / "mixture_singles_table.xlsx"
    _write_excel(p, mix_singles, sheet_name="mixture_singles")
    outputs["mixture_singles_table"] = p

    p = output_dir / "mixture_host_env_summary.xlsx"
    _write_excel(p, mix_host_env, sheet_name="mixture_host_env_summary")
    outputs["mixture_host_env_summary"] = p

    p = output_dir / "regrowth_summary.xlsx"
    _write_excel(p, regrowth, sheet_name="regrowth_summary")
    outputs["regrowth_summary"] = p

    p = output_dir / "leaf_structure_summary.xlsx"
    _write_excel(p, leaf_structure, sheet_name="leaf_structure_summary")
    outputs["leaf_structure_summary"] = p

    p = output_dir / "host_evolution_summary.xlsx"
    _write_excel(p, host_evolution, sheet_name="host_evolution_summary")
    outputs["host_evolution_summary"] = p

    # Supplementary workbook
    supp_path = output_dir / "Supplementary_Data_S1.xlsx"
    _write_supplementary(
        supp_path,
        sheets={
            "dose_c50_summary": dose_c50_summary,
            "dose_c50_fits_all": dose_c50_fits_all,
            "dose_B_dose_summary": dose_B_summary,
            "dose_B_dose_details": dose_B_details,
            "mixture_synergy_bliss": mix_synergy,
            "mixture_singles": mix_singles,
            "mixture_host_env_summary": mix_host_env,
            "regrowth_summary": regrowth,
            "leaf_structure_summary": leaf_structure,
            "host_evolution_summary": host_evolution,
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
