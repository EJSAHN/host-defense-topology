from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_logit_model(y: pd.Series, x: pd.Series):
    X = sm.add_constant(x.astype(float), has_constant="add")
    return sm.GLM(y.astype(int), X, family=sm.families.Binomial()).fit()


def _fit_logistic_group(g: pd.DataFrame, severe_col: str = "severe") -> Dict:
    """Fit logit(P(severe)) = beta0 + alpha * log10C for one host x isolate group."""
    out: Dict = {
        "observations_used": int(len(g)),
        "dose_levels": int(g["concentration"].nunique()) if "concentration" in g else np.nan,
        "severe_events": int(g[severe_col].sum()) if severe_col in g else np.nan,
        "nonsevere_events": int(len(g) - g[severe_col].sum()) if severe_col in g else np.nan,
    }
    if len(g) == 0 or g[severe_col].nunique() < 2:
        out.update({"fit_success": False, "fit_note": "insufficient severe/non-severe variation"})
        return out
    if g["log10C"].nunique() < 2:
        out.update({"fit_success": False, "fit_note": "insufficient dose variation"})
        return out

    try:
        res = _safe_logit_model(g[severe_col], g["log10C"])
        intercept = float(res.params["const"])
        alpha = float(res.params["log10C"])
        cov = res.cov_params()
        alpha_se = float(res.bse["log10C"])
        alpha_lcl, alpha_ucl = [float(v) for v in res.conf_int().loc["log10C"]]

        log10_C50 = -intercept / alpha if alpha != 0 else np.nan
        grad = np.array([-1.0 / alpha, intercept / (alpha * alpha)], dtype=float)
        vcov = cov.loc[["const", "log10C"], ["const", "log10C"]].to_numpy(dtype=float)
        log10_C50_se = float(np.sqrt(np.maximum(0.0, grad @ vcov @ grad))) if np.isfinite(log10_C50) else np.nan
        log10_C50_lcl = log10_C50 - 1.96 * log10_C50_se if np.isfinite(log10_C50_se) else np.nan
        log10_C50_ucl = log10_C50 + 1.96 * log10_C50_se if np.isfinite(log10_C50_se) else np.nan

        pred = np.asarray(res.predict(sm.add_constant(g["log10C"], has_constant="add")))
        eps = 1e-12
        y = g[severe_col].to_numpy(dtype=float)
        brier = float(np.mean((y - pred) ** 2))
        log_loss = float(-np.mean(y * np.log(np.clip(pred, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - pred, eps, 1 - eps))))
        tjur_r2 = float(np.nan)
        if y.sum() > 0 and (1 - y).sum() > 0:
            tjur_r2 = float(np.mean(pred[y == 1]) - np.mean(pred[y == 0]))

        out.update(
            {
                "fit_success": True,
                "fit_note": "OK",
                "intercept": intercept,
                "alpha": alpha,
                "alpha_se": alpha_se,
                "alpha_lcl_delta": alpha_lcl,
                "alpha_ucl_delta": alpha_ucl,
                "log10_C50": float(log10_C50),
                "log10_C50_se_delta": log10_C50_se,
                "log10_C50_lcl_delta": float(log10_C50_lcl),
                "log10_C50_ucl_delta": float(log10_C50_ucl),
                "C50": float(10 ** log10_C50) if np.isfinite(log10_C50) else np.nan,
                "C50_lcl_delta": float(10 ** log10_C50_lcl) if np.isfinite(log10_C50_lcl) else np.nan,
                "C50_ucl_delta": float(10 ** log10_C50_ucl) if np.isfinite(log10_C50_ucl) else np.nan,
                "aic": float(res.aic),
                "deviance": float(res.deviance),
                "pearson_chi2": float(res.pearson_chi2),
                "brier_score": brier,
                "log_loss": log_loss,
                "tjur_r2": tjur_r2,
            }
        )
    except Exception as e:
        out.update({"fit_success": False, "fit_note": repr(e)})
    return out


def _bootstrap_logistic_group(g: pd.DataFrame, n_boot: int, rng: np.random.Generator, severe_col: str = "severe") -> Dict:
    """Nonparametric bootstrap CI for alpha and log10_C50 within a group."""
    if n_boot <= 0 or len(g) == 0:
        return {}
    vals_alpha: List[float] = []
    vals_log10: List[float] = []
    idx = np.arange(len(g))
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        gb = g.iloc[sample_idx]
        if gb[severe_col].nunique() < 2 or gb["log10C"].nunique() < 2:
            continue
        try:
            res = _safe_logit_model(gb[severe_col], gb["log10C"])
            intercept = float(res.params["const"])
            alpha = float(res.params["log10C"])
            if alpha == 0:
                continue
            log10_c50 = -intercept / alpha
            if np.isfinite(alpha) and np.isfinite(log10_c50):
                vals_alpha.append(alpha)
                vals_log10.append(log10_c50)
        except Exception:
            continue
    out: Dict = {"bootstrap_successful_fits": len(vals_alpha)}
    if len(vals_alpha) >= max(20, int(0.05 * n_boot)):
        a = np.array(vals_alpha, dtype=float)
        c = np.array(vals_log10, dtype=float)
        out.update(
            {
                "alpha_lcl_boot": float(np.nanpercentile(a, 2.5)),
                "alpha_ucl_boot": float(np.nanpercentile(a, 97.5)),
                "log10_C50_lcl_boot": float(np.nanpercentile(c, 2.5)),
                "log10_C50_ucl_boot": float(np.nanpercentile(c, 97.5)),
                "C50_lcl_boot": float(10 ** np.nanpercentile(c, 2.5)),
                "C50_ucl_boot": float(10 ** np.nanpercentile(c, 97.5)),
            }
        )
    else:
        out.update(
            {
                "alpha_lcl_boot": np.nan,
                "alpha_ucl_boot": np.nan,
                "log10_C50_lcl_boot": np.nan,
                "log10_C50_ucl_boot": np.nan,
                "C50_lcl_boot": np.nan,
                "C50_ucl_boot": np.nan,
            }
        )
    return out


def dose_response_uncertainty(dose: pd.DataFrame, severe_cutoff: int = 4, n_boot: int = 1000, seed: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)
    df["log10C"] = np.log10(df["concentration"].astype(float))
    rng = np.random.default_rng(seed)

    rows: List[Dict] = []
    pred_rows: List[pd.DataFrame] = []
    cal_rows: List[Dict] = []
    for (host, isolate), g in df.groupby(["host", "isolate_clean"], sort=True):
        base = {"host": host, "isolate": isolate, "severe_cutoff": severe_cutoff}
        fit = _fit_logistic_group(g)
        boot = _bootstrap_logistic_group(g, n_boot=n_boot, rng=rng) if fit.get("fit_success") else {}
        row = {**base, **fit, **boot}
        rows.append(row)

        if fit.get("fit_success"):
            try:
                res = _safe_logit_model(g["severe"], g["log10C"])
                gp = g.copy()
                gp["predicted_p_severe"] = res.predict(sm.add_constant(gp["log10C"], has_constant="add"))
                gp["host"] = host
                gp["isolate"] = isolate
                pred_rows.append(gp[["host", "isolate", "concentration", "log10C", "score", "severe", "predicted_p_severe"]])
                by_dose = gp.groupby("concentration", sort=True).agg(
                    observations=("severe", "size"),
                    observed_p_severe=("severe", "mean"),
                    predicted_p_severe=("predicted_p_severe", "mean"),
                    mean_score=("score", "mean"),
                    variance_score=("score", "var"),
                ).reset_index()
                by_dose["host"] = host
                by_dose["isolate"] = isolate
                cal_rows.extend(by_dose.to_dict("records"))
            except Exception:
                pass

    fit_uncert = pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    calibration = pd.DataFrame(cal_rows).sort_values(["host", "isolate", "concentration"]).reset_index(drop=True) if cal_rows else pd.DataFrame()
    return fit_uncert, predictions, calibration


def high_dose_barrier_uncertainty(dose: pd.DataFrame, severe_cutoff: int = 4) -> pd.DataFrame:
    if stats is None:
        raise RuntimeError("scipy is required for beta confidence intervals")
    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)
    rows: List[Dict] = []
    for (host, isolate), g in df.groupby(["host", "isolate_clean"], sort=True):
        cmax = float(g["concentration"].max())
        gm = g[g["concentration"] == cmax]
        n = int(len(gm))
        k = int(gm["severe"].sum())
        p = k / n if n else np.nan
        p_jeff = (k + 0.5) / (n + 1) if n else np.nan
        p_lcl = float(stats.beta.ppf(0.025, k + 0.5, n - k + 0.5)) if n else np.nan
        p_ucl = float(stats.beta.ppf(0.975, k + 0.5, n - k + 0.5)) if n else np.nan
        B_empirical = float(-np.log(p)) if p and p > 0 else np.inf
        B_jeff = float(-np.log(p_jeff)) if np.isfinite(p_jeff) and p_jeff > 0 else np.nan
        B_lcl = float(-np.log(p_ucl)) if np.isfinite(p_ucl) and p_ucl > 0 else np.inf
        B_ucl = float(-np.log(p_lcl)) if np.isfinite(p_lcl) and p_lcl > 0 else np.inf
        rows.append(
            {
                "host": host,
                "isolate": isolate,
                "severe_cutoff": severe_cutoff,
                "C_max": cmax,
                "observations_at_Cmax": n,
                "severe_events_at_Cmax": k,
                "p_severe_max": p,
                "p_severe_max_lcl_jeffreys": p_lcl,
                "p_severe_max_ucl_jeffreys": p_ucl,
                "B_dose_empirical": B_empirical,
                "B_dose_jeffreys": B_jeff,
                "B_dose_lcl_jeffreys": B_lcl,
                "B_dose_ucl_jeffreys": B_ucl,
                "finite_empirical_B": np.isfinite(B_empirical),
            }
        )
    return pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)


def _parse_mixture_isolates(row: pd.Series) -> List[str]:
    isolates: List[str] = []
    for col in ["isolate_1", "isolate_2", "isolate_3"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            isolates.append(val.strip())
    if not isolates and isinstance(row.get("mixture_label"), str):
        isolates = [x.strip() for x in row["mixture_label"].split("+") if x.strip()]
    return isolates


def mixed_isolate_bliss_uncertainty(mix: pd.DataFrame, severe_cutoff: int = 4, n_boot: int = 5000, seed: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = mix.copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)
    rng = np.random.default_rng(seed)

    singles = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    single_groups = {key: g["severe"].to_numpy(dtype=int) for key, g in singles.groupby(["host", "environment", "isolate_clean"], sort=True)}
    single_controls = singles.groupby(["host", "environment", "isolate_clean"], sort=True).agg(
        observations_used=("severe", "size"),
        severe_events=("severe", "sum"),
        p_severe=("severe", "mean"),
        mean_score=("score", "mean"),
        source_files=("source_file", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
    ).reset_index().rename(columns={"isolate_clean": "isolate"})

    mix_rows = df[df["is_mixture_clean"] == True].copy()
    rows: List[Dict] = []
    composition_rows: List[Dict] = []
    for (host, env, mix_label), g in mix_rows.groupby(["host", "environment", "mixture_label"], sort=True):
        ex_row = g.iloc[0]
        isolates = _parse_mixture_isolates(ex_row)
        missing = []
        ps = []
        ns_controls = []
        for iso in isolates:
            key = (host, env, iso)
            arr = single_groups.get(key)
            if arr is None:
                missing.append(iso)
            else:
                ps.append(float(np.mean(arr)))
                ns_controls.append(int(len(arr)))
        if ps:
            prod = 1.0
            for p in ps:
                prod *= (1 - p)
            p_exp = 1 - prod
        else:
            p_exp = np.nan
        obs_arr = g["severe"].to_numpy(dtype=int)
        p_obs = float(np.mean(obs_arr)) if len(obs_arr) else np.nan
        delta = p_obs - p_exp if np.isfinite(p_exp) else np.nan

        boot_vals = []
        if n_boot > 0 and not missing and len(obs_arr):
            for _ in range(n_boot):
                obs_b = float(np.mean(rng.choice(obs_arr, size=len(obs_arr), replace=True)))
                prod_b = 1.0
                ok = True
                for iso in isolates:
                    arr = single_groups.get((host, env, iso))
                    if arr is None or len(arr) == 0:
                        ok = False
                        break
                    p_b = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
                    prod_b *= (1.0 - p_b)
                if ok:
                    boot_vals.append(obs_b - (1.0 - prod_b))
        if len(boot_vals) >= max(20, int(0.05 * n_boot)):
            lcl = float(np.nanpercentile(boot_vals, 2.5))
            ucl = float(np.nanpercentile(boot_vals, 97.5))
        else:
            lcl = np.nan
            ucl = np.nan
        rows.append(
            {
                "host": host,
                "environment": env,
                "mixture_label": mix_label,
                "isolate_components": "+".join(isolates),
                "observed_p_severe": p_obs,
                "expected_p_severe": p_exp,
                "delta_bliss": delta,
                "delta_bliss_lcl_boot": lcl,
                "delta_bliss_ucl_boot": ucl,
                "mixture_observations": int(len(g)),
                "single_control_observations_min": int(np.min(ns_controls)) if ns_controls else np.nan,
                "single_control_observations_total": int(np.sum(ns_controls)) if ns_controls else 0,
                "missing_single_controls": "+".join(missing) if missing else "",
                "bootstrap_successful_draws": len(boot_vals),
            }
        )
        composition_rows.append(
            {
                "host": host,
                "environment": env,
                "mixture_label": mix_label,
                "isolate_1": isolates[0] if len(isolates) > 0 else "",
                "isolate_2": isolates[1] if len(isolates) > 1 else "",
                "isolate_3": isolates[2] if len(isolates) > 2 else "",
                "mixture_observations": int(len(g)),
                "single_controls_available": len(missing) == 0,
                "missing_single_controls": "+".join(missing) if missing else "",
            }
        )
    summary = pd.DataFrame(rows).sort_values(["host", "environment", "mixture_label"]).reset_index(drop=True)
    comp = pd.DataFrame(composition_rows).sort_values(["host", "environment", "mixture_label"]).reset_index(drop=True)
    return summary, single_controls, comp


def regrowth_uncertainty(barrier: pd.DataFrame, n_boot: int = 5000, seed: int = 1) -> pd.DataFrame:
    if stats is None:
        raise RuntimeError("scipy is required for Fisher exact tests")
    rng = np.random.default_rng(seed)
    reg = barrier[barrier["experiment"] == "regrowth"].copy()
    rows: List[Dict] = []
    for (host, isolate), g in reg.groupby(["host", "isolate_clean"], sort=True):
        rounds = {int(r["round"]): r for _, r in g.iterrows() if pd.notna(r.get("round"))}
        if 1 not in rounds or 2 not in rounds:
            continue
        r1, r2 = rounds[1], rounds[2]
        n1, n2 = int(r1["n"]), int(r2["n"])
        p1, p2 = float(r1["p_severe"]), float(r2["p_severe"])
        k1, k2 = int(round(p1 * n1)), int(round(p2 * n2))
        B1_empirical = float(r1["B"]) if pd.notna(r1["B"]) else np.inf
        B2_empirical = float(r2["B"]) if pd.notna(r2["B"]) else np.inf
        dB_empirical = B2_empirical - B1_empirical if np.isfinite(B1_empirical) and np.isfinite(B2_empirical) else np.nan
        # Jeffreys-corrected values provide finite sensitivity estimates when p_severe is 0.
        p1_jeff = (k1 + 0.5) / (n1 + 1.0)
        p2_jeff = (k2 + 0.5) / (n2 + 1.0)
        B1_jeff = float(-np.log(p1_jeff))
        B2_jeff = float(-np.log(p2_jeff))
        dB_jeff = B2_jeff - B1_jeff
        # Bootstrap from binomial counts. Use Jeffreys correction in each draw to avoid infinite B.
        vals = []
        for _ in range(n_boot):
            kb1 = rng.binomial(n1, p1)
            kb2 = rng.binomial(n2, p2)
            pb1 = (kb1 + 0.5) / (n1 + 1.0)
            pb2 = (kb2 + 0.5) / (n2 + 1.0)
            vals.append(-np.log(pb2) - (-np.log(pb1)))
        lcl, ucl = float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
        try:
            _, fisher_p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        except Exception:
            fisher_p = np.nan
        rows.append(
            {
                "host": host,
                "isolate": isolate,
                "round1_observations": n1,
                "round2_observations": n2,
                "round1_severe_events": k1,
                "round2_severe_events": k2,
                "round1_p_severe": p1,
                "round2_p_severe": p2,
                "B_round1_empirical": B1_empirical,
                "B_round2_empirical": B2_empirical,
                "delta_B_empirical": dB_empirical,
                "B_round1_jeffreys": B1_jeff,
                "B_round2_jeffreys": B2_jeff,
                "delta_B": dB_jeff,
                "delta_B_lcl_boot": lcl,
                "delta_B_ucl_boot": ucl,
                "fisher_exact_p": fisher_p,
            }
        )
    return pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)


def cutoff_sensitivity(dose: pd.DataFrame, mix: pd.DataFrame, barrier: pd.DataFrame, cutoffs: Sequence[int] = (3, 4, 5)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dose_rows: List[pd.DataFrame] = []
    barrier_rows: List[pd.DataFrame] = []
    bliss_rows: List[pd.DataFrame] = []
    for cutoff in cutoffs:
        dfit, _, _ = dose_response_uncertainty(dose, severe_cutoff=cutoff, n_boot=0)
        dfit["severe_cutoff"] = cutoff
        dose_rows.append(dfit)
        bd = high_dose_barrier_uncertainty(dose, severe_cutoff=cutoff)
        bd["severe_cutoff"] = cutoff
        barrier_rows.append(bd)
        bl, _, _ = mixed_isolate_bliss_uncertainty(mix, severe_cutoff=cutoff, n_boot=0)
        bl["severe_cutoff"] = cutoff
        bliss_rows.append(bl)
    return pd.concat(dose_rows, ignore_index=True), pd.concat(barrier_rows, ignore_index=True), pd.concat(bliss_rows, ignore_index=True)




def pooled_dose_model_sensitivity(dose: pd.DataFrame, severe_cutoff: int = 4) -> pd.DataFrame:
    """Pooled logistic sensitivity models to address experimental structure.

    These are not intended to replace the host x isolate descriptive fits. They provide
    extended sensitivity checks with host and isolate fixed effects and cluster-robust
    covariance or GEE grouping where possible.
    """
    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["severe"] = (df["score"] >= severe_cutoff).astype(int)
    df["log10C"] = np.log10(df["concentration"].astype(float))
    if "replicate_id" in df.columns:
        rep = df["replicate_id"].fillna("no_replicate_id").astype(str)
    else:
        rep = pd.Series(["no_replicate_id"] * len(df), index=df.index)
    src = df["source_file"].fillna("unknown_source").astype(str) if "source_file" in df.columns else pd.Series(["unknown_source"] * len(df), index=df.index)
    df["cluster_id"] = src + "|" + df["host"].astype(str) + "|" + rep
    rows: List[Dict] = []

    formulas = [
        ("host_by_dose", "severe ~ log10C * C(host)"),
        ("host_by_dose_plus_isolate_fixed", "severe ~ log10C * C(host) + C(isolate_clean)"),
    ]
    for label, formula in formulas:
        try:
            mod = smf.glm(formula, data=df, family=sm.families.Binomial())
            res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["cluster_id"]})
            rows.append({
                "model": label,
                "method": "GLM binomial with cluster-robust covariance",
                "formula": formula,
                "observations_used": int(len(df)),
                "clusters": int(df["cluster_id"].nunique()),
                "aic": float(res.aic),
                "deviance": float(res.deviance),
                "fit_success": True,
                "fit_note": "OK",
            })
        except Exception as e:
            rows.append({
                "model": label,
                "method": "GLM binomial with cluster-robust covariance",
                "formula": formula,
                "observations_used": int(len(df)),
                "clusters": int(df["cluster_id"].nunique()),
                "aic": np.nan,
                "deviance": np.nan,
                "fit_success": False,
                "fit_note": repr(e),
            })

    try:
        gee = smf.gee("severe ~ log10C * C(host)", groups="cluster_id", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
        gres = gee.fit()
        rows.append({
            "model": "host_by_dose_GEE",
            "method": "GEE binomial, exchangeable working correlation by source_file|host|replicate_id",
            "formula": "severe ~ log10C * C(host)",
            "observations_used": int(len(df)),
            "clusters": int(df["cluster_id"].nunique()),
            "aic": np.nan,
            "deviance": np.nan,
            "fit_success": True,
            "fit_note": "OK",
        })
    except Exception as e:
        rows.append({
            "model": "host_by_dose_GEE",
            "method": "GEE binomial, exchangeable working correlation by source_file|host|replicate_id",
            "formula": "severe ~ log10C * C(host)",
            "observations_used": int(len(df)),
            "clusters": int(df["cluster_id"].nunique()),
            "aic": np.nan,
            "deviance": np.nan,
            "fit_success": False,
            "fit_note": repr(e),
        })
    return pd.DataFrame(rows)


def continuous_score_trend_sensitivity(dose: pd.DataFrame) -> pd.DataFrame:
    """Use the full 1-5 score scale as a continuous sensitivity analysis.

    The original scores are sometimes averaged and therefore not always true ordinal observations;
    this analysis preserves score magnitude without claiming a proportional-odds ordinal model.
    """
    df = dose.copy()
    df = df[(df["is_mixture_clean"] == False) & (df["isolate_clean"].notna())].copy()
    df = df[(df["concentration"] > 0) & (df["score"].notna())].copy()
    df["log10C"] = np.log10(df["concentration"].astype(float))
    rows: List[Dict] = []
    for (host, isolate), g in df.groupby(["host", "isolate_clean"], sort=True):
        row = {"host": host, "isolate": isolate, "observations_used": int(len(g)), "dose_levels": int(g["concentration"].nunique())}
        if len(g) < 3 or g["log10C"].nunique() < 2:
            row.update({"fit_success": False, "fit_note": "insufficient data"})
            rows.append(row)
            continue
        try:
            X = sm.add_constant(g["log10C"], has_constant="add")
            res = sm.OLS(g["score"].astype(float), X).fit(cov_type="HC3")
            row.update({
                "fit_success": True,
                "fit_note": "OK",
                "score_intercept": float(res.params["const"]),
                "score_slope_per_log10C": float(res.params["log10C"]),
                "score_slope_se_HC3": float(res.bse["log10C"]),
                "score_slope_p_HC3": float(res.pvalues["log10C"]),
                "r_squared": float(res.rsquared),
            })
        except Exception as e:
            row.update({"fit_success": False, "fit_note": repr(e)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["host", "isolate"]).reset_index(drop=True)

def data_quality_checks(dose: pd.DataFrame, mix: pd.DataFrame, barrier: pd.DataFrame) -> pd.DataFrame:
    """Flag potential metadata issues without modifying the master data.

    These flags are intended for audit only. They are useful for finding typographical
    inconsistencies in treatment labels or mixtures before revising manuscript summaries.
    """
    rows: List[Dict] = []

    def _tokens_from_label(x) -> List[str]:
        if not isinstance(x, str) or not x.strip():
            return []
        for sep in ["&", ",", ";"]:
            x = x.replace(sep, "+")
        return [t.strip() for t in x.split("+") if t.strip()]

    # Mixture label audit: compare mixture_label tokens with isolate_1/2/3 columns.
    if not mix.empty:
        m = mix[mix.get("is_mixture_clean") == True].copy()
        key_cols = [c for c in ["host", "environment", "treatment_raw", "mixture_label", "isolate_1", "isolate_2", "isolate_3", "source_file"] if c in m.columns]
        for vals, g in m[key_cols].drop_duplicates().iterrows():
            label_tokens = set(_tokens_from_label(g.get("mixture_label")))
            iso_tokens = set([str(g.get(c)).strip() for c in ["isolate_1", "isolate_2", "isolate_3"] if c in g and isinstance(g.get(c), str) and str(g.get(c)).strip()])
            if label_tokens and iso_tokens and label_tokens != iso_tokens:
                rows.append({
                    "source_table": "mixture_master",
                    "issue_type": "mixture_label_component_mismatch",
                    "host": g.get("host", ""),
                    "environment": g.get("environment", ""),
                    "treatment_raw": g.get("treatment_raw", ""),
                    "mixture_label": g.get("mixture_label", ""),
                    "isolate_columns": "+".join(sorted(iso_tokens)),
                    "message": "mixture_label tokens do not match isolate_1/isolate_2/isolate_3; verify whether this is a label typo or a parsing issue.",
                    "source_file": g.get("source_file", ""),
                })
        # Missing single controls for mixture rows under host/environment.
        singles = m.iloc[0:0]
        if "is_mixture_clean" in mix.columns:
            singles = mix[(mix["is_mixture_clean"] == False) & (mix.get("isolate_clean").notna())].copy()
        single_keys = set(zip(singles.get("host", []), singles.get("environment", []), singles.get("isolate_clean", [])))
        for _, g in m[[c for c in ["host", "environment", "mixture_label", "isolate_1", "isolate_2", "isolate_3", "source_file"] if c in m.columns]].drop_duplicates().iterrows():
            missing = []
            for iso in [g.get(c) for c in ["isolate_1", "isolate_2", "isolate_3"] if c in g]:
                if isinstance(iso, str) and iso.strip() and (g.get("host"), g.get("environment"), iso.strip()) not in single_keys:
                    missing.append(iso.strip())
            if missing:
                rows.append({
                    "source_table": "mixture_master",
                    "issue_type": "missing_single_control",
                    "host": g.get("host", ""),
                    "environment": g.get("environment", ""),
                    "mixture_label": g.get("mixture_label", ""),
                    "isolate_columns": "+".join([str(g.get(c)) for c in ["isolate_1", "isolate_2", "isolate_3"] if c in g and isinstance(g.get(c), str)]),
                    "message": f"Missing single-control rows for component(s): {'+'.join(missing)} under the same host/environment.",
                    "source_file": g.get("source_file", ""),
                })

    # Basic missingness/validity checks.
    for table_name, df, required in [
        ("dose_response_master", dose, ["host", "concentration", "score"]),
        ("mixture_master", mix, ["host", "environment", "score"]),
        ("barrier_summary_master", barrier, ["experiment", "host", "n", "p_severe", "B"]),
    ]:
        for col in required:
            if col in df.columns:
                n_missing = int(df[col].isna().sum())
                if n_missing:
                    rows.append({
                        "source_table": table_name,
                        "issue_type": "missing_required_value",
                        "column": col,
                        "message": f"{n_missing} missing values in required audit column {col}.",
                    })
        if "score" in df.columns:
            bad = df[(df["score"].notna()) & ((df["score"] < 1) | (df["score"] > 5))]
            if len(bad):
                rows.append({
                    "source_table": table_name,
                    "issue_type": "score_out_of_range",
                    "message": f"{len(bad)} scores outside the expected 1-5 range.",
                })
    return pd.DataFrame(rows)

def design_audit_tables(dose: pd.DataFrame, mix: pd.DataFrame, barrier: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Host summary across all masters.
    hosts = sorted(set(dose.get("host", pd.Series(dtype=str)).dropna()) | set(mix.get("host", pd.Series(dtype=str)).dropna()) | set(barrier.get("host", pd.Series(dtype=str)).dropna()))
    host_rows: List[Dict] = []
    for host in hosts:
        host_rows.append(
            {
                "host": host,
                "host_group_inferred": "Johnsongrass" if str(host).startswith("SH") else "Sorghum",
                "in_dose_response_master": host in set(dose.get("host", [])),
                "in_mixture_master": host in set(mix.get("host", [])),
                "in_barrier_summary_master": host in set(barrier.get("host", [])),
                "dose_response_observations": int((dose.get("host") == host).sum()) if "host" in dose else 0,
                "mixture_observations": int((mix.get("host") == host).sum()) if "host" in mix else 0,
                "barrier_summary_rows": int((barrier.get("host") == host).sum()) if "host" in barrier else 0,
            }
        )
    host_table = pd.DataFrame(host_rows)

    isolates = sorted(
        set(dose.get("isolate_clean", pd.Series(dtype=str)).dropna())
        | set(mix.get("isolate_clean", pd.Series(dtype=str)).dropna())
        | set(mix.get("isolate_1", pd.Series(dtype=str)).dropna())
        | set(mix.get("isolate_2", pd.Series(dtype=str)).dropna())
        | set(mix.get("isolate_3", pd.Series(dtype=str)).dropna())
        | set(barrier.get("isolate_clean", pd.Series(dtype=str)).dropna())
    )
    iso_rows: List[Dict] = []
    for iso in isolates:
        in_mix_component = pd.Series(False, index=mix.index)
        for col in ["isolate_1", "isolate_2", "isolate_3"]:
            if col in mix:
                in_mix_component = in_mix_component | (mix[col] == iso)
        iso_rows.append(
            {
                "isolate": iso,
                "dose_response_rows": int((dose.get("isolate_clean") == iso).sum()) if "isolate_clean" in dose else 0,
                "mixture_single_control_rows": int(((mix.get("isolate_clean") == iso) & (mix.get("is_mixture_clean") == False)).sum()) if "isolate_clean" in mix else 0,
                "mixture_component_rows": int(in_mix_component.sum()) if len(mix) else 0,
                "barrier_summary_rows": int((barrier.get("isolate_clean") == iso).sum()) if "isolate_clean" in barrier else 0,
                "hosts_observed": "; ".join(sorted(set(pd.concat([
                    dose.loc[dose.get("isolate_clean") == iso, "host"] if "isolate_clean" in dose else pd.Series(dtype=str),
                    mix.loc[(mix.get("isolate_clean") == iso) | in_mix_component, "host"] if "isolate_clean" in mix else pd.Series(dtype=str),
                    barrier.loc[barrier.get("isolate_clean") == iso, "host"] if "isolate_clean" in barrier else pd.Series(dtype=str),
                ]).dropna().astype(str))))
            }
        )
    isolate_table = pd.DataFrame(iso_rows)

    dose_design = dose.copy()
    if not dose_design.empty:
        dose_design["severe_score_ge_4"] = (dose_design["score"] >= 4).astype(int)
        dose_design = dose_design[(dose_design["is_mixture_clean"] == False) & dose_design["isolate_clean"].notna()].groupby(
            ["host", "isolate_clean", "concentration"], sort=True
        ).agg(
            observations_used=("score", "size"),
            severe_events=("severe_score_ge_4", "sum"),
            mean_score=("score", "mean"),
            variance_score=("score", "var"),
            replicate_ids=("replicate_id", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
            source_files=("source_file", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
        ).reset_index().rename(columns={"isolate_clean": "isolate"})

    barrier_design = barrier.copy()
    if not barrier_design.empty:
        if "isolate_clean" in barrier_design.columns:
            barrier_design["isolate_for_analysis"] = barrier_design["isolate_clean"].where(
                barrier_design["isolate_clean"].notna(), barrier_design.get("isolate")
            )
        elif "isolate" in barrier_design.columns:
            barrier_design["isolate_for_analysis"] = barrier_design["isolate"]
        cols = [c for c in ["experiment", "host", "isolate_for_analysis", "round", "n", "mean_score", "p_severe", "B", "tissue", "surface_treatment", "growth_stage", "leaf_part", "thickness_mean", "leaf_angle_mean"] if c in barrier_design]
        barrier_design = barrier_design[cols].sort_values([c for c in ["experiment", "host", "isolate_for_analysis", "round"] if c in cols]).reset_index(drop=True)

    return {
        "host_backgrounds": host_table,
        "isolate_inventory": isolate_table,
        "dose_response_design": dose_design,
        "barrier_summary_design": barrier_design,
    }


def run_extended_analysis(input_dir: Path, output_dir: Path, severe_cutoff: int = 4, n_bootstrap: int = 2000, seed: int = 1) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    tables_dir = output_dir / "tables"
    _ensure_dir(tables_dir)

    dose = pd.read_csv(input_dir / "dose_response_master.csv")
    mix = pd.read_csv(input_dir / "mixture_master.csv")
    barrier = pd.read_csv(input_dir / "barrier_summary_master.csv")

    dose_fit, dose_pred, dose_cal = dose_response_uncertainty(dose, severe_cutoff=severe_cutoff, n_boot=n_bootstrap, seed=seed)
    barrier_unc = high_dose_barrier_uncertainty(dose, severe_cutoff=severe_cutoff)
    bliss_unc, single_controls, mixture_comp = mixed_isolate_bliss_uncertainty(mix, severe_cutoff=severe_cutoff, n_boot=n_bootstrap, seed=seed + 17)
    regrowth_unc = regrowth_uncertainty(barrier, n_boot=n_bootstrap, seed=seed + 23)
    cutoff_dose, cutoff_barrier, cutoff_bliss = cutoff_sensitivity(dose, mix, barrier, cutoffs=(3, 4, 5))
    pooled_model_sensitivity = pooled_dose_model_sensitivity(dose, severe_cutoff=severe_cutoff)
    full_score_sensitivity = continuous_score_trend_sensitivity(dose)
    design_tables = design_audit_tables(dose, mix, barrier)
    data_quality = data_quality_checks(dose, mix, barrier)

    # Host-level mean summaries with uncertainty-aware ingredients.
    host_alpha = dose_fit[dose_fit["fit_success"] == True].groupby("host", sort=True).agg(
        alpha_mean=("alpha", "mean"),
        alpha_min=("alpha", "min"),
        alpha_max=("alpha", "max"),
        fitted_host_isolate_curves=("alpha", "size"),
    ).reset_index()
    host_B = barrier_unc.replace([np.inf, -np.inf], np.nan).groupby("host", sort=True).agg(
        B_dose_jeffreys_mean=("B_dose_jeffreys", "mean"),
        B_dose_jeffreys_min=("B_dose_jeffreys", "min"),
        B_dose_jeffreys_max=("B_dose_jeffreys", "max"),
        high_dose_host_isolate_combinations=("B_dose_jeffreys", "size"),
    ).reset_index()
    lab_bliss = bliss_unc[bliss_unc["environment"] == "lab"].groupby("host", sort=True).agg(
        mean_delta_bliss=("delta_bliss", "mean"),
        min_delta_bliss=("delta_bliss", "min"),
        max_delta_bliss=("delta_bliss", "max"),
        lab_mixtures=("delta_bliss", "size"),
    ).reset_index()
    host_response = host_alpha.merge(host_B, on="host", how="outer").merge(lab_bliss, on="host", how="outer")

    outputs = {
        "dose_response_fit_uncertainty": dose_fit,
        "dose_response_predictions": dose_pred,
        "dose_response_calibration_by_dose": dose_cal,
        "high_dose_barrier_uncertainty": barrier_unc,
        "mixed_isolate_bliss_uncertainty": bliss_unc,
        "mixed_isolate_single_controls": single_controls,
        "mixed_isolate_composition_audit": mixture_comp,
        "regrowth_deltaB_uncertainty": regrowth_unc,
        "cutoff_sensitivity_dose_response": cutoff_dose,
        "cutoff_sensitivity_barrier": cutoff_barrier,
        "cutoff_sensitivity_bliss": cutoff_bliss,
        "host_response_summary_extended": host_response,
        "pooled_dose_model_sensitivity": pooled_model_sensitivity,
        "continuous_score_trend_sensitivity": full_score_sensitivity,
        "data_quality_audit": data_quality,
        **design_tables,
    }

    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)

    with pd.ExcelWriter(output_dir / "Extended_Analysis_Tables.xlsx", engine="openpyxl") as xw:
        for name, df in outputs.items():
            df.to_excel(xw, index=False, sheet_name=name[:31])


    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "severe_cutoff_primary": severe_cutoff,
        "sensitivity_cutoffs": [3, 4, 5],
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "generated_tables": sorted([f.name for f in tables_dir.glob("*.csv")]),
        "notes": [
            "Primary analyses use severe disease score >= severe_cutoff.",
            "Bootstrap intervals are nonparametric for dose response and Bliss, and binomial-parametric for regrowth summary counts.",
            "High-dose barrier intervals use Jeffreys beta-binomial intervals for p_severe_max and transform to B=-ln(p).",
            "These outputs provide uncertainty summaries, diagnostics, sensitivity checks, and data-structure audits derived from the master data.",
        ],
    }
    with open(output_dir / "extended_analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
