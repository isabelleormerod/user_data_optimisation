#!/usr/bin/env python3
"""
fit_mixed_models.py -- how do the hand-derived metrics vary across HEIGHTS and
PROTOTYPE PARAMETERS, accounting for the repeated-measures structure?

Input: place_metrics_combined.csv (one row per Place event) from metrics.py,
with columns: participant, Length, Size, Weight, Angle, height, + metric columns.

Two analyses:

1. STRATIFIED prototype-factor matrix (the main table).
   Within each height stratum, for each metric, fit ONE mixed model with all
   prototype factors jointly and a participant random intercept:

       metric ~ Length + Size + Weight + C(Angle) + (1 | participant)

   Each factor's p-value is a Wald test on that factor's fixed-effect
   coefficient(s) from the single fitted model (a joint Wald chi-square for the
   two Angle contrasts). Fitting ONCE and testing with Wald avoids the
   instability of refitting reduced models for a likelihood-ratio test, which on
   boundary-prone mixed fits produces spurious p-values. Joint fitting controls
   for the other factors; the model is within one height (removes the height
   confound) with a participant random intercept (removes pseudoreplication).

2. HEIGHT effect (pooled). Per metric, fit
       metric ~ C(height) + Length + Size + Weight + C(Angle) + (1 | participant)
   and Wald-test the height contrasts, so you get whether each metric varies
   across heights.

Why mixed effects: multiple Place events per participant are not independent; a
participant random intercept models that dependence. Because each participant ran
the full 2^3 of Length/Size/Weight, those effects are within-subject; Angle is
balanced-incomplete across participants. Random intercept only is the defensible
baseline (a by-participant random slope is a sensible extension where the data
support it, but often will not converge at this n).

Outputs (under <out-dir>, default alongside the input):
  mixed_stratified_tests.csv   tidy: stratum, factor, metric, p_value, effect, n
  mixed_height_tests.csv       tidy: metric, p_height, n
  console: the p-value matrix in Factor x Metric x stratum form.

Usage:
  python fit_mixed_models.py --metrics-csv .../place_metrics_combined.csv
  python fit_mixed_models.py --landmarks-root A:\\Automated_chain_BETA\\Participant_Landmarks
  python fit_mixed_models.py --metrics-csv ... --participants P007 P008
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
import statsmodels.formula.api as smf

METRIC_LABELS = {
    "duration_s":         "Duration",
    "perp_mean_deg":      "Perpendicularity (mean)",
    "leftright_mean_deg": "Left/right tilt (mean)",
    "updown_mean_deg":    "Up/down tilt (mean)",
    "pos_jitter_mm":      "Positional jitter",
    "ang_jitter_deg":     "Angular jitter",
    "aperture_mm":        "Aperture",
    "aperture_comfort":   "Aperture comfort",
    "rula_score":         "RULA",
}
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
HEIGHT_ORDER = ["High", "Medium", "Low"]


def fit_full(data, response, factors):
    """Fit metric ~ sum(C(factor)) + (1|participant); try optimisers in turn."""
    present = [f for f in factors if data[f].nunique() >= 2]
    if not present:
        return None, present
    formula = f"{response} ~ " + " + ".join(f"C({f})" for f in present)
    for method in ("lbfgs", "powell", "cg"):
        try:
            res = smf.mixedlm(formula, data, groups=data["participant"]).fit(
                reml=False, method=method)
            if np.isfinite(res.llf):
                return res, present
        except Exception:
            continue
    return None, present


def wald_factor(res, factor):
    """Joint Wald p-value + effect size for one factor's coefficient(s)."""
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    if not names:
        return np.nan, np.nan
    b = res.fe_params[names].values
    V = res.cov_params().loc[names, names].values
    try:
        W = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    p = float(stats.chi2.sf(W, len(names)))
    effect = float(max(b, key=abs))          # largest contrast (metric units)
    return p, effect


def stratified_matrix(df, metrics):
    recs = []
    for stratum in [h for h in HEIGHT_ORDER if h in df["height"].unique()]:
        sub = df[df["height"] == stratum]
        for metric in metrics:
            d = sub.dropna(subset=[metric] + PARAM_FACTORS).copy().rename(
                columns={metric: "_y"})
            res, present = (None, [])
            if d["participant"].nunique() >= 2 and len(d) >= 8:
                res, present = fit_full(d, "_y", PARAM_FACTORS)
            for f in PARAM_FACTORS:
                p, eff = (wald_factor(res, f) if res is not None else (np.nan, np.nan))
                recs.append(dict(stratum=stratum, factor=f, metric=metric,
                                 p_value=p, effect=eff, n=len(d)))
    return pd.DataFrame(recs)


def height_effect(df, metrics):
    recs = []
    for metric in metrics:
        d = df.dropna(subset=[metric, "height"] + PARAM_FACTORS).copy().rename(
            columns={metric: "_y"})
        p = np.nan
        if d["height"].nunique() >= 2 and d["participant"].nunique() >= 2:
            res, _ = fit_full(d, "_y", PARAM_FACTORS + ["height"])
            if res is not None:
                p, _ = wald_factor(res, "height")
        recs.append(dict(metric=metric, p_height=p, n=len(d)))
    return pd.DataFrame(recs)


def fmt_p(p):
    if pd.isna(p):
        return f"{'n/a':>8}"
    star = "*" if p < 0.05 else " "
    return f"{'<0.001':>7}{star}" if p < 0.001 else f"{p:>7.3f}{star}"


def print_matrix(strat, metrics):
    strata = [h for h in HEIGHT_ORDER if h in strat["stratum"].unique()]
    print("\nFull p-value matrix (prototype factors x metrics x stratum):")
    print(f"  {'Factor':<10} {'Metric':<26}" + "".join(f"  {s:>8}" for s in strata))
    print(f"  {'-'*10} {'-'*26}" + "".join(f"  {'-'*8}" for _ in strata))
    for factor in PARAM_FACTORS:
        for metric in metrics:
            label = METRIC_LABELS.get(metric, metric)
            cells = []
            for s in strata:
                row = strat[(strat.stratum == s) & (strat.factor == factor)
                            & (strat.metric == metric)]
                cells.append(fmt_p(row.iloc[0]["p_value"] if not row.empty else np.nan))
            print(f"  {factor:<10} {label:<26}" + "".join(f"  {c}" for c in cells))
    print("\n  * = p < 0.05  (Wald test, mixed model, participant random intercept)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-csv", type=Path, default=None)
    ap.add_argument("--landmarks-root", type=Path, default=None,
                    help="Reads <root>/metrics/place_metrics_combined.csv")
    ap.add_argument("--participants", nargs="+", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.metrics_csv:
        csv = args.metrics_csv
    elif args.landmarks_root:
        csv = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
    else:
        sys.exit("Provide --metrics-csv or --landmarks-root")
    if not csv.is_file():
        sys.exit(f"Not found: {csv}")
    out_dir = args.out_dir or csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    if args.participants:
        df = df[df["participant"].isin(args.participants)]
    metrics = [m for m in METRIC_LABELS if m in df.columns]
    if not metrics:
        sys.exit("No known metric columns found in the CSV.")
    print(f"Loaded {len(df)} place events, {df['participant'].nunique()} participant(s), "
          f"{df['trial'].nunique() if 'trial' in df else '?'} trial(s)")
    print(f"Metrics: {metrics}")

    strat = stratified_matrix(df, metrics)
    strat.to_csv(out_dir / "mixed_stratified_tests.csv", index=False)
    hgt = height_effect(df, metrics)
    hgt.to_csv(out_dir / "mixed_height_tests.csv", index=False)

    print_matrix(strat, metrics)

    print("\nHeight effect (pooled model, Wald test on height):")
    print(f"  {'Metric':<26} {'p(height)':>10}")
    for _, r in hgt.iterrows():
        print(f"  {METRIC_LABELS.get(r['metric'], r['metric']):<26} "
              f"{fmt_p(r['p_height']).strip():>10}")

    print(f"\nWrote {out_dir/'mixed_stratified_tests.csv'} and "
          f"{out_dir/'mixed_height_tests.csv'}")
    print("Note: p-values screen significance; the 'effect' column in the tidy CSV "
          "gives magnitude/direction (metric units) -- that is 'varies most'.")


if __name__ == "__main__":
    main()
