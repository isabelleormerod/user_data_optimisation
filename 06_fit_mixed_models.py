#!/usr/bin/env python3
"""
Mixed-effects models of Place-event performance.

Design (per participant): a random subset of 8 of the 24 parameter
combinations (3 Angle x 2 Weight x 2 Size x 2 Length), each crossed with 3
Heights, 5 trials each = 120 trials/participant. 10 participants.

Because the parameter subset is randomised PER participant, the design is
incomplete and unbalanced — a mixed-effects model is the appropriate tool: it
pools across participants and accounts for the repeated trials within each
person via a random intercept.

This script:
  1. Reads the combined Place-event metrics (place_metrics_combined.csv).
  2. Parses trial parameters (Length, Size, Weight, Angle) from the stem.
  3. AGGREGATES to one value per trial (mean of that trial's Place events).
  4. Fits, per metric, a random-intercept model:
         metric ~ C(Length) + C(Size) + C(Weight) + C(Angle) + C(height)
         random intercept: participant
     (Main effects only; Angle & Height categorical.)
  5. Writes coefficient tables (fixed effects, with p-values) and a model
     summary per metric, plus an overall significant-effects digest.

Outputs (under <root>/metrics/models/):
    <metric>_fixed_effects.csv     coefficient, SE, z, p, 95% CI per term
    <metric>_summary.txt           full statsmodels summary
    model_overview.csv             every metric x term, p-value, significance
    aggregated_trial_data.csv      the per-trial table the models were fit on

Requires: statsmodels, pandas, scipy, numpy.

Usage:
    python 06_fit_mixed_models.py --landmarks-root "A:/Automated_chain_BETA/Participant_Landmarks"
    python fit_mixed_models.py --metrics-csv path/to/place_metrics_combined.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ImportError:
    sys.exit("statsmodels is required. Install with:  pip install statsmodels")


METRICS = [
    ("duration_s", "Duration (s)"),
    ("perp_mean_deg", "Perpendicularity mean (deg)"),
    ("leftright_mean_deg", "Left/right tilt mean (deg)"),
    ("updown_mean_deg", "Up/down tilt mean (deg)"),
    ("pos_jitter_mm", "Positional jitter (mm)"),
    ("ang_jitter_deg", "Angular jitter (deg)"),
]

# Fixed-effect factors (all categorical, main effects only)
FACTORS = ["Length", "Size", "Weight", "Angle", "height"]

# Reference levels so coefficients read naturally (effect relative to these)
REFERENCE = {
    "Length": "Short",
    "Size": "Small",
    "Weight": "Not_weighted",
    "height": "Medium",
    # Angle reference: the smallest angle present (set at fit time)
}


# --------------------------------------------------------------------------- #
# Parameter parsing (matches the corrected naming: Weight = Front_weighted /
# Not_weighted; no Position factor).
# --------------------------------------------------------------------------- #
def parse_params(trial: str) -> dict:
    out = {"Length": None, "Size": None, "Weight": None, "Angle": None}
    tokens = trial.split("_")
    joined = "_".join(tokens)
    if "Not_weighted" in joined:
        out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined:
        out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:])
            break
    for tok in tokens:
        if tok in ("Long", "Short"):
            out["Length"] = tok
        elif tok in ("Large", "Small"):
            out["Size"] = tok
    return out


# --------------------------------------------------------------------------- #
# Load + aggregate
# --------------------------------------------------------------------------- #
def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("Metrics CSV is empty.")

    # Parse params
    params = df["trial"].apply(parse_params).apply(pd.Series)
    for c in ("Length", "Size", "Weight", "Angle"):
        df[c] = params[c]

    metric_cols = [m for m, _ in METRICS if m in df.columns]

    # Aggregate to one row per (participant, trial, height): mean of the
    # trial's Place events at that height. height is part of the grouping
    # because each trial is a single height here, but we keep it explicit.
    group_keys = ["participant", "trial", "height",
                  "Length", "Size", "Weight", "Angle"]
    group_keys = [k for k in group_keys if k in df.columns]
    agg = (df.groupby(group_keys, dropna=False)[metric_cols]
             .mean()
             .reset_index())
    # Number of Place events that went into each trial-mean (for reporting)
    counts = (df.groupby(group_keys, dropna=False).size()
                .reset_index(name="n_place_events"))
    agg = agg.merge(counts, on=group_keys, how="left")
    return agg


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def build_formula(metric: str, df: pd.DataFrame):
    """Construct a patsy formula with explicit reference levels, including
    only factors that have >=2 levels present."""
    terms = []
    for f in FACTORS:
        if f not in df.columns:
            continue
        levels = df[f].dropna().unique()
        if len(levels) < 2:
            continue
        if f == "Angle":
            ref = REFERENCE.get("Angle")
            if ref is None:
                ref = sorted(levels)[0]
            terms.append(f"C(Angle, Treatment(reference={int(ref)}))")
        else:
            ref = REFERENCE.get(f)
            if ref in set(levels):
                terms.append(f"C({f}, Treatment(reference='{ref}'))")
            else:
                terms.append(f"C({f})")
    if not terms:
        return None
    return f"{metric} ~ " + " + ".join(terms)


def tidy_term_name(name: str) -> str:
    """Make statsmodels' verbose term names readable."""
    # e.g. C(Weight, Treatment(reference='Not_weighted'))[T.Front_weighted]
    import re
    m = re.match(r"C\((\w+).*?\)\[T\.([^\]]+)\]", name)
    if m:
        return f"{m.group(1)}: {m.group(2)}"
    return name


def fit_metric(metric: str, df: pd.DataFrame):
    """Fit one random-intercept model. Returns (result, formula, notes) or
    (None, reason, notes)."""
    import warnings as _warnings
    factor_cols = [f for f in FACTORS if f in df.columns]
    sub = df[[metric, "participant"] + factor_cols].copy()

    # Drop rows with a missing metric, participant, OR any factor used in the
    # model. Then RESET THE INDEX so it's clean and contiguous — statsmodels'
    # mixedlm can index groups/exog positionally, and a gappy index (left over
    # from dropped rows) causes 'index out of bounds' errors.
    formula_factors = []
    for f in factor_cols:
        if sub[f].dropna().nunique() >= 2:
            formula_factors.append(f)
    sub = sub.dropna(subset=[metric, "participant"] + formula_factors)
    sub = sub.reset_index(drop=True)

    if sub[metric].nunique() < 2:
        return None, "no variation in metric", []
    if sub["participant"].nunique() < 2:
        return None, "need >=2 participants for a random intercept", []

    formula = build_formula(metric, sub)
    if formula is None:
        return None, "no factors with >=2 levels", []

    notes = []
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            groups = sub["participant"].to_numpy()
            model = smf.mixedlm(formula, sub, groups=groups)
            try:
                result = model.fit(reml=True, method="lbfgs")
            except Exception:
                # Fallback optimiser — 'lbfgs' can fail on awkward designs;
                # 'powell'/'nm' are more forgiving.
                result = model.fit(reml=True, method="powell")
        for w in caught:
            msg = str(w.message)
            if "singular" in msg.lower():
                notes.append("participant random-effect variance ~0 "
                             "(little between-participant difference for this metric)")
            elif "boundary" in msg.lower() or "Hessian" in msg:
                notes.append("optimiser hit a boundary (often the same ~0 "
                             "variance issue); fixed-effect estimates still valid")
        notes = sorted(set(notes))
    except Exception as e:
        import traceback
        return None, (f"fit failed: {type(e).__name__}: {e}\n"
                      f"      formula: {formula}\n"
                      f"      rows={len(sub)}, participants="
                      f"{sub['participant'].nunique()}\n"
                      f"      {traceback.format_exc().splitlines()[-1]}"), []
    return result, formula, notes


def extract_fixed_effects(result) -> pd.DataFrame:
    fe = result.fe_params                      # Series indexed by term name
    bse = result.bse                           # Series (fixed + random params)
    pvals = result.pvalues                     # Series
    ci = result.conf_int()                     # DataFrame indexed by term
    records = []
    for term in fe.index:
        records.append({
            "term": tidy_term_name(term),
            "raw_term": term,
            "coef": float(fe[term]),
            # Label-based lookups: safe even though bse/pvalues also contain
            # the random-effect (Group Var) entries.
            "std_err": float(bse[term]) if term in bse.index else np.nan,
            "p_value": float(pvals[term]) if term in pvals.index else np.nan,
            "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
            "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Stratified fitting — prototype parameters within each height stratum
# --------------------------------------------------------------------------- #
def build_formula_no_height(metric: str, df: pd.DataFrame):
    """Formula with ONLY prototype parameters — no height term.
    Used for within-stratum models where height is held constant by filtering,
    so adding height as a predictor would be meaningless (it has no variance)."""
    terms = []
    proto_factors = [f for f in FACTORS if f != "height"]
    for f in proto_factors:
        if f not in df.columns:
            continue
        levels = df[f].dropna().unique()
        if len(levels) < 2:
            continue
        if f == "Angle":
            # Use the smallest angle present as reference
            num_levels = [v for v in levels if str(v).lstrip("-").isdigit()]
            ref = min(int(v) for v in num_levels) if num_levels else None
            term = (f"C(Angle, Treatment(reference={ref}))"
                    if ref is not None else "C(Angle)")
        else:
            ref = REFERENCE.get(f)
            term = (f"C({f}, Treatment(reference='{ref}'))"
                    if ref in set(levels) else f"C({f})")
        terms.append(term)
    if not terms:
        return None
    return f"{metric} ~ " + " + ".join(terms)


def fit_stratified(metric: str, df: pd.DataFrame,
                   stratum_col: str = "height") -> list[dict]:
    """Fit one random-intercept model per stratum of stratum_col.

    Returns a list of result dicts, one per stratum:
        stratum, metric, result_obj, formula, notes, skip_reason
    """
    import warnings as _warnings

    strata = sorted(df[stratum_col].dropna().unique(),
                    key=lambda s: {"High": 0, "Medium": 1, "Low": 2}.get(s, 9))
    outputs = []
    proto_factors = [f for f in FACTORS if f != stratum_col and f in df.columns]

    for stratum in strata:
        sub = df[df[stratum_col] == stratum].copy()
        # Keep only the columns we need; reset index for statsmodels safety
        keep = [metric, "participant"] + proto_factors
        sub = sub[[c for c in keep if c in sub.columns]].copy()
        formula_factors = [f for f in proto_factors
                           if sub[f].dropna().nunique() >= 2]
        sub = sub.dropna(subset=[metric, "participant"] + formula_factors
                         ).reset_index(drop=True)

        if len(sub) < 5:
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": None, "formula": None, "notes": [],
                             "skip_reason": "too few rows"})
            continue
        if sub["participant"].nunique() < 2:
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": None, "formula": None, "notes": [],
                             "skip_reason": "need ≥2 participants"})
            continue
        if sub[metric].nunique() < 2:
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": None, "formula": None, "notes": [],
                             "skip_reason": "no variation in metric"})
            continue

        formula = build_formula_no_height(metric, sub)
        if formula is None:
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": None, "formula": None, "notes": [],
                             "skip_reason": "no prototype factors with ≥2 levels"})
            continue

        notes = []
        try:
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                model = smf.mixedlm(formula, sub,
                                    groups=sub["participant"].to_numpy())
                try:
                    result = model.fit(reml=True, method="lbfgs")
                except Exception:
                    result = model.fit(reml=True, method="powell")
            for w in caught:
                msg = str(w.message)
                if "singular" in msg.lower():
                    notes.append("random-effect variance ~0")
                elif "boundary" in msg.lower() or "Hessian" in msg:
                    notes.append("optimiser boundary (estimates still valid)")
            notes = sorted(set(notes))
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": result, "formula": formula,
                             "notes": notes, "skip_reason": None})
        except Exception as e:
            import traceback
            outputs.append({"stratum": stratum, "metric": metric,
                             "result": None, "formula": formula, "notes": [],
                             "skip_reason": f"fit failed: {type(e).__name__}: {e} | "
                                            f"{traceback.format_exc().splitlines()[-1]}"})
    return outputs
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--metrics-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.metrics_csv:
        csv_path = args.metrics_csv
        out_dir = csv_path.parent / "models"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        out_dir = args.landmarks_root / "metrics" / "models"
    else:
        sys.exit("Provide --landmarks-root or --metrics-csv")

    if not csv_path.is_file():
        sys.exit(f"Metrics CSV not found: {csv_path}\nRun place_metrics.py first.")

    agg = load_and_aggregate(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "aggregated_trial_data.csv", index=False)

    n_part = agg["participant"].nunique()
    n_trials = len(agg)
    print(f"Aggregated to {n_trials} trial-level rows across {n_part} participant(s)")
    for f in FACTORS:
        if f in agg.columns:
            print(f"  {f}: {sorted(map(str, agg[f].dropna().unique()))}")
    print()

    if n_part < 2:
        sys.exit("Need at least 2 participants to fit a mixed model. "
                 "(With one participant, use the descriptive comparison instead.)")

    overview = []
    for metric, label in METRICS:
        if metric not in agg.columns:
            continue
        result, info, notes = fit_metric(metric, agg)
        if result is None:
            print(f"[skip] {label}: {info}")
            overview.append({"metric": metric, "metric_label": label,
                             "term": "(model not fit)", "coef": np.nan,
                             "p_value": np.nan, "note": info})
            continue

        fe = extract_fixed_effects(result)
        fe.insert(0, "metric", metric)
        fe.to_csv(out_dir / f"{metric}_fixed_effects.csv", index=False)
        with open(out_dir / f"{metric}_summary.txt", "w") as f:
            f.write(f"Model for {label}\nFormula: {info}\n\n")
            f.write(str(result.summary()))
            if notes:
                f.write("\n\nNotes:\n" + "\n".join(f"  - {n}" for n in notes))

        # console: significant fixed effects (exclude intercept)
        sig = fe[(fe["raw_term"] != "Intercept") &
                 (fe["p_value"].notna()) & (fe["p_value"] < 0.05)]
        tag = "" if len(sig) == 0 else "  *"
        print(f"[ok] {label}{tag}")
        for _, r in sig.iterrows():
            arrow = "higher" if r["coef"] > 0 else "lower"
            print(f"      {r['term']}: {r['coef']:+.3f} ({arrow}), p={r['p_value']:.4f}")
        for n in notes:
            print(f"      note: {n}")

        for _, r in fe.iterrows():
            if r["raw_term"] == "Intercept":
                continue
            overview.append({
                "metric": metric, "metric_label": label,
                "term": r["term"], "coef": round(float(r["coef"]), 4),
                "p_value": (round(float(r["p_value"]), 5)
                            if r["p_value"] == r["p_value"] else np.nan),
                "significant": (r["p_value"] < 0.05
                                if r["p_value"] == r["p_value"] else False),
            })

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--metrics-csv", type=Path, default=None)
    ap.add_argument("--no-stratify", action="store_true",
                    help="Skip the height-stratified prototype-parameter models")
    args = ap.parse_args()

    if args.metrics_csv:
        csv_path = args.metrics_csv
        out_dir = csv_path.parent / "models"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        out_dir = args.landmarks_root / "metrics" / "models"
    else:
        sys.exit("Provide --landmarks-root or --metrics-csv")

    if not csv_path.is_file():
        sys.exit(f"Metrics CSV not found: {csv_path}\nRun place_metrics.py first.")

    agg = load_and_aggregate(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "aggregated_trial_data.csv", index=False)

    n_part = agg["participant"].nunique()
    n_trials = len(agg)
    print(f"Aggregated to {n_trials} trial-level rows across {n_part} participant(s)")
    for f in FACTORS:
        if f in agg.columns:
            print(f"  {f}: {sorted(map(str, agg[f].dropna().unique()))}")
    print()

    if n_part < 2:
        sys.exit("Need at least 2 participants to fit a mixed model.")

    # ------------------------------------------------------------------ #
    # Pooled models (existing behaviour — height as a main effect)
    # ------------------------------------------------------------------ #
    overview = []
    for metric, label in METRICS:
        if metric not in agg.columns:
            continue
        result, info, notes = fit_metric(metric, agg)
        if result is None:
            print(f"[skip] {label}: {info}")
            overview.append({"metric": metric, "metric_label": label,
                             "term": "(model not fit)", "coef": np.nan,
                             "p_value": np.nan, "note": info})
            continue

        fe = extract_fixed_effects(result)
        fe.insert(0, "metric", metric)
        fe.to_csv(out_dir / f"{metric}_fixed_effects.csv", index=False)
        with open(out_dir / f"{metric}_summary.txt", "w") as f:
            f.write(f"Model for {label}\nFormula: {info}\n\n")
            f.write(str(result.summary()))
            if notes:
                f.write("\n\nNotes:\n" + "\n".join(f"  - {n}" for n in notes))

        sig = fe[(fe["raw_term"] != "Intercept") &
                 (fe["p_value"].notna()) & (fe["p_value"] < 0.05)]
        tag = "" if len(sig) == 0 else "  *"
        print(f"[ok] {label}{tag}")
        for _, r in sig.iterrows():
            arrow = "higher" if r["coef"] > 0 else "lower"
            print(f"      {r['term']}: {r['coef']:+.3f} ({arrow}), p={r['p_value']:.4f}")
        for n in notes:
            print(f"      note: {n}")

        for _, r in fe.iterrows():
            if r["raw_term"] == "Intercept":
                continue
            overview.append({
                "metric": metric, "metric_label": label,
                "term": r["term"], "coef": round(float(r["coef"]), 4),
                "p_value": (round(float(r["p_value"]), 5)
                            if r["p_value"] == r["p_value"] else np.nan),
                "significant": (r["p_value"] < 0.05
                                if r["p_value"] == r["p_value"] else False),
            })

    pd.DataFrame(overview).to_csv(out_dir / "model_overview.csv", index=False)
    print(f"\nWrote pooled model outputs to {out_dir}")
    print("  - <metric>_fixed_effects.csv  (coefficients, p-values, 95% CI)")
    print("  - <metric>_summary.txt        (full model summary)")
    print("  - model_overview.csv          (all effects at a glance)")
    print("  - aggregated_trial_data.csv   (the per-trial data modelled)")

    # ------------------------------------------------------------------ #
    # Stratified models — prototype parameters within each height stratum
    # ------------------------------------------------------------------ #
    if not args.no_stratify and "height" in agg.columns:
        strata = sorted(agg["height"].dropna().unique(),
                        key=lambda s: {"High": 0, "Medium": 1, "Low": 2}.get(s, 9))

        print(f"\n{'='*65}")
        print("STRATIFIED MODELS  —  prototype parameters within each height")
        print(f"{'='*65}")
        print("Formula per stratum: metric ~ Length + Size + Weight + Angle")
        print("  (height removed — held constant by filtering to one stratum)")
        print("  Random intercept: participant\n")

        strat_overview = []   # one row per (stratum, metric, term)
        strat_dir = out_dir / "stratified"
        strat_dir.mkdir(exist_ok=True)

        for metric, label in METRICS:
            if metric not in agg.columns:
                continue
            results = fit_stratified(metric, agg, "height")
            for res in results:
                stratum = res["stratum"]
                if res["result"] is None:
                    print(f"  [skip] {label} @ {stratum}: {res['skip_reason']}")
                    continue
                fe = extract_fixed_effects(res["result"])
                fe.insert(0, "stratum", stratum)
                fe.insert(0, "metric", metric)
                # Save per-stratum CSV
                fe.to_csv(
                    strat_dir / f"{metric}_{stratum}_fixed_effects.csv",
                    index=False)
                # Collect into overview
                for _, r in fe.iterrows():
                    if r["raw_term"] == "Intercept":
                        continue
                    strat_overview.append({
                        "stratum":      stratum,
                        "metric":       metric,
                        "metric_label": label,
                        "term":         r["term"],
                        "coef":         round(float(r["coef"]), 4),
                        "p_value":      (round(float(r["p_value"]), 5)
                                         if r["p_value"] == r["p_value"]
                                         else np.nan),
                        "ci_low":       round(float(r["ci_low"]), 4)
                                        if r["ci_low"] == r["ci_low"] else np.nan,
                        "ci_high":      round(float(r["ci_high"]), 4)
                                        if r["ci_high"] == r["ci_high"] else np.nan,
                        "significant":  (r["p_value"] < 0.05
                                         if r["p_value"] == r["p_value"]
                                         else False),
                    })

        strat_df = pd.DataFrame(strat_overview)
        strat_df.to_csv(out_dir / "stratified_model_overview.csv", index=False)
        print(f"\nWrote {out_dir / 'stratified_model_overview.csv'}"
              f"  ({len(strat_df)} rows)")
        print(f"Wrote per-stratum CSVs to {strat_dir}/\n")

        # ---- p-value matrix ----
        print("P-value matrix  (mixed model, prototype factors × metrics × stratum):")
        print("  Reference levels: Short / Small / Not_weighted / smallest Angle")
        print(f"  {'Factor':<10} {'Metric':<28}", end="")
        for s in strata:
            print(f"  {s:>9}", end="")
        print()
        print(f"  {'-'*10} {'-'*28}", end="")
        for _ in strata:
            print(f"  {'---------':>9}", end="")
        print()

        proto_factors = [f for f in FACTORS if f != "height"]
        for factor in proto_factors:
            for col, label in METRICS:
                row_vals = []
                any_val = False
                for s in strata:
                    match = strat_df[
                        (strat_df["stratum"] == s) &
                        (strat_df["metric"]  == col) &
                        strat_df["term"].str.startswith(factor)]
                    if match.empty:
                        row_vals.append("      n/a")
                    else:
                        # For a 2-level factor (e.g. Length) there's one row;
                        # for Angle (3 levels) there are two — report the min p
                        p = match["p_value"].min()
                        if p != p:
                            row_vals.append("      n/a")
                        else:
                            any_val = True
                            marker = "*" if p < 0.05 else " "
                            row_vals.append(f"{p:>8.3f}{marker}")
                if any_val:
                    print(f"  {factor:<10} {label:<28}", end="")
                    for v in row_vals:
                        print(f"  {v:>9}", end="")
                    print()

        print("\n  * = p < 0.05  |  values are min p across levels of that factor")

        # ---- significant findings ----
        sig_s = strat_df[strat_df["significant"]]
        if len(sig_s):
            print("\nSignificant prototype effects (p < 0.05) within each stratum:")
            for stratum in strata:
                sub = sig_s[sig_s["stratum"] == stratum].sort_values("p_value")
                if sub.empty:
                    print(f"  [{stratum:6}]  none")
                    continue
                print(f"  [{stratum:6}]")
                for _, r in sub.iterrows():
                    arrow = "↑" if r["coef"] > 0 else "↓"
                    print(f"    {r['term']:<30} -> {r['metric_label']:<28} "
                          f"{arrow}{abs(r['coef']):.3f}  p={r['p_value']:.4f}")
        else:
            print("\nNo prototype effects reached p < 0.05 within any stratum.")
            print("Check effect sizes (coef column) in stratified_model_overview.csv.")

    print("\nInterpretation notes:")
    print("  Pooled model: height is a main effect alongside the prototype params.")
    print("  Stratified models: height is removed; each model is fit on one height")
    print("    stratum only. The p-value for Length at High is the effect of Length")
    print("    *within High placements*, with participant variation absorbed by the")
    print("    random intercept. This is the cleanest test of prototype effects.")
    print("  Both views are complementary: pooled gives overall picture,")
    print("  stratified gives the within-condition signal you actually care about.")


if __name__ == "__main__":
    main()