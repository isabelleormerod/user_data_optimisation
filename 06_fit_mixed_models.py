#!/usr/bin/env python3
"""
Mixed-effects models of Place-event performance with Interaction Effects.

USAGE GUIDELINES
----------------
Dependencies:
    - python 3.7+
    - pandas
    - numpy
    - statsmodels
    - scipy (required by statsmodels)
    Install via: pip install pandas numpy statsmodels scipy

Input:
    Requires the combined metrics table output by place_metrics.py:
    <landmarks_root>/metrics/place_metrics_combined.csv
    Must contain columns: participant, trial, height, plus your performance metrics.

Execution:
    Option A (Root Directory):
        python 06_fit_mixed_models.py --landmarks-root /path/to/Participant_Landmarks
    Option B (Direct CSV):
        python 06_fit_mixed_models.py --metrics-csv /path/to/place_metrics_combined.csv

Outputs:
    Saves the following to <landmarks_root>/metrics/models/ or <csv_dir>/models/:
    1. aggregated_trial_data.csv (the per-trial dataset used for modeling)
    2. model_overview.csv (summary of all significant/non-significant effects)
    3. <metric>_fixed_effects.csv (coefficients and p-values for a specific metric)
    4. <metric>_summary.txt (raw statsmodels terminal output)

Design Notes:
    Fits a random-intercept model with interaction terms:
    metric ~ (C(Length) + C(Size) + C(Weight) + C(Angle)) * C(height)
    Random intercept: participant
"""

import argparse
import sys
from pathlib import Path
import re

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

FACTORS = ["Length", "Size", "Weight", "Angle", "height"]

REFERENCE = {
    "Length": "Short",
    "Size": "Small",
    "Weight": "Not_weighted",
    "height": "Medium",
}

# --------------------------------------------------------------------------- #
# Parameter parsing
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
        if tok and tok.upper() == "A" and tok[1:].isdigit():
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

    params = df["trial"].apply(parse_params).apply(pd.Series)
    for c in ("Length", "Size", "Weight", "Angle"):
        df[c] = params[c]

    metric_cols = [m for m, _ in METRICS if m in df.columns]

    group_keys = ["participant", "trial", "height",
                  "Length", "Size", "Weight", "Angle"]
    group_keys = [k for k in group_keys if k in df.columns]
    
    agg = (df.groupby(group_keys, dropna=False)[metric_cols]
             .mean()
             .reset_index())
    
    counts = (df.groupby(group_keys, dropna=False).size()
                .reset_index(name="n_place_events"))
    agg = agg.merge(counts, on=group_keys, how="left")
    return agg

# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def build_formula(metric: str, df: pd.DataFrame):
    """Construct a patsy formula with explicit reference levels AND height interactions."""
    terms = []
    height_term = None
    
    for f in FACTORS:
        if f not in df.columns:
            continue
        levels = df[f].dropna().unique()
        if len(levels) < 2:
            continue
            
        if f == "Angle":
            ref = REFERENCE.get("Angle")
            if ref is None:
                ref = sorted(levels)
            term = f"C(Angle, Treatment(reference={int(ref)}))"
        else:
            ref = REFERENCE.get(f)
            if ref in set(levels):
                term = f"C({f}, Treatment(reference='{ref}'))"
            else:
                term = f"C({f})"
                
        if f == "height":
            height_term = term
        else:
            terms.append(term)

    if not terms and not height_term:
        return None
        
    # Apply Interaction logic: (Length + Size + Weight + Angle) * height
    if terms and height_term:
        prototype_effects = " + ".join(terms)
        return f"{metric} ~ ({prototype_effects}) * {height_term}"
    elif terms:
        return f"{metric} ~ " + " + ".join(terms)
    elif height_term:
        return f"{metric} ~ {height_term}"
        
    return None

def tidy_term_name(name: str) -> str:
    """Clean up statsmodels' verbose categorical interaction term names."""
    # Converts "C(Weight, Treatment(...))[T.Front_weighted]:C(height...)[T.High]"
    # into "Weight[Front_weighted]:height[High]"
    clean = re.sub(r"C\(([a-zA-Z0-9_]+)[^\)]*\)\[T\.([^\]]+)\]", r"\1[\2]", name)
    return clean

def fit_metric(metric: str, df: pd.DataFrame):
    """Fit one random-intercept model."""
    import warnings as _warnings
    factor_cols = [f for f in FACTORS if f in df.columns]
    sub = df[[metric, "participant"] + factor_cols].copy()

    formula_factors = [f for f in factor_cols if sub[f].dropna().nunique() >= 2]
    sub = sub.dropna(subset=[metric, "participant"] + formula_factors).reset_index(drop=True)

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
                result = model.fit(reml=True, method="powell")
                
        for w in caught:
            msg = str(w.message)
            if "singular" in msg.lower():
                notes.append("participant random-effect variance ~0")
            elif "boundary" in msg.lower() or "Hessian" in msg:
                notes.append("optimiser hit a boundary; fixed-effect estimates still valid")
        notes = sorted(set(notes))
    except Exception as e:
        import traceback
        return None, (f"fit failed: {e}\n{traceback.format_exc().splitlines()[-1]}"), []
        
    return result, formula, notes

def extract_fixed_effects(result) -> pd.DataFrame:
    fe = result.fe_params                      
    bse = result.bse                           
    pvals = result.pvalues                     
    ci = result.conf_int()                     
    records = []
    for term in fe.index:
        records.append({
            "term": tidy_term_name(term),
            "raw_term": term,
            "coef": float(fe[term]),
            "std_err": float(bse[term]) if term in bse.index else np.nan,
            "p_value": float(pvals[term]) if term in pvals.index else np.nan,
            "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
            "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
        })
    return pd.DataFrame(records)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    if n_part < 2:
        sys.exit("Need at least 2 participants to fit a mixed model.")

    overview = []
    for metric, label in METRICS:
        if metric not in agg.columns:
            continue
            
        result, info, notes = fit_metric(metric, agg)
        if result is None:
            print(f"[skip] {label}: {info}")
            continue

        fe = extract_fixed_effects(result)
        fe.insert(0, "metric", metric)
        fe.to_csv(out_dir / f"{metric}_fixed_effects.csv", index=False)
        
        with open(out_dir / f"{metric}_summary.txt", "w") as f:
            f.write(f"Model for {label}\nFormula: {info}\n\n")
            f.write(str(result.summary()))
            if notes:
                f.write("\n\nNotes:\n" + "\n".join(f"  - {n}" for n in notes))

        sig = fe[(fe["raw_term"] != "Intercept") & (fe["p_value"].notna()) & (fe["p_value"] < 0.05)]
        tag = "" if len(sig) == 0 else "  *"
        print(f"[ok] {label}{tag}")
        for _, r in sig.iterrows():
            arrow = "higher" if r["coef"] > 0 else "lower"
            print(f"      {r['term']}: {r['coef']:+.3f} ({arrow}), p={r['p_value']:.4f}")

        for _, r in fe.iterrows():
            if r["raw_term"] == "Intercept": continue
            overview.append({
                "metric": metric, "metric_label": label,
                "term": r["term"], "coef": round(float(r["coef"]), 4),
                "p_value": (round(float(r["p_value"]), 5) if r["p_value"] == r["p_value"] else np.nan),
                "significant": (r["p_value"] < 0.05 if r["p_value"] == r["p_value"] else False),
            })

    pd.DataFrame(overview).to_csv(out_dir / "model_overview.csv", index=False)
    print(f"\nWrote model outputs to {out_dir}")

if __name__ == "__main__":
    main()