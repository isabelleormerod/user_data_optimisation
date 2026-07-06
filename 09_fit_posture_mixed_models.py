#!/usr/bin/env python3
"""
Mixed-effects models of posture features across prototype parameters.

Mirrors 06_fit_mixed_models.py but for the posture feature table produced
by 07_extract_posture_features.py.

For each posture feature, fits a random-intercept mixed model:
    feature ~ Length + Size + Weight + Angle + height + (1 | participant)

And by default also runs STRATIFIED models (within each height stratum):
    feature ~ Length + Size + Weight + Angle + (1 | participant)

This answers: do the prototype parameters (Length/Size/Weight/Angle) affect
participants' posture, and if so at which heights?

Input:
    <landmarks_root>/metrics/posture_features_combined.csv

Outputs (under <landmarks_root>/metrics/posture_models/):
    aggregated_posture_data.csv
    model_overview.csv
    <feature>_fixed_effects.csv
    stratified_model_overview.csv
    stratified/<feature>_<Height>_fixed_effects.csv

Usage:
    python 09_fit_posture_models.py --landmarks-root A:\\Automated_chain_BETA\\Participant_Landmarks
    python 09_fit_posture_models.py --posture-csv path/to/posture_features_combined.csv
    python 09_fit_posture_models.py --landmarks-root ... --no-stratify
    python 09_fit_posture_models.py --landmarks-root ... --participants P003,P004
"""

import argparse
import sys
import warnings as _warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    sys.exit("statsmodels is required:  pip install statsmodels")


# --------------------------------------------------------------------------- #
# Posture features to model (same list as 08_compare_posture.py)
# --------------------------------------------------------------------------- #
POSTURE_METRICS = [
    ("trunk_flex_mean",            "Trunk flexion (deg)"),
    ("trunk_twist_mean",           "Trunk twist (deg)"),
    ("neck_flex_mean",             "Neck flexion (deg)"),
    ("knee_flex_mean",             "Knee flexion (deg)"),
    ("left_upperarm_flex_mean",    "L upper-arm flex (deg)"),
    ("right_upperarm_flex_mean",   "R upper-arm flex (deg)"),
    ("left_upperarm_abduct_mean",  "L upper-arm abduct (deg)"),
    ("right_upperarm_abduct_mean", "R upper-arm abduct (deg)"),
    ("left_elbow_flex_mean",       "L elbow flex (deg)"),
    ("right_elbow_flex_mean",      "R elbow flex (deg)"),
    ("wrist_neutral_dev_mean",     "Wrist neutral deviation (deg)"),
    ("reach_ratio_mean",           "Reach ratio"),
    ("wrist_elevation_m_mean",     "Wrist elevation (m)"),
    ("reba_score_a",               "REBA Score A"),
    ("reba_grand_right",           "REBA Grand (right)"),
    ("reba_grand_left",            "REBA Grand (left)"),
    ("left_aperture_mean",         "L aperture (m)"),
    ("right_aperture_mean",        "R aperture (m)"),
    ("left_finger_flex_mean",      "L finger flexion (deg)"),
    ("right_finger_flex_mean",     "R finger flexion (deg)"),
    ("left_hand_pos_jitter_mm",    "L hand pos jitter (mm)"),
    ("right_hand_pos_jitter_mm",   "R hand pos jitter (mm)"),
    ("left_hand_orient_jitter_deg","L hand orient jitter (deg)"),
    ("right_hand_orient_jitter_deg","R hand orient jitter (deg)"),
    ("left_wrist_flex_mean",       "L wrist flex (deg)"),
    ("right_wrist_flex_mean",      "R wrist flex (deg)"),
    ("left_wrist_ulnar_dev_mean",  "L wrist ulnar dev (deg)"),
    ("right_wrist_ulnar_dev_mean", "R wrist ulnar dev (deg)"),
]

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
ALL_FACTORS   = PARAM_FACTORS + ["height"]

REFERENCE = {
    "Length": "Short",
    "Size":   "Small",
    "Weight": "Not_weighted",
    "height": "Medium",
}


# --------------------------------------------------------------------------- #
# Parameter parsing (self-contained)
# --------------------------------------------------------------------------- #
def parse_params(trial: str) -> dict:
    out = {k: None for k in PARAM_FACTORS}
    tokens = trial.split("_"); joined = "_".join(tokens)
    if "Not_weighted"   in joined: out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined: out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long","Short"):    out["Length"] = tok
        elif tok in ("Large","Small"): out["Size"]   = tok
    return out


def tidy_term(name: str) -> str:
    import re
    m = re.match(r"C\((\w+).*?\)\[T\.([^\]]+)\]", name)
    return f"{m.group(1)}: {m.group(2)}" if m else name


# --------------------------------------------------------------------------- #
# Data loading — aggregate to trial level per height
# --------------------------------------------------------------------------- #
def load_and_aggregate(csv_path: Path,
                       participants: set = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("Posture features CSV is empty.")

    if participants:
        df = df[df["participant"].astype(str).isin(participants)].copy()
        if df.empty:
            sys.exit(f"No rows for participants {sorted(participants)}")

    params = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = params[c]

    metric_cols = [m for m, _ in POSTURE_METRICS if m in df.columns]
    group_keys  = ["participant", "trial", "height"] + PARAM_FACTORS
    group_keys  = [k for k in group_keys if k in df.columns]

    agg = (df.groupby(group_keys, dropna=False)[metric_cols]
             .mean()
             .reset_index())
    counts = (df.groupby(group_keys, dropna=False)
                .size().reset_index(name="n_place_events"))
    agg = agg.merge(counts, on=group_keys, how="left")
    return agg


# --------------------------------------------------------------------------- #
# Formula builders
# --------------------------------------------------------------------------- #
def build_formula(metric: str, df: pd.DataFrame,
                  include_height: bool = True) -> str | None:
    terms = []
    factors = ALL_FACTORS if include_height else PARAM_FACTORS
    for f in factors:
        if f not in df.columns: continue
        levels = df[f].dropna().unique()
        if len(levels) < 2: continue
        if f == "Angle":
            num = [v for v in levels if str(v).lstrip("-").isdigit()]
            ref = min(int(v) for v in num) if num else None
            term = (f"C(Angle, Treatment(reference={ref}))"
                    if ref is not None else "C(Angle)")
        else:
            ref = REFERENCE.get(f)
            term = (f"C({f}, Treatment(reference='{ref}'))"
                    if ref in set(levels) else f"C({f})")
        terms.append(term)
    if not terms: return None
    return f"Q('{metric}') ~ " + " + ".join(terms)


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def fit_one(metric: str, df: pd.DataFrame,
            include_height: bool = True) -> tuple:
    """Return (result, formula, notes) or (None, reason, [])."""
    factor_cols = [f for f in (ALL_FACTORS if include_height
                               else PARAM_FACTORS) if f in df.columns]
    sub = df[[metric, "participant"] + factor_cols].copy()
    ff  = [f for f in factor_cols if sub[f].dropna().nunique() >= 2]
    sub = sub.dropna(subset=[metric, "participant"] + ff).reset_index(drop=True)

    if len(sub) < 5:
        return None, "too few rows", []
    if sub[metric].nunique() < 2:
        return None, "no variation", []
    if sub["participant"].nunique() < 2:
        return None, "need ≥2 participants", []

    formula = build_formula(metric, sub, include_height)
    if formula is None:
        return None, "no factors with ≥2 levels", []

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
                notes.append("optimiser boundary; estimates still valid")
        notes = sorted(set(notes))
    except Exception as e:
        import traceback
        return None, (f"fit failed: {type(e).__name__}: {e} | "
                      f"{traceback.format_exc().splitlines()[-1]}"), []
    return result, formula, notes


def extract_fe(result) -> pd.DataFrame:
    fe    = result.fe_params
    bse   = result.bse
    pvals = result.pvalues
    ci    = result.conf_int()
    rows  = []
    for term in fe.index:
        rows.append({
            "term":    tidy_term(term),
            "raw_term": term,
            "coef":    float(fe[term]),
            "std_err": float(bse[term])       if term in bse.index   else np.nan,
            "p_value": float(pvals[term])      if term in pvals.index else np.nan,
            "ci_low":  float(ci.loc[term, 0]) if term in ci.index    else np.nan,
            "ci_high": float(ci.loc[term, 1]) if term in ci.index    else np.nan,
        })
    return pd.DataFrame(rows)


def fit_stratified(metric: str, agg: pd.DataFrame,
                   stratum_col: str = "height") -> list:
    strata = sorted(agg[stratum_col].dropna().unique(),
                    key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9))
    out = []
    for stratum in strata:
        sub = agg[agg[stratum_col] == stratum].copy()
        result, info, notes = fit_one(metric, sub, include_height=False)
        out.append({"stratum": stratum, "metric": metric,
                    "result": result, "formula": info,
                    "notes": notes,
                    "skip_reason": info if result is None else None})
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--posture-csv",    type=Path, default=None)
    ap.add_argument("--participants",   type=str,  default=None)
    ap.add_argument("--no-stratify",    action="store_true")
    args = ap.parse_args()

    if args.posture_csv:
        csv_path = args.posture_csv
        out_dir  = csv_path.parent / "posture_models"
    elif args.landmarks_root:
        csv_path = (args.landmarks_root / "metrics" /
                    "posture_features_combined.csv")
        out_dir  = args.landmarks_root / "metrics" / "posture_models"
    else:
        sys.exit("Provide --landmarks-root or --posture-csv")

    if not csv_path.is_file():
        sys.exit(f"Posture features CSV not found: {csv_path}\n"
                 f"Run 07_extract_posture_features.py first.")

    pfilter = None
    if args.participants:
        pfilter = {p.strip() for p in args.participants.split(",") if p.strip()}

    agg = load_and_aggregate(csv_path, pfilter)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "aggregated_posture_data.csv", index=False)

    n_part   = agg["participant"].nunique()
    n_trials = len(agg)
    print(f"Aggregated to {n_trials} trial-level rows across "
          f"{n_part} participant(s)")
    for f in ALL_FACTORS:
        if f in agg.columns:
            print(f"  {f}: {sorted(map(str, agg[f].dropna().unique()))}")
    print()

    if n_part < 2:
        sys.exit("Need ≥2 participants to fit a mixed model.")

    metrics = [(m, l) for m, l in POSTURE_METRICS if m in agg.columns
               and agg[m].dropna().nunique() > 1]
    print(f"Fitting pooled models for {len(metrics)} posture features …\n")

    # ------------------------------------------------------------------ #
    # Pooled models
    # ------------------------------------------------------------------ #
    overview = []
    for metric, label in metrics:
        result, info, notes = fit_one(metric, agg, include_height=True)
        if result is None:
            print(f"  [skip] {label}: {info}")
            continue

        fe = extract_fe(result)
        fe.insert(0, "metric", metric)
        fe.to_csv(out_dir / f"{metric}_fixed_effects.csv", index=False)

        sig = fe[(fe["raw_term"] != "Intercept") &
                 (fe["p_value"].notna()) & (fe["p_value"] < 0.05)]
        tag = "  *" if len(sig) else ""
        print(f"  [ok] {label}{tag}")
        for _, r in sig.iterrows():
            arrow = "↑" if r["coef"] > 0 else "↓"
            print(f"       {r['term']}: {arrow}{abs(r['coef']):.3f}  "
                  f"p={r['p_value']:.4f}")
        for n in notes:
            print(f"       note: {n}")

        for _, r in fe.iterrows():
            if r["raw_term"] == "Intercept": continue
            overview.append({
                "metric": metric, "metric_label": label,
                "term": r["term"],
                "coef": round(float(r["coef"]), 4),
                "p_value": (round(float(r["p_value"]), 5)
                            if r["p_value"] == r["p_value"] else np.nan),
                "significant": (r["p_value"] < 0.05
                                if r["p_value"] == r["p_value"] else False),
            })

    pd.DataFrame(overview).to_csv(out_dir / "model_overview.csv", index=False)
    print(f"\nWrote pooled outputs to {out_dir}")

    # ------------------------------------------------------------------ #
    # Stratified models
    # ------------------------------------------------------------------ #
    if not args.no_stratify and "height" in agg.columns:
        strata = sorted(agg["height"].dropna().unique(),
                        key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9))

        print(f"\n{'='*65}")
        print("STRATIFIED MODELS — prototype effects within each height")
        print(f"{'='*65}\n")

        strat_overview = []
        strat_dir = out_dir / "stratified"
        strat_dir.mkdir(exist_ok=True)

        for metric, label in metrics:
            results = fit_stratified(metric, agg)
            for res in results:
                if res["result"] is None:
                    print(f"  [skip] {label} @ {res['stratum']}: "
                          f"{res['skip_reason']}")
                    continue
                fe = extract_fe(res["result"])
                fe.insert(0, "stratum", res["stratum"])
                fe.insert(0, "metric",  metric)
                fe.to_csv(strat_dir / f"{metric}_{res['stratum']}_fe.csv",
                          index=False)
                for _, r in fe.iterrows():
                    if r["raw_term"] == "Intercept": continue
                    strat_overview.append({
                        "stratum":      res["stratum"],
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
        print(f"Wrote stratified_model_overview.csv  ({len(strat_df)} rows)\n")

        # p-value matrix — only rows with ≥1 significant cell
        print("P-value matrix (posture features × factors × height stratum):")
        print("  Reference: Short / Small / Not_weighted / smallest Angle\n")
        print(f"  {'Feature':<30} {'Factor':<10}", end="")
        for s in strata: print(f"  {s:>9}", end="")
        print()
        print(f"  {'-'*30} {'-'*10}", end="")
        for _ in strata: print(f"  {'---------':>9}", end="")
        print()

        for metric, label in metrics:
            for factor in PARAM_FACTORS:
                row_vals = []; any_sig = False
                for s in strata:
                    match = strat_df[
                        (strat_df["stratum"] == s) &
                        (strat_df["metric"]  == metric) &
                        strat_df["term"].str.startswith(factor)]
                    if match.empty:
                        row_vals.append("      n/a")
                    else:
                        p = match["p_value"].min()
                        if p != p:
                            row_vals.append("      n/a")
                        else:
                            marker = "*" if p < 0.05 else " "
                            if p < 0.05: any_sig = True
                            row_vals.append(f"{p:>8.3f}{marker}")
                if any_sig:
                    lbl = label[:28]
                    print(f"  {lbl:<30} {factor:<10}", end="")
                    for v in row_vals: print(f"  {v:>9}", end="")
                    print()

        print("\n  * = p < 0.05  |  min p across levels of that factor")
        print("  Only rows with ≥1 significant cell are shown\n")

        # Significant findings summary
        sig_s = strat_df[strat_df["significant"]]
        if len(sig_s):
            print("Significant prototype effects on posture:")
            for stratum in strata:
                sub = sig_s[sig_s["stratum"]==stratum].sort_values("p_value")
                if sub.empty:
                    print(f"  [{stratum:6}]  none"); continue
                print(f"  [{stratum:6}]")
                for _, r in sub.iterrows():
                    arrow = "↑" if r["coef"] > 0 else "↓"
                    print(f"    {r['term']:<30} -> {r['metric_label']:<30} "
                          f"{arrow}{abs(r['coef']):.3f}  p={r['p_value']:.4f}")
        else:
            print("No significant prototype effects on posture features.")

    print("\nInterpretation notes:")
    print("  Pooled: height is a main effect. Tells you which features change")
    print("    with height (most of them will — that's expected).")
    print("  Stratified: height removed by filtering. Tells you whether the")
    print("    prototype design (Length/Size/Weight/Angle) changes people's")
    print("    posture within a single height condition — the ergonomically")
    print("    important question.")


if __name__ == "__main__":
    main()