#!/usr/bin/env python3
"""
Stage 1: how do POSTURE features relate to PERFORMANCE? (association models)

Reads the merged per-Place-event table (09_merge.py -> combined_all.csv) and,
for each performance metric and HEIGHT GROUP, models its relationship to the
posture features with a participant random intercept (repeated events per person).

Height is treated as CATEGORICAL. Analysis is run separately for each height,
so you can see which posture–performance links are robust across heights vs
specific to a height group.

Two complementary views (per height group):
  (A) UNIVARIATE screen — each posture feature vs each performance metric:
          performance ~ posture_feature + (1 | participant)
      Clean and interpretable; the safe default. Reports slope, p, and a
      standardised slope (per-SD) so effects are comparable across features.
  (B) MULTIVARIATE model (optional, --multivariate) — per performance metric,
      all (de-correlated) posture features together:
          performance ~ f1 + f2 + ... + (1 | participant)
      Shows which features carry independent predictive value. Features that
      are too collinear (|r| > --corr-threshold) are pruned, keeping the one
      with the stronger univariate link, to avoid unstable estimates.

Standardisation: predictors are z-scored so slopes read as "change in the
metric (in its own units) per 1 SD increase in the posture feature", making
them comparable. Multiple-testing is addressed with a Benjamini-Hochberg FDR
column on the univariate results (per height group).

Input:  <root>/metrics/combined_all.csv
Outputs (under <root>/metrics/stage1/):
    univariate_associations_<height>.csv   per-height metric x feature results
    multivariate_<metric>_<height>.csv     (if --multivariate) per-height coefficients
    significant_links_<height>.csv         FDR-significant links per height

Usage:
    python 10_posture_performance.py --landmarks-root "A:/.../Participant_Landmarks"
    python 10_posture_performance.py --landmarks-root ... --multivariate
    python 10_posture_performance.py --landmarks-root ... --participants P003,P004
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.stats import fdr_bh, zscore
from utils.params import parse_participant_filter

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


# Performance metrics (outcomes)
PERF_METRICS = [
    "duration_s", "perp_mean_deg", "leftright_mean_deg", "updown_mean_deg",
    "pos_jitter_mm", "ang_jitter_deg",
]

# Columns that are keys/metadata or variance/count companions — not predictors
NON_FEATURE = {
    "participant", "trial", "place_index", "height",
    "start_t_s", "stop_t_s", "duration_s", "n_samples",
    "left_hand_n_frames", "right_hand_n_frames", "reba_n_frames",
}
# Also exclude the performance metrics themselves and their variance columns
NON_FEATURE |= set(PERF_METRICS)
NON_FEATURE |= {m.replace("_mean_deg", "_var_deg2") for m in PERF_METRICS}
NON_FEATURE |= {"perp_var_deg2", "leftright_var_deg2", "updown_var_deg2"}



def detect_features(df):
    feats = []
    for c in df.columns:
        if c in NON_FEATURE:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() >= 5 and col.nunique() > 2:
            feats.append(c)
    return feats


def univariate(df, feats, metrics, height_val=None):
    rows = []
    if not HAVE_SM:
        return pd.DataFrame(rows)
    import warnings as _w
    for metric in metrics:
        if metric not in df.columns:
            continue
        y_all = pd.to_numeric(df[metric], errors="coerce")
        for feat in feats:
            sub = pd.DataFrame({
                "y": y_all,
                "x": zscore(df[feat]) if zscore(df[feat]) is not None else np.nan,
                "participant": df["participant"].astype(str),
            }).dropna()
            if len(sub) < 10 or sub["participant"].nunique() < 2 or sub["x"].nunique() < 3:
                continue
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    res = smf.mixedlm("y ~ x", sub,
                                      groups=sub["participant"].to_numpy()
                                      ).fit(reml=True, method="lbfgs")
                slope = float(res.fe_params.get("x", np.nan))
                p = float(res.pvalues.get("x", np.nan))
            except Exception:
                slope, p = np.nan, np.nan
            row = {"metric": metric, "feature": feat,
                   "std_slope": slope, "p_value": p, "n": len(sub)}
            if height_val is not None:
                row["height"] = height_val
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_fdr"] = fdr_bh(out["p_value"].values)
    return out


def prune_collinear(df, feats, uni, metric, corr_threshold):
    """For a given metric, drop features that are too correlated with a
    stronger (lower-p) feature, to stabilise the multivariate model."""
    msub = uni[uni["metric"] == metric].dropna(subset=["p_value"])
    if msub.empty:
        return []
    ranked = msub.sort_values("p_value")["feature"].tolist()
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    kept = []
    for f in ranked:
        if X[f].notna().sum() < 5:
            continue
        too_close = False
        for k in kept:
            r = X[[f, k]].corr().iloc[0, 1]
            if pd.notna(r) and abs(r) > corr_threshold:
                too_close = True
                break
        if not too_close:
            kept.append(f)
    return kept


def multivariate(df, feats, metric, uni, corr_threshold, height_val=None):
    if not HAVE_SM:
        return pd.DataFrame()
    import warnings as _w
    # Filter uni results to this height group if stratified
    if height_val is not None:
        uni_for_metric = uni[(uni["metric"] == metric) & (uni["height"] == height_val)]
    else:
        uni_for_metric = uni[uni["metric"] == metric]
    keep = prune_collinear(df, feats, uni_for_metric, metric, corr_threshold)
    if len(keep) < 1:
        return pd.DataFrame()
    data = {"y": pd.to_numeric(df[metric], errors="coerce"),
            "participant": df["participant"].astype(str)}
    safe_names = {}
    for i, f in enumerate(keep):
        z = zscore(df[f])
        if z is None:
            continue
        nm = f"f{i}"
        safe_names[nm] = f
        data[nm] = z
    sub = pd.DataFrame(data).dropna()
    if len(sub) < 10 or sub["participant"].nunique() < 2:
        return pd.DataFrame()
    terms = " + ".join(safe_names.keys())
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            res = smf.mixedlm(f"y ~ {terms}", sub,
                              groups=sub["participant"].to_numpy()
                              ).fit(reml=True, method="lbfgs")
    except Exception as e:
        row = {"metric": metric, "feature": "(fit failed)",
               "std_slope": np.nan, "p_value": np.nan, "note": str(e)[:80]}
        if height_val is not None:
            row["height"] = height_val
        return pd.DataFrame([row])
    rows = []
    for nm, real in safe_names.items():
        if nm in res.fe_params.index:
            row = {"metric": metric, "feature": real,
                   "std_slope": float(res.fe_params[nm]),
                   "p_value": float(res.pvalues.get(nm, np.nan)),
                   "note": ""}
            if height_val is not None:
                row["height"] = height_val
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--combined-csv", type=Path, default=None)
    ap.add_argument("--participants", type=str, default=None)
    ap.add_argument("--multivariate", action="store_true",
                    help="Also fit a per-metric multivariate model")
    ap.add_argument("--corr-threshold", type=float, default=0.8,
                    help="Prune features with |r| above this in the "
                         "multivariate model (default 0.8)")
    args = ap.parse_args()

    if args.combined_csv:
        csv_path = args.combined_csv
        out_dir = csv_path.parent / "stage1"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "combined_all.csv"
        out_dir = args.landmarks_root / "metrics" / "stage1"
    else:
        sys.exit("Provide --landmarks-root or --combined-csv")

    if not csv_path.is_file():
        sys.exit(f"Combined table not found: {csv_path}\nRun 09_merge.py first.")
    if not HAVE_SM:
        sys.exit("statsmodels is required:  pip install statsmodels")

    df = pd.read_csv(csv_path)
    keep = parse_participant_filter(args.participants)
    if keep:
        df = df[df["participant"].astype(str).isin(keep)].copy()
        if df.empty:
            sys.exit(f"No rows for participants {sorted(keep)}")

    feats = detect_features(df)
    metrics = [m for m in PERF_METRICS if m in df.columns]
    
    # Detect height groups (categorical)
    height_groups = sorted(df["height"].dropna().unique())
    print(f"{len(df)} events, {df['participant'].nunique()} participants")
    print(f"Height groups: {height_groups}")
    print(f"Performance metrics: {len(metrics)}; posture features: {len(feats)}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis separately for each height group
    all_uni = []
    all_sig = []
    for height_val in height_groups:
        df_h = df[df["height"] == height_val].copy()
        print(f"--- Height = {height_val} ({len(df_h)} events) ---")
        
        uni = univariate(df_h, feats, metrics, height_val=height_val)
        if uni.empty:
            print(f"  No univariate models fit for height {height_val}")
            continue
        
        uni = uni.sort_values(["metric", "p_value"])
        uni.to_csv(out_dir / f"univariate_associations_{height_val}.csv", index=False)
        print(f"  Wrote univariate_associations_{height_val}.csv ({len(uni)} rows)")
        all_uni.append(uni)

        sig = uni[(uni["q_value_fdr"].notna()) & (uni["q_value_fdr"] < 0.05)] \
            .sort_values("q_value_fdr")
        sig.to_csv(out_dir / f"significant_links_{height_val}.csv", index=False)
        print(f"  Wrote significant_links_{height_val}.csv ({len(sig)} FDR-significant)")
        
        # Print significant links for this height to terminal
        if len(sig) > 0:
            print(f"\n  Top FDR-significant links (height={height_val}):")
            for _, r in sig.head(10).iterrows():
                direction = "+" if r["std_slope"] > 0 else "-"
                print(f"    {r['feature']:<28} -> {r['metric']:<20} "
                      f"{direction}{abs(r['std_slope']):.3f}/SD  q={r['q_value_fdr']:.4f}")
        else:
            print(f"  No FDR-significant links for height={height_val}")
        
        all_sig.append(sig)

        if args.multivariate:
            for metric in metrics:
                mv = multivariate(df_h, feats, metric, uni, args.corr_threshold, height_val=height_val)
                if len(mv):
                    mv.to_csv(out_dir / f"multivariate_{metric}_{height_val}.csv", index=False)
        print()

    # Merged view across all heights
    print("=" * 80)
    print("SUMMARY ACROSS ALL HEIGHT GROUPS")
    print("=" * 80 + "\n")
    
    if all_uni:
        merged_uni = pd.concat(all_uni, ignore_index=True)
        merged_uni.to_csv(out_dir / "univariate_associations_all_heights.csv", index=False)
        print(f"Wrote univariate_associations_all_heights.csv ({len(merged_uni)} rows)")
        
        # Show per-height summary
        print("\nUnivariate results per height:")
        for height_val in height_groups:
            count = len(merged_uni[merged_uni["height"] == height_val])
            print(f"  Height {height_val}: {count} feature-metric pairs tested")

    if all_sig:
        merged_sig = pd.concat(all_sig, ignore_index=True).sort_values("q_value_fdr")
        merged_sig.to_csv(out_dir / "significant_links_all_heights.csv", index=False)
        print(f"\nFDR-significant links (q < 0.05) across all heights: {len(merged_sig)} total\n")
        
        # Show per-height summary
        print("FDR-significant links per height:")
        for height_val in height_groups:
            count = len(merged_sig[merged_sig["height"] == height_val])
            print(f"  Height {height_val}: {count} significant links")
        
        print("\nTop 30 FDR-significant links (sorted by q-value):\n")
        for _, r in merged_sig.head(30).iterrows():
            direction = "+" if r["std_slope"] > 0 else "-"
            print(f"  [{r['height']}] {r['feature']:<28} -> {r['metric']:<20} "
                  f"{direction}{abs(r['std_slope']):.3f}/SD  q={r['q_value_fdr']:.4f}")
    else:
        print("No FDR-significant links found across any height group.")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("std_slope = change in the metric (its own units) per +1 SD of the posture feature")
    print("q = Benjamini-Hochberg FDR q-value (corrected p-value within each height group)")
    print("Results are stratified by height so you can compare links across groups.")
    print(f"\nOutput files saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
