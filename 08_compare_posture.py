#!/usr/bin/env python3
"""
Compare posture features across participants, heights, and prototype parameters.

Mirrors 05_compare_performance.py but for the posture feature table produced
by 07_extract_posture_features.py.

Input:
    <landmarks_root>/metrics/posture_features_combined.csv

For every posture feature the script:
  - tabulates group means / SD / n for each factor
  - runs an omnibus non-parametric test per factor
      2 groups  -> Mann-Whitney U
      >2 groups -> Kruskal-Wallis H
  - draws box plots of each feature by each factor

--stratify-by height (default ON):
  Re-runs the prototype parameter tests (Length/Size/Weight/Angle) separately
  within each height stratum, removing the height confound.

Outputs (under <landmarks_root>/metrics/posture_comparison/):
    group_summary.csv
    stat_tests.csv
    stratified_stat_tests.csv
    stratified_summary.csv
    by_<factor>_<feature>.png
    by_<factor>_<feature>_stratified.png

Usage:
    python 08_compare_posture.py --landmarks-root A:\\Automated_chain_BETA\\Participant_Landmarks
    python 08_compare_posture.py --posture-csv path/to/posture_features_combined.csv
    python 08_compare_posture.py --landmarks-root ... --no-stratify
    python 08_compare_posture.py --landmarks-root ... --participants P003,P004
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# --------------------------------------------------------------------------- #
# Posture features to analyse
# Body features that are meaningful to compare across conditions
# (means only — SDs, n_frames, and raw REBA sub-scores excluded)
# --------------------------------------------------------------------------- #
POSTURE_METRICS = [
    # Body / REBA
    ("trunk_flex_mean",           "Trunk flexion",           "deg"),
    ("trunk_twist_mean",          "Trunk twist",             "deg"),
    ("neck_flex_mean",            "Neck flexion",            "deg"),
    ("knee_flex_mean",            "Knee flexion",            "deg"),
    ("left_upperarm_flex_mean",   "L upper-arm flex",        "deg"),
    ("right_upperarm_flex_mean",  "R upper-arm flex",        "deg"),
    ("left_upperarm_abduct_mean", "L upper-arm abduct",      "deg"),
    ("right_upperarm_abduct_mean","R upper-arm abduct",      "deg"),
    ("left_elbow_flex_mean",      "L elbow flex",            "deg"),
    ("right_elbow_flex_mean",     "R elbow flex",            "deg"),
    ("wrist_neutral_dev_mean",    "Wrist neutral deviation", "deg"),
    ("reach_ratio_mean",          "Reach ratio",             "ratio"),
    ("wrist_elevation_m_mean",    "Wrist elevation",         "m"),
    ("reba_score_a",              "REBA Score A",            "score"),
    ("reba_grand_right",          "REBA Grand (right)",      "score"),
    ("reba_grand_left",           "REBA Grand (left)",       "score"),
    # Hand
    ("left_aperture_mean",        "L aperture",              "m"),
    ("right_aperture_mean",       "R aperture",              "m"),
    ("left_finger_flex_mean",     "L finger flexion",        "deg"),
    ("right_finger_flex_mean",    "R finger flexion",        "deg"),
    ("left_hand_pos_jitter_mm",   "L hand pos jitter",       "mm"),
    ("right_hand_pos_jitter_mm",  "R hand pos jitter",       "mm"),
    ("left_hand_orient_jitter_deg","L hand orient jitter",   "deg"),
    ("right_hand_orient_jitter_deg","R hand orient jitter",  "deg"),
    ("left_wrist_flex_mean",      "L wrist flex",            "deg"),
    ("right_wrist_flex_mean",     "R wrist flex",            "deg"),
    ("left_wrist_ulnar_dev_mean", "L wrist ulnar dev",       "deg"),
    ("right_wrist_ulnar_dev_mean","R wrist ulnar dev",       "deg"),
]

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
ALL_FACTORS   = ["participant", "height"] + PARAM_FACTORS


# --------------------------------------------------------------------------- #
# Parameter parsing (self-contained — no utils dependency)
# --------------------------------------------------------------------------- #
def parse_params(trial: str) -> dict:
    out = {k: None for k in PARAM_FACTORS}
    tokens = trial.split("_")
    joined = "_".join(tokens)
    if "Not_weighted" in joined:
        out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined:
        out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long", "Short"):   out["Length"] = tok
        elif tok in ("Large", "Small"): out["Size"]   = tok
    return out


def add_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = parsed[c]
    return df


def available_metrics(df: pd.DataFrame) -> list:
    """Return only those POSTURE_METRICS whose column is present and has variance."""
    out = []
    for col, label, unit in POSTURE_METRICS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) >= 4 and vals.nunique() > 1:
            out.append((col, label, unit))
    return out


# --------------------------------------------------------------------------- #
# Summaries + stats (identical logic to 05)
# --------------------------------------------------------------------------- #
def group_summary(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    records = []
    for factor in ALL_FACTORS:
        if factor not in df.columns: continue
        for level, sub in df.groupby(factor, dropna=True):
            for col, label, unit in metrics:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if not len(vals): continue
                records.append({
                    "factor": factor, "level": level,
                    "metric": col, "metric_label": label, "unit": unit,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "sd":   float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                })
    return pd.DataFrame(records)


def stat_tests(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    records = []
    for factor in ALL_FACTORS:
        if factor not in df.columns: continue
        for col, label, unit in metrics:
            groups, levels = [], []
            for level, sub in df.groupby(factor, dropna=True):
                vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
                if len(vals) >= 1:
                    groups.append(vals); levels.append(str(level))
            usable = [g for g in groups if len(g) >= 2]
            if len(usable) < 2:
                records.append({"factor": factor, "metric": col,
                                 "metric_label": label,
                                 "test": "skipped",
                                 "statistic": np.nan, "p_value": np.nan,
                                 "n_groups": len(groups),
                                 "levels": ", ".join(levels)})
                continue
            try:
                if len(usable) == 2:
                    stat, p = stats.mannwhitneyu(usable[0], usable[1],
                                                  alternative="two-sided")
                    test = "Mann-Whitney U"
                else:
                    stat, p = stats.kruskal(*usable)
                    test = "Kruskal-Wallis H"
            except ValueError as e:
                stat, p, test = np.nan, np.nan, f"error: {e}"
            records.append({
                "factor": factor, "metric": col, "metric_label": label,
                "test": test,
                "statistic": float(stat) if stat == stat else np.nan,
                "p_value":   float(p)    if p    == p    else np.nan,
                "n_groups": len(usable),
                "levels": ", ".join(levels),
            })
    return pd.DataFrame(records)


def stat_tests_stratified(df: pd.DataFrame, metrics: list,
                           stratum_col: str = "height") -> pd.DataFrame:
    if stratum_col not in df.columns: return pd.DataFrame()
    records = []
    for stratum, sub_df in df.groupby(stratum_col, dropna=True):
        for factor in PARAM_FACTORS:
            if factor not in sub_df.columns: continue
            for col, label, unit in metrics:
                groups, levels = [], []
                for level, grp in sub_df.groupby(factor, dropna=True):
                    vals = pd.to_numeric(grp[col], errors="coerce").dropna().values
                    if len(vals) >= 1:
                        groups.append(vals); levels.append(str(level))
                usable = [g for g in groups if len(g) >= 2]
                if len(usable) < 2:
                    records.append({
                        "stratum": stratum, "factor": factor,
                        "metric": col, "metric_label": label,
                        "test": "skipped", "statistic": np.nan,
                        "p_value": np.nan, "n_groups": len(groups),
                        "levels": ", ".join(levels),
                        "n_events": len(sub_df)})
                    continue
                try:
                    if len(usable) == 2:
                        stat, p = stats.mannwhitneyu(usable[0], usable[1],
                                                      alternative="two-sided")
                        test = "Mann-Whitney U"
                    else:
                        stat, p = stats.kruskal(*usable)
                        test = "Kruskal-Wallis H"
                except ValueError as e:
                    stat, p, test = np.nan, np.nan, f"error: {e}"
                records.append({
                    "stratum": stratum, "factor": factor,
                    "metric": col, "metric_label": label, "test": test,
                    "statistic": float(stat) if stat == stat else np.nan,
                    "p_value":   float(p)    if p    == p    else np.nan,
                    "n_groups": len(usable), "levels": ", ".join(levels),
                    "n_events": int(len(sub_df))})
    return pd.DataFrame(records)


def group_summary_stratified(df: pd.DataFrame, metrics: list,
                              stratum_col: str = "height") -> pd.DataFrame:
    if stratum_col not in df.columns: return pd.DataFrame()
    records = []
    for stratum, sub_df in df.groupby(stratum_col, dropna=True):
        for factor in PARAM_FACTORS:
            if factor not in sub_df.columns: continue
            for level, grp in sub_df.groupby(factor, dropna=True):
                for col, label, unit in metrics:
                    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                    if not len(vals): continue
                    records.append({
                        "stratum": stratum, "factor": factor, "level": level,
                        "metric": col, "metric_label": label, "unit": unit,
                        "n": int(len(vals)),
                        "mean":   float(vals.mean()),
                        "sd":     float(vals.std(ddof=1)) if len(vals)>1 else 0.,
                        "median": float(vals.median()),
                    })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
def order_levels(factor, levels):
    orders = {
        "height": ["High", "Medium", "Low"],
        "Length": ["Short", "Long"],
        "Size":   ["Small", "Large"],
        "Weight": ["Not_weighted", "Front_weighted"],
    }
    if factor in orders:
        known = [l for l in orders[factor] if l in levels]
        rest  = [l for l in levels if l not in known]
        return known + sorted(rest)
    try:    return sorted(levels, key=lambda x: float(x))
    except: return sorted(levels, key=str)


def make_graphs(df, metrics, out_dir, p_lookup):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for factor in ALL_FACTORS:
        if factor not in df.columns: continue
        levels = order_levels(factor, list(df[factor].dropna().unique()))
        if len(levels) < 2: continue
        for col, label, unit in metrics:
            data, tick_labels = [], []
            for lv in levels:
                vals = pd.to_numeric(
                    df.loc[df[factor]==lv, col], errors="coerce").dropna().values
                if len(vals):
                    data.append(vals)
                    tick_labels.append(f"{lv}\n(n={len(vals)})")
            if len(data) < 2: continue
            fig, ax = plt.subplots(figsize=(max(6, len(data)*1.3), 5))
            ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
            for i, vals in enumerate(data, 1):
                jit = (np.random.rand(len(vals)) - 0.5) * 0.15
                ax.scatter(np.full(len(vals), i)+jit, vals,
                           alpha=0.5, s=18, color="#1f77b4", zorder=3)
            ax.set_ylabel(f"{label} ({unit})")
            ax.set_xlabel(factor)
            p = p_lookup.get((factor, col))
            title = f"{label} by {factor}"
            if p is not None and p == p:
                title += f"   (p={p:.3f})"
            ax.set_title(title); ax.grid(axis="y", alpha=0.3)
            path = out_dir / f"by_{factor}_{col}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
            made.append(path)
    return made


def make_graphs_stratified(df, metrics, out_dir, sp_lookup,
                            stratum_col="height"):
    if stratum_col not in df.columns: return []
    strata = sorted(df[stratum_col].dropna().unique(),
                    key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s, 9))
    out_dir.mkdir(parents=True, exist_ok=True); made = []
    for factor in PARAM_FACTORS:
        if factor not in df.columns: continue
        all_levels = order_levels(factor, list(df[factor].dropna().unique()))
        if len(all_levels) < 2: continue
        for col, label, unit in metrics:
            fig, axes = plt.subplots(
                1, len(strata),
                figsize=(max(5, len(all_levels)*1.5)*len(strata), 5),
                sharey=True)
            if len(strata) == 1: axes = [axes]
            for ax, stratum in zip(axes, strata):
                sub = df[df[stratum_col]==stratum]
                data, tick_labels = [], []
                for lv in all_levels:
                    vals = pd.to_numeric(
                        sub.loc[sub[factor]==lv, col],
                        errors="coerce").dropna().values
                    if len(vals):
                        data.append(vals)
                        tick_labels.append(f"{lv}\n(n={len(vals)})")
                if len(data) >= 2:
                    ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
                    for i, vals in enumerate(data, 1):
                        jit = (np.random.rand(len(vals))-0.5)*0.15
                        ax.scatter(np.full(len(vals),i)+jit, vals,
                                   alpha=0.5, s=18, color="#1f77b4", zorder=3)
                p = sp_lookup.get((stratum, factor, col))
                p_str = f"   p={p:.3f}" if (p is not None and p==p) else ""
                ax.set_title(f"{stratum}{p_str}", fontsize=10)
                ax.set_xlabel(factor); ax.grid(axis="y", alpha=0.3)
            axes[0].set_ylabel(f"{label} ({unit})")
            fig.suptitle(f"{label}  by  {factor}  — stratified by {stratum_col}",
                         fontsize=11)
            fig.tight_layout()
            path = out_dir / f"by_{factor}_{col}_stratified.png"
            fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
            made.append(path)
    return made


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--posture-csv",    type=Path, default=None)
    ap.add_argument("--participants",   type=str,  default=None)
    ap.add_argument("--no-graphs",      action="store_true")
    ap.add_argument("--no-stratify",    action="store_true")
    ap.add_argument("--stratify-by",    default="height")
    args = ap.parse_args()

    if args.posture_csv:
        csv_path = args.posture_csv
        out_dir  = csv_path.parent / "posture_comparison"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
        out_dir  = args.landmarks_root / "metrics" / "posture_comparison"
    else:
        sys.exit("Provide --landmarks-root or --posture-csv")

    if not csv_path.is_file():
        sys.exit(f"Posture features CSV not found: {csv_path}\n"
                 f"Run 07_extract_posture_features.py first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("Posture features CSV is empty.")

    if args.participants:
        keep = {p.strip() for p in args.participants.split(",") if p.strip()}
        df = df[df["participant"].astype(str).isin(keep)].copy()
        if df.empty:
            sys.exit(f"No rows for participants {sorted(keep)}")

    df = add_parameter_columns(df)
    metrics = available_metrics(df)

    print(f"Loaded {len(df)} Place events, "
          f"{df['participant'].nunique()} participant(s), "
          f"{df['trial'].nunique()} trial(s)")
    print(f"Posture features available for analysis: {len(metrics)}")
    for f in PARAM_FACTORS:
        vals = df[f].dropna().unique()
        print(f"  {f}: {sorted(map(str,vals))}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pooled ----
    summary = group_summary(df, metrics)
    summary.to_csv(out_dir / "group_summary.csv", index=False)
    print(f"Wrote group_summary.csv  ({len(summary)} rows)")

    tests = stat_tests(df, metrics)
    tests.to_csv(out_dir / "stat_tests.csv", index=False)
    print(f"Wrote stat_tests.csv  ({len(tests)} rows)")

    sig = tests[(tests["p_value"].notna()) & (tests["p_value"] < 0.05)]
    if len(sig):
        print("\nPooled significant differences (p < 0.05):")
        for _, r in sig.sort_values("p_value").iterrows():
            print(f"  {r['factor']:>12} -> {r['metric_label']:<26} "
                  f"{r['test']}  p={r['p_value']:.4f}")
    else:
        print("\nPooled: no differences reached p < 0.05.")

    if not args.no_graphs:
        p_lookup = {(r["factor"], r["metric"]): r["p_value"]
                    for _, r in tests.iterrows()}
        made = make_graphs(df, metrics, out_dir, p_lookup)
        print(f"Wrote {len(made)} pooled graph(s)")

    # ---- stratified ----
    sc = args.stratify_by
    if not args.no_stratify and sc in df.columns:
        strata = sorted(df[sc].dropna().unique(),
                        key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9))
        print(f"\n{'='*60}")
        print(f"STRATIFIED ANALYSIS — prototype factors within each {sc}")
        print(f"{'='*60}")

        strat_tests_df = stat_tests_stratified(df, metrics, sc)
        strat_tests_df.to_csv(out_dir / "stratified_stat_tests.csv", index=False)

        strat_sum = group_summary_stratified(df, metrics, sc)
        strat_sum.to_csv(out_dir / "stratified_summary.csv", index=False)

        print(f"Wrote stratified_stat_tests.csv  ({len(strat_tests_df)} rows)")
        print(f"Wrote stratified_summary.csv  ({len(strat_sum)} rows)\n")

        # p-value matrix
        print("P-value matrix (posture features × prototype factors × stratum):")
        print(f"  {'Feature':<30} {'Factor':<10}", end="")
        for s in strata: print(f"  {s:>8}", end="")
        print()
        print(f"  {'-'*30} {'-'*10}", end="")
        for _ in strata: print(f"  {'--------':>8}", end="")
        print()

        for col, label, unit in metrics:
            for factor in PARAM_FACTORS:
                row_vals = []
                any_sig = False
                for s in strata:
                    match = strat_tests_df[
                        (strat_tests_df["stratum"] == s) &
                        (strat_tests_df["factor"]  == factor) &
                        (strat_tests_df["metric"]  == col)]
                    if match.empty:
                        row_vals.append("     n/a")
                    else:
                        p = match.iloc[0]["p_value"]
                        if p != p:
                            row_vals.append("     n/a")
                        else:
                            marker = "*" if p < 0.05 else " "
                            if p < 0.05: any_sig = True
                            row_vals.append(f"{p:>7.3f}{marker}")
                if any_sig:
                    print(f"  {label:<30} {factor:<10}", end="")
                    for v in row_vals: print(f"  {v:>8}", end="")
                    print()

        print("\n  * = p < 0.05  (only rows with ≥1 significant cell shown)")

        # Significant findings list
        sig_s = strat_tests_df[strat_tests_df["p_value"].notna() &
                               (strat_tests_df["p_value"] < 0.05)]
        if len(sig_s):
            print("\nSignificant prototype effects on posture (p < 0.05) "
                  "within each stratum:")
            for stratum in strata:
                sub = sig_s[sig_s["stratum"]==stratum].sort_values("p_value")
                if sub.empty:
                    print(f"  [{stratum:6}]  none"); continue
                print(f"  [{stratum:6}]")
                for _, r in sub.iterrows():
                    print(f"    {r['factor']:>8} -> {r['metric_label']:<26} "
                          f"p={r['p_value']:.4f}")
        else:
            print("\nNo prototype effects on posture reached p < 0.05.")

        if not args.no_graphs:
            sp_lookup = {(r["stratum"], r["factor"], r["metric"]): r["p_value"]
                         for _, r in strat_tests_df.iterrows()}
            made_s = make_graphs_stratified(df, metrics, out_dir, sp_lookup, sc)
            print(f"\nWrote {len(made_s)} stratified graph(s)")

    print("\nNote: these tests treat each Place event as independent. "
          "Use 09_fit_posture_models.py for mixed-effects models that account "
          "for repeated events per participant.")


if __name__ == "__main__":
    main()