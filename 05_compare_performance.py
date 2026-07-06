#!/usr/bin/env python3
"""
Compare Place-event performance across participants, heights, and trial
parameters (Length, Size, Position, Weight, Angle).

Input: the combined metrics table written by place_metrics.py
       <landmarks_root>/metrics/place_metrics_combined.csv
       (one row per Place event, with columns participant, trial, height,
        duration_s, perp_mean_deg, leftright_mean_deg, updown_mean_deg,
        pos_jitter_mm, ang_jitter_deg, ...)

The trial stem is parsed into parameters:
   <PID>_<Length>_<Size>_<Position>_<Weight>_A<Angle>
e.g. P004_Long_Large_Front_weighted_A135  ->
   Length=Long, Size=Large, Position=Front, Weight=weighted, Angle=135
'Not_weighted' is handled (the weight field may be two tokens).

For every metric, the script:
  - tabulates group means / SD / n for each factor
  - runs an omnibus test per factor:
        * 2 groups  -> Mann-Whitney U (non-parametric)
        * >2 groups -> Kruskal-Wallis H (non-parametric)
    Non-parametric tests are used because per-cell samples are small and not
    guaranteed normal; group means + SD are reported for effect size.
  - draws box plots of each metric by each factor

--stratify-by height (default):
  Re-runs the PROTOTYPE PARAMETER tests (Length/Size/Weight/Angle) separately
  within each height stratum (High / Medium / Low), so the height confound is
  removed and you get clean p-values for each prototype factor at each height.
  Outputs:
    stratified_stat_tests.csv    factor, stratum, metric, p_value ...
    stratified_summary.csv       means/SDs within each stratum
    by_<factor>_<metric>_by_height.png  one panel per height

Outputs (under <landmarks_root>/metrics/comparison/):
  group_summary.csv         tidy table: factor, level, metric, mean, sd, n
  stat_tests.csv            factor, metric, test, statistic, p_value, n_groups
  stratified_stat_tests.csv prototype factor tests within each height stratum
  stratified_summary.csv    group means/SDs within each height stratum
  by_<factor>_<metric>.png  box plots (pooled)
  by_<factor>_<metric>_stratified.png  box plots split by height stratum

Usage:
  python 05_compare_performance.py --landmarks-root A:\\Automated_chain_BETA\\Participant_Landmarks
  python 05_compare_performance.py --metrics-csv A:\\...\\place_metrics_combined.csv
  python 05_compare_performance.py --landmarks-root ... --no-stratify   # skip stratification
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


METRICS = [
    ("duration_s", "Duration", "s"),
    ("perp_mean_deg", "Perpendicularity (mean)", "deg"),
    ("leftright_mean_deg", "Left/right tilt (mean)", "deg"),
    ("updown_mean_deg", "Up/down tilt (mean)", "deg"),
    ("pos_jitter_mm", "Positional jitter", "mm"),
    ("ang_jitter_deg", "Angular jitter", "deg"),
]

# Factors to compare across. 'participant' and 'height' come straight from the
# table; the rest are parsed from the trial stem.
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
ALL_FACTORS = ["participant", "height"] + PARAM_FACTORS


# --------------------------------------------------------------------------- #
# Parameter parsing
# --------------------------------------------------------------------------- #
def parse_params(trial: str) -> dict:
    """Parse trial parameters by matching known values.

    Known vocabularies:
        Length : Long | Short
        Size   : Large | Small
        Weight : Front_weighted | Not_weighted
        Angle  : A<digits>  (stored as int)

    Note: the weight descriptor is two tokens ('Front_weighted' or
    'Not_weighted'); 'Front' is part of the weight, NOT a separate position
    factor. Returns a dict; fields not found are None.
    """
    out = {k: None for k in PARAM_FACTORS}
    tokens = trial.split("_")
    joined = "_".join(tokens)

    # Weight: match the two-token descriptors
    if "Not_weighted" in joined:
        out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined:
        out["Weight"] = "Front_weighted"
    elif "weighted" in tokens:
        # Fallback: bare 'weighted' with an unknown prefix
        out["Weight"] = "weighted"

    # Angle: token like A135
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:])
            break

    # Categorical fields matched against their known vocabularies
    vocab = {
        "Length": {"Long", "Short"},
        "Size":   {"Large", "Small"},
    }
    for field, allowed in vocab.items():
        for tok in tokens:
            if tok in allowed:
                out[field] = tok
                break

    return out


def add_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = parsed[c]
    return df


# --------------------------------------------------------------------------- #
# Summaries + stats
# --------------------------------------------------------------------------- #
def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy table: factor, level, metric, mean, sd, n."""
    records = []
    for factor in ALL_FACTORS:
        if factor not in df.columns:
            continue
        for level, sub in df.groupby(factor, dropna=True):
            for col, label, unit in METRICS:
                if col not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                records.append({
                    "factor": factor,
                    "level": level,
                    "metric": col,
                    "metric_label": label,
                    "unit": unit,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                })
    return pd.DataFrame(records)


def stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Omnibus non-parametric test per factor per metric."""
    records = []
    for factor in ALL_FACTORS:
        if factor not in df.columns:
            continue
        for col, label, unit in METRICS:
            if col not in df.columns:
                continue
            groups = []
            levels = []
            for level, sub in df.groupby(factor, dropna=True):
                vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
                if len(vals) >= 1:
                    groups.append(vals)
                    levels.append(str(level))
            # Need at least 2 groups, each with enough data
            usable = [g for g in groups if len(g) >= 2]
            if len(usable) < 2:
                records.append({
                    "factor": factor, "metric": col, "metric_label": label,
                    "test": "skipped (insufficient data)",
                    "statistic": np.nan, "p_value": np.nan,
                    "n_groups": len(groups),
                    "levels": ", ".join(levels),
                })
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
                "test": test, "statistic": float(stat) if stat == stat else np.nan,
                "p_value": float(p) if p == p else np.nan,
                "n_groups": len(usable),
                "levels": ", ".join(levels),
            })
    return pd.DataFrame(records)


def stat_tests_stratified(df: pd.DataFrame,
                          stratum_col: str = "height") -> pd.DataFrame:
    """Run prototype-parameter tests within each stratum of stratum_col.

    For each combination of (stratum, prototype factor, metric) run the same
    non-parametric omnibus test as stat_tests(), but only on the rows that
    belong to that stratum.  This removes the stratum confound so the p-value
    reflects a genuine within-condition effect of the prototype parameter.

    Returns a tidy DataFrame with an extra 'stratum' column.
    """
    if stratum_col not in df.columns:
        return pd.DataFrame()

    records = []
    strata = sorted(df[stratum_col].dropna().unique())

    for stratum in strata:
        sub_df = df[df[stratum_col] == stratum].copy()

        for factor in PARAM_FACTORS:
            if factor not in sub_df.columns:
                continue
            for col, label, unit in METRICS:
                if col not in sub_df.columns:
                    continue
                groups, levels = [], []
                for level, grp in sub_df.groupby(factor, dropna=True):
                    vals = pd.to_numeric(grp[col],
                                         errors="coerce").dropna().values
                    if len(vals) >= 1:
                        groups.append(vals)
                        levels.append(str(level))

                usable = [g for g in groups if len(g) >= 2]
                if len(usable) < 2:
                    records.append({
                        "stratum": stratum, "factor": factor,
                        "metric": col, "metric_label": label,
                        "test": "skipped (insufficient data)",
                        "statistic": np.nan, "p_value": np.nan,
                        "n_groups": len(groups),
                        "levels": ", ".join(levels),
                        "n_events": len(sub_df),
                    })
                    continue

                try:
                    if len(usable) == 2:
                        stat, p = stats.mannwhitneyu(
                            usable[0], usable[1], alternative="two-sided")
                        test = "Mann-Whitney U"
                    else:
                        stat, p = stats.kruskal(*usable)
                        test = "Kruskal-Wallis H"
                except ValueError as e:
                    stat, p, test = np.nan, np.nan, f"error: {e}"

                records.append({
                    "stratum":      stratum,
                    "factor":       factor,
                    "metric":       col,
                    "metric_label": label,
                    "test":         test,
                    "statistic":    float(stat) if stat == stat else np.nan,
                    "p_value":      float(p)    if p    == p    else np.nan,
                    "n_groups":     len(usable),
                    "levels":       ", ".join(levels),
                    "n_events":     int(len(sub_df)),
                })

    return pd.DataFrame(records)


def group_summary_stratified(df: pd.DataFrame,
                             stratum_col: str = "height") -> pd.DataFrame:
    """group_summary() but with an extra 'stratum' column, restricted to
    PARAM_FACTORS so the table stays focused on the prototype effects."""
    if stratum_col not in df.columns:
        return pd.DataFrame()

    records = []
    for stratum, sub_df in df.groupby(stratum_col, dropna=True):
        for factor in PARAM_FACTORS:
            if factor not in sub_df.columns:
                continue
            for level, grp in sub_df.groupby(factor, dropna=True):
                for col, label, unit in METRICS:
                    if col not in grp.columns:
                        continue
                    vals = pd.to_numeric(grp[col],
                                          errors="coerce").dropna()
                    if len(vals) == 0:
                        continue
                    records.append({
                        "stratum":      stratum,
                        "factor":       factor,
                        "level":        level,
                        "metric":       col,
                        "metric_label": label,
                        "unit":         unit,
                        "n":            int(len(vals)),
                        "mean":         float(vals.mean()),
                        "sd":           float(vals.std(ddof=1))
                                        if len(vals) > 1 else 0.0,
                        "median":       float(vals.median()),
                    })
    return pd.DataFrame(records)



def order_levels(factor, levels):
    """Sensible ordering for known factors."""
    orders = {
        "height": ["High", "Medium", "Low"],
        "Length": ["Short", "Long"],
        "Size": ["Small", "Large"],
        "Weight": ["Not_weighted", "Front_weighted"],
    }
    if factor in orders:
        known = [l for l in orders[factor] if l in levels]
        rest = [l for l in levels if l not in known]
        return known + sorted(rest)
    # Angle numeric, participant alphabetical
    try:
        return sorted(levels, key=lambda x: float(x))
    except (TypeError, ValueError):
        return sorted(levels, key=str)


def make_graphs(df: pd.DataFrame, out_dir: Path, p_lookup: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for factor in ALL_FACTORS:
        if factor not in df.columns:
            continue
        levels = [l for l in df[factor].dropna().unique()]
        if len(levels) < 2:
            continue
        levels = order_levels(factor, list(levels))
        for col, label, unit in METRICS:
            if col not in df.columns:
                continue
            data, labels = [], []
            for lv in levels:
                vals = pd.to_numeric(
                    df.loc[df[factor] == lv, col], errors="coerce").dropna().values
                if len(vals):
                    data.append(vals)
                    labels.append(f"{lv}\n(n={len(vals)})")
            if len(data) < 2:
                continue
            fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.3), 5))
            ax.boxplot(data, tick_labels=labels, showmeans=True)
            for i, vals in enumerate(data, 1):
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           alpha=0.5, s=18, color="#1f77b4", zorder=3)
            ax.set_ylabel(f"{label} ({unit})")
            ax.set_xlabel(factor)
            title = f"{label} by {factor}"
            p = p_lookup.get((factor, col))
            if p is not None and p == p:
                title += f"   (p = {p:.3f})"
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
            path = out_dir / f"by_{factor}_{col}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            made.append(path)
    return made


def make_graphs_stratified(df: pd.DataFrame, out_dir: Path,
                           p_lookup: dict,
                           stratum_col: str = "height"):
    """One figure per (factor, metric): sub-plots side-by-side for each stratum.

    p_lookup keys are (stratum, factor, metric).
    """
    if stratum_col not in df.columns:
        return []
    strata  = sorted(df[stratum_col].dropna().unique(),
                     key=lambda s: {"High": 0, "Medium": 1, "Low": 2}.get(s, 9))
    if not strata:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    for factor in PARAM_FACTORS:
        if factor not in df.columns:
            continue
        all_levels = order_levels(factor,
                                  list(df[factor].dropna().unique()))
        if len(all_levels) < 2:
            continue

        for col, label, unit in METRICS:
            if col not in df.columns:
                continue

            fig, axes = plt.subplots(
                1, len(strata),
                figsize=(max(5, len(all_levels) * 1.5) * len(strata), 5),
                sharey=True)
            if len(strata) == 1:
                axes = [axes]

            for ax, stratum in zip(axes, strata):
                sub = df[df[stratum_col] == stratum]
                data, tick_labels = [], []
                for lv in all_levels:
                    vals = pd.to_numeric(
                        sub.loc[sub[factor] == lv, col],
                        errors="coerce").dropna().values
                    if len(vals):
                        data.append(vals)
                        tick_labels.append(f"{lv}\n(n={len(vals)})")

                if len(data) >= 2:
                    ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
                    for i, vals in enumerate(data, 1):
                        jit = (np.random.rand(len(vals)) - 0.5) * 0.15
                        ax.scatter(np.full(len(vals), i) + jit, vals,
                                   alpha=0.5, s=18, color="#1f77b4", zorder=3)
                elif data:
                    ax.text(0.5, 0.5, "< 2 groups",
                            ha="center", va="center", transform=ax.transAxes)

                p = p_lookup.get((stratum, factor, col))
                p_str = f"   p={p:.3f}" if (p is not None and p == p) else ""
                ax.set_title(f"{stratum}{p_str}", fontsize=10)
                ax.set_xlabel(factor)
                ax.grid(axis="y", alpha=0.3)

            axes[0].set_ylabel(f"{label} ({unit})")
            fig.suptitle(f"{label}  by  {factor}  —  stratified by {stratum_col}",
                         fontsize=11)
            fig.tight_layout()
            path = out_dir / f"by_{factor}_{col}_stratified.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            made.append(path)

    return made



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None,
                    help="Root; reads <root>/metrics/place_metrics_combined.csv")
    ap.add_argument("--metrics-csv", type=Path, default=None,
                    help="Path to place_metrics_combined.csv (overrides root)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-stratify", action="store_true",
                    help="Skip the height-stratified prototype-parameter tests")
    ap.add_argument("--stratify-by", default="height",
                    help="Column to stratify by (default: height)")
    args = ap.parse_args()

    if args.metrics_csv:
        csv_path = args.metrics_csv
        out_dir = csv_path.parent / "comparison"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        out_dir = args.landmarks_root / "metrics" / "comparison"
    else:
        sys.exit("Provide --landmarks-root or --metrics-csv")

    if not csv_path.is_file():
        sys.exit(f"Metrics CSV not found: {csv_path}\n"
                 f"Run place_metrics.py first to generate it.")

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("Metrics CSV is empty.")

    df = add_parameter_columns(df)
    print(f"Loaded {len(df)} Place events from {len(df['trial'].unique())} trial(s), "
          f"{len(df['participant'].unique())} participant(s)")
    for f in PARAM_FACTORS:
        vals = df[f].dropna().unique()
        print(f"  {f}: {sorted(map(str, vals))}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pooled analysis (existing behaviour)
    # ------------------------------------------------------------------ #
    summary = group_summary(df)
    summary.to_csv(out_dir / "group_summary.csv", index=False)
    print(f"Wrote {out_dir / 'group_summary.csv'}  ({len(summary)} rows)")

    tests = stat_tests(df)
    tests.to_csv(out_dir / "stat_tests.csv", index=False)
    print(f"Wrote {out_dir / 'stat_tests.csv'}  ({len(tests)} rows)")

    sig = tests[(tests["p_value"].notna()) & (tests["p_value"] < 0.05)]
    if len(sig):
        print("\nPooled significant differences (p < 0.05):")
        for _, r in sig.sort_values("p_value").iterrows():
            print(f"  {r['factor']:>12} -> {r['metric_label']:<24} "
                  f"{r['test']}  p={r['p_value']:.4f}")
    else:
        print("\nPooled: no differences reached p < 0.05.")

    if not args.no_graphs:
        p_lookup = {(r["factor"], r["metric"]): r["p_value"]
                    for _, r in tests.iterrows()}
        made = make_graphs(df, out_dir, p_lookup)
        print(f"Wrote {len(made)} pooled graph(s)")

    # ------------------------------------------------------------------ #
    # Stratified analysis — prototype factors within each height
    # ------------------------------------------------------------------ #
    if not args.no_stratify and args.stratify_by in df.columns:
        sc = args.stratify_by
        strata = sorted(df[sc].dropna().unique())
        print(f"\n{'='*60}")
        print(f"STRATIFIED ANALYSIS  —  prototype factors within each {sc}")
        print(f"{'='*60}")
        print(f"Strata: {strata}")
        print("Confound removed: p-values now reflect prototype effects "
              "within a single height condition.\n")

        strat_tests = stat_tests_stratified(df, sc)
        strat_tests.to_csv(out_dir / "stratified_stat_tests.csv", index=False)
        print(f"Wrote {out_dir / 'stratified_stat_tests.csv'}"
              f"  ({len(strat_tests)} rows)")

        strat_summary = group_summary_stratified(df, sc)
        strat_summary.to_csv(out_dir / "stratified_summary.csv", index=False)
        print(f"Wrote {out_dir / 'stratified_summary.csv'}"
              f"  ({len(strat_summary)} rows)\n")

        # Console: significant results per stratum
        sig_s = strat_tests[
            (strat_tests["p_value"].notna()) & (strat_tests["p_value"] < 0.05)]

        if len(sig_s):
            print("Significant prototype-parameter effects (p < 0.05) "
                  "within each stratum:")
            # Print as a tidy table grouped by stratum
            for stratum in strata:
                sub = sig_s[sig_s["stratum"] == stratum].sort_values("p_value")
                if sub.empty:
                    print(f"  [{stratum:6}]  none")
                    continue
                print(f"  [{stratum:6}]")
                for _, r in sub.iterrows():
                    print(f"    {r['factor']:>8} -> {r['metric_label']:<24} "
                          f"{r['test']}  p={r['p_value']:.4f}")
        else:
            print("No prototype-parameter effects reached p < 0.05 within "
                  "any stratum.\n"
                  "This is common with small samples; check the effect sizes "
                  "(means/SDs) in stratified_summary.csv.")

        # Full p-value table across all strata for quick overview
        print(f"\nFull p-value matrix (prototype factors × metrics × stratum):")
        print(f"  {'Factor':<10} {'Metric':<26}", end="")
        for s in strata:
            print(f"  {s:>8}", end="")
        print()
        print(f"  {'-'*10} {'-'*26}", end="")
        for _ in strata:
            print(f"  {'--------':>8}", end="")
        print()
        for factor in PARAM_FACTORS:
            for col, label, _ in METRICS:
                row_vals = []
                any_val = False
                for s in strata:
                    match = strat_tests[
                        (strat_tests["stratum"] == s) &
                        (strat_tests["factor"]  == factor) &
                        (strat_tests["metric"]  == col)]
                    if match.empty:
                        row_vals.append("     n/a")
                    else:
                        p = match.iloc[0]["p_value"]
                        if p != p:
                            row_vals.append("     n/a")
                        else:
                            any_val = True
                            marker = "*" if p < 0.05 else " "
                            row_vals.append(f"{p:>7.3f}{marker}")
                if any_val:
                    print(f"  {factor:<10} {label:<26}", end="")
                    for v in row_vals:
                        print(f"  {v:>8}", end="")
                    print()

        print("\n  * = p < 0.05  (non-parametric omnibus test within stratum)")

        if not args.no_graphs:
            sp_lookup = {
                (r["stratum"], r["factor"], r["metric"]): r["p_value"]
                for _, r in strat_tests.iterrows()}
            made_s = make_graphs_stratified(df, out_dir, sp_lookup, sc)
            print(f"\nWrote {len(made_s)} stratified graph(s)")

    elif not args.no_stratify:
        print(f"\nWarning: --stratify-by column '{args.stratify_by}' "
              f"not found in data; skipping stratification.")

    print("\nNote: stratified tests still compare one factor at a time and treat "
          "each Place event as independent. For a model that accounts for the "
          "repeated-measures structure (multiple events per participant) use "
          "06_fit_mixed_models.py — the stratified p-values here are "
          "descriptive screening.")


if __name__ == "__main__":
    main()