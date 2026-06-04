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

Outputs (under <landmarks_root>/metrics/comparison/):
  group_summary.csv         tidy table: factor, level, metric, mean, sd, n
  stat_tests.csv            factor, metric, test, statistic, p_value, n_groups
  by_<factor>_<metric>.png  box plots

Usage:
  python 05_compare_performance.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python 05_compare_performance.py --metrics-csv A:\Automated_chain_BETA\Participant_Landmarks\metrics\place_metrics_combined.csv
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


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None,
                    help="Root; reads <root>/metrics/place_metrics_combined.csv")
    ap.add_argument("--metrics-csv", type=Path, default=None,
                    help="Path to place_metrics_combined.csv (overrides root)")
    ap.add_argument("--no-graphs", action="store_true")
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
    # Report parsed parameter coverage
    for f in PARAM_FACTORS:
        vals = df[f].dropna().unique()
        print(f"  {f}: {sorted(map(str, vals))}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = group_summary(df)
    summary_path = out_dir / "group_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(summary)} rows)")

    tests = stat_tests(df)
    tests_path = out_dir / "stat_tests.csv"
    tests.to_csv(tests_path, index=False)
    print(f"Wrote {tests_path}  ({len(tests)} rows)")

    # Console: significant results (p < 0.05)
    sig = tests[(tests["p_value"].notna()) & (tests["p_value"] < 0.05)]
    if len(sig):
        print("\nSignificant differences (p < 0.05):")
        for _, r in sig.sort_values("p_value").iterrows():
            print(f"  {r['factor']:>12} -> {r['metric_label']:<24} "
                  f"{r['test']}  p={r['p_value']:.4f}")
    else:
        print("\nNo differences reached p < 0.05 "
              "(small samples make this likely; check effect sizes in the summary).")

    if not args.no_graphs:
        p_lookup = {(r["factor"], r["metric"]): r["p_value"]
                    for _, r in tests.iterrows()}
        made = make_graphs(df, out_dir, p_lookup)
        print(f"\nWrote {len(made)} graph(s) to {out_dir}")

    print("\nNote: each test compares one factor at a time. If your design isn't "
          "balanced (e.g. most 'weighted' trials are also 'Large'), factors can "
          "be confounded and a single-factor effect may really reflect another. "
          "Treat these as descriptive screening, not a substitute for a full "
          "model (e.g. mixed-effects with participant as a random effect) if you "
          "need confirmatory results.")


if __name__ == "__main__":
    main()
