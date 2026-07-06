#!/usr/bin/env python3
"""
Weighted Multi-Criteria Prototype Ranking with Sensitivity Analysis.

Ranks prototype configurations per height condition using a weighted composite
cost score, where weights are derived from how much each metric actually varies
across prototype conditions (eta-squared from the screening analyses).

THREE WEIGHTING SCHEMES are compared for sensitivity analysis:
  1. eta2      Empirically derived: each metric weighted by its eta-squared
               across the prototype parameters (Length/Size/Weight/Angle),
               averaged across those four factors. Metrics that barely vary
               with prototype get low weight; those that differentiate
               prototypes get high weight. Most defensible.
  2. equal     Equal weights (Z-score sum). Simple baseline.
  3. custom    User-specified weights via --custom-weights (optional).

A prototype is flagged as ROBUST if it ranks in the top 3 under ALL schemes.
That robustness flag is the key output — a prototype that wins only under one
scheme is fragile; one that wins under all three is a genuine recommendation.

Weight sources:
  Performance metrics: <metrics>/comparison/group_summary.csv (from 05_compare)
                       or inferred from the data directly if not available.
  Posture features:    <metrics>/screening/posture_eta_squared.csv (from 08).

Inputs:
    <root>/metrics/combined_all.csv            (from 09_merge.py)
    <root>/metrics/comparison/group_summary.csv (from 05_compare)
    <root>/metrics/screening/posture_eta_squared.csv (from 08_screen_posture)

Outputs (under <root>/metrics/ranking/):
    prototype_rankings_by_height.csv     all schemes, all heights, all configs
    robust_prototypes.csv                configs flagged robust across schemes
    weights_used.csv                     the actual weights applied per scheme
    sensitivity_summary.txt              human-readable interpretation

Usage:
    python 11_rank_prototypes.py --landmarks-root "A:/Automated_chain_BETA/Participant_Landmarks"
    python 11_rank_prototypes.py --landmarks-root ... \\
        --custom-weights 1 1 1 2 1 1 3 2 1 1 1 1
    python 11_rank_prototypes.py --landmarks-root ... --reba-side left
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Full cost function — all 12 metrics, lower is better.
# Signed metrics (tilt, ulnar deviation) are abs-valued before Z-scoring
# so that deviation in either direction contributes equally to cost.
DEFAULT_COST_METRICS = [
    # --- Task performance ---
    "duration_s",           # time taken per placement
    "pos_jitter_mm",        # positional stability of pen tip
    "ang_jitter_deg",       # angular stability of pen orientation
    "perp_mean_deg",        # how far pen deviates from perpendicular to surface
    "leftright_mean_deg",   # left/right tilt (abs-valued)
    "updown_mean_deg",      # up/down tilt (abs-valued)
    # --- Ergonomic risk ---
    "reba_grand_right",     # REBA whole-body risk score
    # --- Wrist / hand posture (Quest) ---
    "wrist_neutral_dev_mean",    # wrist orientation from neutral (body frame)
    "reach_ratio_mean",          # arm extension fraction (1 = fully extended)
    "wrist_elevation_m_mean",    # wrist height above shoulder (m)
    "left_wrist_flex_mean",      # wrist dorsal-palmar flexion (Quest)
    "left_wrist_ulnar_dev_mean", # radial/ulnar deviation (abs-valued)
]

# Metrics that are signed and should be absolute-valued before Z-scoring.
# Deviation in either direction is equally costly.
ABS_VALUE_METRICS = {
    "leftright_mean_deg",        # left OR right tilt are equally bad
    "updown_mean_deg",           # up OR down tilt are equally bad
    "left_wrist_ulnar_dev_mean", # radial OR ulnar deviation are equally bad
    # NOTE: wrist_elevation_m_mean stays SIGNED:
    #   negative = wrist below shoulder (fine), positive = above (strain).
    #   "lower is better" already holds in the signed form.
}

# Prototype factors (what varies between prototypes, not including height)
PROTO_FACTORS = ["Length", "Size", "Weight", "Angle"]

# How many top prototypes to display per height
TOP_N = 3


# --------------------------------------------------------------------------- #
# Parameter parsing (Weight = Front_weighted / Not_weighted; no Position)
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
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long", "Short"):
            out["Length"] = tok
        elif tok in ("Large", "Small"):
            out["Size"] = tok
    return out


# --------------------------------------------------------------------------- #
# Weight derivation from screening outputs
# --------------------------------------------------------------------------- #
def eta2_weights_from_screening(metrics_dir: Path,
                                cost_metrics: list) -> dict:
    """
    For each cost metric, compute its mean eta-squared across the prototype
    factors (Length/Size/Weight/Angle).

    Reads from TWO sources:
      - metrics/comparison/group_summary.csv        (performance metrics, from 05)
      - metrics/posture_comparison/group_summary.csv (posture metrics, from 08)

    Falls back to computing eta2 directly from combined_all.csv if neither
    source is available for a given metric.
    """
    def eta2_from_gs(gs: pd.DataFrame, metric: str) -> float:
        """Compute mean eta2 across prototype factors from a group_summary table."""
        eta2s = []
        for fac in PROTO_FACTORS:
            sub = gs[(gs["factor"] == fac) & (gs["metric"] == metric)]
            if len(sub) < 2:
                continue
            ns    = sub["n"].values.astype(float)
            means = sub["mean"].values.astype(float)
            sds   = sub["sd"].values.astype(float)
            grand = np.average(means, weights=ns)
            ss_b  = np.sum(ns * (means - grand) ** 2)
            ss_w  = np.sum((ns - 1) * sds ** 2)
            ss_t  = ss_b + ss_w
            if ss_t > 0:
                eta2s.append(ss_b / ss_t)
        return float(np.mean(eta2s)) if eta2s else 0.0

    # Load both group_summary files where available
    perf_gs    = None
    posture_gs = None

    perf_path    = metrics_dir / "comparison" / "group_summary.csv"
    posture_path = metrics_dir / "posture_comparison" / "group_summary.csv"

    if perf_path.is_file():
        perf_gs = pd.read_csv(perf_path)
    if posture_path.is_file():
        posture_gs = pd.read_csv(posture_path)

    weights = {}
    for metric in cost_metrics:
        # Try performance table first, then posture table
        val = None
        if perf_gs is not None and metric in perf_gs["metric"].values:
            val = eta2_from_gs(perf_gs, metric)
        elif posture_gs is not None and metric in posture_gs["metric"].values:
            val = eta2_from_gs(posture_gs, metric)
        if val is not None:
            weights[metric] = val

    return weights


def compute_eta2_from_data(df: pd.DataFrame, metrics: list) -> dict:
    """Fallback: compute eta2 for each metric across prototype factors
    directly from the merged dataframe."""
    from scipy import stats
    weights = {}
    for metric in metrics:
        col = pd.to_numeric(df[metric], errors="coerce").dropna()
        if col.empty:
            weights[metric] = 0.0
            continue
        eta2s = []
        for fac in PROTO_FACTORS:
            if fac not in df.columns:
                continue
            groups = [g[metric].dropna().values
                      for _, g in df.groupby(fac)
                      if len(g[metric].dropna()) >= 2]
            if len(groups) < 2:
                continue
            grand = col.mean()
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ss_t = float(((col - grand) ** 2).sum())
            if ss_t > 0:
                eta2s.append(ss_b / ss_t)
        weights[metric] = float(np.mean(eta2s)) if eta2s else 0.0
    return weights


def normalise_weights(w: dict) -> dict:
    """Normalise to sum to 1, assign minimum 0.01 floor so nothing is silent."""
    vals = np.array([max(v, 0.01) for v in w.values()], dtype=float)
    total = vals.sum()
    return {k: float(v / total) for k, v in zip(w.keys(), vals / total * len(vals))}


# --------------------------------------------------------------------------- #
# Composite cost computation
# --------------------------------------------------------------------------- #
def compute_costs(df: pd.DataFrame, metrics: list,
                  weight_schemes: dict) -> pd.DataFrame:
    """
    For each weighting scheme, compute the weighted composite cost per row.
    Signed metrics in ABS_VALUE_METRICS are absolute-valued first, then
    all metrics are Z-scored globally, then weighted.
    Returns df with added cost columns.
    """
    z = {}
    for m in metrics:
        col = pd.to_numeric(df[m], errors="coerce")
        # Take absolute value for directionally symmetric metrics
        if m in ABS_VALUE_METRICS:
            col = col.abs()
        sd = col.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            z[m] = pd.Series(np.zeros(len(df)), index=df.index)
        else:
            z[m] = (col - col.mean()) / sd

    for scheme, weights in weight_schemes.items():
        cost = sum(weights[m] * z[m] for m in metrics)
        df[f"cost_{scheme}"] = cost
    return df


# --------------------------------------------------------------------------- #
# Ranking per height per scheme
# --------------------------------------------------------------------------- #
def rank_by_height(df: pd.DataFrame, schemes: list,
                   cost_metrics: list) -> pd.DataFrame:
    """Aggregate to prototype level (mean cost per config per height)."""
    group_keys = ["height"] + PROTO_FACTORS
    agg_dict = {f"cost_{s}": "mean" for s in schemes}
    for m in cost_metrics:
        agg_dict[m] = "mean"

    ranked = (df.groupby(group_keys)[list(agg_dict.keys())]
                .mean()
                .reset_index())

    # Rank within each height for each scheme
    for s in schemes:
        ranked[f"rank_{s}"] = (ranked.groupby("height")[f"cost_{s}"]
                                .rank(method="min", ascending=True))

    # Robustness: top-N under ALL schemes
    n_schemes = len(schemes)
    ranked["top_n_count"] = sum(
        (ranked[f"rank_{s}"] <= TOP_N).astype(int) for s in schemes)
    ranked["robust"] = ranked["top_n_count"] == n_schemes

    # Sort within height by the eta2 cost (primary), then equal (tiebreak)
    ranked = ranked.sort_values(
        by=["height", f"rank_{schemes[0]}", f"rank_{schemes[1]}"],
        ascending=True)
    return ranked


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def print_results(ranked: pd.DataFrame, schemes: list, weights_used: dict):
    print("\n" + "=" * 70)
    print("  PROTOTYPE RANKINGS BY HEIGHT")
    print("=" * 70)
    print(f"  Schemes: {', '.join(schemes)}")
    print(f"  Robust = top {TOP_N} under ALL {len(schemes)} schemes\n")

    for height in ["High", "Medium", "Low"]:
        hdata = ranked[ranked["height"] == height].head(TOP_N * 2)
        if hdata.empty:
            continue
        print(f"--- {height.upper()} ---")
        for _, row in hdata.iterrows():
            config = (f"{row.get('Length','?')} / {row.get('Size','?')} / "
                      f"{row.get('Weight','?')} / A{row.get('Angle','?')}")
            ranks = "  ".join(f"{s}=#{int(row[f'rank_{s}'])}"
                              for s in schemes)
            robust = "  *** ROBUST ***" if row["robust"] else ""
            print(f"  {config:<45} {ranks}{robust}")
        print()

    print("--- ROBUST WINNERS (top 3 under ALL schemes) ---")
    robust = ranked[ranked["robust"]]
    if robust.empty:
        print("  None — no configuration ranked top 3 under all schemes.")
        print("  (This means the ranking is sensitive to weighting choice.)")
        print("  Consider the eta2-scheme ranking as the primary recommendation.")
    else:
        for _, row in robust.sort_values("height").iterrows():
            config = (f"{row.get('Length','?')} / {row.get('Size','?')} / "
                      f"{row.get('Weight','?')} / A{row.get('Angle','?')}")
            print(f"  [{row['height']:6}] {config}")


def sensitivity_text(ranked: pd.DataFrame, schemes: list,
                     weights_used: dict) -> str:
    lines = ["SENSITIVITY ANALYSIS", "=" * 60, ""]
    lines.append(
        "A robust prototype ranks in the top 3 under ALL weighting schemes.")
    lines.append(
        "If rankings differ substantially between schemes, the 'best' prototype")
    lines.append(
        "is sensitive to how you weight the metrics — report both the eta2 and")
    lines.append("equal-weight rankings with a note on the weighting choice.")
    lines.append("")
    lines.append("Weights used per scheme:")
    for s, w in weights_used.items():
        lines.append(f"  [{s}]")
        for m, v in w.items():
            lines.append(f"    {m:<30} {v:.4f}")
    lines.append("")
    lines.append("Scheme definitions:")
    lines.append("  eta2   : weighted by mean eta-squared of each metric across")
    lines.append("           prototype parameters (Length/Size/Weight/Angle).")
    lines.append("           Metrics that don't differentiate prototypes get low weight.")
    lines.append("  equal  : equal Z-score sum (all metrics weighted 1.0).")
    lines.append("  domain : domain-knowledge weights for a drill-type precision task:")
    lines.append("           perp_mean_deg + reba_grand = 3x (primary outcomes)")
    lines.append("           duration + pos/ang jitter  = 2x (precision/efficiency)")
    lines.append("           tilt metrics               = 1.5x (secondary quality)")
    lines.append("           wrist/reach metrics        = 1x (ergonomic refinements)")
    lines.append("           Override with --custom-weights if priorities differ.")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    for height in ["High", "Medium", "Low"]:
        hdata = ranked[ranked["height"] == height]
        top_eta = hdata.nsmallest(1, "rank_eta2")
        top_eq = hdata.nsmallest(1, "rank_equal")
        if top_eta.empty:
            continue
        conf_eta = (f"{top_eta.iloc[0].get('Length','?')} / "
                    f"{top_eta.iloc[0].get('Size','?')} / "
                    f"{top_eta.iloc[0].get('Weight','?')} / "
                    f"A{top_eta.iloc[0].get('Angle','?')}")
        conf_eq = (f"{top_eq.iloc[0].get('Length','?')} / "
                   f"{top_eq.iloc[0].get('Size','?')} / "
                   f"{top_eq.iloc[0].get('Weight','?')} / "
                   f"A{top_eq.iloc[0].get('Angle','?')}")
        agree = (conf_eta == conf_eq)
        lines.append(f"  {height}: eta2 best = {conf_eta}")
        lines.append(f"         equal best = {conf_eq}")
        lines.append(f"         Agreement: {'YES' if agree else 'NO — ranking is weight-sensitive'}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--cost-metrics", nargs="+", default=DEFAULT_COST_METRICS,
                    help=f"Metrics to include in the cost function "
                         f"(default: {DEFAULT_COST_METRICS})")
    ap.add_argument("--custom-weights", nargs="+", type=float, default=None,
                    help="Custom weights for each --cost-metric, in the same "
                         "order. If provided, a third 'custom' scheme is added.")
    ap.add_argument("--reba-side", choices=["right", "left"], default="right",
                    help="Which REBA grand score to use (default: right)")
    args = ap.parse_args()

    # Substitute the correct REBA side
    cost_metrics = [
        m.replace("reba_grand_right", f"reba_grand_{args.reba_side}")
         .replace("reba_grand_left", f"reba_grand_{args.reba_side}")
        for m in args.cost_metrics
    ]

    metrics_dir = args.landmarks_root / "metrics"
    combined_path = metrics_dir / "combined_all.csv"

    if not combined_path.is_file():
        sys.exit(f"combined_all.csv not found: {combined_path}\n"
                 "Run 09_merge.py first.")

    df = pd.read_csv(combined_path)

    # Check metrics exist
    missing = [m for m in cost_metrics if m not in df.columns]
    if missing:
        sys.exit(f"These cost metrics are not in the data: {missing}\n"
                 f"Available columns: {list(df.columns)}")

    # Parse prototype parameters
    params = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PROTO_FACTORS:
        df[c] = params[c]

    # ------------------------------------------------------------------ #
    # Build weight schemes
    # ------------------------------------------------------------------ #
    # Scheme 1: eta2 from screening outputs (preferred)
    raw_eta2 = eta2_weights_from_screening(metrics_dir, cost_metrics)
    if not raw_eta2 or all(v == 0 for v in raw_eta2.values()):
        print("Screening outputs not found or all-zero; computing eta2 from data.")
        raw_eta2 = compute_eta2_from_data(df, cost_metrics)

    eta2_w = normalise_weights(raw_eta2)

    # Scheme 2: equal
    equal_w = {m: 1.0 for m in cost_metrics}

    # Scheme 3: domain-knowledge weights (default, overridable via --custom-weights)
    # Rationale for a drill-type precision placing task:
    #   perpendicularity    -> primary task-quality metric (3x)
    #   reba_grand          -> primary ergonomic risk metric (3x)
    #   pos/ang jitter      -> placement precision (2x)
    #   duration            -> efficiency, important but secondary (2x)
    #   tilt metrics        -> secondary to perpendicularity (1.5x)
    #   wrist metrics       -> ergonomic refinements, secondary to REBA (1x)
    DOMAIN_WEIGHTS = {
        "duration_s":               2.0,
        "pos_jitter_mm":            2.0,
        "ang_jitter_deg":           2.0,
        "perp_mean_deg":            3.0,
        "leftright_mean_deg":       1.5,
        "updown_mean_deg":          1.5,
        "reba_grand_right":         3.0,
        "wrist_neutral_dev_mean":   1.0,
        "reach_ratio_mean":         1.0,
        "wrist_elevation_m_mean":   1.0,
        "left_wrist_flex_mean":     1.0,
        "left_wrist_ulnar_dev_mean": 1.0,
    }
    if args.custom_weights:
        if len(args.custom_weights) != len(cost_metrics):
            sys.exit(f"--custom-weights needs {len(cost_metrics)} values "
                     f"(one per metric); got {len(args.custom_weights)}")
        domain_w = {m: w for m, w in zip(cost_metrics, args.custom_weights)}
        print("Using user-supplied custom weights as the domain-knowledge scheme.")
    else:
        domain_w = {m: DOMAIN_WEIGHTS.get(m, 1.0) for m in cost_metrics}

    weight_schemes = {"eta2": eta2_w, "equal": equal_w, "domain": domain_w}

    schemes = list(weight_schemes.keys())

    print(f"Loaded {len(df)} events, "
          f"{df['participant'].nunique()} participants")
    print(f"Cost metrics ({len(cost_metrics)}): {cost_metrics}")
    print(f"Weighting schemes: {schemes}")
    print()

    # ------------------------------------------------------------------ #
    # Full per-factor eta2 breakdown
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("  FEATURE WEIGHTS — ETA2 PER PROTOTYPE FACTOR")
    print("=" * 70)
    print("  (†) = absolute value taken before Z-scoring (deviation in either")
    print("        direction is equally costly)")
    print(f"  {'Metric':<34} {'Length':>8} {'Size':>8} "
          f"{'Weight':>8} {'Angle':>8}  {'MEAN':>8}  {'Norm.wt':>8}")
    print(f"  {'-'*34} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}  {'-'*8}")

    # Re-compute the per-factor eta2 for the breakdown table
    per_factor_eta2 = {}
    for metric in cost_metrics:
        col = pd.to_numeric(df[metric], errors="coerce").dropna()
        fac_eta2 = {}
        for fac in PROTO_FACTORS:
            if fac not in df.columns:
                fac_eta2[fac] = np.nan
                continue
            grand = col.mean()
            ss_t = float(((col - grand) ** 2).sum())
            if ss_t <= 0:
                fac_eta2[fac] = np.nan
                continue
            groups = [pd.to_numeric(g[metric], errors="coerce").dropna().values
                      for _, g in df.groupby(fac)
                      if len(pd.to_numeric(g[metric], errors="coerce").dropna()) >= 2]
            if len(groups) < 2:
                fac_eta2[fac] = np.nan
                continue
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            fac_eta2[fac] = float(ss_b / ss_t)
        per_factor_eta2[metric] = fac_eta2

    for metric in cost_metrics:
        fe = per_factor_eta2[metric]
        vals = [fe.get(f, np.nan) for f in PROTO_FACTORS]
        mean_val = float(np.nanmean(vals)) if any(v == v for v in vals) else np.nan
        norm_val = eta2_w.get(metric, np.nan)
        fmt = lambda v: f"{v:>8.4f}" if v == v else "     n/a"
        abs_marker = " †" if metric in ABS_VALUE_METRICS else "  "
        label = f"{metric}{abs_marker}"
        print(f"  {label:<34} {fmt(vals[0])} {fmt(vals[1])} "
              f"{fmt(vals[2])} {fmt(vals[3])}  {fmt(mean_val)}  {fmt(norm_val)}")

    print()
    print("  Norm.wt = normalised eta2 weight used in the eta2 cost scheme.")
    print("  Metrics with low eta2 across all prototype factors barely vary")
    print("  with prototype choice and contribute little to ranking.")
    print()

    # ------------------------------------------------------------------ #
    # All weighting schemes side by side
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("  ALL WEIGHTING SCHEMES")
    print("=" * 70)
    header = f"  {'Metric':<32}"
    for s in schemes:
        header += f"  {s:>10}"
    print(header)
    print(f"  {'-'*32}" + "  ----------" * len(schemes))
    for metric in cost_metrics:
        row_str = f"  {metric:<32}"
        for s in schemes:
            row_str += f"  {weight_schemes[s].get(metric, 0):>10.4f}"
        # Mark domain rationale inline
        if metric in ("perp_mean_deg", "reba_grand_right"):
            row_str += "  <- primary (3x)"
        elif metric in ("duration_s", "pos_jitter_mm", "ang_jitter_deg"):
            row_str += "  <- precision/efficiency (2x)"
        elif metric in ("leftright_mean_deg", "updown_mean_deg"):
            row_str += "  <- secondary quality (1.5x)"
        print(row_str)
    print()

    print(f"Weights (eta2 scheme — derived from prototype-parameter variation):")
    for m, w in eta2_w.items():
        bar = "█" * max(1, int(w * 4))
        raw = raw_eta2.get(m, 0)
        print(f"  {m:<30} {bar:<20} {w:.3f}  (raw eta2={raw:.4f})")
    print()

    # ------------------------------------------------------------------ #
    # Compute costs and rank
    # ------------------------------------------------------------------ #
    df = compute_costs(df, cost_metrics, weight_schemes)
    ranked = rank_by_height(df, schemes, cost_metrics)

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    out_dir = metrics_dir / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(out_dir / "prototype_rankings_by_height.csv", index=False)

    robust = ranked[ranked["robust"]]
    robust.to_csv(out_dir / "robust_prototypes.csv", index=False)

    # Weights table
    w_rows = []
    for s, w in weight_schemes.items():
        for m, v in w.items():
            w_rows.append({"scheme": s, "metric": m, "weight": v,
                           "raw_eta2": raw_eta2.get(m, np.nan)})
    pd.DataFrame(w_rows).to_csv(out_dir / "weights_used.csv", index=False)

    sens_text = sensitivity_text(ranked, schemes, weight_schemes)
    (out_dir / "sensitivity_summary.txt").write_text(sens_text, encoding="utf-8")

    print_results(ranked, schemes, weight_schemes)

    print()
    print("=" * 70)
    print(sens_text)

    print(f"\nWrote outputs to {out_dir}")
    print(f"  prototype_rankings_by_height.csv  ({len(ranked)} rows)")
    print(f"  robust_prototypes.csv             ({len(robust)} rows)")
    print(f"  weights_used.csv")
    print(f"  sensitivity_summary.txt")


if __name__ == "__main__":
    main()