#!/usr/bin/env python3
"""
Screen which POSTURE features actually vary across the experimental factors.

This is the "what moves?" step: before correlating posture with performance or
optimising prototypes, find which of the ~25 posture features respond to
participant / height / prototype parameters at all, and which are flat.

It mirrors the two views used for the performance metrics:
  (A) DESCRIPTIVE eta-squared  (like 05_compare_performance.py): participant is
      treated as just another factor. eta^2 = fraction of a feature's total
      variance explained by that factor (one-way). Quick, assumption-light.
  (B) MIXED-MODEL  (like 06_fit_mixed_models.py): prototype params + height as
      fixed effects, participant as a random intercept. Reports each fixed
      factor's significance (Wald) accounting for repeated trials per person.

Input:  <root>/metrics/posture_features_combined.csv   (from 07_…)
Outputs (under <root>/metrics/screening/):
    posture_eta_squared.csv      feature x factor -> eta^2 (+ p from one-way)
    posture_mixed_effects.csv     feature x fixed factor -> p, coef, significant
    posture_screening_summary.csv one row per feature: strongest factor (both
                                  views), peak eta^2, whether it varies at all
    heatmap_eta_squared.png       feature x factor eta^2 heatmap

Usage:
    python 08_screen_posture.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks --participants P003,P004
    python 08_screen_posture.py --metrics-csv path/to/posture_features_combined.csv
    python 08_screen_posture.py --metrics-csv path/to/posture_features_combined.csv --participants P003,P004
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

from scipy import stats


# Columns that are keys/metadata, not posture features
NON_FEATURE = {"participant", "trial", "place_index", "height",
               "start_t_s", "stop_t_s", "duration_s",
               "left_hand_n_frames", "right_hand_n_frames", "reba_n_frames"}

# Factors, mirroring the performance scripts
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
DESCRIPTIVE_FACTORS = ["participant", "height"] + PARAM_FACTORS   # eta^2 (05-style)
MIXED_FIXED = ["Length", "Size", "Weight", "Angle", "height"]     # 06-style

REFERENCE = {"Length": "Short", "Size": "Small",
             "Weight": "Not_weighted", "height": "Medium"}


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


def feature_columns(df):
    feats = []
    for c in df.columns:
        if c in NON_FEATURE or c in PARAM_FACTORS:
            continue
        # numeric only
        if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
            feats.append(c)
    return feats


# --------------------------------------------------------------------------- #
# (A) Descriptive eta-squared (one-way), participant treated as a factor
# --------------------------------------------------------------------------- #
def eta_squared_oneway(values, groups):
    """eta^2 = SS_between / SS_total for a one-way grouping. Returns (eta2, p)
    using a one-way ANOVA F-test (parametric) for the p-value."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    if df["g"].nunique() < 2 or len(df) < 3:
        return np.nan, np.nan
    grand = df["y"].mean()
    ss_total = ((df["y"] - grand) ** 2).sum()
    if ss_total <= 0:
        return np.nan, np.nan
    ss_between = 0.0
    group_arrays = []
    for _, sub in df.groupby("g"):
        ss_between += len(sub) * (sub["y"].mean() - grand) ** 2
        group_arrays.append(sub["y"].values)
    eta2 = ss_between / ss_total
    # p-value from one-way ANOVA (or Kruskal if you prefer non-parametric)
    try:
        if len(group_arrays) == 2:
            _, p = stats.f_oneway(*group_arrays)
        else:
            _, p = stats.f_oneway(*group_arrays)
    except Exception:
        p = np.nan
    return float(eta2), float(p) if p == p else np.nan


def descriptive_table(df, feats):
    rows = []
    for feat in feats:
        vals = pd.to_numeric(df[feat], errors="coerce")
        for factor in DESCRIPTIVE_FACTORS:
            if factor not in df.columns:
                continue
            eta2, p = eta_squared_oneway(vals.values, df[factor].values)
            rows.append({"feature": feat, "factor": factor,
                         "eta_squared": eta2, "p_value": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (B) Mixed model, participant random intercept (06-style)
# --------------------------------------------------------------------------- #
def build_formula(feat, df):
    terms = []
    for f in MIXED_FIXED:
        if f not in df.columns:
            continue
        levels = df[f].dropna().unique()
        if len(levels) < 2:
            continue
        if f == "Angle":
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
    return f"Q('{feat}') ~ " + " + ".join(terms)


def tidy_term(name):
    import re
    m = re.match(r"C\((\w+).*?\)\[T\.([^\]]+)\]", name)
    if m:
        return f"{m.group(1)}: {m.group(2)}"
    return name


def mixed_table(df, feats):
    rows = []
    if not HAVE_SM:
        return pd.DataFrame(rows)
    import warnings as _w
    for feat in feats:
        sub = df[[feat, "participant"] + [f for f in MIXED_FIXED if f in df.columns]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        keep_fixed = [f for f in MIXED_FIXED
                      if f in sub.columns and sub[f].dropna().nunique() >= 2]
        sub = sub.dropna(subset=[feat, "participant"] + keep_fixed).reset_index(drop=True)
        if sub[feat].nunique() < 2 or sub["participant"].nunique() < 2:
            rows.append({"feature": feat, "term": "(not fit)",
                         "coef": np.nan, "p_value": np.nan, "significant": False,
                         "note": "insufficient variation"})
            continue
        formula = build_formula(feat, sub)
        if formula is None:
            continue
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = smf.mixedlm(formula, sub,
                                  groups=sub["participant"].to_numpy()
                                  ).fit(reml=True, method="lbfgs")
        except Exception as e:
            rows.append({"feature": feat, "term": "(fit failed)",
                         "coef": np.nan, "p_value": np.nan, "significant": False,
                         "note": str(e)[:80]})
            continue
        fe = res.fe_params
        pv = res.pvalues
        for term in fe.index:
            if term == "Intercept":
                continue
            p = float(pv[term]) if term in pv.index else np.nan
            rows.append({"feature": feat, "term": tidy_term(term),
                         "coef": float(fe[term]),
                         "p_value": p,
                         "significant": (p < 0.05) if p == p else False,
                         "note": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary + heatmap
# --------------------------------------------------------------------------- #
def build_summary(eta_df, mixed_df, feats):
    rows = []
    for feat in feats:
        e = eta_df[eta_df["feature"] == feat]
        # strongest factor by eta^2 (excluding participant to see condition effects)
        cond = e[e["factor"] != "participant"].dropna(subset=["eta_squared"])
        part = e[e["factor"] == "participant"]
        peak_cond = cond.loc[cond["eta_squared"].idxmax()] if len(cond) else None
        part_eta = float(part["eta_squared"].iloc[0]) if len(part) and part["eta_squared"].notna().any() else np.nan

        # mixed-model: any significant fixed factor?
        m = mixed_df[(mixed_df["feature"] == feat) & (mixed_df["significant"])] \
            if len(mixed_df) else pd.DataFrame()
        sig_terms = ", ".join(sorted(set(m["term"].tolist()))) if len(m) else ""

        rows.append({
            "feature": feat,
            "strongest_condition_factor": peak_cond["factor"] if peak_cond is not None else "",
            "peak_condition_eta2": round(float(peak_cond["eta_squared"]), 4) if peak_cond is not None else np.nan,
            "participant_eta2": round(part_eta, 4) if part_eta == part_eta else np.nan,
            "varies_with_condition": (peak_cond is not None and peak_cond["eta_squared"] >= 0.06),
            "mostly_participant": (part_eta == part_eta and peak_cond is not None
                                   and part_eta > peak_cond["eta_squared"]),
            "mixed_significant_factors": sig_terms,
        })
    return pd.DataFrame(rows)


def heatmap(eta_df, feats, out_path):
    factors = DESCRIPTIVE_FACTORS
    mat = np.full((len(feats), len(factors)), np.nan)
    for i, feat in enumerate(feats):
        for j, fac in enumerate(factors):
            sub = eta_df[(eta_df["feature"] == feat) & (eta_df["factor"] == fac)]
            if len(sub) and sub["eta_squared"].notna().any():
                mat[i, j] = sub["eta_squared"].iloc[0]
    fig_h = max(4, len(feats) * 0.32)
    fig, ax = plt.subplots(figsize=(1.2 * len(factors) + 3, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors, rotation=30, ha="right")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("Posture features: variance explained (eta^2) by factor")
    for i in range(len(feats)):
        for j in range(len(factors)):
            if mat[i, j] == mat[i, j]:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="eta^2")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--metrics-csv", type=Path, default=None)
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated participant IDs to restrict the "
                         "screening to (e.g. 'P003,P004'). Default: all.")
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args()

    if args.metrics_csv:
        csv_path = args.metrics_csv
        out_dir = csv_path.parent / "screening"
    elif args.landmarks_root:
        csv_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
        out_dir = args.landmarks_root / "metrics" / "screening"
    else:
        sys.exit("Provide --landmarks-root or --metrics-csv")

    if not csv_path.is_file():
        sys.exit(f"Posture features CSV not found: {csv_path}\nRun 07 first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("Posture features CSV is empty.")

    # Restrict to requested participants, if any
    if args.participants:
        keep = {p.strip() for p in args.participants.split(",") if p.strip()}
        before = len(df)
        df = df[df["participant"].isin(keep)].copy()
        if df.empty:
            sys.exit(f"No rows for participants {sorted(keep)} in {csv_path}")
        print(f"Restricted to participants {sorted(keep)}: "
              f"{len(df)}/{before} rows")

    # Parse prototype parameters from trial stem
    params = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = params[c]

    feats = feature_columns(df)
    print(f"Loaded {len(df)} Place events; screening {len(feats)} posture features")
    print(f"  participants: {df['participant'].nunique()}, "
          f"trials: {df['trial'].nunique()}")
    if not HAVE_SM:
        print("  (statsmodels not installed -> mixed-model table skipped; "
              "eta^2 still produced)")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    eta_df = descriptive_table(df, feats)
    eta_df.to_csv(out_dir / "posture_eta_squared.csv", index=False)
    print(f"Wrote posture_eta_squared.csv ({len(eta_df)} rows)")

    mixed_df = mixed_table(df, feats)
    if len(mixed_df):
        mixed_df.to_csv(out_dir / "posture_mixed_effects.csv", index=False)
        print(f"Wrote posture_mixed_effects.csv ({len(mixed_df)} rows)")

    summary = build_summary(eta_df, mixed_df, feats)
    summary = summary.sort_values("peak_condition_eta2", ascending=False)
    summary.to_csv(out_dir / "posture_screening_summary.csv", index=False)
    print(f"Wrote posture_screening_summary.csv ({len(summary)} rows)")

    if not args.no_graphs:
        heatmap(eta_df, feats, out_dir / "heatmap_eta_squared.png")
        print("Wrote heatmap_eta_squared.png")

    # Console digest: features that vary most with condition (not participant)
    print("\nFeatures most driven by experimental CONDITION (top by eta^2, "
          "excluding participant):")
    top = summary.dropna(subset=["peak_condition_eta2"]).head(12)
    for _, r in top.iterrows():
        flag = " [mostly participant]" if r["mostly_participant"] else ""
        sig = f"  mixed-sig: {r['mixed_significant_factors']}" if r["mixed_significant_factors"] else ""
        print(f"  {r['feature']:<28} {r['strongest_condition_factor']:<12} "
              f"eta^2={r['peak_condition_eta2']:.3f}{flag}{sig}")

    flat = summary[(summary["peak_condition_eta2"] < 0.06) |
                   (summary["peak_condition_eta2"].isna())]
    if len(flat):
        print(f"\nFeatures that barely vary with condition (eta^2 < 0.06) — "
              f"candidates to drop:")
        for _, r in flat.iterrows():
            print(f"  {r['feature']}")

    print("\nGuide: eta^2 ~ 0.01 small, 0.06 medium, 0.14 large (Cohen). "
          "'mostly_participant' means the feature differs more between people "
          "than between conditions — relevant to your individual-differences "
          "question, less so to prototype optimisation.")


if __name__ == "__main__":
    main()
