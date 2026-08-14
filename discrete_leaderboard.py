
#!/usr/bin/env python3
r"""
discrete_consistency.py - Discrete Metric Consistency (Between-Participant Agreement)

This script completes the 2x2 comparison grid for the thesis. Where 
06_rank_prototypes.py calculates the "Normative" score (closeness to the mean), 
this script calculates "Consistency" (between-participant agreement) for your 
discrete point-metrics (REBA, SPARC, Macro-Angles).

THE MATH:
  1. Standardizes all discrete metrics within a height stratum so angles (deg) 
     and distances (m/mm) can be mathematically combined.
  2. Averages each metric per participant (fixing pseudoreplication).
  3. Calculates the standard deviation ACROSS participants for each prototype.
  4. Lower SD = Tighter human convergence (more consistent).

USAGE:
  python discrete_leaderboard.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEIGHTS = ["High", "Medium", "Low"]

# We use the same parameter parsing as your MCDA script
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]

def parse_params(trial_val) -> dict:
    out = {k: "Other" for k in PARAM_FACTORS}
    if pd.isna(trial_val): return out
    clean_str = str(trial_val).strip()
    tokens = [t.strip() for t in clean_str.split("_") if t.strip()]
    joined_low = "_".join(tokens).lower()

    if "not_weighted" in joined_low or "notweighted" in joined_low: out["Weight"] = "Not_weighted"
    elif "front_weighted" in joined_low or "frontweighted" in joined_low: out["Weight"] = "Front_weighted"
    
    for tok in tokens:
        t_low = tok.lower(); t_cap = tok.capitalize()
        if t_cap in ("Long", "Short"): out["Length"] = t_cap
        elif t_cap in ("Large", "Small"): out["Size"] = t_cap
        elif t_low.startswith("a") and t_low[1:].isdigit(): out["Angle"] = f"A{t_low[1:]}"
        elif tok.isdigit() and int(tok) in (0, 45, 90, 135, 180): out["Angle"] = f"A{tok}"
    return out

def add_prototype_label(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    parsed = df_clean["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df_clean[c] = parsed[c]
    df_clean["Prototype_Config"] = df_clean[["Length", "Size", "Weight", "Angle"]].agg("_".join, axis=1)
    return df_clean

def calculate_discrete_consistency(df_stratum, metrics, min_participants=2):
    """Calculates Between-Participant Consistency for a single height stratum."""
    z_df = df_stratum.copy()
    for m in metrics:
        mu = df_stratum[m].mean()
        sigma = df_stratum[m].std(ddof=1)
        z_df[f"Z_{m}"] = (df_stratum[m] - mu) / sigma if sigma > 1e-9 else 0.0

    z_metrics = [f"Z_{m}" for m in metrics]
    
    # Aggregate to Participant Level
    part_df = z_df.groupby(["Prototype_Config", "participant"])[z_metrics].mean().reset_index()

    configs = part_df["Prototype_Config"].unique()
    rows = []

    for cfg in configs:
        cfg_data = part_df[part_df["Prototype_Config"] == cfg]
        n_part = len(cfg_data)
        
        consistency = np.nan
        if n_part >= min_participants:
            between_sd = cfg_data[z_metrics].std(ddof=1)
            consistency = np.sqrt(np.nanmean(between_sd ** 2))
            
        rows.append({
            "Prototype_Config": cfg,
            "N_Participants": n_part,
            "Discrete_Consistency_SD": consistency
        })

    lb = pd.DataFrame(rows)
    lb["Consistency_Rank"] = lb["Discrete_Consistency_SD"].rank(method="min", ascending=True)
    lb["Consistency_Score_0_100"] = 100 - (lb["Discrete_Consistency_SD"] / lb["Discrete_Consistency_SD"].max() * 100)
    
    return lb.sort_values("Discrete_Consistency_SD", ascending=True).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-root", type=Path, required=True)
    args = ap.parse_args()

    pen_path = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
    posture_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
    out_dir = args.landmarks_root / "metrics" / "prototype_rankings"

    if not pen_path.exists() and not posture_path.exists():
        sys.exit("Error: No metric CSVs found.")

    df_pen = pd.read_csv(pen_path) if pen_path.exists() else pd.DataFrame()
    df_posture = pd.read_csv(posture_path) if posture_path.exists() else pd.DataFrame()

    common = [c for c in ["participant", "trial", "place_index", "height"] if c in df_pen.columns and c in df_posture.columns]
    df = pd.merge(df_pen, df_posture, on=common, how="inner") if not df_pen.empty and not df_posture.empty else (df_pen if not df_pen.empty else df_posture)
    
    df = add_prototype_label(df)
    metrics = [c for c in df.select_dtypes(include=[np.number]).columns if c not in common + PARAM_FACTORS + ["t_s", "start_t_s", "stop_t_s"]]
    
    out_dir.mkdir(parents=True, exist_ok=True)

    for height in HEIGHTS:
        df_stratum = df[df["height"] == height].copy()
        if len(df_stratum) < 4: continue
        
        lb = calculate_discrete_consistency(df_stratum, metrics)
        out_file = out_dir / f"discrete_consistency_{height}.csv"
        lb.to_csv(out_file, index=False)
        
        print(f"\nDISCRETE CONSISTENCY -- {height.upper()}")
        print(f" {'Prototype Configuration':<32} {'Consist. SD':>12} {'Rank':>6} {'Np':>4}")
        for _, r in lb.head(10).iterrows():
            cons = f"{r['Discrete_Consistency_SD']:>12.3f}" if pd.notna(r['Discrete_Consistency_SD']) else f"{'n/a':>12}"
            print(f" {r['Prototype_Config']:<32} {cons} {r['Consistency_Rank']:>6.0f} {int(r['N_Participants']):>4}")

if __name__ == "__main__":
    main()