#!/usr/bin/env python3
r"""
validity_check.py - Convergent Validity: Consistency vs MCDA Favourability

Checks how well human consistency (both Continuous/fPCA and Discrete/MCDA) 
correlates with calibrated ergonomic favourability.

CROSS-CORRELATIONS RUN:
  1. Continuous (fPCA) Consistency vs. Directional/Normative MCDA
  2. Discrete (MCDA) Consistency vs. Directional/Normative MCDA
  3. Continuous (fPCA) vs. Discrete (MCDA) Consistency (Pipeline Bridge)

USAGE:
  python validity_check.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HEIGHTS = ["High", "Medium", "Low"]

def canon_config(cfg):
    """Normalise the Angle token to 'A###'."""
    return re.sub(r"_A?(\d+)$", r"_A\1", str(cfg))

def load_csv_dict(rankings_dir: Path, file_template: str):
    """Generic loader for height-stratified leaderboards."""
    out = {}
    for h in HEIGHTS:
        p = rankings_dir / file_template.format(h=h)
        if p.is_file():
            df = pd.read_csv(p)
            if "Prototype_Config" in df.columns:
                df["key"] = df["Prototype_Config"].map(canon_config)
                out[h] = df
    return out

def run_spearman_comparison(df_a_by_height, df_b_by_height, col_a, col_b, name_a, name_b):
    """Runs a Spearman correlation between two columns across all heights."""
    rows = []
    for h in HEIGHTS:
        if h not in df_a_by_height or h not in df_b_by_height:
            continue
        df_a, df_b = df_a_by_height[h].copy(), df_b_by_height[h].copy()
        
        j = pd.merge(df_a[["key", col_a]], df_b[["key", col_b]], on="key", how="inner").dropna()
        
        rho, p = stats.spearmanr(j[col_a], j[col_b]) if len(j) >= 3 else (np.nan, np.nan)
        rows.append({"height": h, "Metric_A": name_a, "Metric_B": name_b, 
                     "n_configs": len(j), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)

def print_validity(df, title):
    print(f"\n{'='*85}\n{title}\n{'='*85}")
    print(f" {'Height':<8} {'Metric A':<22} {'Metric B':<22} {'n':>4} {'rho':>7} {'p':>7}")
    print(f" {'-'*8} {'-'*22} {'-'*22} {'-'*4} {'-'*7} {'-'*7}")
    for _, r in df.iterrows():
        rho = f"{r['spearman_rho']:>+7.3f}" if pd.notna(r['spearman_rho']) else f"{'n/a':>7}"
        pv = f"{r['p_value']:>7.3f}" if pd.notna(r['p_value']) else f"{'n/a':>7}"
        print(f" {r['height']:<8} {r['Metric_A']:<22} {r['Metric_B']:<22} {int(r['n_configs']):>4} {rho} {pv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-root", type=Path, required=True)
    args = ap.parse_args()

    rankings_dir = args.landmarks_root / "metrics" / "prototype_rankings"
    if not rankings_dir.exists():
        sys.exit(f"ERROR: Directory not found at {rankings_dir}.")

    print(f"\nLoading leaderboards from {rankings_dir.name}...")
    
    # Load all three sets of leaderboards
    mcda_dir = load_csv_dict(rankings_dir, "directional_rankings_stratified_{h}.csv")
    mcda_norm = load_csv_dict(rankings_dir, "normative_rankings_stratified_{h}.csv")
    cons_cont = load_csv_dict(rankings_dir, "consistency_leaderboard_{h}.csv")
    cons_disc = load_csv_dict(rankings_dir, "discrete_consistency_{h}.csv")

    if not cons_disc:
        print("  [!] Discrete Consistency not found. Run discrete_consistency.py first.")
    
    # 1. Discrete Consistency vs MCDA (Favourability & Normative)
    if cons_disc and mcda_dir:
        res = run_spearman_comparison(cons_disc, mcda_dir, "Consistency_Score_0_100", "Grand_Score", "Discrete Consistency", "MCDA Directional")
        print_validity(res, "VALIDITY: Discrete Consistency vs. Directional MCDA Favourability")
        res.to_csv(rankings_dir / "validity_discrete_vs_directional.csv", index=False)

    if cons_disc and mcda_norm:
        res = run_spearman_comparison(cons_disc, mcda_norm, "Consistency_Score_0_100", "Grand_Score", "Discrete Consistency", "MCDA Normative")
        print_validity(res, "VALIDITY: Discrete Consistency vs. Normative MCDA (Closeness to Mean)")
        res.to_csv(rankings_dir / "validity_discrete_vs_normative.csv", index=False)

    # 2. Continuous Consistency vs MCDA (Original comparisons)
    if cons_cont and mcda_dir:
        res = run_spearman_comparison(cons_cont, mcda_dir, "Posture_Consistency_Score", "Grand_Score", "Continuous Posture", "MCDA Directional")
        print_validity(res, "VALIDITY: Continuous Consistency vs. Directional MCDA Favourability")
        res.to_csv(rankings_dir / "validity_continuous_vs_directional.csv", index=False)

    # 3. THE BRIDGE: Discrete vs Continuous Consistency
    if cons_disc and cons_cont:
        res = run_spearman_comparison(cons_disc, cons_cont, "Consistency_Score_0_100", "Posture_Consistency_Score", "Discrete Consistency", "Continuous Consistency")
        print_validity(res, "PIPELINE BRIDGE: Discrete vs. Continuous Consistency")
        print("\n *(rho > 0 means the discrete point-metrics and continuous fPCA curves agree on which tools force humans to move uniformly.)*")
        res.to_csv(rankings_dir / "validity_discrete_vs_continuous.csv", index=False)

if __name__ == "__main__":
    main()