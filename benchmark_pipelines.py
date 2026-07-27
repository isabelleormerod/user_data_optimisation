#!/usr/bin/env python3
r"""
benchmark_pipelines.py

A quantitative evaluation engine to benchmark the Metric-Driven MCDA pipeline
against the Data-Driven Hierarchical mfPCA pipeline across 4 key operational pillars:
  1. Reliability (Robustness to Simulated Sensor Noise & Jitter)
  2. Data Intensity (Minimum Viable Dataset via Participant Subsampling)
  3. Kinematic Sensitivity (Intraclass Correlation Coefficient / Participant Bias)
  4. Actionability & Traceability (Dimensionality & Entropy)

STATISTICAL ENGINE:
  To maintain strict methodological consistency with your evaluation and ranking 
  scripts, all benchmarking loops apply participant-level aggregation before computing 
  Kruskal-Wallis Epsilon-Squared (E_R^2) weights or Spearman's rank correlations (\rho). 
  This prevents repeated Place events from introducing pseudoreplication into the stress tests.

Usage:
  python benchmark_pipelines.py --metrics-csv A:\Automated_chain_BETA\Participant_Landmarks\metrics\posture_features_combined.csv --fpca-csv A:\Automated_chain_BETA\Participant_Landmarks\metrics\fpca_results\all_fpca_scores_stratified.csv

  # Run from repository root (assumes default output paths):
  python benchmark_pipelines.py --metrics-csv Participant_Landmarks\metrics\posture_features_combined.csv --fpca-csv Participant_Landmarks\metrics\fpca_results\all_fpca_scores_stratified.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# =========================================================================== #
# DYNAMIC METRIC DISCOVERY ENGINE
# =========================================================================== #

def get_active_metrics(df_metrics: pd.DataFrame) -> tuple:
    """
    Dynamically imports metric definitions and directions from rank_prototypes.py,
    filtering strictly for columns that exist in the currently loaded CSV.
    Falls back to intelligent pattern matching if the module cannot be imported.
    """
    active_cols = []
    metric_dirs = {}

    try:
        from rank_prototypes import METRIC_REGISTRY
        for col, meta in METRIC_REGISTRY.items():
            if col in df_metrics.columns and pd.to_numeric(df_metrics[col], errors="coerce").dropna().nunique() > 1:
                active_cols.append(col)
                metric_dirs[col] = meta["dir"]
        print(f" -> Successfully loaded {len(active_cols)} active metrics from canonical rank_prototypes.METRIC_REGISTRY.")
    except ImportError:
        print(" -> [WARN] Could not import 'rank_prototypes.py'. Discovering metrics dynamically from CSV headers...")
        ignore_cols = {"participant", "trial", "place_index", "height", "start_t_s", "stop_t_s", "duration_s", "Prototype_Config", "Length", "Size", "Weight", "Angle"}
        for col in df_metrics.columns:
            if col in ignore_cols or col.startswith("Global_Synergy_"):
                continue
            if pd.to_numeric(df_metrics[col], errors="coerce").dropna().nunique() > 1:
                active_cols.append(col)
                # Assign maximization to comfort and SPARC smoothness; minimization to all errors/angles/strain
                if any(w in col.lower() for w in ("comfort", "sparc", "smoothness")):
                    metric_dirs[col] = "max"
                else:
                    metric_dirs[col] = "min"
        print(f" -> Dynamically discovered {len(active_cols)} numeric metric columns.")

    if not active_cols:
        sys.exit("Error: No valid numeric metric columns discovered in the provided metrics CSV!")
        
    return active_cols, metric_dirs


def get_fpca_cols(df: pd.DataFrame) -> list:
    """Extracts all Global Synergy feature columns from the fPCA dataset."""
    cols = [c for c in df.columns if c.startswith("Global_Synergy_")]
    if not cols:
        sys.exit("Error: No columns starting with 'Global_Synergy_' found in the fPCA CSV!")
    return cols


# =========================================================================== #
# RAPID RANKING ENGINES (FOR MONTE CARLO & STRESS LOOPING)
# =========================================================================== #

def quick_rank_metrics(df: pd.DataFrame, metric_cols: list, metric_dirs: dict) -> pd.DataFrame:
    """Calculates rapid MCDA desirability leaderboard for the Metric-Driven approach."""
    df_calc = df.copy()
    score_cols = []
    
    for col in metric_cols:
        if col not in df_calc.columns: continue
        vals = pd.to_numeric(df_calc[col], errors="coerce").dropna()
        if len(vals) < 2: continue
        
        v_min, v_max = vals.min(), vals.max()
        if abs(v_max - v_min) < 1e-9: continue
        
        sc_col = f"{col}_sc"
        if metric_dirs.get(col, "min") == "min":
            df_calc[sc_col] = 100.0 * (v_max - df_calc[col]) / (v_max - v_min)
        else:
            df_calc[sc_col] = 100.0 * (df_calc[col] - v_min) / (v_max - v_min)
        score_cols.append(sc_col)
        
    return _apply_weights_and_rank(df_calc, score_cols)


def quick_rank_fpca(df: pd.DataFrame, fpca_cols: list) -> pd.DataFrame:
    """Calculates rapid MCDA leaderboard for fPCA by minimizing absolute anomaly deviation."""
    df_calc = df.copy()
    score_cols = []
    
    for col in fpca_cols:
        vals = df_calc[col].abs()
        v_min, v_max = vals.min(), vals.max()
        if abs(v_max - v_min) < 1e-9: continue
        
        sc_col = f"{col}_sc"
        df_calc[sc_col] = 100.0 * (v_max - vals) / (v_max - v_min)
        score_cols.append(sc_col)
        
    return _apply_weights_and_rank(df_calc, score_cols)


def _apply_weights_and_rank(df_calc: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """Applies participant-aggregated E_R^2 sensitivity weighting to prevent pseudoreplication."""
    df_agg = df_calc.groupby(["Prototype_Config", "participant"], dropna=False, as_index=False).mean(numeric_only=True)
    n = len(df_agg)
    
    weights = {}
    for col in score_cols:
        groups = [grp[col].dropna().values for _, grp in df_agg.groupby("Prototype_Config", dropna=False) if len(grp[col].dropna()) > 0]
        if len(groups) >= 2 and n > 1:
            try:
                h_stat, _ = stats.kruskal(*groups)
                weights[col] = max(0.01, float(h_stat / (n - 1)))
            except ValueError:
                weights[col] = 0.01
        else:
            weights[col] = 0.01
            
    tot_w = sum(weights.values()) or 1.0
    df_calc["Grand_Score"] = 0.0
    for c in score_cols:
        df_calc["Grand_Score"] += df_calc[c].fillna(0.0) * (weights[c] / tot_w)
    
    ranks = df_calc.groupby("Prototype_Config", dropna=False)["Grand_Score"].mean().sort_values(ascending=False).reset_index()
    ranks["Rank"] = ranks.index + 1
    return ranks


# =========================================================================== #
# EVALUATION PILLAR MODULES
# =========================================================================== #

def test_robustness_to_noise(df_metrics: pd.DataFrame, df_fpca: pd.DataFrame, metric_cols: list, metric_dirs: dict, fpca_cols: list, noise_level: float = 0.15, iterations: int = 10) -> tuple:
    """Simulates 15% random sensor noise/jitter and evaluates leaderboard rank stability."""
    print(" -> Pillar 1: Simulating sensor jitter and tracking unreliability (15% Gaussian noise)...")
    
    base_rank_m = quick_rank_metrics(df_metrics, metric_cols, metric_dirs)
    base_rank_f = quick_rank_fpca(df_fpca, fpca_cols)
    
    m_corrs, f_corrs = [], []
    
    for _ in range(iterations):
        # Inject noise into Metrics
        noisy_m = df_metrics.copy()
        for c in metric_cols:
            if c in noisy_m.columns:
                std_val = noisy_m[c].std()
                if pd.notna(std_val) and std_val > 0:
                    noise = np.random.normal(0, std_val * noise_level, len(noisy_m))
                    noisy_m[c] = noisy_m[c] + noise
                
        # Inject noise into fPCA scores
        noisy_f = df_fpca.copy()
        for c in fpca_cols:
            std_val = noisy_f[c].std()
            if pd.notna(std_val) and std_val > 0:
                noise = np.random.normal(0, std_val * noise_level, len(noisy_f))
                noisy_f[c] = noisy_f[c] + noise
            
        rank_m = quick_rank_metrics(noisy_m, metric_cols, metric_dirs)
        rank_f = quick_rank_fpca(noisy_f, fpca_cols)
        
        m_merge = pd.merge(base_rank_m, rank_m, on="Prototype_Config", suffixes=("_base", "_noise"))
        rho_m, _ = stats.spearmanr(m_merge["Rank_base"], m_merge["Rank_noise"])
        m_corrs.append(rho_m if pd.notna(rho_m) else 0.0)
        
        f_merge = pd.merge(base_rank_f, rank_f, on="Prototype_Config", suffixes=("_base", "_noise"))
        rho_f, _ = stats.spearmanr(f_merge["Rank_base"], f_merge["Rank_noise"])
        f_corrs.append(rho_f if pd.notna(rho_f) else 0.0)
        
    return float(np.mean(m_corrs)), float(np.mean(f_corrs))


def test_data_intensity(df_metrics: pd.DataFrame, df_fpca: pd.DataFrame, metric_cols: list, metric_dirs: dict, fpca_cols: list, fractions: list = [0.4, 0.6, 0.8]) -> tuple:
    """Tests Minimum Viable Dataset (MVD) stability via random participant subsampling."""
    print(" -> Pillar 2: Evaluating Data Intensity via participant subsampling (MVD stress testing)...")
    
    base_rank_m = quick_rank_metrics(df_metrics, metric_cols, metric_dirs)
    base_rank_f = quick_rank_fpca(df_fpca, fpca_cols)
    
    participants = df_metrics["participant"].dropna().unique()
    m_results, f_results = {}, {}
    
    for frac in fractions:
        n_keep = max(2, int(len(participants) * frac))
        m_corrs, f_corrs = [], []
        
        for _ in range(10): # 10 random shuffles per fraction
            keep_p = np.random.choice(participants, n_keep, replace=False)
            
            sub_m = df_metrics[df_metrics["participant"].isin(keep_p)]
            sub_f = df_fpca[df_fpca["participant"].isin(keep_p)]
            
            rank_m = quick_rank_metrics(sub_m, metric_cols, metric_dirs)
            rank_f = quick_rank_fpca(sub_f, fpca_cols)
            
            m_merge = pd.merge(base_rank_m, rank_m, on="Prototype_Config", suffixes=("_base", "_sub"))
            rho_m, _ = stats.spearmanr(m_merge["Rank_base"], m_merge["Rank_sub"])
            m_corrs.append(rho_m if pd.notna(rho_m) else 0.0)
            
            f_merge = pd.merge(base_rank_f, rank_f, on="Prototype_Config", suffixes=("_base", "_sub"))
            rho_f, _ = stats.spearmanr(f_merge["Rank_base"], f_merge["Rank_sub"])
            f_corrs.append(rho_f if pd.notna(rho_f) else 0.0)
            
        m_results[f"{int(frac*100)}%"] = float(np.mean(m_corrs))
        f_results[f"{int(frac*100)}%"] = float(np.mean(f_corrs))
        
    return m_results, f_results


def test_kinematic_sensitivity(df_metrics: pd.DataFrame, df_fpca: pd.DataFrame, metric_cols: list, fpca_cols: list) -> tuple:
    """Calculates Intraclass Correlation Coefficient (ICC) to isolate participant bias vs hardware effect."""
    print(" -> Pillar 3: Extracting Mixed-Effects Variance Components (Participant ICC)...")
    
    def get_avg_icc(df, cols):
        iccs = []
        for c in cols:
            if c not in df.columns: continue
            d = df.dropna(subset=[c, "participant"]).copy()
            if len(d["participant"].unique()) < 2 or len(d) < 10: continue
            try:
                md = smf.mixedlm(f"{c} ~ 1", d, groups=d["participant"]).fit(reml=False, method="lbfgs")
                var_re = float(md.cov_re.iloc[0, 0])
                var_resid = float(md.scale)
                if (var_re + var_resid) > 0:
                    iccs.append(var_re / (var_re + var_resid))
            except Exception:
                pass
        return float(np.mean(iccs)) if iccs else 0.0

    icc_m = get_avg_icc(df_metrics, metric_cols)
    icc_f = get_avg_icc(df_fpca, fpca_cols)
    return icc_m, icc_f


def ensure_prototype_config(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the unified Prototype_Config column exists for aggregation."""
    if "Prototype_Config" not in df.columns:
        if all(c in df.columns for c in ["Length", "Size", "Weight", "Angle"]):
            df["Prototype_Config"] = df[["Length", "Size", "Weight", "Angle"]].astype(str).agg("_".join, axis=1)
        elif "trial" in df.columns:
            df["Prototype_Config"] = df["trial"].astype(str)
        else:
            sys.exit("Error: Could not construct 'Prototype_Config' column from available metadata!")
    return df


# =========================================================================== #
# MAIN EXECUTION & ASCII REPORTING
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-csv", type=Path, required=True, help="Path to posture_features_combined.csv")
    ap.add_argument("--fpca-csv", type=Path, required=True, help="Path to all_fpca_scores_stratified.csv")
    args = ap.parse_args()

    if not args.metrics_csv.is_file():
        sys.exit(f"Error: Metrics CSV not found at {args.metrics_csv}")
    if not args.fpca_csv.is_file():
        sys.exit(f"Error: fPCA CSV not found at {args.fpca_csv}")

    df_metrics = ensure_prototype_config(pd.read_csv(args.metrics_csv))
    df_fpca = ensure_prototype_config(pd.read_csv(args.fpca_csv))

    print(f"\n{'='*75}\nRUNNING 4-PILLAR QUANTITATIVE PIPELINE BENCHMARK\n{'='*75}")

    # Discover metrics dynamically
    metric_cols, metric_dirs = get_active_metrics(df_metrics)
    fpca_cols = get_fpca_cols(df_fpca)

    # 1. Noise Robustness
    rho_m_noise, rho_f_noise = test_robustness_to_noise(df_metrics, df_fpca, metric_cols, metric_dirs, fpca_cols, noise_level=0.15)
    
    # 2. Data Intensity (Subsampling)
    mvd_m, mvd_f = test_data_intensity(df_metrics, df_fpca, metric_cols, metric_dirs, fpca_cols)

    # 3. Kinematic Sensitivity (ICC)
    icc_m, icc_f = test_kinematic_sensitivity(df_metrics, df_fpca, metric_cols, fpca_cols)
    
    # 4. Actionability (Dimensionality)
    n_metrics = len(metric_cols)
    n_fpca = len(fpca_cols)

    print(f"\n{'='*75}\nQUANTITATIVE PIPELINE COMPARISON REPORT\n{'='*75}")
    print(f" {'Evaluation Pillar':<32} | {'Metric-Driven':>18} | {'Data-Driven (fPCA)':>18}")
    print(f" {'-'*32}-+-{'-'*18}-+-{'-'*18}")
    
    # Pillar 1
    print(f" {'1. Reliability (Noise Robustness)':<32} | {'':>18} | {'':>18}")
    print(f" {'   Spearman Rho (w/ 15% Noise)':<32} | {rho_m_noise:>18.3f} | {rho_f_noise:>18.3f}")
    
    # Pillar 2
    print(f" {'2. Data Intensity (MVD Stability)':<32} | {'':>18} | {'':>18}")
    for frac in mvd_m.keys():
        print(f" {'   Spearman Rho at '+frac+' Dataset':<32} | {mvd_m[frac]:>18.3f} | {mvd_f[frac]:>18.3f}")
        
    # Pillar 3
    print(f" {'3. Sensitivity (Participant Bias)':<32} | {'':>18} | {'':>18}")
    print(f" {'   Participant ICC (Lower=Better)':<32} | {icc_m:>18.3f} | {icc_f:>18.3f}")
    
    # Pillar 4
    print(f" {'4. Actionability & Dimensionality':<32} | {'':>18} | {'':>18}")
    print(f" {'   Active Variables Evaluated':<32} | {n_metrics:>18} | {n_fpca:>18}")
    print(f" {'   Direct Kinematic Traceability':<32} | {'High (1-to-1)':>18} | {'Low (Synergies)':>18}")

    print("\n* METHODOLOGICAL INTERPRETATION:")
    print(" - Reliability: Higher Spearman's Rho indicates the ranking leaderboard is resilient to sensor tracking jitter.")
    print(" - Data Intensity: High correlation at reduced dataset percentages indicates fewer participants are needed to reach a stable decision.")
    print(" - Sensitivity (ICC): Lower Intraclass Correlation indicates the metric variance is driven by prototype hardware changes rather than personal participant habits.")
    print(" - Traceability: Metric-driven variables map directly to single anatomical joints, whereas fPCA synergies require interpreting multi-joint eigenvector loadings.")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()