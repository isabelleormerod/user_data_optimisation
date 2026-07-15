#!/usr/bin/env python3
"""
rank_prototypes.py

100% Data-Driven Multi-Criteria Decision Analysis (MCDA) for Prototype Selection.
Eliminates human preference weights entirely. Uses Kruskal-Wallis Epsilon-Squared
(E_R^2) effect sizes to automatically generate weights from the design space at both
the individual metric level AND the aggregate domain level.

Usage:
  python rank_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python rank_prototypes.py --pen-csv path/to/place_metrics.csv --posture-csv path/to/posture.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================================== #
# SECTION 1: METRIC REGISTRY & OPTIMALITY DIRECTION
# =========================================================================== #

METRIC_REGISTRY = {
    # --- DOMAIN 1: Task Performance (min = lower error/duration is better) ---
    "duration_s":                 {"domain": "Performance",     "dir": "min", "label": "Duration (s)"},
    "perp_mean_deg":              {"domain": "Performance",     "dir": "min", "label": "Perp Error (deg)"},
    "leftright_mean_deg":         {"domain": "Performance",     "dir": "min", "label": "L/R Tilt (deg)"},
    "updown_mean_deg":            {"domain": "Performance",     "dir": "min", "label": "U/D Tilt (deg)"},
    "pos_jitter_mm":              {"domain": "Performance",     "dir": "min", "label": "Pos Jitter (mm)"},
    "ang_jitter_deg":             {"domain": "Performance",     "dir": "min", "label": "Ang Jitter (deg)"},

    # --- DOMAIN 2: Postural Risk / REBA (min = lower strain is better) ---
    "reba_score_a":               {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score A"},
    "reba_score_b_right":         {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (R)"},
    "reba_score_b_left":          {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (L)"},
    "reba_grand_right":           {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (R)"},
    "reba_grand_left":            {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (L)"},
    "trunk_flex_mean":            {"domain": "Postural_Risk",   "dir": "min", "label": "Trunk Flex (deg)"},
    "neck_flex_mean":             {"domain": "Postural_Risk",   "dir": "min", "label": "Neck Flex (deg)"},
    "wrist_elevation_m_mean":     {"domain": "Postural_Risk",   "dir": "min", "label": "Wrist Elev (m)"},

    # --- DOMAIN 3: Grip Ergonomics (min dev from 50mm, max comfort) ---
    "right_grip_span_dev_mm":     {"domain": "Grip_Ergonomics", "dir": "min", "label": "R Grip Dev (mm)"},
    "left_grip_span_dev_mm":      {"domain": "Grip_Ergonomics", "dir": "min", "label": "L Grip Dev (mm)"},
    "right_grip_comfort_score":   {"domain": "Grip_Ergonomics", "dir": "max", "label": "R Grip Comfort"},
    "left_grip_comfort_score":    {"domain": "Grip_Ergonomics", "dir": "max", "label": "L Grip Comfort"},

    # --- DOMAIN 4: Motor Control / SPARC (max = closer to 0 is smoother) ---
    "right_sparc_linear":         {"domain": "Motor_Control",   "dir": "max", "label": "R Linear SPARC"},
    "left_sparc_linear":          {"domain": "Motor_Control",   "dir": "max", "label": "L Linear SPARC"},
    "right_sparc_angular":        {"domain": "Motor_Control",   "dir": "max", "label": "R Angular SPARC"},
    "left_sparc_angular":         {"domain": "Motor_Control",   "dir": "max", "label": "L Angular SPARC"},
}

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]


# =========================================================================== #
# SECTION 2: ROBUST PARSING & DESIRABILITY SCORING ENGINE
# =========================================================================== #

def parse_params(trial_val) -> dict:
    """Robustly parses prototype parameters from trial strings without formatting sensitivity."""
    out = {k: "Other" for k in PARAM_FACTORS}
    if trial_val is None or pd.isna(trial_val): return out
    clean_str = str(trial_val).strip()
    tokens = [t.strip() for t in clean_str.split("_") if t.strip()]
    joined_low = "_".join(tokens).lower()

    if "not_weighted" in joined_low or "notweighted" in joined_low: out["Weight"] = "Not_weighted"
    elif "front_weighted" in joined_low or "frontweighted" in joined_low: out["Weight"] = "Front_weighted"
    elif "weighted" in tokens: out["Weight"] = "weighted"

    for tok in tokens:
        t_low = tok.lower(); t_cap = tok.capitalize()
        if t_cap in ("Long", "Short"): out["Length"] = t_cap
        elif t_cap in ("Large", "Small"): out["Size"] = t_cap
        elif t_low.startswith("a") and t_low[1:].isdigit(): out["Angle"] = f"A{t_low[1:]}"
        elif tok.isdigit() and int(tok) in (0, 45, 90, 135, 180, 225, 270, 315): out["Angle"] = f"A{tok}"
    return out


def add_prototype_label(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a unified 'Prototype_Config' column, using existing columns or parsing from 'trial'."""
    df_clean = df.copy()
    parsed = df_clean["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        if c not in df_clean.columns or df_clean[c].isna().all() or (df_clean[c] == "Unknown").any():
            df_clean[c] = parsed[c]
        else:
            df_clean[c] = df_clean[c].fillna(parsed[c])
        df_clean[c] = df_clean[c].fillna("Other").astype(str).str.strip()
    df_clean["Prototype_Config"] = df_clean[["Length", "Size", "Weight", "Angle"]].agg("_".join, axis=1)
    return df_clean


def calculate_desirability_scores(df: pd.DataFrame) -> tuple:
    """Converts raw metrics into a standardized 0-100 Desirability Score."""
    df_scored = df.copy()
    active_domains = defaultdict(list)
    for col, meta in METRIC_REGISTRY.items():
        if col not in df_scored.columns: continue
        vals = pd.to_numeric(df_scored[col], errors="coerce")
        if vals.dropna().empty or vals.nunique() <= 1: continue
        if any(w in col for w in ("deg", "dev", "flex", "tilt")): vals = vals.abs()
        v_min, v_max = vals.min(), vals.max()
        if abs(v_max - v_min) < 1e-9: continue
        
        score_col = f"{col}_dscore"
        df_scored[score_col] = 100.0 * (v_max - vals)/(v_max - v_min) if meta["dir"] == "min" else 100.0 * (vals - v_min)/(v_max - v_min)
        active_domains[meta["domain"]].append(score_col)
    return df_scored, active_domains


def compute_epsilon_weights(df: pd.DataFrame, active_domains: dict) -> dict:
    """Computes objective Epsilon-Squared (E_R^2 = H / (n-1)) sensitivity weights directly on desirability scores."""
    n = len(df)
    weights = {}
    for domain, cols in active_domains.items():
        for col in cols:
            groups = [grp[col].dropna().values for _, grp in df.groupby("Prototype_Config") if len(grp[col].dropna()) > 0]
            if len(groups) >= 2 and n > 1:
                try:
                    h_stat, _ = stats.kruskal(*groups)
                    e_sq = max(0.01, float(h_stat / (n - 1)))
                except ValueError: e_sq = 0.01
            else: e_sq = 0.01
            weights[col] = e_sq
    return weights


def score_and_rank_pure_data(df: pd.DataFrame, active_domains: dict, metric_weights: dict, custom_domain_override: dict = None) -> pd.DataFrame:
    """
    100% Data-Driven Scoring Engine.
    Intra-Domain weights = exact E_R^2 of each metric.
    Inter-Domain weights = exact mean E_R^2 of that domain (unless explicitly overridden for stress-testing).
    """
    df_calc = df.copy()
    domain_names = list(active_domains.keys())
    
    # 1. Intra-domain weighted average (governed strictly by E_R^2 data sensitivity)
    for domain, cols in active_domains.items():
        dom_w_sum = sum(metric_weights.get(c, 0.01) for c in cols)
        df_calc[f"Domain_{domain}"] = 0.0
        for c in cols:
            w = metric_weights.get(c, 0.01) / dom_w_sum
            df_calc[f"Domain_{domain}"] += df_calc[c].fillna(0.0) * w

    # 2. Inter-domain weighted average (governed strictly by Domain Average E_R^2)
    domain_data_weights = {}
    for d, cols in active_domains.items():
        avg_sensitivity = np.mean([metric_weights.get(c, 0.01) for c in cols])
        if custom_domain_override and d in custom_domain_override:
            domain_data_weights[d] = custom_domain_override[d]
        else:
            domain_data_weights[d] = avg_sensitivity
        
    tot_dom_w = sum(domain_data_weights.values())
    
    df_calc["Grand_Score"] = 0.0
    for d in domain_names:
        w = domain_data_weights[d] / tot_dom_w
        df_calc["Grand_Score"] += df_calc[f"Domain_{d}"] * w

    agg_dict = {"Grand_Score": ["mean", "std", "count"]}
    for d in domain_names: agg_dict[f"Domain_{d}"] = "mean"

    rankings = df_calc.groupby("Prototype_Config", dropna=False).agg(agg_dict)
    rankings.columns = ["Grand_Score", "Score_SD", "N_Events"] + domain_names
    rankings = rankings.reset_index().sort_values(by="Grand_Score", ascending=False).reset_index(drop=True)
    rankings["Rank"] = rankings.index + 1
    
    cols = ["Rank", "Prototype_Config", "Grand_Score", "Score_SD", "N_Events"] + domain_names
    return rankings[cols]


# =========================================================================== #
# SECTION 3: DECISION SENSITIVITY ANALYSIS ENGINE
# =========================================================================== #

def run_sensitivity_analysis(df: pd.DataFrame, active_domains: dict, metric_weights: dict, out_dir: Path, top_n: int = 8):
    """Executes Pure Data Scenario Stress-Testing, LODO Stability, and Monte Carlo Weight Simulation."""
    print(f"\n{'='*85}\nDECISION SENSITIVITY ANALYSIS (Robustness & Confidence Testing)\n{'='*85}")
    
    domain_names = list(active_domains.keys())
    base_ranks = score_and_rank_pure_data(df, active_domains, metric_weights)
    top_configs = base_ranks["Prototype_Config"].head(top_n).tolist()

    # ----------------------------------------------------------------------- #
    # 1. OPERATIONAL SCENARIO STRESS-TESTING
    # ----------------------------------------------------------------------- #
    # Calculate pure data domain weights for reference
    pure_dom_w = {d: np.mean([metric_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}
    
    scenarios = {
        "Pure Data (E_sq)": None, # Uses 100% data-generated weights
        "Speed Only":     {d: (1.0 if d == "Performance" else 0.0) for d in domain_names},
        "Ergo Only":      {d: (1.0 if d in ("Postural_Risk", "Grip_Ergonomics") else 0.0) for d in domain_names},
        "Smoothness Only":{d: (1.0 if d == "Motor_Control" else 0.0) for d in domain_names},
    }
    
    scenario_matrix = {cfg: [] for cfg in top_configs}
    for sc_name, sc_override in scenarios.items():
        sc_ranks = score_and_rank_pure_data(df, active_domains, metric_weights, custom_domain_override=sc_override)
        for cfg in top_configs:
            rk = sc_ranks[sc_ranks["Prototype_Config"] == cfg]["Rank"].values
            scenario_matrix[cfg].append(int(rk[0]) if len(rk) > 0 else np.nan)

    print(f"\n--- TABLE 5A: Operational Scenario Stress-Testing (Top {top_n} Ranks) ---")
    sc_headers = "  ".join([f"{k[:16]:>16}" for k in scenarios.keys()])
    print(f" {'Prototype Configuration':<32} | {sc_headers}")
    print(f" {'-'*32}-+-{'-'*len(sc_headers)}")
    
    sc_records = []
    for cfg in top_configs:
        ranks_str = "  ".join([f"{r:>16}" for r in scenario_matrix[cfg]])
        print(f" {cfg:<32} | {ranks_str}")
        rec = {"Prototype_Config": cfg}; rec.update(zip(scenarios.keys(), scenario_matrix[cfg]))
        sc_records.append(rec)
    pd.DataFrame(sc_records).to_csv(out_dir / "sensitivity_scenarios.csv", index=False)

    # ----------------------------------------------------------------------- #
    # 2. LEAVE-ONE-DOMAIN-OUT (LODO) STABILITY
    # ----------------------------------------------------------------------- #
    lodo_matrix = {cfg: [] for cfg in top_configs}
    for dropped_dom in domain_names:
        lodo_override = {d: (0.0 if d == dropped_dom else pure_dom_w[d]) for d in domain_names}
        lodo_ranks = score_and_rank_pure_data(df, active_domains, metric_weights, custom_domain_override=lodo_override)
        for cfg in top_configs:
            rk = lodo_ranks[lodo_ranks["Prototype_Config"] == cfg]["Rank"].values
            lodo_matrix[cfg].append(int(rk[0]) if len(rk) > 0 else np.nan)

    print(f"\n--- TABLE 5B: Leave-One-Domain-Out (LODO) Rank Stability ---")
    lodo_headers = "  ".join([f"-{d[:10]:>10}" for d in domain_names])
    print(f" {'Prototype Configuration':<32} | {'Pure Data':>10} {lodo_headers}")
    print(f" {'-'*32}-+-{'-'*11}-{'-'*len(lodo_headers)}")
    
    lodo_records = []
    for cfg in top_configs:
        base_rk = int(base_ranks[base_ranks["Prototype_Config"] == cfg]["Rank"].values[0])
        ranks_str = "  ".join([f"{r:>10}" for r in lodo_matrix[cfg]])
        print(f" {cfg:<32} | {base_rk:>10} {ranks_str}")
        rec = {"Prototype_Config": cfg, "Pure_Data_Rank": base_rk}; rec.update(zip([f"Drop_{d}" for d in domain_names], lodo_matrix[cfg]))
        lodo_records.append(rec)
    pd.DataFrame(lodo_records).to_csv(out_dir / "sensitivity_lodo.csv", index=False)

    # ----------------------------------------------------------------------- #
    # 3. MONTE CARLO WEIGHT PERTURBATION (1,000 SIMULATIONS)
    # ----------------------------------------------------------------------- #
    print(f"\n--- TABLE 5C: Monte Carlo Rank Confidence (1,000 Random Data-Weight Simulations) ---")
    print(" Simulating +/- 50% random perturbation around automatically generated E_R^2 domain weights...")
    
    n_sims = 1000
    win_counts = defaultdict(int); top3_counts = defaultdict(int)
    
    np.random.seed(42)
    for _ in range(n_sims):
        sim_override = {}
        for d in domain_names:
            base_w = pure_dom_w[d]
            sim_override[d] = max(0.001, base_w * np.random.uniform(0.5, 1.5))
            
        sim_ranks = score_and_rank_pure_data(df, active_domains, metric_weights, custom_domain_override=sim_override)
        winner = sim_ranks.iloc[0]["Prototype_Config"]
        win_counts[winner] += 1
        for _, r in sim_ranks.head(3).iterrows():
            top3_counts[r["Prototype_Config"]] += 1

    print(f" {'Prototype Configuration':<32} | {'Win Rate (%)':>14} {'Top-3 Finish (%)':>18} {'Robustness Status':>20}")
    print(f" {'-'*32}-+-{'-'*14}-{'-'*18}-{'-'*20}")
    
    mc_records = []
    all_cfgs_sorted = sorted(list(set(list(win_counts.keys()) + list(top3_counts.keys()) + top_configs)), key=lambda x: win_counts[x], reverse=True)
    
    for cfg in all_cfgs_sorted[:top_n]:
        w_pct = (win_counts[cfg] / n_sims) * 100.0; t3_pct = (top3_counts[cfg] / n_sims) * 100.0
        if w_pct >= 60.0: status = "★★ DOMINANT CHOICE"
        elif t3_pct >= 75.0: status = "★ HIGHLY ROBUST"
        elif t3_pct >= 30.0: status = "  COMPETITIVE"
        else: status = "  SENSITIVE / NICHE"
        
        print(f" {cfg:<32} | {w_pct:>13.1f}% {t3_pct:>17.1f}% {status:>20}")
        mc_records.append({"Prototype_Config": cfg, "Win_Rate_Pct": w_pct, "Top3_Rate_Pct": t3_pct, "Status": status.strip()})
        
    pd.DataFrame(mc_records).to_csv(out_dir / "sensitivity_monte_carlo.csv", index=False)


# =========================================================================== #
# SECTION 4: FORMATTED ASCII REPORTING
# =========================================================================== #

def print_ascii_leaderboard(rankings: pd.DataFrame, title: str, top_n: int = 10):
    """Prints a formatted leaderboard showing ranks, overall scores, and domain breakdown."""
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    domains = [d for d in rankings.columns if d not in ("Rank", "Prototype_Config", "Grand_Score", "Score_SD", "N_Events")]
    domain_headers = "  ".join([f"{d[:10]:>10}" for d in domains])
    
    print(f" {'Rk':<3} {'Prototype Configuration':<32} {'Grand':>6} {'(SD)':>6} {'N':>4} | {domain_headers}")
    print(f" {'-'*3} {'-'*32} {'-'*6} {'-'*6} {'-'*4}-+-{'-'*len(domain_headers)}")

    for _, r in rankings.head(top_n).iterrows():
        domain_vals = "  ".join([f"{r[d]:>10.1f}" for d in domains])
        print(f" {int(r['Rank']):<3} {r['Prototype_Config']:<32} {r['Grand_Score']:>6.1f} ({r['Score_SD']:>4.1f}) {int(r['N_Events']):>4} | {domain_vals}")
    
    if len(rankings) > top_n:
        print(f" ... and {len(rankings) - top_n} more configurations.")
    print(f" *(Scores 0-100; 100 = Optimal observed baseline across active metrics)*")


# =========================================================================== #
# SECTION 5: COMMAND LINE INTERFACE & MAIN EXECUTION
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--pen-csv", type=Path, default=None, help="Explicit path to place_metrics_combined.csv")
    ap.add_argument("--posture-csv", type=Path, default=None, help="Explicit path to posture_features_combined.csv")
    args = ap.parse_args()

    if args.landmarks_root:
        pen_path     = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        posture_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
        out_dir      = args.landmarks_root / "metrics" / "prototype_rankings"
    else:
        pen_path     = args.pen_csv
        posture_path = args.posture_csv
        out_dir      = (pen_path or posture_path).parent / "prototype_rankings"

    if not pen_path and not posture_path:
        sys.exit("Error: Provide --landmarks-root OR specify explicit paths via --pen-csv / --posture-csv.")

    df_pen = pd.read_csv(pen_path) if (pen_path and pen_path.is_file()) else pd.DataFrame()
    df_posture = pd.read_csv(posture_path) if (posture_path and posture_path.is_file()) else pd.DataFrame()

    if df_pen.empty and df_posture.empty:
        sys.exit("Error: Both pen and posture CSVs are empty or missing. Nothing to evaluate.")

    if not df_pen.empty and not df_posture.empty:
        common = [c for c in ["participant", "trial", "place_index", "height"] if c in df_pen.columns and c in df_posture.columns]
        df = pd.merge(df_pen, df_posture, on=common, how="inner", suffixes=("", "_posture"))
        print(f"Merged Pen ({len(df_pen)}) and Posture ({len(df_posture)}) into {len(df)} total Place events.")
    else:
        df = df_pen if not df_pen.empty else df_posture
        print(f"Loaded {len(df)} Place events from single available data source.")

    df = add_prototype_label(df)
    df_scored, active_domains = calculate_desirability_scores(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute Global Sensitivity Weights & Rankings
    global_weights = compute_epsilon_weights(df_scored, active_domains)
    global_ranks = score_and_rank_pure_data(df_scored, active_domains, global_weights)
    global_ranks.to_csv(out_dir / "rankings_global_all_rounder.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 1A: AUTOMATICALLY GENERATED DOMAIN VOTING POWER
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*75}\nAUTOMATICALLY GENERATED DOMAIN VOTING POWER (From Design Space Sensitivity)\n{'='*75}")
    print(f" {'Domain Name':<20} {'Mean Sensitivity (E_R^2)':>25} {'Automatic Share of Grand Score (%)':>32}")
    print(f" {'-'*20} {'-'*25} {'-'*32}")
    
    dom_sens = {d: np.mean([global_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}
    tot_sens = sum(dom_sens.values())
    
    for d, sens in sorted(dom_sens.items(), key=lambda item: item[1], reverse=True):
        share_pct = (sens / tot_sens) * 100.0
        print(f" {d:<20} {sens:>25.4f} {share_pct:>31.1f}%")
    print(f" *The design space automatically grants higher voting power to domains that react more strongly to prototype changes.*")

    # 2. Compute Stratified Sensitivity Weights & Rankings
    strata = sorted(df_scored["height"].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9)) if "height" in df_scored.columns else []
    stratified_results = {}
    for s in strata:
        sub_df = df_scored[df_scored["height"] == s]
        if sub_df.empty: continue
        s_weights = compute_epsilon_weights(sub_df, active_domains)
        s_ranks = score_and_rank_pure_data(sub_df, active_domains, s_weights)
        stratified_results[s] = s_ranks
        s_ranks.to_csv(out_dir / f"rankings_stratified_{s}.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 1B: Side-by-Side Metric Sensitivity (E_R^2) Comparison
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*80}\nMETRIC SENSITIVITY WEIGHTS (E_R^2) BY STRATEGY\n{'='*80}")
    header = f" {'Domain':<15} {'Metric':<26} {'Global':>8}" + "".join([f" {s:>8}" for s in strata])
    print(header); print(f" {'-'*15} {'-'*26} {'-'*8}" + "".join([f" {'-'*8}" for _ in strata]))
    
    sens_records = []
    for domain, cols in active_domains.items():
        for col in cols:
            raw = col.replace("_dscore", "")
            row_str = f" {domain:<15} {raw:<26} {global_weights.get(col, 0):>8.3f}"
            rec = {"Domain": domain, "Metric": raw, "Global_E_sq": global_weights.get(col, 0)}
            for s in strata:
                sw = compute_epsilon_weights(df_scored[df_scored["height"] == s], active_domains)
                val = sw.get(col, 0)
                row_str += f" {val:>8.3f}"
                rec[f"{s}_E_sq"] = val
            print(row_str)
            sens_records.append(rec)
            
    pd.DataFrame(sens_records).to_csv(out_dir / "metric_sensitivity_weights.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 2 & 3: Global Leaderboard & Stratified Leaderboards
    # ----------------------------------------------------------------------- #
    print_ascii_leaderboard(global_ranks, "GLOBAL PROTOTYPE LEADERBOARD (100% Data-Driven All-Rounders)", top_n=10)
    
    for s in strata:
        if s in stratified_results:
            print_ascii_leaderboard(stratified_results[s], f"STRATIFIED LEADERBOARD — {s.upper()} WORKSTATION", top_n=5)

    # ----------------------------------------------------------------------- #
    # TABLE 4: Prototype Ranking Shift Matrix
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*85}\nPROTOTYPE RANKING SHIFT MATRIX (Global vs. Height-Stratified Models)\n{'='*85}")
    r_header = f" {'Prototype Configuration':<32} | {'Global Rk':>10} {'(Score)':>8} |" + "".join([f" {s[:3]} Rk (Scr) |" for s in strata])
    print(r_header); print(f" {'-'*32}-+-{'-'*10}-{'-'*8}-+" + "".join([f"{'-'*13}-+" for _ in strata]))

    shift_records = []
    for _, gr in global_ranks.iterrows():
        p_cfg = gr["Prototype_Config"]
        row_str = f" {p_cfg:<32} | {int(gr['Rank']):>10} {gr['Grand_Score']:>8.1f} |"
        rec = {"Prototype_Config": p_cfg, "Global_Rank": int(gr['Rank']), "Global_Score": gr['Grand_Score']}
        
        for s in strata:
            sr = stratified_results[s]
            match = sr[sr["Prototype_Config"] == p_cfg]
            if not match.empty:
                r_val, s_val = int(match.iloc[0]["Rank"]), match.iloc[0]["Grand_Score"]
                row_str += f" {r_val:>5} ({s_val:>4.1f}) |"
                rec[f"{s}_Rank"] = r_val; rec[f"{s}_Score"] = s_val
            else:
                row_str += f" {'n/a':>11} |"
                rec[f"{s}_Rank"] = np.nan; rec[f"{s}_Score"] = np.nan
        print(row_str)
        shift_records.append(rec)
        
    pd.DataFrame(shift_records).to_csv(out_dir / "rankings_shift_matrix.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 5: 3-PILLAR DECISION SENSITIVITY ANALYSIS
    # ----------------------------------------------------------------------- #
    run_sensitivity_analysis(df_scored, active_domains, global_weights, out_dir, top_n=8)

    print(f"\nAll ranking evaluations, shift matrices, and sensitivity tables saved to:\n  -> {out_dir}")


if __name__ == "__main__":
    main()