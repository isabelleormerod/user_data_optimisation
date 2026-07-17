#!/usr/bin/env python3
r"""
run_hierarchical_fpca.py

Two-Block Hierarchical Functional PCA (mfPCA) for Biomechanics with Synergy Ranking.

STATISTICAL ENGINE:
  Block 1 (Domain Distillation): Time-normalizes continuous kinematic curves and runs 
  fPCA independently on the Pen, Body, and Hand.
  
  Block 2 (Global Synergy Fusion): Concatenates the distilled domain components and 
  runs a second, top-level PCA to discover Cross-Domain Synergies.
  
  Sensitivity Evaluation: Evaluates these global synergies within each workstation height 
  using a linear mixed-effects model (synergy ~ Length + Size + Weight + Angle + 1|participant).

  python run_hierarchical_fpca.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  
  Synergy Ranking: Calculates the absolute deviation of each synergy from the grand mean (0),
  scores it from 0-100 (where 100 = natural/average movement), applies participant-aggregated
  E_R^2 sensitivity weights, and outputs a final data-driven Prototype Leaderboard.
"""

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Restore native discovery modules
try:
    from utils.discovery import find_labelled_pen, iter_trials_labelled
    from utils.params import parse_participant_filter
except ImportError:
    sys.exit("Error: Could not import 'utils.discovery' or 'utils.params'.")

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]

# =========================================================================== #
# 1. VECTORIZED MATH & TIME-NORMALIZATION
# =========================================================================== #

def time_normalize_curve(t, y, n_points=100):
    valid = ~np.isnan(y)
    if np.sum(valid) < 2:
        return np.full(n_points, np.nan)
    t_val, y_val = t[valid], y[valid]
    if t_val[-1] == t_val[0]:
        return np.full(n_points, y_val[0])
    
    t_norm = (t_val - t_val[0]) / (t_val[-1] - t_val[0])
    interpolator = interp1d(t_norm, y_val, kind='linear', bounds_error=False, fill_value=(y_val[0], y_val[-1]))
    return interpolator(np.linspace(0.0, 1.0, n_points))

def vec_angle_3pt(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine = np.sum(v1 * v2, axis=1) / (n1 * n2)
        cosine = np.clip(cosine, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
    return angles

def vec_dist(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def parse_factors_from_stem(stem: str) -> dict:
    out = {"Length": "Other", "Size": "Other", "Weight": "Other", "Angle": "Other"}
    tokens = [t.strip() for t in stem.split("_") if t.strip()]
    joined_low = "_".join(tokens).lower()
    if "not_weighted" in joined_low or "notweighted" in joined_low: out["Weight"] = "Not_weighted"
    elif "front_weighted" in joined_low or "frontweighted" in joined_low: out["Weight"] = "Front_weighted"
    elif "weighted" in tokens: out["Weight"] = "weighted"
    for tok in tokens:
        t_low, t_cap = tok.lower(), tok.capitalize()
        if t_cap in ("Long", "Short"): out["Length"] = t_cap
        elif t_cap in ("Large", "Small"): out["Size"] = t_cap
        elif t_low.startswith("a") and t_low[1:].isdigit(): out["Angle"] = f"A{t_low[1:]}"
        elif tok.isdigit() and int(tok) in (0, 45, 90, 135, 180, 225, 270, 315): out["Angle"] = f"A{tok}"
    return out


# =========================================================================== #
# 2. BLOCK 1: TIME-SERIES EXTRACTION ENGINE
# =========================================================================== #

def extract_functional_curves(root_dir: Path, participants_str: str = None):
    print(f"\n{'='*75}\nPHASE 1: EXTRACTING FUNCTIONAL TIME-SERIES CURVES\n{'='*75}")
    pfilter = parse_participant_filter(participants_str)
    trials = list(iter_trials_labelled(root_dir, pfilter))

    if not trials:
        sys.exit("Error: No trial folders found.")

    events_meta = []
    domain_curves = {"Pen": defaultdict(list), "Body": defaultdict(list), "Hand": defaultdict(list)}
    
    for stem, pid, trial_dir in trials:
        pen_file = find_labelled_pen(trial_dir, stem)
        if not pen_file or not pen_file.is_file(): continue
            
        df_pen = pd.read_csv(pen_file)
        if "Place" not in df_pen.columns or "t_s" not in df_pen.columns: continue
            
        places, in_run, start, prev_t = [], False, None, None
        for _, r in df_pen.iterrows():
            t, flag = r["t_s"], str(r["Place"]).strip() in ("1", "1.0", "True", "true")
            if flag and not in_run: in_run, start = True, t
            elif not flag and in_run: in_run = False; places.append((start, prev_t))
            prev_t = t
        if in_run: places.append((start, prev_t))

        # Extract height stratum per timestamp
        height_runs = {}
        for h in ("High", "Medium", "Low"):
            if h in df_pen.columns:
                runs, in_run, start_h, prev_h = [], False, None, None
                for _, r in df_pen.iterrows():
                    t, flag = r["t_s"], str(r[h]).strip() in ("1", "1.0", "True", "true")
                    if flag and not in_run: in_run, start_h = True, t
                    elif not flag and in_run: in_run = False; runs.append((start_h, prev_h))
                    prev_h = t
                if in_run: runs.append((start_h, prev_h))
                height_runs[h] = runs

        def get_height(t_mid):
            for h, runs in height_runs.items():
                for s, e in runs:
                    if s <= t_mid <= e: return h
            return "Unknown"

        body_file = trial_dir / f"{stem}_body.csv"
        hand_file = trial_dir / f"{stem}_hand.csv"
        df_b = pd.read_csv(body_file) if body_file.is_file() else pd.DataFrame()
        df_h = pd.read_csv(hand_file) if hand_file.is_file() else pd.DataFrame()

        factors = parse_factors_from_stem(stem)

        for i, (s, e) in enumerate(places, 1):
            h_label = get_height((s + e) / 2)
            events_meta.append({
                "participant": pid, "Trial": stem, "Place_ID": i, "height": h_label,
                "Prototype_Config": f"{factors['Length']}_{factors['Size']}_{factors['Weight']}_{factors['Angle']}",
                **factors
            })
            
            # --- PEN DOMAIN ---
            sub_p = df_pen[(df_pen["t_s"] >= s) & (df_pen["t_s"] <= e)]
            if len(sub_p) > 2:
                t_arr = sub_p["t_s"].values
                for col in ("perp_mean_deg", "updown_mean_deg", "leftright_mean_deg"):
                    if col in sub_p.columns:
                        domain_curves["Pen"][col].append(time_normalize_curve(t_arr, sub_p[col].values))
                    else:
                        domain_curves["Pen"][col].append(np.full(100, np.nan))
            else:
                for col in ("perp_mean_deg", "updown_mean_deg", "leftright_mean_deg"):
                    domain_curves["Pen"][col].append(np.full(100, np.nan))

            # --- BODY DOMAIN ---
            sub_b = df_b[(df_b["t_s"] >= s - 0.1) & (df_b["t_s"] <= e + 0.1)] if not df_b.empty and "t_s" in df_b.columns else pd.DataFrame()
            if len(sub_b) > 2:
                t_arr = sub_b["t_s"].values
                def get_pts(name): return sub_b[[f"{name}_x", f"{name}_y", f"{name}_z"]].values
                for side in ("Left", "Right"):
                    try:
                        sh, el, wr = get_pts(f"{side}Shoulder"), get_pts(f"{side}Elbow"), get_pts(f"{side}Wrist")
                        hp, kn, an = get_pts(f"{side}Hip"), get_pts(f"{side}Knee"), get_pts(f"{side}Ankle")
                        domain_curves["Body"][f"{side}_Elbow"].append(time_normalize_curve(t_arr, vec_angle_3pt(sh, el, wr)))
                        domain_curves["Body"][f"{side}_Shoulder"].append(time_normalize_curve(t_arr, vec_angle_3pt(hp, sh, el)))
                    except KeyError:
                        domain_curves["Body"][f"{side}_Elbow"].append(np.full(100, np.nan))
                        domain_curves["Body"][f"{side}_Shoulder"].append(np.full(100, np.nan))
            else:
                for side in ("Left", "Right"):
                    domain_curves["Body"][f"{side}_Elbow"].append(np.full(100, np.nan))
                    domain_curves["Body"][f"{side}_Shoulder"].append(np.full(100, np.nan))

            # --- HAND DOMAIN ---
            sub_h = df_h[(df_h["t_s"] >= s - 0.05) & (df_h["t_s"] <= e + 0.05)] if not df_h.empty and "t_s" in df_h.columns else pd.DataFrame()
            if len(sub_h) > 2:
                t_arr = sub_h["t_s"].values
                try:
                    for side in ("Right",):
                        wrist = sub_h[[f"{side}_HandWristRoot_x", f"{side}_HandWristRoot_y", f"{side}_HandWristRoot_z"]].values
                        for f in ("Thumb", "Index", "Middle"):
                            j1 = sub_h[[f"{side}_Hand{f}1_x", f"{side}_Hand{f}1_y", f"{side}_Hand{f}1_z"]].values
                            j2 = sub_h[[f"{side}_Hand{f}2_x", f"{side}_Hand{f}2_y", f"{side}_Hand{f}2_z"]].values
                            domain_curves["Hand"][f"{side}_{f}_MCP"].append(time_normalize_curve(t_arr, vec_angle_3pt(wrist, j1, j2)))
                        
                        tt = sub_h[[f"{side}_HandThumbTip_x", f"{side}_HandThumbTip_y", f"{side}_HandThumbTip_z"]].values
                        it = sub_h[[f"{side}_HandIndexTip_x", f"{side}_HandIndexTip_y", f"{side}_HandIndexTip_z"]].values
                        domain_curves["Hand"][f"{side}_Aperture"].append(time_normalize_curve(t_arr, vec_dist(tt, it)))
                except KeyError:
                    for f in ("Thumb", "Index", "Middle"): domain_curves["Hand"][f"Right_{f}_MCP"].append(np.full(100, np.nan))
                    domain_curves["Hand"][f"Right_Aperture"].append(np.full(100, np.nan))
            else:
                for f in ("Thumb", "Index", "Middle"): domain_curves["Hand"][f"Right_{f}_MCP"].append(np.full(100, np.nan))
                domain_curves["Hand"][f"Right_Aperture"].append(np.full(100, np.nan))

    print(f"Extracted curves for {len(events_meta)} Place Events.")
    
    for dom in domain_curves:
        for feat in domain_curves[dom]:
            mat = np.array(domain_curves[dom][feat])
            mean_curve = np.nanmean(mat, axis=0)
            if np.isnan(mean_curve).all(): mean_curve = np.zeros(100)
            inds = np.where(np.isnan(mat))
            mat[inds] = np.take(mean_curve, inds[1])
            domain_curves[dom][feat] = mat
            
    return pd.DataFrame(events_meta), domain_curves


# =========================================================================== #
# 3. MIXED-EFFECTS STATISTICAL ENGINE & STRATIFIED PLOTTING
# =========================================================================== #

def _fit_mixed_main(data: pd.DataFrame, response: str, factors: list):
    present = [f for f in factors if data[f].nunique() >= 2]
    if not present: return None, present, "no factor has >=2 levels"
    formula = f"{response} ~ " + " + ".join(f"C({f})" for f in present)
    last_err = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for method in ("lbfgs", "powell", "cg"):
            try:
                res = smf.mixedlm(formula, data, groups=data["participant"]).fit(reml=False, method=method)
                if np.isfinite(res.llf): return res, present, None
                last_err = f"non-finite log-likelihood ({method})"
            except Exception as ex:
                last_err = f"{type(ex).__name__}: {ex}"
    return None, present, (last_err or "unknown fit failure")

def _term_wald(res, factor: str):
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    if not names: return np.nan, np.nan
    b = res.fe_params[names].values
    try:
        V = res.cov_params().loc[names, names].values
        W = float(b @ np.linalg.solve(V, b))
    except Exception: return np.nan, len(names)
    p = float(stats.chi2.sf(W, len(names)))
    return p, len(names)

def mixed_tests_synergies(df: pd.DataFrame, synergies: list, factors: list, stratum_col: str = "height", min_n: int = 8):
    records = []
    groups = df.groupby(stratum_col, dropna=True) if stratum_col else [("All", df)]
    for stratum, sub_df in groups:
        for syn in synergies:
            d = sub_df.dropna(subset=[syn, "participant"] + factors).copy().rename(columns={syn: "_y"})
            n_ppt = d["participant"].nunique()
            res, present, note = (None, [], None)
            if n_ppt >= 2 and len(d) >= min_n:
                res, present, note = _fit_mixed_main(d, "_y", factors)
            
            for f in factors:
                if res is not None: p, dfree = _term_wald(res, f)
                else: p, dfree = np.nan, np.nan
                rec = {"stratum": stratum, "factor": f, "synergy": syn, "p_value": p, "n": len(d)}
                records.append(rec)
    return pd.DataFrame(records)

def order_levels(factor, levels):
    orders = {"Length": ["Short", "Long"], "Size": ["Small", "Large"], "Weight": ["Not_weighted", "Front_weighted"]}
    if factor in orders:
        known = [l for l in orders[factor] if l in levels]
        return known + sorted([l for l in levels if l not in known])
    try: return sorted(levels, key=lambda x: float(x))
    except: return sorted(levels, key=str)

def make_synergy_graphs_stratified(df, synergies, out_dir, p_lookup, stratum_col="height"):
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = sorted(df[stratum_col].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9))
    
    print("\nGenerating stratified diagnostic box plots for each synergy...")
    for factor in PARAM_FACTORS:
        if factor not in df.columns: continue
        levels = order_levels(factor, list(df[factor].dropna().unique()))
        if len(levels) < 2: continue
        
        for syn in synergies:
            fig, axes = plt.subplots(1, len(strata), figsize=(max(5, len(levels)*1.5)*len(strata), 5), sharey=True)
            if len(strata) == 1: axes = [axes]
            
            for ax, stratum in zip(axes, strata):
                sub = df[df[stratum_col] == stratum]
                data, tick_labels = [], []
                for lv in levels:
                    vals = pd.to_numeric(sub.loc[sub[factor]==lv, syn], errors="coerce").dropna().values
                    if len(vals): 
                        data.append(vals)
                        tick_labels.append(f"{lv}\n(n={len(vals)})")
                
                if len(data) >= 2:
                    ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
                    for i, vals in enumerate(data, 1):
                        jit = (np.random.rand(len(vals))-0.5)*0.15
                        ax.scatter(np.full(len(vals),i)+jit, vals, alpha=0.5, s=15, color="#1f77b4", zorder=3)
                
                p = p_lookup.get((stratum, factor, syn), np.nan)
                has_p = p is not None and not pd.isna(p)
                title = f"{stratum}   (p={p:.3f}*)" if has_p and p < 0.05 else (f"{stratum}   (p={p:.3f})" if has_p else f"{stratum}")
                ax.set_title(title, fontsize=10, fontweight='bold' if (has_p and p < 0.05) else 'normal')
                ax.set_xlabel(factor)
                ax.grid(axis="y", alpha=0.3)
                
            axes[0].set_ylabel(f"{syn} Score")
            fig.suptitle(f"{syn} by {factor} — stratified by {stratum_col}", fontsize=11)
            fig.tight_layout()
            
            path = out_dir / f"by_{factor}_{syn}_stratified.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)


# =========================================================================== #
# 4. BLOCK 2: DOMAIN fPCA & GLOBAL SYNERGY FUSION
# =========================================================================== #

def perform_domain_fpca(domain_name, curves_dict, variance_target=0.90):
    feature_names = list(curves_dict.keys())
    if not feature_names: return pd.DataFrame(), None
    
    stacked_matrix = np.hstack([curves_dict[f] for f in feature_names])
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(stacked_matrix)
    
    pca = PCA(n_components=variance_target, random_state=42)
    scores = pca.fit_transform(scaled_matrix)
    
    df_scores = pd.DataFrame(scores, columns=[f"{domain_name}_fPC{i+1}" for i in range(scores.shape[1])])
    print(f" -> {domain_name} Domain: {len(feature_names)} curve types compressed into {scores.shape[1]} fPCs ({sum(pca.explained_variance_ratio_)*100:.1f}% variance)")
    return df_scores, pca

def evaluate_synergies_stratified(df_meta, df_global_scores, out_dir):
    df = pd.concat([df_meta, df_global_scores], axis=1)
    synergies = df_global_scores.columns.tolist()
    
    valid_strata = {"High", "Medium", "Low"}
    unknown_mask = ~df["height"].isin(valid_strata) | df["height"].isna()
    if unknown_mask.any():
        print(f"\n[QUARANTINE] Filtering out {unknown_mask.sum()} Place events that failed to match a valid High/Medium/Low window.")
        df = df[~unknown_mask].copy()

    strata = sorted(df["height"].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s, 9))
    
    print(f"\n{'='*75}\nPHASE 3: STRATIFIED ANALYSIS — Synergy LMM within each workstation height\n{'='*75}")
    strat_tests_df = mixed_tests_synergies(df, synergies, PARAM_FACTORS, stratum_col="height")
    
    print(f"Full p-value matrix (prototype factors x synergies x stratum):")
    header_str = f"  {'Factor':<10} {'Synergy':<30}"
    for s in strata: header_str += f" {s:>8}"
    print(header_str)
    
    sep_str = f"  {'-'*10} {'-'*30}"
    for _ in strata: sep_str += f" {'-'*8}"
    print(sep_str)

    for factor in PARAM_FACTORS:
        for syn in synergies:
            row_vals = []
            has_data = False
            for s in strata:
                match = strat_tests_df[
                    (strat_tests_df["stratum"] == s) &
                    (strat_tests_df["factor"]  == factor) &
                    (strat_tests_df["synergy"] == syn)
                ]
                if match.empty or pd.isna(match.iloc[0]["p_value"]):
                    row_vals.append("     n/a")
                else:
                    p = match.iloc[0]["p_value"]
                    has_data = True
                    marker = "*" if p < 0.05 else " "
                    row_vals.append(f"{p:>7.3f}{marker}")

            if has_data:
                row_str = f"  {factor:<10} {syn:<30}"
                for v in row_vals: row_str += f" {v:>8}"
                print(row_str)

    print("\n  * = p < 0.05  (Wald test, mixed model, participant random intercept, main effects only)")

    p_lookup = {(r["stratum"], r["factor"], r["synergy"]): r["p_value"] for _, r in strat_tests_df.iterrows()}
    make_synergy_graphs_stratified(df, synergies, out_dir, p_lookup, stratum_col="height")
    
    return strat_tests_df, df


# =========================================================================== #
# 5. PHASE 4: PROTOTYPE RANKING (SYNERGY DEVIATION SCORING)
# =========================================================================== #

def rank_prototypes_by_synergy(df_clean: pd.DataFrame, synergies: list):
    """
    Ranks prototypes based on how well they minimize extreme PCA synergy deviations.
    A PCA score of 0 represents the mathematical 'average' movement profile of the dataset.
    Prototypes that force users into high absolute score magnitudes are forcing extreme anomalies.
    """
    print(f"\n{'='*75}\nPHASE 4: SYNERGY-BASED PROTOTYPE RANKING\n{'='*75}")
    
    df_scored = df_clean.copy()
    score_cols = []
    
    # 1. Calculate 0-100 Desirability Scores (Minimizing Absolute Deviation from 0)
    for syn in synergies:
        vals = df_scored[syn].abs()
        v_min, v_max = vals.min(), vals.max()
        if abs(v_max - v_min) < 1e-9: continue
        
        score_col = f"{syn}_dscore"
        df_scored[score_col] = 100.0 * (v_max - vals) / (v_max - v_min)
        score_cols.append(score_col)
        
    # 2. Compute Participant-Aggregated E_R^2 Weights
    df_agg = df_scored.groupby(["Prototype_Config", "participant"], dropna=False, as_index=False).mean(numeric_only=True)
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

    # 3. Compute Grand Synergy Score
    df_calc = df_scored.copy()
    tot_w = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    
    df_calc["Grand_Synergy_Score"] = 0.0
    for c in score_cols:
        w = weights.get(c, 0.01) / tot_w
        df_calc["Grand_Synergy_Score"] += df_calc[c].fillna(0.0) * w
        
    # 4. Aggregate Leaderboard
    agg_dict = {"Grand_Synergy_Score": ["mean", "std", "count"]}
    for c in score_cols: agg_dict[c] = "mean"
        
    rankings = df_calc.groupby("Prototype_Config", dropna=False).agg(agg_dict)
    rankings.columns = ["Grand_Score", "Score_SD", "N_Events"] + [c.replace("_dscore", "") for c in score_cols]
    rankings = rankings.reset_index().sort_values(by="Grand_Score", ascending=False).reset_index(drop=True)
    rankings["Rank"] = rankings.index + 1
    
    # 5. Output Leaderboard
    print(f" {'Rk':<3} {'Prototype Configuration':<32} {'Grand':>6} {'(SD)':>6} {'N':>4}")
    print(f" {'-'*3} {'-'*32} {'-'*6} {'-'*6} {'-'*4}")

    for _, r in rankings.head(10).iterrows():
        print(f" {int(r['Rank']):<3} {r['Prototype_Config']:<32} {r['Grand_Score']:>6.1f} ({r['Score_SD']:>4.1f}) {int(r['N_Events']):>4}")
    
    if len(rankings) > 10:
        print(f" ... and {len(rankings) - 10} more configurations.")
    print(f" *(Scores 0-100; 100 = Natural/Average Dataset Baseline; 0 = Extreme Postural/Kinematic Anomaly)*")
    
    return rankings, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-root", type=Path, required=True, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--participants", type=str, default=None, help="Comma-separated list of participant IDs")
    args = ap.parse_args()

    if not args.landmarks_root.is_dir():
        sys.exit(f"Error: Invalid directory: {args.landmarks_root}")

    out_dir = args.landmarks_root / "metrics" / "fpca_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_meta, domain_curves = extract_functional_curves(args.landmarks_root, args.participants)
    
    print(f"\n{'='*75}\nPHASE 2: HIERARCHICAL fPCA COMPRESSION\n{'='*75}")
    domain_scores = []
    for dom in ["Pen", "Body", "Hand"]:
        scores_df, _ = perform_domain_fpca(dom, domain_curves[dom])
        domain_scores.append(scores_df)
        
    df_all_domain_scores = pd.concat(domain_scores, axis=1)

    print("\nRunning Top-Level Global fPCA on concatenated domain components...")
    scaler_global = StandardScaler()
    scaled_global = scaler_global.fit_transform(df_all_domain_scores)
    
    pca_global = PCA(n_components=0.90, random_state=42)
    global_scores = pca_global.fit_transform(scaled_global)
    synergies = [f"Global_Synergy_{i+1}" for i in range(global_scores.shape[1])]
    df_global_scores = pd.DataFrame(global_scores, columns=synergies)
    print(f" -> Fused into {global_scores.shape[1]} Global Cross-Domain Synergies.")

    # Phase 3: Evaluate Stratified Mixed Models
    strat_tests_df, df_fused_clean = evaluate_synergies_stratified(df_meta, df_global_scores, out_dir)
    
    # Phase 4: Rank Prototypes
    synergy_rankings, synergy_weights = rank_prototypes_by_synergy(df_fused_clean, synergies)
    
    # Save outputs
    df_fused_clean.to_csv(out_dir / "all_fpca_scores_stratified.csv", index=False)
    strat_tests_df.to_csv(out_dir / "stratified_synergy_sensitivities_lmm.csv", index=False)
    synergy_rankings.to_csv(out_dir / "synergy_prototype_rankings.csv", index=False)
    
    pd.DataFrame([{"Synergy": k.replace("_dscore", ""), "E_R2_Weight": v} for k, v in synergy_weights.items()]).to_csv(out_dir / "synergy_ranking_weights.csv", index=False)
    print(f"\nResults saved to: {out_dir}")

if __name__ == "__main__":
    main()