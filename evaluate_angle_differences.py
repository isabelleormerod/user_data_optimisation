#!/usr/bin/env python3
r"""
evaluate_angle_differences.py

100% Data-Driven Stratified Biomechanical & Kinematic Evaluation Engine.
Extracts systematic, coordinate-free joint angles, internal hand distances, AND tool 
orientation angles during Place events. Executes stratified Kruskal-Wallis tests across 
prototype factors (Length, Size, Weight, Angle) and workstation strata (High, Medium, Low).

Outputs a clean ASCII p-value matrix and exports full statistical ledgers to CSV.

Usage:
  python evaluate_angle_differences.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python evaluate_angle_differences.py --landmarks-root ... --participants P001,P002
"""

import argparse
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Restore native discovery modules
try:
    from utils.discovery import find_labelled_pen, iter_trials_labelled
    from utils.params import parse_participant_filter
except ImportError:
    sys.exit("Error: Could not import 'utils.discovery' or 'utils.params'. Ensure this script is run from within your repo root.")


# =========================================================================== #
# 1. SYSTEMATIC ANGLE & GEOMETRY MATH
# =========================================================================== #

def angle_between_3_points(p1, p2, p3):
    """Calculates translation-invariant internal angle at p2 formed by p1-p2 and p3-p2."""
    if any(p is None or np.any(np.isnan(p)) for p in (p1, p2, p3)):
        return np.nan
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def dist_between(p1, p2):
    """Calculates internal Euclidean distance (e.g., grip span aperture)."""
    if p1 is None or p2 is None or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return np.nan
    return float(np.linalg.norm(p1 - p2))

def vector_angle_with_vertical(v):
    """Calculates tilt angle (in degrees) between a vector and true vertical [0, 0, 1]."""
    if v is None or np.any(np.isnan(v)):
        return np.nan
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.nan
    cosine = np.clip(np.dot(v / norm, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# =========================================================================== #
# 2. METADATA & PARAMETER PARSERS
# =========================================================================== #

def parse_height_stratum(trial_dir: Path, stem: str, df_pen: pd.DataFrame) -> str:
    """Robustly locates workstation height (High/Medium/Low) from CSV columns, folder names, or stem."""
    # 1. Check CSV columns first
    for col in ("height", "stratum", "Workstation_Height", "table_height"):
        if col in df_pen.columns and not df_pen[col].isna().all():
            val = str(df_pen[col].dropna().iloc[0]).strip().capitalize()
            if val in ("High", "Medium", "Low"):
                return val
                
    # 2. Check directory path strings and filenames
    full_path_str = f"{str(trial_dir)}_{stem}".lower()
    if "_high" in full_path_str or "/high" in full_path_str or "\\high" in full_path_str: return "High"
    if "_medium" in full_path_str or "/medium" in full_path_str or "\\medium" in full_path_str: return "Medium"
    if "_low" in full_path_str or "/low" in full_path_str or "\\low" in full_path_str: return "Low"
    
    # 3. Fallback regex for standalone height tokens
    for token in stem.split("_"):
        if token.capitalize() in ("High", "Medium", "Low"):
            return token.capitalize()
            
    return "Unknown"

def parse_factors_from_stem(stem: str) -> dict:
    """Robustly extracts Length, Size, Weight, and Angle from trial names."""
    out = {"Length": "Other", "Size": "Other", "Weight": "Other", "Angle": "Other"}
    tokens = [t.strip() for t in stem.split("_") if t.strip()]
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


# =========================================================================== #
# 3. FEATURE EXTRACTION ENGINE (INTEGRATING PEN, BODY, AND HANDS)
# =========================================================================== #

def extract_all_kinematics_and_angles(root_dir: Path, participants_str: str = None):
    print(f"\n{'='*75}\nEXTRACTING PEN KINEMATICS & ANATOMICAL ANGLES\n{'='*75}")
    
    pfilter = parse_participant_filter(participants_str)
    trials = list(iter_trials_labelled(root_dir, pfilter))

    if not trials:
        sys.exit("Error: No trial folders with labelled pen files found! Check your --landmarks-root path.")

    print(f"Processing {len(trials)} trial(s) located via native discovery...")
    all_events = []
    
    for stem, pid, trial_dir in trials:
        pen_file = find_labelled_pen(trial_dir, stem)
        if not pen_file or not pen_file.is_file():
            continue
            
        body_file = trial_dir / f"{stem}_body.csv"
        hand_file = trial_dir / f"{stem}_hand.csv"
        
        df_pen = pd.read_csv(pen_file)
        if "Place" not in df_pen.columns or "t_s" not in df_pen.columns:
            continue
            
        height_stratum = parse_height_stratum(trial_dir, stem, df_pen)
        factors = parse_factors_from_stem(stem)
        
        # Find Place intervals
        places, in_run, start, prev_t = [], False, None, None
        for _, r in df_pen.iterrows():
            t = r["t_s"]
            flag = str(r["Place"]).strip() in ("1", "1.0", "True", "true")
            if flag and not in_run: in_run, start = True, t
            elif not flag and in_run: in_run = False; places.append((start, prev_t))
            prev_t = t
        if in_run: places.append((start, prev_t))
        
        df_body = pd.read_csv(body_file) if body_file.is_file() else pd.DataFrame()
        df_hand = pd.read_csv(hand_file) if hand_file.is_file() else pd.DataFrame()

        for i, (s, e) in enumerate(places, 1):
            event = {
                "Trial": stem, "Place_ID": i, "Stratum": height_stratum,
                "Length": factors["Length"], "Size": factors["Size"],
                "Weight": factors["Weight"], "Angle": factors["Angle"]
            }
            
            # --- 1. EXTRACT PEN ORIENTATION & TILT ANGLES ---
            sub_p = df_pen[(df_pen["t_s"] >= s) & (df_pen["t_s"] <= e)]
            if len(sub_p) > 0:
                # If pre-calculated tilt columns exist, take their mean
                for col in ("perp_mean_deg", "updown_mean_deg", "leftright_mean_deg", "tilt_angle", "perpendicularity"):
                    if col in sub_p.columns:
                        event[f"Pen_{col.capitalize()}"] = float(np.nanmean(sub_p[col].values))
                        
                # Otherwise, calculate tool tilt from raw pen tip/tail vectors if available
                if "pen_tip_x" in sub_p.columns and "pen_tail_x" in sub_p.columns:
                    tip = np.nanmean(sub_p[["pen_tip_x", "pen_tip_y", "pen_tip_z"]].values, axis=0)
                    tail = np.nanmean(sub_p[["pen_tail_x", "pen_tail_y", "pen_tail_z"]].values, axis=0)
                    event["Pen_Vertical_Tilt_Angle"] = vector_angle_with_vertical(tail - tip)

            # --- 2. EXTRACT ROBUST BODY ANGLES (WITH TIMESTAMPS OVERLAP DEFENSE) ---
            if not df_body.empty and "t_s" in df_body.columns:
                sub_b = df_body[(df_body["t_s"] >= s - 0.1) & (df_body["t_s"] <= e + 0.1)] # 100ms tolerance
                if len(sub_b) > 0:
                    def get_pt(name):
                        if f"{name}_x" not in sub_b.columns: return None
                        vals = sub_b[[f"{name}_x", f"{name}_y", f"{name}_z"]].values
                        return None if np.all(np.isnan(vals)) else np.nanmean(vals, axis=0)

                    for side in ("Left", "Right"):
                        sh, el, wr = get_pt(f"{side}Shoulder"), get_pt(f"{side}Elbow"), get_pt(f"{side}Wrist")
                        hp, kn, an = get_pt(f"{side}Hip"), get_pt(f"{side}Knee"), get_pt(f"{side}Ankle")
                        
                        event[f"Body_{side}_Elbow_Angle"] = angle_between_3_points(sh, el, wr)
                        event[f"Body_{side}_Shoulder_Angle"] = angle_between_3_points(hp, sh, el)
                        event[f"Body_{side}_Hip_Angle"] = angle_between_3_points(sh, hp, kn)
                        event[f"Body_{side}_Knee_Angle"] = angle_between_3_points(hp, kn, an)

            # --- 3. EXTRACT HAND ANGLES & INTERNAL DISTANCES ---
            if not df_hand.empty and "t_s" in df_hand.columns:
                sub_h = df_hand[(df_hand["t_s"] >= s - 0.05) & (df_hand["t_s"] <= e + 0.05)]
                if len(sub_h) > 0:
                    for side in ("Left", "Right"):
                        def get_hpt(name):
                            if f"{side}_{name}_x" not in sub_h.columns: return None
                            vals = sub_h[[f"{side}_{name}_x", f"{side}_{name}_y", f"{side}_{name}_z"]].values
                            return None if np.all(np.isnan(vals)) else np.nanmean(vals, axis=0)

                        wrist = get_hpt("HandWristRoot")
                        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                            j1, j2, j3 = get_hpt(f"Hand{finger}1"), get_hpt(f"Hand{finger}2"), get_hpt(f"Hand{finger}3")
                            tip = get_hpt(f"Hand{finger}Tip")
                            
                            event[f"Hand_{side}_{finger}_MCP_Angle"] = angle_between_3_points(wrist, j1, j2)
                            event[f"Hand_{side}_{finger}_PIP_Angle"] = angle_between_3_points(j1, j2, j3)
                            event[f"Hand_{side}_{finger}_DIP_Angle"] = angle_between_3_points(j2, j3, tip)
                        
                        t_tip, i_tip = get_hpt("HandThumbTip"), get_hpt("HandIndexTip")
                        event[f"Hand_{side}_Aperture_Dist"] = dist_between(t_tip, i_tip)

            all_events.append(event)
            
    if not all_events:
        sys.exit("Error: 0 Place events extracted! Halting script.")

    df = pd.DataFrame(all_events).dropna(axis=1, how='all')
    # Identify all analytical feature columns (Pen, Body, and Hand)
    feature_cols = sorted([c for c in df.columns if c.startswith("Pen_") or c.startswith("Body_") or c.startswith("Hand_")])
    
    print(f"Extracted {len(df)} Place Events across {len(feature_cols)} systematic kinematic & angle features.")
    stratum_counts = df["Stratum"].value_counts().to_dict()
    print(f"Stratum Distribution: {stratum_counts}")
    return df, feature_cols


# =========================================================================== #
# 4. STRATIFIED KRUSKAL-WALLIS STATISTICAL ENGINE
# =========================================================================== #

def evaluate_stratified_differences(df: pd.DataFrame, feature_cols: list, out_dir: Path):
    print(f"\nFull p-value matrix (prototype factors × angle metrics × stratum):")
    
    factors = ["Length", "Size", "Weight", "Angle"]
    strata = ["High", "Medium", "Low", "Unknown"]
    
    print(f"  {'Factor':<10} {'Metric':<35} {'High':>8} {'Medium':>8} {'Low':>8} {'Unknown':>8}")
    print(f"  {'-'*10} {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    full_records = []
    
    for factor in factors:
        if factor not in df.columns or df[factor].nunique() <= 1:
            continue
            
        for metric in feature_cols:
            row_str = f"  {factor:<10} {metric[:35]:<35}"
            record = {"Factor": factor, "Metric": metric}
            
            for s in strata:
                sub_df = df[df["Stratum"] == s]
                groups = [grp[metric].dropna().values for _, grp in sub_df.groupby(factor) if len(grp[metric].dropna()) > 0]
                
                if len(groups) >= 2 and sum(len(g) for g in groups) > len(groups):
                    try:
                        h_stat, p_val = stats.kruskal(*groups)
                        n_tot = sum(len(g) for g in groups)
                        e_sq = max(0.0, float(h_stat / (n_tot - 1))) if n_tot > 1 else 0.0
                        
                        if p_val < 0.0005: p_str = "0.000*"
                        elif p_val < 0.05: p_str = f"{p_val:.3f}*"
                        else: p_str = f"{p_val:.3f} "
                    except ValueError:
                        p_str, e_sq, p_val = "n/a", np.nan, np.nan
                else:
                    p_str, e_sq, p_val = "n/a", np.nan, np.nan
                    
                row_str += f" {p_str:>8}"
                record[f"P_Value_{s}"] = p_val
                record[f"E_R2_{s}"] = e_sq
                record[f"Sig_{s}"] = "*" if (not np.isnan(p_val) and p_val < 0.05) else ""
                
            print(row_str)
            full_records.append(record)
            
    df_results = pd.DataFrame(full_records)
    out_file = out_dir / "stratified_angle_differences_matrix.csv"
    df_results.to_csv(out_file, index=False)
    print(f"\nComplete stratified statistical ledger saved to:\n  -> {out_file}")
    return df_results


# =========================================================================== #
# 5. COMMAND LINE INTERFACE
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True, help="Root directory containing trial folders and metrics/ CSVs")
    ap.add_argument("--participants", type=str, default=None, help="Comma-separated list of participant IDs to include")
    args = ap.parse_args()

    if not args.landmarks_root.is_dir():
        sys.exit(f"Error: Invalid directory: {args.landmarks_root}")
        
    df, feature_cols = extract_all_kinematics_and_angles(args.landmarks_root, args.participants)
    if df.empty:
        sys.exit("No data extracted.")
        
    out_dir = args.landmarks_root / "metrics" / "feature_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_stratified_differences(df, feature_cols, out_dir)

if __name__ == "__main__":
    main()