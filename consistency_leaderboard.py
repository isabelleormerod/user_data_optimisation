#!/usr/bin/env python3
r"""
consistency_leaderboard.py - Between-Participant Movement Consistency Back-End

A calibration-free scoring engine that evaluates prototype ergonomics based on 
between-participant agreement. It measures how tightly different participants 
converge on the same physical movement when using a specific prototype.

TWO SEPARATE CHANNELS (Different units, not pooled):
  1. POSTURE (Degrees): Between-participant dispersion of the warped angle 
     skeleton. Measures whether people adopt the SAME POSTURES.
  2. TIMING (% of duration): Between-participant dispersion of the SRVF warping 
     functions. Measures whether people PHASE the movement the same way in time.

USAGE:
  python consistency_leaderboard.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks

This is a fully standalone script. It reads the pre-extracted `curves.npz` cache, 
performs its own elastic curve registration to separate phase and amplitude, 
and outputs configuration leaderboards per workstation height.
"""

import argparse
import sys
from pathlib import Path

import fdasrsf as fs
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =========================================================================== #
# 1. CACHE LOADING & DYNAMIC FEATURE DISCOVERY
# =========================================================================== #

def load_event_cache(cache_dir: Path):
    """Natively reads the pre-extracted curve cache from disk."""
    meta_path = cache_dir / "meta.csv"
    curves_path = cache_dir / "curves.npz"
    info_path = cache_dir / "cache_info.csv"

    if not (meta_path.exists() and curves_path.exists() and info_path.exists()):
        sys.exit(f"ERROR: Cached extraction not found at {cache_dir}.\n"
                 f"Ensure the fpca extraction script has been run first.")

    info = pd.read_csv(info_path)
    n_grid = int(info.iloc[0]["n_grid"])

    meta = pd.read_csv(meta_path).set_index("event_id")
    meta.index.name = None

    npz = np.load(curves_path)
    event_data = {eid: {"ref": npz[f"e{eid}_ref"], "curves": {}} for eid in meta.index}
    
    for key in npz.files:
        if key.endswith("_ref"):
            continue
        eid_str, rest = key.split("_", 1)
        eid = int(eid_str[1:])
        domain, feat = rest.split("__", 1)
        event_data[eid]["curves"][(domain, feat)] = npz[key]

    return meta, event_data, n_grid


def discover_features(event_data: dict) -> dict:
    """Dynamically determines domains and features from the loaded cache."""
    feat_names = {}
    if not event_data:
        return feat_names
    any_eid = next(iter(event_data))
    for domain, feat in event_data[any_eid]["curves"].keys():
        feat_names.setdefault(domain, []).append(feat)
    return feat_names


# =========================================================================== #
# 2. ELASTIC REGISTRATION (SRVF)
# =========================================================================== #

def register_stratum(event_ids, event_data, n_grid):
    """Aligns curves temporally to separate phase (timing) from amplitude (posture)."""
    grid = np.linspace(0.0, 1.0, n_grid)
    F = np.array([event_data[eid]["ref"] for eid in event_ids]).T
    
    obj = fs.fdawarp(F, grid)
    obj.srsf_align(parallel=False, MaxItr=15, verbose=False)
    gam = obj.gam

    registered = {}
    gam_by_eid = {}
    
    for i, eid in enumerate(event_ids):
        gam_i = gam[:, i]
        gam_by_eid[eid] = gam_i
        reg = {}
        for key, curve in event_data[eid]["curves"].items():
            if np.isnan(curve).all():
                reg[key] = curve
                continue
            interp = interp1d(grid, curve, kind="linear", bounds_error=False, fill_value=(curve[0], curve[-1]))
            reg[key] = interp(gam_i)
        registered[eid] = reg
        
    return registered, gam_by_eid


# =========================================================================== #
# 3. CONSISTENCY MATH ENGINE
# =========================================================================== #

def _between_participant_sd(templates):
    """Returns RMS-over-grid of the between-participant standard deviation."""
    T = templates.shape[1]
    sd_t = np.full(T, np.nan)
    for t in range(T):
        col = templates[:, t]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            sd_t[t] = np.std(col, ddof=1)
    if np.all(np.isnan(sd_t)):
        return np.nan
    return float(np.sqrt(np.nanmean(sd_t ** 2)))


def _config_dispersion(get_curve, by_pid, n_grid):
    """Calculates between-participant dispersion for a specific prototype."""
    templates = []
    for events in by_pid.values():
        stack = [np.asarray(get_curve(e), float) for e in events if get_curve(e) is not None]
        templates.append(np.nanmean(np.vstack(stack), axis=0) if stack else np.full(n_grid, np.nan))
    templates = np.vstack(templates)
    return _between_participant_sd(templates) if templates.shape[0] >= 2 else np.nan


def _score_0_100(disp_series):
    """Standardizes dispersion into a 0-100 score (100 = most convergent/consistent)."""
    out = pd.Series(np.nan, index=disp_series.index)
    valid = disp_series.notna()
    if valid.sum() >= 1:
        v = disp_series[valid]
        lo, hi = v.min(), v.max()
        out[valid] = 100.0 if (hi - lo) < 1e-9 else 100.0 * (hi - disp_series[valid]) / (hi - lo)
    return out


def generate_leaderboard(registered, ids, sub_meta, feat_names, gam_by_eid, n_grid, min_participants=2):
    """Constructs the consistency leaderboard dataframe."""
    configs = sub_meta.loc[ids, "Prototype_Config"].unique()
    rows, feat_rows = [], []

    for cfg in configs:
        cfg_ids = [e for e in ids if sub_meta.loc[e, "Prototype_Config"] == cfg]
        by_pid = {}
        for e in cfg_ids:
            by_pid.setdefault(sub_meta.loc[e, "participant"], []).append(e)
            
        n_part = len(by_pid)
        enough = n_part >= min_participants

        # ---- POSTURE Channel (Degrees) ----
        domain_disp = {}
        for domain, feats in feat_names.items():
            feat_sds = []
            for f in feats:
                fsd = _config_dispersion(lambda e: registered[e].get((domain, f)), by_pid, n_grid) if enough else np.nan
                feat_sds.append(fsd)
                feat_rows.append({"Prototype_Config": cfg, "domain": domain, "feature": f,
                                  "between_pp_sd_deg": fsd, "n_participants": n_part})
                
            feat_sds = np.array(feat_sds, float)
            domain_disp[domain] = float(np.sqrt(np.nanmean(feat_sds ** 2))) if not np.all(np.isnan(feat_sds)) else np.nan
            
        dvals = np.array([domain_disp.get(d, np.nan) for d in feat_names], float)
        posture = float(np.nanmean(dvals)) if not np.all(np.isnan(dvals)) else np.nan

        # ---- TIMING Channel (% of duration) ----
        timing = np.nan
        if gam_by_eid and enough:
            t = _config_dispersion(lambda e: gam_by_eid.get(e), by_pid, n_grid)
            timing = t * 100.0 if not np.isnan(t) else np.nan

        rows.append({
            "Prototype_Config": cfg, "N_Participants": n_part, "N_Events": len(cfg_ids),
            "Posture_Disp_deg": posture, "Timing_Disp_pct": timing,
            "pen_disp_deg": domain_disp.get("pen"), "body_disp_deg": domain_disp.get("body"),
            "hand_disp_deg": domain_disp.get("hand"),
        })

    lb = pd.DataFrame(rows)
    lb["Posture_Consistency_Score"] = _score_0_100(lb["Posture_Disp_deg"])
    lb["Timing_Consistency_Score"] = _score_0_100(lb["Timing_Disp_pct"])
    lb["Posture_Rank"] = lb["Posture_Disp_deg"].rank(method="min").where(lb["Posture_Disp_deg"].notna())
    lb["Timing_Rank"] = lb["Timing_Disp_pct"].rank(method="min").where(lb["Timing_Disp_pct"].notna())

    lb = lb.sort_values("Posture_Disp_deg", ascending=True, na_position="last").reset_index(drop=True)
    
    cols = ["Prototype_Config", "Posture_Consistency_Score", "Posture_Disp_deg", "Posture_Rank",
            "Timing_Consistency_Score", "Timing_Disp_pct", "Timing_Rank",
            "N_Participants", "N_Events", "pen_disp_deg", "body_disp_deg", "hand_disp_deg"]
    
    return lb[[c for c in cols if c in lb.columns]], pd.DataFrame(feat_rows)


def print_ascii_leaderboard(lb, height, top_n=10):
    print(f"\n{'='*88}\nCONSISTENCY LEADERBOARD -- {height.upper()} "
          f"(between-participant convergence; lower dispersion = better)\n{'='*88}")
    print(f" {'Prototype Configuration':<32} {'PostCons':>8} {'PostDeg':>7} "
          f"{'TimeCons':>8} {'Time%':>6} {'Np':>3} {'Ne':>4}")
    print(f" {'-'*32} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*3} {'-'*4}")
    
    for _, r in lb.head(top_n).iterrows():
        pc = f"{r['Posture_Consistency_Score']:>8.1f}" if pd.notna(r['Posture_Consistency_Score']) else f"{'n/a':>8}"
        pd_ = f"{r['Posture_Disp_deg']:>7.1f}" if pd.notna(r['Posture_Disp_deg']) else f"{'n/a':>7}"
        tc = f"{r['Timing_Consistency_Score']:>8.1f}" if pd.notna(r['Timing_Consistency_Score']) else f"{'n/a':>8}"
        tp = f"{r['Timing_Disp_pct']:>6.1f}" if pd.notna(r['Timing_Disp_pct']) else f"{'n/a':>6}"
        print(f" {r['Prototype_Config']:<32} {pc} {pd_} {tc} {tp} "
              f"{int(r['N_Participants']):>3} {int(r['N_Events']):>4}")
              
    print(" *(PostCons/TimeCons 0-100, 100=most convergent. PostDeg = between-participant SD in degrees;")
    print("   Time% = between-participant SD of the warping functions, % of normalised duration. Separate")
    print("   channels: posture = which postures, timing = when. Not pooled -- different units.)*")


# =========================================================================== #
# 4. COMMAND LINE INTERFACE
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Standalone Consistency Leaderboard Engine")
    ap.add_argument("--landmarks-root", type=Path, required=True, 
                    help="Path to Participant_Landmarks root directory")
    ap.add_argument("--cache-dir", type=Path, default=None, 
                    help="Explicit path to fpca_cache (defaults to <landmarks-root>/metrics/fpca_cache)")
    ap.add_argument("--out-dir", type=Path, default=None, 
                    help="Explicit path to output rankings (defaults to <landmarks-root>/metrics/prototype_rankings)")
    args = ap.parse_args()

    cache_dir = args.cache_dir or (args.landmarks_root / "metrics" / "fpca_cache")
    out_dir = args.out_dir or (args.landmarks_root / "metrics" / "prototype_rankings")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nLOADING CACHE & COMPUTING CONSISTENCY\n{'='*70}")
    
    meta, event_data, n_grid = load_event_cache(cache_dir)
    print(f"Loaded {len(meta)} place events from {cache_dir}")
    
    feat_names = discover_features(event_data)
    print(f"Discovered features across {len(feat_names)} domains: {list(feat_names.keys())}")

    for height in ["High", "Medium", "Low"]:
        ids = meta.index[meta["height"] == height].tolist()
        if len(ids) < 4:
            continue

        print(f"\nProcessing {height.upper()} stratum ({len(ids)} events)...")
        
        # 1. Align curves
        registered, gam_by_eid = register_stratum(ids, event_data, n_grid)
        
        # 2. Compute Leaderboards
        sub_meta = meta.loc[ids]
        lb, feat_df = generate_leaderboard(
            registered, ids, sub_meta, feat_names, gam_by_eid, n_grid
        )

        # 3. Output
        print_ascii_leaderboard(lb, height)
        
        lb_file = out_dir / f"consistency_leaderboard_{height}.csv"
        feat_file = out_dir / f"consistency_per_feature_{height}.csv"
        lb.to_csv(lb_file, index=False)
        feat_df.to_csv(feat_file, index=False)
        print(f"\nSaved:\n -> {lb_file.name}\n -> {feat_file.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()