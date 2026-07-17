#!/usr/bin/env python3
"""
evaluate_difference.py

A unified master script for Ergonomic Posture Extraction and Statistical Comparison.
Combines pen performance metrics and streamlined posture/hand metrics (REBA, Preferred
Grip Span Model, and SPARC movement smoothness) into a single analytical pipeline.

STATISTICAL ENGINE (mixed-effects, MAIN EFFECTS ONLY):
  Within each height stratum, for every metric, ONE mixed model is fit with all
  four prototype factors and a participant random intercept:

      metric ~ C(Length) + C(Size) + C(Weight) + C(Angle) + (1 | participant)

  Each factor gets a Wald test read off that single fit (via patsy's per-term
  coefficient slices). This deliberately does NOT include interaction terms --
  an earlier version added all pairwise interactions, but that made several
  models (especially the coarser REBA/grip metrics) singular or unstable, which
  is not worth the opacity it introduces. Main-effects-only mixed models are the
  simple, robust, trustworthy baseline: they fit cleanly on this design and
  directly answer "does this factor affect this metric, within this height,
  once we account for repeated Place events per participant." A pooled model
  per metric (across all heights, height added as a fifth factor) is also fit.

  Why mixed effects at all: multiple Place events per participant are not
  independent (same person, same trial); a random intercept for participant
  models that dependence, avoiding the pseudoreplication of treating every
  Place event as an independent sample (which a plain Mann-Whitney/Kruskal-Wallis
  screen over raw events would do).

MODES OF OPERATION:
  1. Compare Only (Default):
     python evaluate_difference.py --mode compare --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks

  2. Extract Only (Re-run posture extraction with simplified SPARC & Grip Comfort models):
     python evaluate_difference.py --mode extract --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks

  3. End-to-End Pipeline (Extract features then run statistical comparisons & plots):
     python evaluate_difference.py --mode all --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks

  Optional Arguments:
     --pen-csv path/to/place_metrics.csv       (Override default pen metrics path)
     --posture-csv path/to/posture.csv         (Override default posture features path)
     --participants P001,P002                  (Filter by specific participant IDs)
     --no-graphs                               (Skip generating box plots)
     --no-stratify                             (Skip height-stratified mixed models)
     --stratify-by height                      (Column to stratify by, default: height)
     --min-n 8                                 (Min rows required to attempt a model fit)
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq

warnings.simplefilter("ignore")
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Restore native discovery modules to ensure labelled files are found accurately
try:
    from utils.discovery import find_labelled_pen, iter_trials_labelled
    from utils.params import parse_participant_filter
except ImportError:
    sys.exit("Error: Could not import 'utils.discovery' or 'utils.params'. Ensure this script is run from within your repo root or that the utils folder is in your PYTHONPATH.")


# =========================================================================== #
# SECTION 1: STREAMLINED METRICS REGISTRY & PARAMETERS
# =========================================================================== #

PEN_METRICS = [
    ("duration_s",          "Duration",               "s"),
    ("perp_mean_deg",       "Perpendicularity (mean)","deg"),
    ("leftright_mean_deg",  "Left/right tilt (mean)", "deg"),
    ("updown_mean_deg",     "Up/down tilt (mean)",    "deg"),
    ("pos_jitter_mm",       "Positional jitter",      "mm"),
    ("ang_jitter_deg",      "Angular jitter",         "deg"),
]

POSTURE_METRICS = [
    # 1. Full-Body Ergonomic Risk (REBA)
    ("reba_score_a",              "REBA Score A (Trunk/Neck/Legs)",    "score"),
    ("reba_score_b_right",        "REBA Score B (Right Arm/Wrist)",    "score"),
    ("reba_score_b_left",         "REBA Score B (Left Arm/Wrist)",     "score"),
    ("reba_grand_right",          "REBA Grand Score (Right)",          "score"),
    ("reba_grand_left",           "REBA Grand Score (Left)",           "score"),

    # 2. Key Macro-Angles (Body Frame)
    ("trunk_flex_mean",           "Trunk Flexion",                     "deg"),
    ("neck_flex_mean",            "Neck Flexion",                      "deg"),
    ("reach_ratio_mean",          "Reach Ratio",                       "ratio"),
    ("wrist_elevation_m_mean",    "Wrist Elevation Above Shoulder",    "m"),

    # 3. Grasp Comfort (Preferred Grip Span Model)
    ("right_grip_span_dev_mm",    "R Grip Span Deviation (from 50mm)", "mm"),
    ("left_grip_span_dev_mm",     "L Grip Span Deviation (from 50mm)", "mm"),
    ("right_grip_comfort_score",  "R Grip Comfort Index",              "score"),
    ("left_grip_comfort_score",   "L Grip Comfort Index",              "score"),

    # 4. Movement Smoothness & Micro-Jitter (SPARC)
    ("right_sparc_linear",        "R Linear Smoothness (SPARC)",       "val"),
    ("left_sparc_linear",         "L Linear Smoothness (SPARC)",       "val"),
    ("right_sparc_angular",       "R Angular Smoothness (SPARC)",      "val"),
    ("left_sparc_angular",        "L Angular Smoothness (SPARC)",      "val"),
]

ALL_METRICS   = PEN_METRICS + POSTURE_METRICS
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
ALL_FACTORS   = ["participant", "height"] + PARAM_FACTORS

BODY_UP  = np.array([-1.0, 0.0, 0.0])  # MediaPipe body: 'up' is -X
HAND_UP  = np.array([0.0, 1.0, 0.0])   # Quest hand: 'up' is +Y
CONF_MIN = 0.3                         # MediaPipe landmark confidence threshold


# =========================================================================== #
# SECTION 2: MATHEMATICAL MODELS (SPARC, GRIP SPAN, REBA)
# =========================================================================== #

def compute_sparc(speed_profile: np.ndarray, fs: float, pad_level: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """
    Compute Spectral Arc Length (SPARC) on a 1D speed or angular velocity profile.
    Values closer to 0 indicate smooth movement; more negative values indicate jitter/tremor.
    """
    if len(speed_profile) < 10 or fs <= 0:
        return np.nan
    n_fft = int(2 ** np.ceil(np.log2(len(speed_profile)) + pad_level))
    V_mags = np.abs(rfft(speed_profile, n=n_fft))
    freqs = rfftfreq(n_fft, d=1.0/fs)
    v_max = np.max(V_mags)
    if v_max < 1e-9:
        return np.nan
    V_norm = V_mags / v_max
    valid_idx = np.where((freqs <= fc) & (V_norm >= amp_th))[0]
    if len(valid_idx) < 2:
        return np.nan
    freqs_c = freqs[valid_idx]
    V_norm_c = V_norm[valid_idx]
    omega_norm = freqs_c / fc
    d_omega = np.diff(omega_norm)
    d_V = np.diff(V_norm_c)
    arc_length = np.sum(np.sqrt(d_omega**2 + d_V**2))
    return float(-arc_length)


def grip_comfort_model(aperture_m: float, optimal_span_m: float = 0.050, max_tol_m: float = 0.035) -> tuple:
    """
    Preferred Grip Span Model. Evaluates cylinder grasp against an optimal 50mm span.
    Returns: (deviation_mm, comfort_index_0_to_100)
    """
    if aperture_m is None or np.isnan(aperture_m):
        return np.nan, np.nan
    dev_m = abs(aperture_m - optimal_span_m)
    dev_mm = dev_m * 1000.0
    penalty = (dev_m / max_tol_m) ** 2
    comfort_score = max(0.0, min(100.0, (1.0 - penalty) * 100.0))
    return float(dev_mm), float(comfort_score)


# Canonical REBA Lookup Tables (Hignett & McAtamney 2000)
REBA_TABLE_A = {
    1: {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [3, 3, 5, 6]},
    2: {1: [2, 3, 4, 5], 2: [3, 4, 5, 6], 3: [4, 5, 6, 7]},
    3: {1: [2, 4, 5, 6], 2: [4, 5, 6, 7], 3: [5, 6, 7, 8]},
    4: {1: [3, 5, 6, 7], 2: [5, 6, 7, 8], 3: [6, 7, 8, 9]},
    5: {1: [4, 6, 7, 8], 2: [6, 7, 8, 9], 3: [7, 8, 9, 9]},
}
REBA_TABLE_B = {
    1: {1: [1, 2, 2], 2: [1, 2, 3]},
    2: {1: [1, 2, 3], 2: [2, 3, 4]},
    3: {1: [3, 4, 5], 2: [4, 5, 5]},
    4: {1: [4, 5, 5], 2: [5, 6, 7]},
    5: {1: [6, 7, 8], 2: [7, 8, 8]},
    6: {1: [7, 8, 8], 2: [8, 9, 9]},
}
REBA_TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]

def reba_table_a(trunk, neck, legs):
    return REBA_TABLE_A[min(max(trunk, 1), 5)][min(max(neck, 1), 3)][min(max(legs, 1), 4) - 1]

def reba_table_b(upper, lower, wrist):
    return REBA_TABLE_B[min(max(upper, 1), 6)][min(max(lower, 1), 2)][min(max(wrist, 1), 3) - 1]

def reba_table_c(score_a, score_b):
    return REBA_TABLE_C[min(max(int(round(score_a)), 1), 12) - 1][min(max(int(round(score_b)), 1), 12) - 1]

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

def quat_angular_distance(q1, q2):
    return np.degrees(2.0 * np.arccos(min(abs(float(np.dot(q1, q2))), 1.0)))


# =========================================================================== #
# SECTION 3: FEATURE EXTRACTION ENGINE (`--mode extract`)
# =========================================================================== #

def extract_posture_features(root_dir: Path, participants_str: str = None):
    """Reads raw trial CSVs via native discovery and extracts REBA, Preferred Grip Span, and SPARC metrics."""
    print(f"\n{'='*65}\nSTARTING POSTURE & HAND FEATURE EXTRACTION\n{'='*65}")

    pfilter = parse_participant_filter(participants_str)
    trials = list(iter_trials_labelled(root_dir, pfilter))

    if not trials:
        sys.exit("Error: No trial folders with labelled pen files found! Check your --landmarks-root path.")

    print(f"Processing {len(trials)} trial(s) located via native discovery...")
    all_rows = []

    for stem, pid, trial_dir in trials:
        pen_path  = find_labelled_pen(trial_dir, stem)
        body_path = trial_dir / f"{stem}_body.csv"
        hand_path = trial_dir / f"{stem}_hand.csv"

        if not pen_path or not pen_path.is_file():
            print(f"  [WARN] {stem}: Labelled pen file not found.")
            continue

        df_pen = pd.read_csv(pen_path)
        if "Place" not in df_pen.columns or "t_s" not in df_pen.columns:
            print(f"  [WARN] {stem}: 'Place' or 't_s' column missing in {pen_path.name}")
            continue

        # Extract Place runs
        places, in_run, start, prev_t = [], False, None, None
        for _, r in df_pen.iterrows():
            t = r["t_s"]
            flag = str(r["Place"]).strip() in ("1", "1.0", "True", "true")
            if flag and not in_run:
                in_run, start = True, t
            elif not flag and in_run:
                in_run = False; places.append((start, prev_t))
            prev_t = t
        if in_run: places.append((start, prev_t))

        if not places:
            print(f"  [WARN] {stem}: 0 Place events identified in pen file.")
            continue

        # Extract height stratum per timestamp
        height_runs = {}
        for h in ("High", "Medium", "Low"):
            if h in df_pen.columns:
                runs, in_run, start, prev_t = [], False, None, None
                for _, r in df_pen.iterrows():
                    t, flag = r["t_s"], str(r[h]).strip() in ("1", "1.0", "True", "true")
                    if flag and not in_run: in_run, start = True, t
                    elif not flag and in_run: in_run = False; runs.append((start, prev_t))
                    prev_t = t
                if in_run: runs.append((start, prev_t))
                height_runs[h] = runs

        def get_height(t_mid):
            for h, runs in height_runs.items():
                for s, e in runs:
                    if s <= t_mid <= e: return h
            return "Unknown"

        df_body = pd.read_csv(body_path) if body_path.is_file() else pd.DataFrame()
        df_hand = pd.read_csv(hand_path) if hand_path.is_file() else pd.DataFrame()

        # Compute each place event's height BEFORE assigning place_index, and
        # number place_index WITHIN each height (resetting to 1 per height) --
        # matching metrics.py's convention exactly (df.groupby("height").cumcount()+1).
        # Previously this counted continuously across the WHOLE trial regardless
        # of height, so pen and posture tables numbered the same physical place
        # event differently unless that height happened to be first in the
        # session's randomised order -- causing the pen/posture merge to
        # silently mismatch or duplicate rows for any height tested 2nd or 3rd.
        heights_for_places = [get_height((s + e) / 2) for s, e in places]
        height_counters = {}
        place_indices = []
        for h in heights_for_places:
            height_counters[h] = height_counters.get(h, 0) + 1
            place_indices.append(height_counters[h])

        for (i, (s, e)), height, place_idx in zip(enumerate(places, 1), heights_for_places, place_indices):
            dur = e - s
            row = {"participant": pid, "trial": stem, "place_index": place_idx,
                   "height": height, "start_t_s": round(s, 4),
                   "stop_t_s": round(e, 4), "duration_s": round(dur, 4)}

            # --- Hand Extraction (Grip Comfort & SPARC) ---
            if not df_hand.empty and "t_s" in df_hand.columns:
                sub_h = df_hand[(df_hand["t_s"] >= s) & (df_hand["t_s"] <= e)]
                if len(sub_h) >= 4:
                    fs = float(1.0 / np.median(np.diff(sub_h["t_s"]))) if len(sub_h) > 1 else 30.0
                    for side in ("Left", "Right"):
                        pref = side.lower()
                        if f"{side}_HandThumbTip_x" in sub_h.columns and f"{side}_HandIndexTip_x" in sub_h.columns:
                            tt = sub_h[[f"{side}_HandThumbTip_{ax}" for ax in ("x","y","z")]].values
                            it = sub_h[[f"{side}_HandIndexTip_{ax}" for ax in ("x","y","z")]].values
                            apertures = np.linalg.norm(tt - it, axis=1)
                            # Silences RuntimeWarning when an idle hand is untracked (all NaNs)
                            m_ap = float(np.nanmean(apertures)) if not np.all(np.isnan(apertures)) else np.nan
                            dev_mm, comfort = grip_comfort_model(m_ap)
                            row[f"{pref}_aperture_mean"] = m_ap
                            row[f"{pref}_grip_span_dev_mm"] = dev_mm
                            row[f"{pref}_grip_comfort_score"] = comfort

                        if f"{side}_HandWristRoot_x" in sub_h.columns:
                            wp = sub_h[[f"{side}_HandWristRoot_{ax}" for ax in ("x","y","z")]].values
                            if len(wp) >= 10:
                                lin_vel = np.gradient(wp, sub_h["t_s"].values[:len(wp)], axis=0)
                                row[f"{pref}_sparc_linear"] = compute_sparc(np.linalg.norm(lin_vel, axis=1), fs=fs, fc=10.0)
                            if f"{side}_HandWristRoot_qw" in sub_h.columns:
                                wq = sub_h[[f"{side}_HandWristRoot_{ax}" for ax in ("qw","qx","qy","qz")]].values
                                if len(wq) >= 10:
                                    ang_dists = [quat_angular_distance(wq[k], wq[k-1]) for k in range(1, len(wq))]
                                    dt = np.diff(sub_h["t_s"].values[:len(wq)])
                                    dt = np.where(dt <= 0, 1.0/fs, dt)
                                    row[f"{pref}_sparc_angular"] = compute_sparc(np.array(ang_dists)/dt, fs=fs, fc=6.0)

            # --- Body Extraction (REBA & Macro Angles) ---
            if not df_body.empty and "t_s" in df_body.columns:
                sub_b = df_body[(df_body["t_s"] >= s) & (df_body["t_s"] <= e)]
                if len(sub_b) >= 1:
                    def get_j(name):
                        if f"{name}_x" not in sub_b.columns: return None
                        pts = sub_b[[f"{name}_{ax}" for ax in ("x","y","z")]].values
                        return np.nanmean(pts, axis=0)

                    ls, rs, lh, rh = get_j("LeftShoulder"), get_j("RightShoulder"), get_j("LeftHip"), get_j("RightHip")
                    le, re = get_j("LeftEar"), get_j("RightEar")

                    trunk_flex = angle_between((ls + rs)/2 - (lh + rh)/2, BODY_UP) if ls is not None and lh is not None else np.nan
                    neck_flex  = angle_between((le + re)/2 - (ls + rs)/2, BODY_UP) - trunk_flex if le is not None and ls is not None else np.nan

                    row["trunk_flex_mean"] = trunk_flex
                    row["neck_flex_mean"]  = neck_flex

                    # REBA Score A
                    ts = 1 if abs(trunk_flex) < 5 else (2 if abs(trunk_flex) <= 20 else (3 if abs(trunk_flex) <= 60 else 4))
                    ns = 1 if (0 <= neck_flex <= 20) else 2
                    score_a = reba_table_a(ts, ns, 1) + 0  # 0 assumed load
                    row["reba_score_a"] = score_a

                    for side in ("Left", "Right"):
                        pref = side.lower()
                        sh, el, wr = get_j(f"{side}Shoulder"), get_j(f"{side}Elbow"), get_j(f"{side}Wrist")
                        if sh is not None and wr is not None:
                            row["wrist_elevation_m_mean"] = float(np.dot(sh - wr, BODY_UP))
                            arm_len = np.linalg.norm(el - sh) + np.linalg.norm(wr - el) if el is not None else np.nan
                            row["reach_ratio_mean"] = float(np.linalg.norm(wr - sh) / arm_len) if arm_len > 1e-6 else np.nan

                        if sh is not None and el is not None:
                            uf = angle_between(el - sh, -(((ls + rs)/2 - (lh + rh)/2) if ls is not None else BODY_UP))
                            ef = angle_between(sh - el, wr - el) if wr is not None else 90.0
                            ua_s = 1 if abs(uf) <= 20 else (2 if uf <= 45 else (3 if uf <= 90 else 4))
                            la_s = 1 if (60 <= ef <= 100) else 2
                            score_b = reba_table_b(ua_s, la_s, 1) + 0  # 1 wrist neutral, 0 coupling
                            row[f"reba_score_b_{pref}"] = score_b
                            row[f"reba_grand_{pref}"] = reba_table_c(score_a, score_b) + (1 if dur > 60.0 else 0)

            all_rows.append(row)
        print(f"  [{len(places):>4}] {stem}: Extracted {len(places)} Place event(s)")

    if not all_rows:
        sys.exit("Error: 0 Place events extracted across all trials! Halting script to prevent overwriting CSVs with empty data.")

    df_out = pd.DataFrame(all_rows)
    out_dir = root_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "posture_features_combined.csv"
    df_out.to_csv(out_file, index=False)
    print(f"\nExtraction Complete! Successfully saved {len(df_out)} rows to {out_file}")
    return df_out


# =========================================================================== #
# SECTION 4: STATISTICAL COMPARISON & PLOTTING (`--mode compare`)
#   Mixed-effects models, MAIN EFFECTS ONLY (see module docstring for why the
#   interaction-term version was dropped in favour of this simpler, robust one).
# =========================================================================== #

def parse_params(trial: str) -> dict:
    """Parses Length/Size/Weight/Angle from a trial name. Weight is left as None
    (not silently bucketed into a third category) if it doesn't cleanly match
    'Not_weighted' or 'Front_weighted' -- see add_parameter_columns, which
    reports and quarantines any trial that fails to parse cleanly, rather than
    letting an inconsistently-named trial silently form a spurious third Weight
    level that would corrupt every downstream model and verdict built on it."""
    out = {k: None for k in PARAM_FACTORS}
    tokens = trial.split("_")
    joined = "_".join(tokens)
    if "Not_weighted" in joined: out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined: out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long", "Short"): out["Length"] = tok
        elif tok in ("Large", "Small"): out["Size"] = tok
    return out

def add_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses PARAM_FACTORS from the trial name, then LOUDLY reports and
    quarantines (drops) any row where a factor could not be cleanly determined
    -- rather than the previous behaviour of silently assigning a fallback
    value that would appear as a spurious extra category in every downstream
    model, test, and verdict. A naming inconsistency should be visible and
    fixable at load time, not discovered three analysis stages later as an
    unexplained statistical anomaly."""
    parsed = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = parsed[c]

    print("\nObserved factor levels (sanity check -- each should show exactly the "
          "expected set, with no unexpected extra category):")
    for c in PARAM_FACTORS:
        print(f"  {c}: {sorted(df[c].dropna().unique().tolist(), key=str)}")

    bad_mask = df[PARAM_FACTORS].isna().any(axis=1)
    if bad_mask.any():
        bad_trials = df.loc[bad_mask, "trial"].unique()
        print(f"\n  [WARN] {bad_mask.sum()} row(s) across {len(bad_trials)} distinct trial name(s) "
              f"could not be cleanly parsed into Length/Size/Weight/Angle and are being QUARANTINED "
              f"(dropped) rather than risk a spurious extra category:")
        for t in bad_trials[:10]:
            print(f"      '{t}'")
        if len(bad_trials) > 10:
            print(f"      ... and {len(bad_trials) - 10} more (see the full list in df['trial'] for "
                  f"rows where any of Length/Size/Weight/Angle is null)")
        df = df[~bad_mask].copy()
    return df

def available_metrics(df: pd.DataFrame) -> list:
    out = []
    for col, label, unit in ALL_METRICS:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").dropna().nunique() > 1:
            out.append((col, label, unit))
    return out

def group_summary(df: pd.DataFrame, metrics: list, stratum_col: str = None) -> pd.DataFrame:
    """Descriptive means/SDs per factor level -- purely descriptive, independent
    of which statistical test is used, so kept as-is."""
    records = []
    groups = df.groupby(stratum_col, dropna=True) if stratum_col else [("All", df)]
    for stratum, sub_df in groups:
        for factor in PARAM_FACTORS if stratum_col else ALL_FACTORS:
            if factor not in sub_df.columns: continue
            for level, grp in sub_df.groupby(factor, dropna=True):
                for col, label, unit in metrics:
                    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                    if not len(vals): continue
                    rec = {"factor": factor, "level": level, "metric": col, "metric_label": label,
                           "unit": unit, "n": int(len(vals)), "mean": float(vals.mean()),
                           "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "median": float(vals.median())}
                    if stratum_col: rec["stratum"] = stratum
                    records.append(rec)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Mixed-model engine (main effects only)
# --------------------------------------------------------------------------- #
def _fit_mixed_main(data: pd.DataFrame, response: str, factors: list):
    """Fit response ~ C(f1) + C(f2) + ... + (1|participant), main effects only.
    Tries a few optimisers; ConvergenceWarnings suppressed locally (robust even
    if something else in the environment has reset the global warning filters).
    Returns (result or None, factors actually used, failure reason or None)."""
    present = [f for f in factors if data[f].nunique() >= 2]
    if not present:
        return None, present, "no factor has >=2 levels in this subset"
    formula = f"{response} ~ " + " + ".join(f"C({f})" for f in present)
    last_err = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for method in ("lbfgs", "powell", "cg"):
            try:
                res = smf.mixedlm(formula, data, groups=data["participant"]).fit(reml=False, method=method)
                if np.isfinite(res.llf):
                    return res, present, None
                last_err = f"non-finite log-likelihood ({method})"
            except Exception as ex:
                last_err = f"{type(ex).__name__}: {ex}"
    return None, present, (last_err or "unknown fit failure")


def _term_wald(res, factor: str):
    """Joint Wald p-value for one factor's coefficient(s) (tests all levels of
    that factor simultaneously -- used for significance screening)."""
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    if not names:
        return np.nan, np.nan
    b = res.fe_params[names].values
    try:
        V = res.cov_params().loc[names, names].values
        W = float(b @ np.linalg.solve(V, b))
    except Exception:
        return np.nan, len(names)
    p = float(stats.chi2.sf(W, len(names)))
    return p, len(names)


def _term_level_effects(res, factor: str):
    """Per-LEVEL signed coefficients for one factor, each relative to the
    reference level. Necessary for multi-level factors (e.g. Angle, 3 levels ->
    2 coefficients): collapsing to a single 'largest effect' loses which level
    that was and discards the other contrast, which makes it impossible to
    later determine which specific level is preferable. Returns a list of
    dicts: [{level, effect, se}, ...], one per non-reference level.
    Coefficient names from patsy look like 'C(Angle)[T.135]' or
    'C(Angle)[T.135.0]' (float-coded) -- the level label is parsed out of the
    '[T. ... ]' bracket."""
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    out = []
    for n in names:
        m = re.search(r"\[T\.(.+?)\]", n)
        level_raw = m.group(1) if m else n
        # tidy up float-coded labels like "135.0" -> "135"
        try:
            level_raw = str(int(float(level_raw)))
        except ValueError:
            pass
        b = float(res.fe_params[n])
        se = float(np.sqrt(res.cov_params().loc[n, n])) if n in res.cov_params().index else np.nan
        out.append({"level": level_raw, "effect": b, "se": se})
    return out


def mixed_tests(df: pd.DataFrame, metrics: list, factors: list,
                stratum_col: str = None, min_n: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Tidy table: [stratum,] factor, level, metric, p_value (joint, per factor),
    effect (signed, per LEVEL), se, n. One mixed model per (metric, stratum), all
    factors as main effects; a joint Wald test per factor for significance
    screening, PLUS one row per individual level so the sign and magnitude of
    every level's own contrast is preserved (needed for multi-level factors like
    Angle -- see _term_level_effects). Binary factors get exactly one level row;
    Angle gets two. Prints a fit-diagnostic per model."""
    records = []
    diag = {"fit": 0, "skip": 0, "fail": 0}
    groups = df.groupby(stratum_col, dropna=True) if stratum_col else [("All", df)]
    for stratum, sub_df in groups:
        for col, label, unit in metrics:
            d = sub_df.dropna(subset=[col, "participant"] + factors).copy().rename(columns={col: "_y"})
            n_ppt = d["participant"].nunique()
            tag = f"[{stratum}] {label}"
            res, present, note = (None, [], None)
            if n_ppt < 2 or len(d) < min_n:
                diag["skip"] += 1
                note = f"skipped: n={len(d)} rows, {n_ppt} participant(s) (need >=2 participants, >={min_n} rows)"
                if verbose: print(f"    {tag}: {note}")
            else:
                res, present, note = _fit_mixed_main(d, "_y", factors)
                if res is None:
                    diag["fail"] += 1
                    if verbose: print(f"    {tag}: fit failed: {note}  (n={len(d)}, {n_ppt} participants)")
                else:
                    diag["fit"] += 1
            for f in factors:
                if res is not None:
                    p, dfree = _term_wald(res, f)
                    level_effects = _term_level_effects(res, f)
                else:
                    p, dfree, level_effects = np.nan, np.nan, []
                if not level_effects:
                    # no data / fit failed / factor absent -- still emit one placeholder row
                    level_effects = [{"level": None, "effect": np.nan, "se": np.nan}]
                for le in level_effects:
                    rec = {"factor": f, "level": le["level"], "metric": col, "metric_label": label,
                           "p_value": p, "df": dfree, "effect": le["effect"], "se": le["se"],
                           "n": len(d), "n_participants": n_ppt,
                           "fit_status": "ok" if res is not None else (note or "failed")}
                    if stratum_col: rec["stratum"] = stratum
                    records.append(rec)
    if verbose:
        total = sum(diag.values())
        print(f"  Model fit summary: {diag['fit']}/{total} fitted OK, "
              f"{diag['skip']}/{total} skipped (too little data), "
              f"{diag['fail']}/{total} failed to converge in any optimiser.")
    return pd.DataFrame(records)


def order_levels(factor, levels):
    orders = {"height": ["High", "Medium", "Low"], "Length": ["Short", "Long"],
              "Size": ["Small", "Large"], "Weight": ["Not_weighted", "Front_weighted"]}
    if factor in orders:
        known = [l for l in orders[factor] if l in levels]
        return known + sorted([l for l in levels if l not in known])
    try: return sorted(levels, key=lambda x: float(x))
    except: return sorted(levels, key=str)

def make_graphs(df, metrics, out_dir, p_lookup, stratum_col=None):
    out_dir.mkdir(parents=True, exist_ok=True); made = []
    strata = sorted(df[stratum_col].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9)) if stratum_col else [None]
    for factor in PARAM_FACTORS if stratum_col else ALL_FACTORS:
        if factor not in df.columns: continue
        all_levels = order_levels(factor, list(df[factor].dropna().unique()))
        if len(all_levels) < 2: continue
        for col, label, unit in metrics:
            fig, axes = plt.subplots(1, len(strata), figsize=(max(5, len(all_levels)*1.5)*len(strata), 5), sharey=True)
            if len(strata) == 1: axes = [axes]
            for ax, stratum in zip(axes, strata):
                sub = df[df[stratum_col] == stratum] if stratum else df
                data, tick_labels = [], []
                for lv in all_levels:
                    vals = pd.to_numeric(sub.loc[sub[factor]==lv, col], errors="coerce").dropna().values
                    if len(vals): data.append(vals); tick_labels.append(f"{lv}\n(n={len(vals)})")
                if len(data) >= 2:
                    ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
                    for i, vals in enumerate(data, 1):
                        jit = (np.random.rand(len(vals))-0.5)*0.15
                        ax.scatter(np.full(len(vals),i)+jit, vals, alpha=0.5, s=18, color="#1f77b4", zorder=3)
                p = p_lookup.get((stratum, factor, col) if stratum else (factor, col), np.nan)
                has_p = p is not None and not pd.isna(p)
                if stratum:
                    title = f"{stratum}   (p={p:.3f})" if has_p else f"{stratum}"
                else:
                    title = f"{label} by {factor}   (p={p:.3f})" if has_p else f"{label} by {factor}"
                ax.set_title(title, fontsize=10); ax.set_xlabel(factor); ax.grid(axis="y", alpha=0.3)
            axes[0].set_ylabel(f"{label} ({unit})")
            if stratum: fig.suptitle(f"{label} by {factor} — stratified by {stratum_col}", fontsize=11)
            fig.tight_layout()
            path = out_dir / (f"by_{factor}_{col}_stratified.png" if stratum else f"by_{factor}_{col}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
            made.append(path)
    return made


def compare_metrics(df_pen: pd.DataFrame, df_posture: pd.DataFrame, out_dir: Path, args):
    """Merges datasets, fits main-effects-only mixed models, outputs tables/plots."""
    print(f"\n{'='*65}\nSTARTING STATISTICAL COMPARISON (mixed-effects, main effects only)\n{'='*65}")
    if not df_pen.empty and not df_posture.empty:
        # Normalise a known naming drift between the two upstream extraction
        # scripts (metrics.py historically used 'trial_num' where this
        # script's own posture extraction uses 'place_index' for the same
        # concept -- see metrics.py for the fix at the source).
        for d in (df_pen, df_posture):
            if "place_index" not in d.columns and "trial_num" in d.columns:
                d.rename(columns={"trial_num": "place_index"}, inplace=True)

        intended_key = ["participant", "trial", "place_index", "height"]
        missing_pen = [c for c in intended_key if c not in df_pen.columns]
        missing_posture = [c for c in intended_key if c not in df_posture.columns]
        if missing_pen or missing_posture:
            sys.exit(f"\nError: the intended merge key {intended_key} is missing column(s) "
                     f"{missing_pen or '[]'} from the pen table and {missing_posture or '[]'} from "
                     f"the posture table. Merging on a SUBSET of this key (e.g. dropping "
                     f"'place_index') would silently produce a non-unique join and a many-to-many "
                     f"merge blowup -- refusing to proceed rather than repeat that bug. Check the "
                     f"column names in place_metrics_combined.csv and posture_features_combined.csv.")
        common_cols = intended_key

        for name, d in (("pen", df_pen), ("posture", df_posture)):
            dup_counts = d.groupby(common_cols).size()
            dups = dup_counts[dup_counts > 1]
            if len(dups):
                print(f"\n  [WARN] {name} table: merge key {common_cols} is NOT unique -- "
                      f"{len(dups)} key combination(s) appear more than once (up to {dups.max()}x).")

        n_pen, n_posture = len(df_pen), len(df_posture)
        df = pd.merge(df_pen, df_posture, on=common_cols, how="outer")
        expected_max = max(n_pen, n_posture)
        if len(df) > 1.2 * expected_max:
            sys.exit(f"\nError: MERGE BLOWUP detected. Pen table has {n_pen} rows, posture table has "
                     f"{n_posture} rows, but the merge produced {len(df)} rows. The merge key "
                     f"{common_cols} is not unique in at least one source table (see [WARN] above) -- "
                     f"refusing to proceed with a corrupted merge. Fix the upstream extraction so "
                     f"(participant, trial, place_index, height) is unique in both files, then rerun.")
        print(f"Merged Pen ({len(df_pen)} rows) and Posture ({len(df_posture)} rows) into {len(df)} Place events.")
    else:
        df = df_pen if not df_pen.empty else df_posture
        print(f"Loaded {len(df)} Place events from single available source.")

    if args.participants:
        keep = {p.strip() for p in args.participants.split(",") if p.strip()}
        df = df[df["participant"].astype(str).isin(keep)].copy()
        if df.empty: sys.exit(f"No rows remaining for participants: {sorted(keep)}")

    df = add_parameter_columns(df)
    metrics = available_metrics(df)
    print(f"Total Participants: {df['participant'].nunique()} | Trials: {df['trial'].nunique()}")
    print(f"Active metrics evaluated: {len(metrics)}")

    if "height" in df.columns:
        counts = df["height"].fillna("<blank>").value_counts()
        print("\nPlace events per height label:")
        for h, n in counts.items():
            flag = "  <-- not High/Medium/Low; could not match this event's timestamp to a labelled height window" \
                  if h not in ("High", "Medium", "Low") else ""
            print(f"  {h:>10}: {n}{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pooled Analysis
    group_summary(df, metrics).to_csv(out_dir / "group_summary.csv", index=False)
    pooled_factors = PARAM_FACTORS + (["height"] if "height" in df.columns else [])
    tests = mixed_tests(df, metrics, pooled_factors, min_n=args.min_n)
    tests.to_csv(out_dir / "stat_tests.csv", index=False)
    print(f"Wrote {out_dir / 'stat_tests.csv'} ({len(tests)} rows)")

    sig = tests[(tests["p_value"].notna()) & (tests["p_value"] < 0.05)]
    if len(sig):
        print("\nPooled Significant Differences (p < 0.05):")
        for _, r in sig.sort_values("p_value").iterrows():
            print(f"  {r['factor']:>12} -> {r['metric_label']:<35} p={r['p_value']:.4f}")

    if not args.no_graphs:
        p_lookup = {(r["factor"], r["metric"]): r["p_value"] for _, r in tests.iterrows()}
        print(f"Wrote {len(make_graphs(df, metrics, out_dir, p_lookup))} pooled graph(s)")

    # Stratified Analysis (ASCII Table Format)
    sc = args.stratify_by
    if not args.no_stratify and sc in df.columns:
        strata = sorted(df[sc].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s, 9))
        print(f"\n{'='*65}\nSTRATIFIED ANALYSIS — prototype factors within each {sc}\n{'='*65}")

        strat_tests_df = mixed_tests(df, metrics, PARAM_FACTORS, stratum_col=sc, min_n=args.min_n)
        strat_tests_df.to_csv(out_dir / "stratified_stat_tests.csv", index=False)
        group_summary(df, metrics, stratum_col=sc).to_csv(out_dir / "stratified_summary.csv", index=False)
        print(f"Wrote {out_dir / 'stratified_stat_tests.csv'} ({len(strat_tests_df)} rows)")
        print(f"Wrote {out_dir / 'stratified_summary.csv'}\n")

        # --- FULL P-VALUE MATRIX OUTPUT ---
        print(f"Full p-value matrix (prototype factors x metrics x stratum):")

        header_str = f"  {'Factor':<10} {'Metric':<35}"
        for s in strata: header_str += f" {s:>8}"
        print(header_str)

        sep_str = f"  {'-'*10} {'-'*35}"
        for _ in strata: sep_str += f" {'-'*8}"
        print(sep_str)

        for factor in PARAM_FACTORS:
            for col, label, _ in metrics:
                row_vals = []
                has_data = False
                for s in strata:
                    match = strat_tests_df[
                        (strat_tests_df["stratum"] == s) &
                        (strat_tests_df["factor"]  == factor) &
                        (strat_tests_df["metric"]  == col)
                    ]
                    if match.empty or pd.isna(match.iloc[0]["p_value"]):
                        row_vals.append("     n/a")
                    else:
                        p = match.iloc[0]["p_value"]
                        has_data = True
                        marker = "*" if p < 0.05 else " "
                        row_vals.append(f"{p:>7.3f}{marker}")

                if has_data:
                    row_str = f"  {factor:<10} {label:<35}"
                    for v in row_vals: row_str += f" {v:>8}"
                    print(row_str)

        print("\n  * = p < 0.05  (Wald test, mixed model, participant random intercept, main effects only)")

        if not args.no_graphs:
            sp_lookup = {(r["stratum"], r["factor"], r["metric"]): r["p_value"] for _, r in strat_tests_df.iterrows()}
            print(f"Wrote {len(make_graphs(df, metrics, out_dir, sp_lookup, stratum_col=sc))} stratified graph(s)")

    print(f"\nAll comparison outputs successfully generated in:\n  -> {out_dir}")
    print("\nNote: mixed models use a participant random intercept (repeated Place events "
          "per participant are not independent). p-values screen significance; the "
          "'effect' column in the tidy CSVs gives magnitude and direction in the metric's "
          "own units. No interaction terms are fit here -- see module docstring.")


# =========================================================================== #
# SECTION 5: COMMAND LINE INTERFACE & MAIN EXECUTION
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["compare", "extract", "all"], default="compare",
                    help="Operation mode: 'compare' (stats/plots), 'extract' (recompute posture features), or 'all' (both)")
    ap.add_argument("--landmarks-root", type=Path, default=None,
                    help="Root directory containing trial folders and metrics/ CSVs")
    ap.add_argument("--pen-csv", type=Path, default=None,
                    help="Explicit path to place_metrics_combined.csv")
    ap.add_argument("--posture-csv", type=Path, default=None,
                    help="Explicit path to posture_features_combined.csv")
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated list of participant IDs to include")
    ap.add_argument("--no-graphs", action="store_true", help="Skip generating box plots")
    ap.add_argument("--no-stratify", action="store_true", help="Skip height-stratified mixed models")
    ap.add_argument("--stratify-by", default="height", help="Column to stratify by (default: height)")
    ap.add_argument("--min-n", type=int, default=8, help="Min rows required to attempt a model fit")
    args = ap.parse_args()

    if not args.landmarks_root and not (args.pen_csv or args.posture_csv):
        sys.exit("Error: Please provide --landmarks-root OR specify explicit paths via --pen-csv / --posture-csv.")

    df_posture = pd.DataFrame()

    # Step 1: Execute Extraction if requested
    if args.mode in ("extract", "all"):
        if not args.landmarks_root or not args.landmarks_root.is_dir():
            sys.exit("Error: --mode extract/all requires a valid directory passed to --landmarks-root.")
        df_posture = extract_posture_features(args.landmarks_root, participants_str=args.participants)
        if args.mode == "extract":
            return

    # Step 2: Execute Comparison
    if args.mode in ("compare", "all"):
        if args.landmarks_root:
            pen_path     = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
            posture_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
            out_dir      = args.landmarks_root / "metrics" / "combined_comparison"
        else:
            pen_path     = args.pen_csv
            posture_path = args.posture_csv
            out_dir      = (pen_path or posture_path).parent / "combined_comparison"

        df_pen = pd.read_csv(pen_path) if (pen_path and pen_path.is_file()) else pd.DataFrame()
        if df_posture.empty and posture_path and posture_path.is_file():
            df_posture = pd.read_csv(posture_path)

        if df_pen.empty and df_posture.empty:
            sys.exit("Error: Both pen and posture CSVs are missing or empty. Nothing to analyze.")

        compare_metrics(df_pen, df_posture, out_dir, args)


if __name__ == "__main__":
    main()