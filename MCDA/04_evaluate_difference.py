#!/usr/bin/env python3
r"""
evaluate_difference.py

Unified single-pass pipeline: extracts PEN task-performance metrics, POSTURE/REBA,
grip, and SPARC smoothness metrics for every Place event, and runs the statistical
comparison -- all in one script. Pen metrics are computed HERE, not upstream, so
every metric for a Place event lands on ONE row. This removes the pen<->posture
merge entirely (and with it the place_index/height key-matching that merge needed).

CLEANING INTEGRATION (from clean_place_events.py):
  - Body coordinates are read from {stem}_body_zfilt.csv (median-filtered z) when
    present, else raw {stem}_body.csv. Disable with --no-zfilt.
  - Place events listed in excluded_place_events.csv are skipped at extraction and
    filtered at compare. Disable with --no-exclude; relocate with --exclude-csv.

STATISTICAL ENGINE (mixed-effects, MAIN EFFECTS ONLY), within each height stratum:
      metric ~ C(Length) + C(Size) + C(Weight) + C(Angle) + (1 | participant)
  A joint Wald test per factor; per-level signed coefficients retained (needed for
  Angle's two contrasts). No interaction terms (they made the coarse REBA/grip
  models singular). A participant random intercept handles repeated Place events.

PEN GEOMETRY (from the former 04_place_metrics.py):
  - Pen shaft = local +Z rotated to world by the per-sample quaternion.
  - Plane normal/centroid (and, if logged, in-plane axes) come from
    plane_quality_log.csv; in-plane axes are rebuilt the same way flatten did, so
    left/right & up/down match the flattened frame. Trials with no plane entry
    still yield posture metrics; their pen metrics are NaN (with a warning).

MODES:
  1. Compare only (default):  --mode compare  reads metrics/combined_place_metrics.csv
  2. Extract only:            --mode extract  writes metrics/combined_place_metrics.csv
  3. End-to-end:              --mode all      extract then compare

  python MCDA/04_evaluate_difference.py --mode compare --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks

  Optional:
     --combined-csv PATH     Explicit path to combined_place_metrics.csv (compare-only)
     --participants P001,P002
     --no-graphs / --no-stratify / --stratify-by height / --min-n 8
     --no-zfilt              Use raw body even when *_body_zfilt.csv exist
     --no-exclude            Keep events even if in the cleaning manifest
     --exclude-csv PATH      excluded_place_events.csv (default <root>/metrics/cleaning/...)
"""

import argparse
import re
import sys
import warnings
from collections import defaultdict
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from utils.io import parse_float, read_table
    from utils.discovery import find_labelled_pen, iter_trials_labelled
    from utils.params import parse_participant_filter
except ImportError:
    sys.exit("Error: Could not import 'utils.*'. Run from the repo root or add utils to PYTHONPATH.")


# =========================================================================== #
# SECTION 1: METRIC REGISTRY & PARAMETERS
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
    ("reba_score_a",              "REBA Score A (Trunk/Neck/Legs)",    "score"),
    ("reba_score_b_right",        "REBA Score B (Right Arm/Wrist)",    "score"),
    ("reba_score_b_left",         "REBA Score B (Left Arm/Wrist)",     "score"),
    ("reba_grand_right",          "REBA Grand Score (Right)",          "score"),
    ("reba_grand_left",           "REBA Grand Score (Left)",           "score"),
    ("trunk_flex_mean",           "Trunk Flexion",                     "deg"),
    ("neck_flex_mean",            "Neck Flexion",                      "deg"),
    ("reach_ratio_mean",          "Reach Ratio",                       "ratio"),
    ("wrist_elevation_m_mean",    "Wrist Elevation Above Shoulder",    "m"),
    ("right_grip_comfort_score",  "R Grip Comfort Index",              "score"),
    ("right_sparc_linear",        "R Linear Smoothness (SPARC)",       "val"),
    ("right_sparc_angular",       "R Angular Smoothness (SPARC)",      "val"),
]
ALL_METRICS   = PEN_METRICS + POSTURE_METRICS
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
ALL_FACTORS   = ["participant", "height"] + PARAM_FACTORS

BODY_UP  = np.array([-1.0, 0.0, 0.0])   # MediaPipe body: 'up' is -X
HAND_UP  = np.array([0.0, 1.0, 0.0])    # Quest hand: 'up' is +Y
CONF_MIN = 0.3
PEN_LOCAL_AXIS = np.array([0.0, 0.0, 1.0])   # local +Z = pen shaft
HEIGHT_COLS = ["High", "Medium", "Low"]
VALID_HEIGHTS = ["High", "Medium", "Low"]    # the only height strata ever analysed

# Columns carried straight from compute_place_metrics onto each combined row.
PEN_METRIC_KEYS = ["n_samples", "perp_mean_deg", "perp_var_deg2",
                   "leftright_mean_deg", "leftright_var_deg2",
                   "updown_mean_deg", "updown_var_deg2",
                   "pos_jitter_mm", "ang_jitter_deg"]


# =========================================================================== #
# SECTION 2: CLEANING INTEGRATION HELPERS (clean_place_events.py)
# =========================================================================== #

def resolve_body_path(trial_dir: Path, stem: str) -> Path:
    """Prefer the median-filtered body; fall back to raw."""
    zf = trial_dir / f"{stem}_body_zfilt.csv"
    return zf if zf.is_file() else (trial_dir / f"{stem}_body.csv")

def _event_key(participant, trial, height, place_index):
    try:
        pi = int(place_index)
    except (ValueError, TypeError):
        return None
    return (str(participant), str(trial), str(height), pi)

def load_excluded_events(exclude_csv):
    if exclude_csv is None or not Path(exclude_csv).is_file():
        print(f"  [exclude] no cleaning manifest at {exclude_csv}; no events excluded.")
        return set()
    ex = pd.read_csv(exclude_csv)
    need = {"participant", "trial", "height", "place_index_in_height"}
    if not need.issubset(ex.columns):
        print(f"  [exclude] manifest missing {need - set(ex.columns)}; skipping exclusion.")
        return set()
    keys = {k for k in (_event_key(r["participant"], r["trial"], r["height"], r["place_index_in_height"])
                        for _, r in ex.iterrows()) if k is not None}
    print(f"  [exclude] loaded {len(keys)} rejected place event(s) from {Path(exclude_csv).name}")
    return keys


# =========================================================================== #
# SECTION 3: PLANE IO + PEN GEOMETRY (from 04_place_metrics.py)
# =========================================================================== #

def load_plane_log(log_path: Path) -> dict:
    out = {}
    if not log_path.is_file():
        return out
    rows, fields = read_table(log_path)
    have_axes = all(c in fields for c in ("axis_u_x", "axis_v_x", "axis_n_x"))
    for r in rows:
        stem = r.get("trial_stem")
        try:
            normal = np.array([float(r["normal_x"]), float(r["normal_y"]), float(r["normal_z"])])
            centroid = np.array([float(r["centroid_x"]), float(r["centroid_y"]), float(r["centroid_z"])])
        except (TypeError, ValueError, KeyError):
            continue
        entry = {"normal": normal, "centroid": centroid, "u": None, "v": None, "n": None}
        if have_axes:
            try:
                entry["u"] = np.array([float(r["axis_u_x"]), float(r["axis_u_y"]), float(r["axis_u_z"])])
                entry["v"] = np.array([float(r["axis_v_x"]), float(r["axis_v_y"]), float(r["axis_v_z"])])
                entry["n"] = np.array([float(r["axis_n_x"]), float(r["axis_n_y"]), float(r["axis_n_z"])])
            except (TypeError, ValueError, KeyError):
                pass
        out[stem] = entry
    return out


def rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def plane_frame(normal):
    n = normal / np.linalg.norm(normal)
    if n[2] < 0:
        n = -n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def pen_axis_world(qw, qx, qy, qz):
    R = rotmat(qw, qx, qy, qz)
    a = R @ PEN_LOCAL_AXIS
    return a / np.linalg.norm(a)


def best_shaft_axis(pen, place_mask, normal):
    idx = np.where(place_mask)[0]
    if len(idx) == 0:
        return None, None
    aligns = [0.0, 0.0, 0.0]
    for ax in range(3):
        local = np.zeros(3); local[ax] = 1.0
        s = 0.0
        for i in idx:
            R = rotmat(pen["qw"][i], pen["qx"][i], pen["qy"][i], pen["qz"][i])
            s += abs(np.dot(R @ local, normal))
        aligns[ax] = s / len(idx)
    return int(np.argmax(aligns)), aligns


def compute_place_metrics(pen, start, stop, normal, u, v):
    t = pen["t_s"]
    mask = (t >= start) & (t <= stop)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None
    pos = np.column_stack([pen["x"][idx], pen["y"][idx], pen["z"][idx]])
    quats = np.column_stack([pen["qw"][idx], pen["qx"][idx], pen["qy"][idx], pen["qz"][idx]])
    axes = np.array([pen_axis_world(*q) for q in quats])
    dots = np.clip(np.abs(axes @ normal), -1, 1)
    perp_angle = np.degrees(np.arccos(dots))
    a_n = axes @ normal
    a_u = axes @ u
    a_v = axes @ v
    lr_angle = np.degrees(np.arctan2(a_u, np.abs(a_n)))
    ud_angle = np.degrees(np.arctan2(a_v, np.abs(a_n)))
    mean_pos = pos.mean(axis=0)
    dists = np.linalg.norm(pos - mean_pos, axis=1)
    pos_jitter_mm = float(np.sqrt(np.mean(dists**2)) * 1000.0)
    mean_axis = axes.mean(axis=0); mean_axis /= np.linalg.norm(mean_axis)
    ang_dev = np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1)))
    ang_jitter_deg = float(np.sqrt(np.mean(ang_dev**2)))
    return {
        "duration_s": float(stop - start), "n_samples": int(len(idx)),
        "perp_mean_deg": float(perp_angle.mean()), "perp_var_deg2": float(perp_angle.var()),
        "leftright_mean_deg": float(lr_angle.mean()), "leftright_var_deg2": float(lr_angle.var()),
        "updown_mean_deg": float(ud_angle.mean()), "updown_var_deg2": float(ud_angle.var()),
        "pos_jitter_mm": pos_jitter_mm, "ang_jitter_deg": ang_jitter_deg,
    }


# =========================================================================== #
# SECTION 4: LABELLED PEN IO + RUN DETECTION (from 04_place_metrics.py)
# =========================================================================== #

def load_labelled(path: Path):
    rows, fields = read_table(path)
    need = ("t_s", "qw", "qx", "qy", "qz", "x", "y", "z")
    for c in need:
        if c not in fields:
            raise RuntimeError(f"Labelled pen file missing '{c}': {path.name}")
    known = set(need) | {"data_quality", "x_flat", "y_flat", "z_flat", "is_calibration"}
    beh_names = [c for c in fields if c not in known]
    cols = {c: [] for c in need}
    beh = {b: [] for b in beh_names}
    for r in rows:
        vals = {c: parse_float(r.get(c)) for c in need}
        if any(val is None for val in vals.values()):
            continue
        for c in need:
            cols[c].append(vals[c])
        for b in beh_names:
            beh[b].append(1 if str(r.get(b)).strip() in ("1", "1.0", "True", "true") else 0)
    out = {c: np.asarray(cols[c]) for c in need}
    out_beh = {b: np.asarray(beh[b], dtype=int) for b in beh_names}
    return out, out_beh, beh_names


def runs_from_flag(t_s, flag):
    intervals, in_run, start = [], False, None
    for i, val in enumerate(flag):
        if val and not in_run:
            in_run, start = True, t_s[i]
        elif not val and in_run:
            in_run = False; intervals.append((start, t_s[i - 1]))
    if in_run:
        intervals.append((start, t_s[-1]))
    return intervals


# =========================================================================== #
# SECTION 5: POSTURE / GRIP / SPARC MODELS (from evaluate_difference.py)
# =========================================================================== #

def compute_sparc(speed_profile, fs, pad_level=4, fc=10.0, amp_th=0.05):
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
    omega_norm = freqs[valid_idx] / fc
    d_omega = np.diff(omega_norm); d_V = np.diff(V_norm[valid_idx])
    return float(-np.sum(np.sqrt(d_omega**2 + d_V**2)))


def grip_comfort_model(aperture_m, optimal_span_m=0.050, max_tol_m=0.035):
    if aperture_m is None or np.isnan(aperture_m):
        return np.nan, np.nan
    dev_mm = abs(aperture_m - optimal_span_m) * 1000.0
    penalty = (abs(aperture_m - optimal_span_m) / max_tol_m) ** 2
    comfort = max(0.0, min(100.0, (1.0 - penalty) * 100.0))
    return float(dev_mm), float(comfort)


REBA_TABLE_A = {
    1: {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [3, 3, 5, 6]},
    2: {1: [2, 3, 4, 5], 2: [3, 4, 5, 6], 3: [4, 5, 6, 7]},
    3: {1: [2, 4, 5, 6], 2: [4, 5, 6, 7], 3: [5, 6, 7, 8]},
    4: {1: [3, 5, 6, 7], 2: [5, 6, 7, 8], 3: [6, 7, 8, 9]},
    5: {1: [4, 6, 7, 8], 2: [6, 7, 8, 9], 3: [7, 8, 9, 9]},
}
REBA_TABLE_B = {
    1: {1: [1, 2, 2], 2: [1, 2, 3]}, 2: {1: [1, 2, 3], 2: [2, 3, 4]},
    3: {1: [3, 4, 5], 2: [4, 5, 5]}, 4: {1: [4, 5, 5], 2: [5, 6, 7]},
    5: {1: [6, 7, 8], 2: [7, 8, 8]}, 6: {1: [7, 8, 8], 2: [8, 9, 9]},
}
REBA_TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7], [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8], [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9], [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11], [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12], [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12], [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
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
# SECTION 6: SINGLE-PASS EXTRACTION (pen + posture on one row per Place event)
# =========================================================================== #

def _posture_for_event(row, df_body, df_hand, s, e, dur):
    """Fill REBA / grip / SPARC / macro-angle fields for one Place event (from
    the former posture extractor), operating on the body/hand slices."""
    if df_hand is not None and not df_hand.empty and "t_s" in df_hand.columns:
        sub_h = df_hand[(df_hand["t_s"] >= s) & (df_hand["t_s"] <= e)]
        if len(sub_h) >= 4:
            fs = float(1.0 / np.median(np.diff(sub_h["t_s"]))) if len(sub_h) > 1 else 30.0
            for side in ("Left", "Right"):
                pref = side.lower()
                if f"{side}_HandThumbTip_x" in sub_h.columns and f"{side}_HandIndexTip_x" in sub_h.columns:
                    tt = sub_h[[f"{side}_HandThumbTip_{ax}" for ax in ("x", "y", "z")]].values
                    it = sub_h[[f"{side}_HandIndexTip_{ax}" for ax in ("x", "y", "z")]].values
                    apertures = np.linalg.norm(tt - it, axis=1)
                    m_ap = float(np.nanmean(apertures)) if not np.all(np.isnan(apertures)) else np.nan
                    dev_mm, comfort = grip_comfort_model(m_ap)
                    row[f"{pref}_aperture_mean"] = m_ap
                    row[f"{pref}_grip_span_dev_mm"] = dev_mm
                    row[f"{pref}_grip_comfort_score"] = comfort
                if f"{side}_HandWristRoot_x" in sub_h.columns:
                    wp = sub_h[[f"{side}_HandWristRoot_{ax}" for ax in ("x", "y", "z")]].values
                    if len(wp) >= 10:
                        lin_vel = np.gradient(wp, sub_h["t_s"].values[:len(wp)], axis=0)
                        row[f"{pref}_sparc_linear"] = compute_sparc(np.linalg.norm(lin_vel, axis=1), fs=fs, fc=10.0)
                    if f"{side}_HandWristRoot_qw" in sub_h.columns:
                        wq = sub_h[[f"{side}_HandWristRoot_{ax}" for ax in ("qw", "qx", "qy", "qz")]].values
                        if len(wq) >= 10:
                            ang_dists = [quat_angular_distance(wq[k], wq[k-1]) for k in range(1, len(wq))]
                            dt = np.diff(sub_h["t_s"].values[:len(wq)]); dt = np.where(dt <= 0, 1.0/fs, dt)
                            row[f"{pref}_sparc_angular"] = compute_sparc(np.array(ang_dists)/dt, fs=fs, fc=6.0)

    if df_body is not None and not df_body.empty and "t_s" in df_body.columns:
        sub_b = df_body[(df_body["t_s"] >= s) & (df_body["t_s"] <= e)]
        if len(sub_b) >= 1:
            def get_j(name):
                if f"{name}_x" not in sub_b.columns:
                    return None
                return np.nanmean(sub_b[[f"{name}_{ax}" for ax in ("x", "y", "z")]].values, axis=0)
            ls, rs, lh, rh = get_j("LeftShoulder"), get_j("RightShoulder"), get_j("LeftHip"), get_j("RightHip")
            le, re_ = get_j("LeftEar"), get_j("RightEar")
            trunk_flex = angle_between((ls + rs)/2 - (lh + rh)/2, BODY_UP) if ls is not None and lh is not None else np.nan
            neck_flex = angle_between((le + re_)/2 - (ls + rs)/2, BODY_UP) - trunk_flex if le is not None and ls is not None else np.nan
            row["trunk_flex_mean"] = trunk_flex
            row["neck_flex_mean"] = neck_flex
            ts = 1 if abs(trunk_flex) < 5 else (2 if abs(trunk_flex) <= 20 else (3 if abs(trunk_flex) <= 60 else 4))
            ns = 1 if (0 <= neck_flex <= 20) else 2
            score_a = reba_table_a(ts, ns, 1)
            row["reba_score_a"] = score_a
            for side in ("Left", "Right"):
                pref = side.lower()
                sh, el, wr = get_j(f"{side}Shoulder"), get_j(f"{side}Elbow"), get_j(f"{side}Wrist")
                if sh is not None and wr is not None:
                    row["wrist_elevation_m_mean"] = float(np.dot(sh - wr, BODY_UP))
                    arm_len = np.linalg.norm(el - sh) + np.linalg.norm(wr - el) if el is not None else np.nan
                    row["reach_ratio_mean"] = float(np.linalg.norm(wr - sh) / arm_len) if arm_len and arm_len > 1e-6 else np.nan
                if sh is not None and el is not None:
                    uf = angle_between(el - sh, -(((ls + rs)/2 - (lh + rh)/2) if ls is not None else BODY_UP))
                    ef = angle_between(sh - el, wr - el) if wr is not None else 90.0
                    ua_s = 1 if abs(uf) <= 20 else (2 if uf <= 45 else (3 if uf <= 90 else 4))
                    la_s = 1 if (60 <= ef <= 100) else 2
                    score_b = reba_table_b(ua_s, la_s, 1)
                    row[f"reba_score_b_{pref}"] = score_b
                    row[f"reba_grand_{pref}"] = reba_table_c(score_a, score_b) + (1 if dur > 60.0 else 0)
    return row


def extract_features(root_dir: Path, participants_str=None, use_zfilt=True, exclude_keys=None):
    print(f"\n{'='*65}\nSINGLE-PASS EXTRACTION (pen + posture + hand)\n{'='*65}")
    plane_log = load_plane_log(root_dir / "plane_quality_log.csv")
    if not plane_log:
        print("  [WARN] no plane_quality_log.csv -- pen orientation metrics will be NaN.")
    pfilter = parse_participant_filter(participants_str)
    trials = list(iter_trials_labelled(root_dir, pfilter))
    if not trials:
        sys.exit("Error: No trial folders with labelled pen files found! Check --landmarks-root.")

    exclude_keys = exclude_keys or set()
    all_rows, n_zfilt, n_excl, n_noplane, warns = [], 0, 0, 0, []

    for stem, pid, trial_dir in trials:
        pen_path = find_labelled_pen(trial_dir, stem)
        if pen_path is None:
            continue
        try:
            pen, beh, _ = load_labelled(pen_path)
        except RuntimeError as ex:
            warns.append(str(ex)); print(f"  [skip] {stem}: {ex}"); continue
        if "Place" not in beh:
            warns.append(f"{stem}: no 'Place' column"); continue
        t_s = pen["t_s"]
        place_runs = runs_from_flag(t_s, beh["Place"])
        if not place_runs:
            print(f"  [   0] {stem}: no Place events"); continue

        height_runs = {h: runs_from_flag(t_s, beh[h]) for h in HEIGHT_COLS if h in beh}
        def height_at(tmid):
            for h, runs in height_runs.items():
                for a, b in runs:
                    if a <= tmid <= b:
                        return h
            return "Unknown"

        entry = plane_log.get(stem)
        if entry is not None:
            if entry["u"] is not None:
                n = entry["n"]/np.linalg.norm(entry["n"]); u = entry["u"]/np.linalg.norm(entry["u"]); v = entry["v"]/np.linalg.norm(entry["v"])
            else:
                n, u, v = plane_frame(entry["normal"])
        else:
            n = u = v = None; n_noplane += 1
            warns.append(f"{stem}: no plane in quality log -- pen metrics NaN")

        body_path = resolve_body_path(trial_dir, stem) if use_zfilt else (trial_dir / f"{stem}_body.csv")
        if body_path.name.endswith("_body_zfilt.csv"):
            n_zfilt += 1
        df_body = pd.read_csv(body_path) if body_path.is_file() else pd.DataFrame()
        hand_path = trial_dir / f"{stem}_hand.csv"
        df_hand = pd.read_csv(hand_path) if hand_path.is_file() else pd.DataFrame()

        counters = {}
        kept = 0
        for (s, e) in place_runs:
            h = height_at((s + e) / 2.0)
            counters[h] = counters.get(h, 0) + 1
            pidx = counters[h]
            if exclude_keys and _event_key(pid, stem, h, pidx) in exclude_keys:
                n_excl += 1
                continue
            dur = e - s
            row = {"participant": pid, "trial": stem, "place_index": pidx, "height": h,
                   "start_t_s": round(float(s), 4), "stop_t_s": round(float(e), 4),
                   "duration_s": round(float(dur), 4)}
            # pen metrics
            if n is not None:
                m = compute_place_metrics(pen, s, e, n, u, v)
                if m is not None:
                    for k in PEN_METRIC_KEYS:
                        row[k] = m[k]
                else:
                    for k in PEN_METRIC_KEYS:
                        row[k] = np.nan
            else:
                for k in PEN_METRIC_KEYS:
                    row[k] = np.nan
            # posture / hand metrics
            _posture_for_event(row, df_body, df_hand, s, e, dur)
            all_rows.append(row); kept += 1
        print(f"  [{kept:>4}] {stem}: {kept} kept / {len(place_runs)} Place event(s)")

    if not all_rows:
        sys.exit("Error: 0 Place events extracted! Halting to avoid overwriting with empty data.")

    df = pd.DataFrame(all_rows)
    out_dir = root_dir / "metrics"; out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "combined_place_metrics.csv"
    df.to_csv(out_file, index=False)
    print(f"\nExtraction complete: {len(df)} row(s) -> {out_file}")
    print(f"  body source: median-filtered (_zfilt) for {n_zfilt}/{len(trials)} trial(s)"
          + (f"; {n_noplane} trial(s) had no plane (pen metrics NaN)" if n_noplane else "")
          + (f"; excluded {n_excl} event(s) via manifest" if n_excl else ""))
    if warns:
        print("  warnings:")
        for w in warns[:12]:
            print(f"    {w}")
        if len(warns) > 12:
            print(f"    ... and {len(warns) - 12} more")
    return df


# =========================================================================== #
# SECTION 7: STATISTICAL COMPARISON (mixed-effects, main effects only)
# =========================================================================== #

def parse_params(trial: str) -> dict:
    out = {k: None for k in PARAM_FACTORS}
    tokens = trial.split("_")
    low = "_".join(tokens).lower()
    if "not_weighted" in low:
        out["Weight"] = "Not_weighted"
    elif "front_weighted" in low:
        out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long", "Short"): out["Length"] = tok
        elif tok in ("Large", "Small"): out["Size"] = tok
    return out

def add_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        df[c] = parsed[c]
    print("\nObserved factor levels (each should show exactly the expected set):")
    for c in PARAM_FACTORS:
        print(f"  {c}: {sorted(df[c].dropna().unique().tolist(), key=str)}")
    bad_mask = df[PARAM_FACTORS].isna().any(axis=1)
    if bad_mask.any():
        bad_trials = df.loc[bad_mask, "trial"].unique()
        print(f"\n  [WARN] {bad_mask.sum()} row(s) across {len(bad_trials)} trial(s) could not be parsed "
              f"into Length/Size/Weight/Angle -- QUARANTINED (dropped):")
        for t in bad_trials[:10]:
            print(f"      '{t}'")
        df = df[~bad_mask].copy()
    return df

def available_metrics(df: pd.DataFrame) -> list:
    out = []
    for col, label, unit in ALL_METRICS:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").dropna().nunique() > 1:
            out.append((col, label, unit))
    return out

def group_summary(df, metrics, stratum_col=None):
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

def _fit_mixed_main(data, response, factors):
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

def _term_wald(res, factor):
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    if not names:
        return np.nan, np.nan
    b = res.fe_params[names].values
    try:
        V = res.cov_params().loc[names, names].values
        W = float(b @ np.linalg.solve(V, b))
    except Exception:
        return np.nan, len(names)
    return float(stats.chi2.sf(W, len(names))), len(names)

def _term_level_effects(res, factor):
    names = [n for n in res.fe_params.index if n.startswith(f"C({factor})")]
    out = []
    for n in names:
        m = re.search(r"\[T\.(.+?)\]", n)
        level_raw = m.group(1) if m else n
        try:
            level_raw = str(int(float(level_raw)))
        except ValueError:
            pass
        se = float(np.sqrt(res.cov_params().loc[n, n])) if n in res.cov_params().index else np.nan
        out.append({"level": level_raw, "effect": float(res.fe_params[n]), "se": se})
    return out

def mixed_tests(df, metrics, factors, stratum_col=None, min_n=8, verbose=True):
    records, diag = [], {"fit": 0, "skip": 0, "fail": 0}
    groups = df.groupby(stratum_col, dropna=True) if stratum_col else [("All", df)]
    for stratum, sub_df in groups:
        for col, label, unit in metrics:
            d = sub_df.dropna(subset=[col, "participant"] + factors).copy().rename(columns={col: "_y"})
            n_ppt = d["participant"].nunique()
            tag = f"[{stratum}] {label}"
            res, present, note = (None, [], None)
            if n_ppt < 2 or len(d) < min_n:
                diag["skip"] += 1
                note = f"skipped: n={len(d)} rows, {n_ppt} participant(s)"
                if verbose: print(f"    {tag}: {note}")
            else:
                res, present, note = _fit_mixed_main(d, "_y", factors)
                if res is None:
                    diag["fail"] += 1
                    if verbose: print(f"    {tag}: fit failed: {note}")
                else:
                    diag["fit"] += 1
            for f in factors:
                if res is not None:
                    p, dfree = _term_wald(res, f); level_effects = _term_level_effects(res, f)
                else:
                    p, dfree, level_effects = np.nan, np.nan, []
                if not level_effects:
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
        print(f"  Model fit summary: {diag['fit']}/{total} ok, {diag['skip']}/{total} skipped, "
              f"{diag['fail']}/{total} failed.")
    return pd.DataFrame(records)

def order_levels(factor, levels):
    orders = {"height": ["High", "Medium", "Low"], "Length": ["Short", "Long"],
              "Size": ["Small", "Large"], "Weight": ["Not_weighted", "Front_weighted"]}
    if factor in orders:
        known = [l for l in orders[factor] if l in levels]
        return known + sorted([l for l in levels if l not in known])
    try:
        return sorted(levels, key=lambda x: float(x))
    except Exception:
        return sorted(levels, key=str)

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

def run_comparison(df, out_dir, args, exclude_keys=None):
    print(f"\n{'='*65}\nSTATISTICAL COMPARISON (mixed-effects, main effects only)\n{'='*65}")
    # exclusion (covers compare-mode reading a pre-existing CSV)
    if exclude_keys and {"participant", "trial", "height", "place_index"}.issubset(df.columns):
        keys = df.apply(lambda r: _event_key(r["participant"], r["trial"], r["height"], r["place_index"]), axis=1)
        before = len(df); df = df[~keys.isin(exclude_keys)].copy()
        if before - len(df):
            print(f"  [exclude] removed {before - len(df)} rejected place event(s); {len(df)} remain.")

    if args.participants:
        keep = {p.strip() for p in args.participants.split(",") if p.strip()}
        df = df[df["participant"].astype(str).isin(keep)].copy()
        if df.empty: sys.exit(f"No rows remaining for participants: {sorted(keep)}")

    # Quarantine events with no valid working-height label (midpoint fell outside
    # every High/Medium/Low window). The rows remain in combined_place_metrics.csv
    # for audit; they are simply not analysed by height.
    if "height" in df.columns and not args.keep_unknown_height:
        df["height"] = df["height"].astype(str).str.strip()
        bad = ~df["height"].isin(VALID_HEIGHTS)
        if bad.any():
            n_bad = int(bad.sum())
            offenders = df.loc[bad, "trial"].nunique()
            print(f"\n[QUARANTINE] Dropping {n_bad} Place event(s) across {offenders} trial(s) whose "
                  f"midpoint matched no labelled High/Medium/Low window. "
                  f"Use --keep-unknown-height to keep them.")
            df = df[~bad].copy()

    df = add_parameter_columns(df)
    metrics = available_metrics(df)
    print(f"Total Participants: {df['participant'].nunique()} | Trials: {df['trial'].nunique()}")
    print(f"Active metrics evaluated: {len(metrics)}")

    if "height" in df.columns:
        print("\nPlace events per height label:")
        for h, n in df["height"].fillna("<blank>").value_counts().items():
            flag = "  <-- not High/Medium/Low" if h not in ("High", "Medium", "Low") else ""
            print(f"  {h:>10}: {n}{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
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

    sc = args.stratify_by
    if not args.no_stratify and sc in df.columns:
        if sc == "height":
            strata = [h for h in VALID_HEIGHTS if h in set(df[sc].dropna())]
            df_s = df[df[sc].isin(strata)].copy()   # never stratify on Unknown/blank
        else:
            strata = sorted(df[sc].dropna().unique(), key=str)
            df_s = df
        print(f"\n{'='*65}\nSTRATIFIED ANALYSIS — prototype factors within each {sc}\n{'='*65}")
        strat_tests_df = mixed_tests(df_s, metrics, PARAM_FACTORS, stratum_col=sc, min_n=args.min_n)
        strat_tests_df.to_csv(out_dir / "stratified_stat_tests.csv", index=False)
        group_summary(df_s, metrics, stratum_col=sc).to_csv(out_dir / "stratified_summary.csv", index=False)
        print(f"Wrote {out_dir / 'stratified_stat_tests.csv'} ({len(strat_tests_df)} rows)\n")

        print("Full p-value matrix (prototype factors x metrics x stratum):")
        header = f"  {'Factor':<10} {'Metric':<35}"
        for s in strata: header += f" {s:>8}"
        print(header)
        sep = f"  {'-'*10} {'-'*35}"
        for _ in strata: sep += f" {'-'*8}"
        print(sep)
        for factor in PARAM_FACTORS:
            for col, label, _ in metrics:
                row_vals, has_data = [], False
                for s in strata:
                    match = strat_tests_df[(strat_tests_df["stratum"] == s) &
                                           (strat_tests_df["factor"] == factor) &
                                           (strat_tests_df["metric"] == col)]
                    if match.empty or pd.isna(match.iloc[0]["p_value"]):
                        row_vals.append("     n/a")
                    else:
                        p = match.iloc[0]["p_value"]; has_data = True
                        row_vals.append(f"{p:>7.3f}{'*' if p < 0.05 else ' '}")
                if has_data:
                    line = f"  {factor:<10} {label:<35}"
                    for vv in row_vals: line += f" {vv:>8}"
                    print(line)
        print("\n  * = p < 0.05  (Wald test, mixed model, participant random intercept, main effects only)")

        if not args.no_graphs:
            sp_lookup = {(r["stratum"], r["factor"], r["metric"]): r["p_value"] for _, r in strat_tests_df.iterrows()}
            print(f"Wrote {len(make_graphs(df_s, metrics, out_dir, sp_lookup, stratum_col=sc))} stratified graph(s)")

    print(f"\nAll comparison outputs generated in:\n  -> {out_dir}")


# =========================================================================== #
# SECTION 8: CLI
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["compare", "extract", "all"], default="compare")
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--combined-csv", type=Path, default=None,
                    help="Explicit path to combined_place_metrics.csv (compare-only)")
    ap.add_argument("--participants", type=str, default=None)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--keep-unknown-height", action="store_true",
                    help="Keep events whose midpoint matched no High/Medium/Low window "
                         "(default: quarantine them, matching rank_prototypes.py)")
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--stratify-by", default="height")
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--no-zfilt", action="store_true", help="Use raw *_body.csv even when *_body_zfilt.csv exist")
    ap.add_argument("--no-exclude", action="store_true", help="Keep events even if in the cleaning manifest")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="excluded_place_events.csv (default <root>/metrics/cleaning/excluded_place_events.csv)")
    args = ap.parse_args()

    if not args.landmarks_root and not args.combined_csv:
        sys.exit("Error: provide --landmarks-root OR --combined-csv.")
    root = args.landmarks_root

    # cleaning manifest (shared by extract + compare)
    exclude_keys = set()
    if not args.no_exclude:
        exclude_csv = args.exclude_csv
        if exclude_csv is None and root:
            exclude_csv = root / "metrics" / "cleaning" / "excluded_place_events.csv"
        exclude_keys = load_excluded_events(exclude_csv)

    df = None
    if args.mode in ("extract", "all"):
        if not root or not root.is_dir():
            sys.exit("Error: --mode extract/all requires a valid --landmarks-root.")
        df = extract_features(root, participants_str=args.participants,
                              use_zfilt=not args.no_zfilt, exclude_keys=exclude_keys)
        if args.mode == "extract":
            return

    if args.mode in ("compare", "all"):
        if df is None:
            combined = args.combined_csv or (root / "metrics" / "combined_place_metrics.csv")
            if not combined.is_file():
                sys.exit(f"Error: {combined} not found. Run --mode extract (or all) first.")
            df = pd.read_csv(combined)
            print(f"Loaded {len(df)} Place events from {combined}")
        out_dir = (root / "metrics" / "combined_comparison") if root else (args.combined_csv.parent / "combined_comparison")
        run_comparison(df, out_dir, args, exclude_keys=exclude_keys)


if __name__ == "__main__":
    main()