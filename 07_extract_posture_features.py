#!/usr/bin/env python3
"""
Extract posture / interaction features per Place event.

For every Place event (the Place START->STOP intervals, read from the labelled
pen file's 'Place' column) this computes, from the trial's body and hand
tracking CSVs:

BODY (REBA, Hignett & McAtamney 2000):
  - Raw component angles (both arms): upper-arm flexion, upper-arm abduction,
    elbow flexion; central: trunk flexion, neck flexion, knee flexion.
    Mean + SD over hold.
  - REBA Group A (trunk/neck/legs, with measured knee flexion),
    Group B (arm/wrist, per side), and grand score per side via canonical
    Table A / B / C lookups. Wrist flexion fed from Quest hand skeleton.
  - New grip-comfort features (body frame, reliable side):
      wrist_neutral_dev : wrist orientation deviation from neutral (deg)
      reach_ratio       : wrist / arm-length (0=arm at side, 1=fully extended)
      wrist_elevation_m : wrist height above shoulder (m; +ve = above shoulder)
  - Load/force, coupling, activity scores: documented parameters.

HAND (Quest hand skeleton, power/drill grip):
  - Aperture (thumb tip <-> index tip): mean + SD
  - Mean finger flexion per finger + overall
  - Hand positional jitter: wrist-root RMS displacement (mm)
  - Hand orientation jitter: wrist-root quaternion angular variability (deg)
  - Movement smoothness: normalised jerk of the wrist-root path
  - Wrist flexion (HandStart->WristRoot->HandMiddle0): mean + SD
  - Wrist ulnar/radial deviation (knuckle line vs forearm): mean + SD
    (signed: +ve = radial, -ve = ulnar)

COORDINATE FRAMES (verified for this dataset):
  - Body (MediaPipe): vertical axis = X, with 'up' = -X.
  - Hand (Quest): shares the headset frame, vertical = +Y.

Output: <root>/metrics/posture_features_combined.csv  (one row per Place event,
keyed on participant, trial, place_index, height) plus a per-participant copy.
A manifest of which REBA elements were measured / approximated / assumed is
printed and saved.

Usage:
    python 07_extract_posture_features.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python 07_extract_posture_features.py --landmarks-root ... --participants P003,P004
    python 07_extract_posture_features.py --landmarks-root ... --load 1 --coupling 1
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import parse_float, read_table
from utils.discovery import find_labelled_pen, iter_trials_labelled
from utils.stats import summarise
from utils.params import parse_participant_filter


# --------------------------------------------------------------------------- #
# Coordinate-frame conventions (verified earlier in the pipeline)
# --------------------------------------------------------------------------- #
BODY_UP = np.array([-1.0, 0.0, 0.0])   # MediaPipe body: 'up' is -X
HAND_UP = np.array([0.0, 1.0, 0.0])    # Quest hand: 'up' is +Y

CONF_MIN = 0.3   # ignore MediaPipe landmarks below this confidence (if present)


# --------------------------------------------------------------------------- #
# REBA lookup tables (canonical, Hignett & McAtamney 2000)
# --------------------------------------------------------------------------- #
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
    trunk = min(max(trunk, 1), 5); neck = min(max(neck, 1), 3)
    legs = min(max(legs, 1), 4)
    return REBA_TABLE_A[trunk][neck][legs - 1]


def reba_table_b(upper, lower, wrist):
    upper = min(max(upper, 1), 6); lower = min(max(lower, 1), 2)
    wrist = min(max(wrist, 1), 3)
    return REBA_TABLE_B[upper][lower][wrist - 1]


def reba_table_c(score_a, score_b):
    a = min(max(int(round(score_a)), 1), 12)
    b = min(max(int(round(score_b)), 1), 12)
    return REBA_TABLE_C[a - 1][b - 1]


# --------------------------------------------------------------------------- #
# REBA component sub-scores from continuous angles
# --------------------------------------------------------------------------- #
def trunk_subscore(flex_deg, twisted=False):
    a = abs(flex_deg)
    if a < 5:   s = 1
    elif a <= 20: s = 2
    elif a <= 60: s = 3
    else:         s = 4
    return s + (1 if twisted else 0)


def neck_subscore(flex_deg, twisted=False):
    s = 1 if (0 <= flex_deg <= 20) else 2
    return s + (1 if twisted else 0)


def legs_subscore(supported=True, knee_flex_deg=0.0):
    s = 1 if supported else 2
    if knee_flex_deg > 60:   s += 2
    elif knee_flex_deg > 30: s += 1
    return s


def upper_arm_subscore(flex_deg, abducted=False, raised=False, supported=False):
    if -20 <= flex_deg <= 20:    s = 1
    elif flex_deg < -20 or flex_deg <= 45: s = 2
    elif flex_deg <= 90:         s = 3
    else:                        s = 4
    if abducted or raised: s += 1
    if supported:          s -= 1
    return max(s, 1)


def lower_arm_subscore(elbow_deg):
    return 1 if (60 <= elbow_deg <= 100) else 2


def wrist_subscore(flex_deg, deviated=False):
    s = 1 if abs(flex_deg) <= 15 else 2
    return s + (1 if deviated else 0)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def angle_between(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def quat_angular_distance(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def place_runs_from_pen(pen_path):
    rows, fields = read_table(pen_path)
    if "Place" not in fields or "t_s" not in fields:
        return []
    runs = []
    in_run = False; start = None; prev_t = None
    for r in rows:
        t = parse_float(r.get("t_s"))
        flag = str(r.get("Place", "")).strip() in ("1", "1.0", "True", "true")
        if t is None: continue
        if flag and not in_run:
            in_run, start = True, t
        elif not flag and in_run:
            in_run = False; runs.append((start, prev_t))
        prev_t = t
    if in_run:
        runs.append((start, prev_t))
    return runs


def height_runs_from_pen(pen_path):
    rows, fields = read_table(pen_path)
    out = {}
    for h in ("High", "Medium", "Low"):
        if h not in fields: continue
        runs = []; in_run = False; start = None; prev_t = None
        for r in rows:
            t = parse_float(r.get("t_s"))
            if t is None: continue
            flag = str(r.get(h, "")).strip() in ("1", "1.0", "True", "true")
            if flag and not in_run:
                in_run, start = True, t
            elif not flag and in_run:
                in_run = False; runs.append((start, prev_t))
            prev_t = t
        if in_run: runs.append((start, prev_t))
        out[h] = runs
    return out


def height_for(t_mid, height_runs):
    for h, runs in height_runs.items():
        for s, e in runs:
            if s <= t_mid <= e:
                return h
    return "Unknown"


# --------------------------------------------------------------------------- #
# Body: load rows into arrays of named joints
# --------------------------------------------------------------------------- #
BODY_JOINTS = ["Nose", "LeftEar", "RightEar",
               "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
               "LeftWrist", "RightWrist", "LeftHip", "RightHip",
               "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle",
               "LeftIndex", "RightIndex", "LeftPinky", "RightPinky",
               "LeftThumb", "RightThumb"]


def load_body(path):
    rows, fields = read_table(path)
    have_conf = any(f.endswith("_conf") for f in fields)
    t = []
    joints = {j: [] for j in BODY_JOINTS}
    confs  = {j: [] for j in BODY_JOINTS}
    for r in rows:
        tv = parse_float(r.get("t_s"))
        if tv is None: continue
        t.append(tv)
        for j in BODY_JOINTS:
            x = parse_float(r.get(f"{j}_x"))
            y = parse_float(r.get(f"{j}_y"))
            z = parse_float(r.get(f"{j}_z"))
            joints[j].append((x, y, z) if None not in (x, y, z) else (np.nan,)*3)
            c = parse_float(r.get(f"{j}_conf")) if have_conf else 1.0
            confs[j].append(c if c is not None else 1.0)
    t      = np.asarray(t)
    joints = {j: np.asarray(v, dtype=float) for j, v in joints.items()}
    confs  = {j: np.asarray(v, dtype=float) for j, v in confs.items()}
    return t, joints, confs


def reliable_side(confs):
    """Which body side is better tracked (by median confidence across limb joints).
    The camera films side-on, so one side is typically occluded. We use the
    reliable side for grip-comfort features and assume bilateral symmetry."""
    limb = ["Elbow", "Wrist", "Knee", "Ankle"]
    def med(side):
        vals = []
        for part in limb:
            c = confs.get(f"{side}{part}")
            if c is not None and len(c):
                arr = c[~np.isnan(c)]
                if len(arr): vals.append(float(np.median(arr)))
        return np.median(vals) if vals else 0.0
    return "Left" if med("Left") > med("Right") else "Right"


def body_angles_at(joints, confs, i, up):
    def g(j):
        if j not in joints: return None
        p = joints[j][i]
        if np.any(np.isnan(p)) or confs[j][i] < CONF_MIN: return None
        return p
    def have(*xs):
        return all(x is not None for x in xs)
    out = {}

    ls, rs = g("LeftShoulder"), g("RightShoulder")
    lh, rh = g("LeftHip"),      g("RightHip")

    # Trunk
    if have(ls, rs, lh, rh):
        sh_mid  = (ls + rs) / 2; hip_mid = (lh + rh) / 2
        trunk   = sh_mid - hip_mid
        out["trunk_flex"]  = angle_between(trunk, up)
        sh_line  = rs - ls; hip_line = rh - lh
        out["trunk_twist"] = angle_between(
            sh_line - np.dot(sh_line, up) * up,
            hip_line - np.dot(hip_line, up) * up)
    else:
        out["trunk_flex"] = np.nan; out["trunk_twist"] = np.nan

    # Neck (ear-based, relative to trunk lean)
    le, re = g("LeftEar"), g("RightEar")
    if have(ls, rs) and have(le, re):
        sh_mid   = (ls + rs) / 2
        ear_mid  = (le + re) / 2
        neck_from_vert  = angle_between(ear_mid - sh_mid, up)
        if have(lh, rh):
            trunk_from_vert = angle_between(sh_mid - ((lh + rh) / 2), up)
            out["neck_flex"] = (neck_from_vert - trunk_from_vert
                                if not (np.isnan(neck_from_vert)
                                        or np.isnan(trunk_from_vert)) else np.nan)
        else:
            out["neck_flex"] = neck_from_vert
    else:
        out["neck_flex"] = np.nan

    # Arms (both sides)
    for side, sh, el, wr in (("left",  ls, g("LeftElbow"),  g("LeftWrist")),
                             ("right", rs, g("RightElbow"), g("RightWrist"))):
        if have(sh, el):
            trunk_v = (((ls + rs) / 2) - ((lh + rh) / 2)
                       if have(ls, rs, lh, rh) else up)
            upper = el - sh
            out[f"{side}_upperarm_flex"]   = angle_between(upper, -trunk_v)
            side_axis = (rs - ls) if have(ls, rs) else np.array([0., 0., 1.])
            out[f"{side}_upperarm_abduct"] = abs(90 - angle_between(upper, side_axis))
        else:
            out[f"{side}_upperarm_flex"]   = np.nan
            out[f"{side}_upperarm_abduct"] = np.nan
        if have(sh, el, wr):
            out[f"{side}_elbow_flex"] = angle_between(sh - el, wr - el)
        else:
            out[f"{side}_elbow_flex"] = np.nan

    # Knee flexion on the reliable side (set per-trial via body_angles_at._side)
    rs_side   = body_angles_at._side or "Right"
    hipj      = f"{rs_side}Hip";   kneej  = f"{rs_side}Knee"
    anklej    = f"{rs_side}Ankle"
    hp  = g(hipj)  if hipj  in joints else None
    kn  = g(kneej) if kneej in joints else None
    an  = g(anklej) if anklej in joints else None
    if have(hp, kn, an):
        interior = angle_between(hp - kn, an - kn)
        out["knee_flex"] = (180.0 - interior) if not np.isnan(interior) else np.nan
    else:
        out["knee_flex"] = np.nan

    # Grip-comfort features (reliable side)
    sh_r  = g(f"{rs_side}Shoulder")
    el_r  = g(f"{rs_side}Elbow")
    wr_r  = g(f"{rs_side}Wrist")
    idx_r = g(f"{rs_side}Index")

    # Wrist neutrality deviation: angle of hand axis (wrist->index) out of
    # the forearm-up plane. 0=neutral, 90=fully deviated.
    if have(el_r, wr_r, idx_r):
        fore   = wr_r - el_r; fn = np.linalg.norm(fore)
        hand_v = idx_r - wr_r; hn = np.linalg.norm(hand_v)
        if fn > 1e-6 and hn > 1e-6:
            fore_u   = fore / fn; hand_u = hand_v / hn
            lateral  = np.cross(fore_u, up); ln = np.linalg.norm(lateral)
            if ln > 1e-6:
                lateral /= ln
                out["wrist_neutral_dev"] = np.degrees(
                    np.arcsin(np.clip(abs(float(np.dot(hand_u, lateral))), 0, 1)))
            else:
                out["wrist_neutral_dev"] = np.nan
        else:
            out["wrist_neutral_dev"] = np.nan
    else:
        out["wrist_neutral_dev"] = np.nan

    # Reach ratio: |shoulder->wrist| / (upper_arm + forearm)
    if have(sh_r, el_r, wr_r):
        arm_len = np.linalg.norm(el_r - sh_r) + np.linalg.norm(wr_r - el_r)
        reach   = np.linalg.norm(wr_r - sh_r)
        out["reach_ratio"] = float(reach / arm_len) if arm_len > 1e-6 else np.nan
    else:
        out["reach_ratio"] = np.nan

    # Wrist elevation above shoulder (m). Positive = above shoulder = strain.
    if have(sh_r, wr_r):
        out["wrist_elevation_m"] = float(np.dot(sh_r - wr_r, up))
    else:
        out["wrist_elevation_m"] = np.nan

    return out


# Per-trial slot so body_angles_at knows which side to use
body_angles_at._side = None


# --------------------------------------------------------------------------- #
# Hand: load Quest joints
# --------------------------------------------------------------------------- #
def detect_hand_side(fields):
    sides = []
    if any(f.startswith("Left_")  for f in fields): sides.append("Left")
    if any(f.startswith("Right_") for f in fields): sides.append("Right")
    return sides


def load_hand(path):
    rows, fields = read_table(path)
    sides = detect_hand_side(fields)
    t = []; data = []
    for r in rows:
        tv = parse_float(r.get("t_s"))
        if tv is None: continue
        t.append(tv); data.append(r)
    return np.asarray(t), data, fields, sides


def hand_point(row, side, joint):
    p = [parse_float(row.get(f"{side}_{joint}_{ax}")) for ax in ("x", "y", "z")]
    if None in p: return None
    return np.array(p, dtype=float)


def hand_quat(row, side, joint):
    q = [parse_float(row.get(f"{side}_{joint}_{ax}"))
         for ax in ("qw", "qx", "qy", "qz")]
    if None in q: return None
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else None


FINGERS = {
    "Thumb":  ["HandThumb1",  "HandThumb2",  "HandThumb3",  "HandThumbTip"],
    "Index":  ["HandIndex1",  "HandIndex2",  "HandIndex3",  "HandIndexTip"],
    "Middle": ["HandMiddle1", "HandMiddle2", "HandMiddle3", "HandMiddleTip"],
    "Ring":   ["HandRing1",   "HandRing2",   "HandRing3",   "HandRingTip"],
    "Pinky":  ["HandPinky1",  "HandPinky2",  "HandPinky3",  "HandPinkyTip"],
}


def finger_flexion(row, side, joints):
    pts = [hand_point(row, side, j) for j in joints]
    if any(p is None for p in pts): return np.nan
    total = 0.0
    for k in range(1, len(pts) - 1):
        a = angle_between(pts[k-1] - pts[k], pts[k+1] - pts[k])
        if not np.isnan(a): total += (180.0 - a)
    return total


# --------------------------------------------------------------------------- #
# Per-event feature computation
# --------------------------------------------------------------------------- #
def body_features_for_event(t_body, joints, confs, s, e, up,
                            load, coupling, activity_static,
                            wrist_flex_deg=None):
    mask = np.where((t_body >= s) & (t_body <= e))[0]
    if len(mask) == 0: return None

    keys = ["trunk_flex", "trunk_twist", "neck_flex", "knee_flex",
            "wrist_neutral_dev", "reach_ratio", "wrist_elevation_m",
            "left_upperarm_flex",  "left_upperarm_abduct",  "left_elbow_flex",
            "right_upperarm_flex", "right_upperarm_abduct", "right_elbow_flex"]
    series = {k: [] for k in keys}
    for i in mask:
        a = body_angles_at(joints, confs, i, up)
        for k in keys:
            series[k].append(a.get(k, np.nan))

    feat = {}; means = {}
    for k in keys:
        m, sd = summarise(series[k])
        feat[f"{k}_mean"] = m
        feat[f"{k}_sd"]   = sd
        means[k] = m

    # REBA sub-scores from mean angles
    twisted  = (not np.isnan(means["trunk_twist"]) and means["trunk_twist"] > 10)
    trunk_s  = trunk_subscore(means["trunk_flex"], twisted=twisted) \
               if not np.isnan(means["trunk_flex"]) else np.nan
    neck_s   = neck_subscore(means["neck_flex"]) \
               if not np.isnan(means["neck_flex"]) else np.nan
    knee     = means.get("knee_flex", np.nan)
    legs_s   = legs_subscore(supported=True,
                             knee_flex_deg=(knee if not np.isnan(knee) else 0.0))

    if not (np.isnan(trunk_s) or np.isnan(neck_s)):
        posture_a = reba_table_a(int(round(trunk_s)), int(round(neck_s)), legs_s)
        score_a   = posture_a + load
    else:
        score_a = np.nan
    feat["reba_score_a"] = score_a

    # Wrist flex from Quest (passed in)
    wf = wrist_flex_deg if (wrist_flex_deg is not None
                            and not np.isnan(wrist_flex_deg)) else 0.0

    for side in ("left", "right"):
        uf = means[f"{side}_upperarm_flex"]
        ab = means[f"{side}_upperarm_abduct"]
        ef = means[f"{side}_elbow_flex"]
        if np.isnan(uf) or np.isnan(ef):
            feat[f"reba_score_b_{side}"] = np.nan
            feat[f"reba_grand_{side}"]   = np.nan
            continue
        ua       = upper_arm_subscore(uf, abducted=(not np.isnan(ab) and ab > 45))
        la       = lower_arm_subscore(ef)
        wr_score = wrist_subscore(wf)
        posture_b = reba_table_b(ua, la, wr_score)
        score_b   = posture_b + coupling
        feat[f"reba_score_b_{side}"] = score_b
        if not np.isnan(score_a):
            feat[f"reba_grand_{side}"] = reba_table_c(score_a, score_b) + activity_static
        else:
            feat[f"reba_grand_{side}"] = np.nan

    feat["reba_n_frames"] = int(len(mask))
    return feat


def hand_features_for_event(t_hand, hand_rows, sides, s, e):
    idx = np.where((t_hand >= s) & (t_hand <= e))[0]
    if len(idx) == 0: return None
    feat = {}
    for side in sides:
        sl   = side
        pref = sl.lower()
        apertures = []; wrist_pts = []; wrist_quats = []
        finger_curl = {fn: [] for fn in FINGERS}
        for i in idx:
            row = hand_rows[i]
            tt  = hand_point(row, sl, "HandThumbTip")
            it  = hand_point(row, sl, "HandIndexTip")
            if tt is not None and it is not None:
                apertures.append(float(np.linalg.norm(tt - it)))
            wr = hand_point(row, sl, "HandWristRoot")
            if wr is not None: wrist_pts.append(wr)
            wq = hand_quat(row, sl, "HandWristRoot")
            if wq is not None: wrist_quats.append(wq)
            for fn, chain in FINGERS.items():
                finger_curl[fn].append(finger_flexion(row, sl, chain))

        m, sd = summarise(apertures)
        feat[f"{pref}_aperture_mean"] = m
        feat[f"{pref}_aperture_sd"]   = sd

        all_curls = []
        for fn in FINGERS:
            mm, _ = summarise(finger_curl[fn])
            feat[f"{pref}_{fn.lower()}_flex_mean"] = mm
            if not np.isnan(mm): all_curls.append(mm)
        feat[f"{pref}_finger_flex_mean"] = (float(np.mean(all_curls))
                                            if all_curls else np.nan)

        if len(wrist_pts) >= 2:
            wp     = np.array(wrist_pts)
            centre = wp.mean(axis=0)
            d      = np.linalg.norm(wp - centre, axis=1)
            feat[f"{pref}_hand_pos_jitter_mm"] = float(np.sqrt(np.mean(d**2)) * 1000.0)
        else:
            feat[f"{pref}_hand_pos_jitter_mm"] = np.nan

        if len(wrist_quats) >= 2:
            wq      = np.array(wrist_quats)
            ref     = wq[0]
            aligned = np.array([q if np.dot(q, ref) >= 0 else -q for q in wq])
            mean_q  = aligned.mean(axis=0); mean_q /= np.linalg.norm(mean_q)
            devs    = [quat_angular_distance(q, mean_q) for q in aligned]
            feat[f"{pref}_hand_orient_jitter_deg"] = float(
                np.sqrt(np.mean(np.square(devs))))
        else:
            feat[f"{pref}_hand_orient_jitter_deg"] = np.nan

        if len(wrist_pts) >= 4:
            feat[f"{pref}_hand_jerk"] = normalised_jerk(
                np.array(wrist_pts), t_hand[idx][:len(wrist_pts)])
        else:
            feat[f"{pref}_hand_jerk"] = np.nan

        # ---- Wrist flexion (HandStart->WristRoot->HandMiddle0) ----
        # Valid for a power/drill grip. MediaPipe finger landmarks are NOT
        # used: during power gripping the knuckles cluster within ~1cm and
        # those vectors are noise-dominated.
        wrist_flex_vals = []
        for i in idx:
            row = hand_rows[i]
            st  = hand_point(row, sl, "HandStart")
            wr2 = hand_point(row, sl, "HandWristRoot")
            m0  = hand_point(row, sl, "HandMiddle0")
            if st is not None and wr2 is not None and m0 is not None:
                fore = wr2 - st; hand_axis = m0 - wr2
                a = angle_between(fore, hand_axis)
                if not np.isnan(a):
                    wrist_flex_vals.append(180.0 - a)
        wf_m, wf_sd = summarise(wrist_flex_vals)
        feat[f"{pref}_wrist_flex_mean"] = wf_m
        feat[f"{pref}_wrist_flex_sd"]   = wf_sd

        # ---- Radial/ulnar deviation ----
        # Knuckle line (Middle0->Pinky0) tilt vs forearm (HandStart->WristRoot).
        # Neutral for drill grip = knuckle row perpendicular to forearm -> 0 deg.
        # Signed: +ve = radial deviation, -ve = ulnar.
        ul_dev_vals = []
        for i in idx:
            row = hand_rows[i]
            wr2 = hand_point(row, sl, "HandWristRoot")
            m0  = hand_point(row, sl, "HandMiddle0")
            p0  = hand_point(row, sl, "HandPinky0")
            st  = hand_point(row, sl, "HandStart")
            if all(v is not None for v in (wr2, m0, p0, st)):
                fore    = wr2 - st;   fn = np.linalg.norm(fore)
                knuckle = m0  - p0;   kn = np.linalg.norm(knuckle)
                if fn > 1e-6 and kn > 1e-6:
                    proj = float(np.dot(knuckle / kn, fore / fn))
                    ul_dev_vals.append(
                        np.degrees(np.arcsin(np.clip(proj, -1, 1))))
        ud_m, ud_sd = summarise(ul_dev_vals)
        feat[f"{pref}_wrist_ulnar_dev_mean"] = ud_m
        feat[f"{pref}_wrist_ulnar_dev_sd"]   = ud_sd

        feat[f"{pref}_hand_n_frames"] = int(len(idx))
    return feat


def normalised_jerk(positions, times):
    if len(positions) < 4 or len(times) < 4: return np.nan
    t = np.asarray(times, dtype=float); p = np.asarray(positions, dtype=float)
    dt = np.diff(t)
    if np.any(dt <= 0):
        keep = np.concatenate([[True], dt > 0]); t, p = t[keep], p[keep]
        if len(t) < 4: return np.nan
    vel  = np.gradient(p, t, axis=0)
    acc  = np.gradient(vel, t, axis=0)
    jerk = np.gradient(acc, t, axis=0)
    duration = t[-1] - t[0]
    if duration <= 0: return np.nan
    path_len = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    if path_len < 1e-9: return np.nan
    jerk_sq = np.sum(np.linalg.norm(jerk, axis=1) ** 2)
    return float(np.sqrt(0.5 * jerk_sq * (duration ** 5) / (path_len ** 2)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--participants", type=str, default=None)
    ap.add_argument("--load",    type=int,   default=0)
    ap.add_argument("--coupling", type=int,  default=0)
    ap.add_argument("--static-seconds", type=float, default=60.0)
    args = ap.parse_args()

    root    = args.landmarks_root
    if not root.is_dir(): sys.exit(f"Not a directory: {root}")
    pfilter = parse_participant_filter(args.participants)
    trials  = list(iter_trials_labelled(root, pfilter))
    if not trials: sys.exit("No trial folders with a labelled pen file found.")

    print(f"Processing {len(trials)} trial(s)")
    print(f"REBA params: load={args.load}, coupling={args.coupling}, "
          f"static>{args.static_seconds}s -> +1 activity\n")

    all_rows = []; by_participant = defaultdict(list); warnings = []

    for stem, pid, trial_dir in trials:
        pen        = find_labelled_pen(trial_dir, stem)
        body_path  = trial_dir / f"{stem}_body.csv"
        hand_path  = trial_dir / f"{stem}_hand.csv"
        places     = place_runs_from_pen(pen)
        if not places:
            print(f"  [   0] {stem}: no Place events"); continue
        height_runs = height_runs_from_pen(pen)

        t_body = joints = confs = None
        if body_path.is_file():
            t_body, joints, confs = load_body(body_path)
            body_angles_at._side  = reliable_side(confs)
        else:
            warnings.append(f"{stem}: no body CSV")

        t_hand = hand_rows = hand_sides = None
        if hand_path.is_file():
            t_hand, hand_rows, _, hand_sides = load_hand(hand_path)
        else:
            warnings.append(f"{stem}: no hand CSV")

        n_events = 0
        for i, (s, e) in enumerate(places, 1):
            dur             = e - s
            activity_static = 1 if dur > args.static_seconds else 0
            row = {"participant": pid, "trial": stem, "place_index": i,
                   "height":     height_for((s + e) / 2, height_runs),
                   "start_t_s":  round(s, 4), "stop_t_s": round(e, 4),
                   "duration_s": round(dur, 4)}

            # Hand first so wrist flex can feed REBA Group B
            hf = None
            if t_hand is not None and hand_sides:
                hf = hand_features_for_event(t_hand, hand_rows, hand_sides, s, e)
                if hf: row.update(hf)

            wrist_flex = None
            if hf:
                pref_side = (body_angles_at._side or "Right").lower()
                wrist_flex = hf.get(f"{pref_side}_wrist_flex_mean")
                if wrist_flex is None:
                    for sd in ("left", "right"):
                        if hf.get(f"{sd}_wrist_flex_mean") is not None:
                            wrist_flex = hf.get(f"{sd}_wrist_flex_mean"); break

            if t_body is not None:
                bf = body_features_for_event(
                    t_body, joints, confs, s, e, BODY_UP,
                    args.load, args.coupling, activity_static,
                    wrist_flex_deg=wrist_flex)
                if bf: row.update(bf)

            all_rows.append(row)
            by_participant[pid].append(row)
            n_events += 1
        print(f"  [{n_events:>4}] {stem}: {n_events} Place event(s)")

    if not all_rows: sys.exit("No Place events found; nothing written.")

    cols  = ["participant", "trial", "place_index", "height",
             "start_t_s", "stop_t_s", "duration_s"]
    for r in all_rows:
        for k in r:
            if k not in cols: cols.append(k)

    out_dir = root / "metrics"; out_dir.mkdir(parents=True, exist_ok=True)

    def write(path, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    combined = out_dir / "posture_features_combined.csv"
    write(combined, all_rows)
    print(f"\nWrote {combined}  ({len(all_rows)} rows, {len(cols)} columns)")
    for pid, rows in by_participant.items():
        write(out_dir / f"{pid}_posture_features.csv", rows)

    manifest = out_dir / "reba_manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("REBA element provenance\n" + "="*50 + "\n")
        f.write(f"load score       : ASSUMED = {args.load}\n")
        f.write(f"coupling score   : ASSUMED = {args.coupling}\n")
        f.write(f"activity(static) : COMPUTED (Place duration >{args.static_seconds}s -> +1)\n")
        f.write("trunk flexion    : MEASURED (MediaPipe, vertical = -X)\n")
        f.write("neck flexion     : APPROXIMATED (ear-vs-trunk lean)\n")
        f.write("upper-arm flex   : MEASURED (both sides)\n")
        f.write("elbow flexion    : MEASURED (both sides)\n")
        f.write("trunk twist      : APPROXIMATED (shoulder vs hip line)\n")
        f.write("upper-arm abduct : APPROXIMATED (sideways component)\n")
        f.write("knee/legs        : COMPUTED (reliable side; symmetry assumed)\n")
        f.write("wrist_neutral_dev: COMPUTED body frame (forearm-up plane dev)\n")
        f.write("reach ratio      : COMPUTED body frame (wrist/arm-length)\n")
        f.write("wrist elevation  : COMPUTED body frame (above shoulder, m)\n")
        f.write("\nGRIP TYPE: power/drill (cylindrical handle)\n")
        f.write("wrist flex (REBA): MEASURED from Quest (HandStart->WristRoot->\n")
        f.write("                   HandMiddle0). MediaPipe finger landmarks NOT\n")
        f.write("                   used (knuckles cluster ~1cm in power grip).\n")
        f.write("wrist ulnar dev  : MEASURED from Quest (knuckle line vs forearm;\n")
        f.write("                   +ve=radial, -ve=ulnar)\n")
        f.write("legs             : standing/bilateral assumed as base\n")
        f.write("shoulder raised / arm supported / side-bends: ASSUMED 0\n")
    print(f"Wrote {manifest}")

    if warnings:
        print("\nWarnings:")
        for w in warnings[:20]: print(f"  {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings)-20} more")

    print("\nNote: REBA sub-scores computed from MEAN angle over each Place hold.")


if __name__ == "__main__":
    main()