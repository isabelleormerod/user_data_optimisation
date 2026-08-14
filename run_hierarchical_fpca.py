#!/usr/bin/env python3
r"""
run_hierarchical_fpca.py

Two-stage automated feature extraction and ranking pipeline. Reads labelled
tracking CSVs in place (same discovery pattern as evaluate_difference.py),
detecting Place events on the fly -- no separate extraction step or manifest.

  Stage 0 (Registration): elastic (SRVF) curve registration, using pen tip
    speed as the single reference signal, one warping function per place
    event applied to every stream. Registered per height stratum against a
    stratum-specific template.

  Stage 1 (Domain fPCA): functional PCA fitted independently per domain
    (Pen, Body, right Hand) and per height stratum, retaining components up
    to a variance-explained threshold (default 90%).

  Block normalisation: each domain's retained component scores are divided
    by that domain's total retained eigenvalue mass, so no domain dominates
    the combined feature vector purely by having more components.

  Stage 2 (LDA per factor): Fisher's linear discriminant fitted separately
    for each design factor (Length, Size, Weight, Angle) within each height
    stratum, using the combined normalised feature vector as predictors.

  Factor-influence leaderboard (per height stratum): factors are ranked by a
    data-driven weight = each factor's summed LDA eigenvalue (summed over its
    n_levels-1 discriminant axes, so 3-level Angle is not penalised), normalised
    across the four factors. The eigenvalue is scatter-normalised (between- over
    within-level scatter), so a factor scores high only if it separates the
    levels far RELATIVE to within-level spread -- "how differently, and how
    consistently, does this factor move people through the condensed feature
    space." A permutation test on the LDA1 separation gives significance, and a
    2x2 PCA scatter (one panel per factor, coloured by level) visualises the
    clustering. The earlier desirability / distance-from-mean config leaderboard
    has been removed in favour of this.

DISCOVERY (matches evaluate_difference.py):
  Walks <root>/<PID>/<trial>/, finds the labelled pen file
  (*_pen_flattened_labelled.csv, else *_pen_labelled.csv). Body/hand streams
  are read as siblings: *_body_labelled.csv / *_hand_labelled.csv if present,
  else *_body.csv / *_hand.csv. Place events are contiguous Place==1 runs in
  the pen file; height per event is read from the overlapping High/Medium/Low
  columns at the event midpoint.

MODES (matches evaluate_difference.py's extract/compare/all split): the costly
step is walking every trial folder and computing every joint-angle curve, so
that step is cached separately from the registration/fPCA/LDA/ranking that
follows, which is comparatively cheap and depends on CLI parameters you're
likely to iterate on.
  --mode extract  : walk trial folders, build per-event curves, cache to disk,
                    then stop.
  --mode compare  : load the cache and run registration/fPCA/LDA/ranking.
                    Fails with a clear message if no cache exists yet.
  --mode all      : both, in one run.

USAGE:
    python run_hierarchical_fpca.py --mode extract --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python run_hierarchical_fpca.py --mode compare --landmarks-root ...
    python run_hierarchical_fpca.py --mode all --landmarks-root ... --participants P002,P003
    python run_hierarchical_fpca.py --mode compare --landmarks-root ... --n-components-hand 5 --n-components-body 2

NOTES / ASSUMPTIONS TO VERIFY:
  - Pen orientation angles (perpendicularity, up/down, left/right) are derived
    from the pen quaternion assuming the pen's local long axis is (0,0,1) and
    world Z is the calibration-plane normal (true once flattened). Confirm this
    matches your rig.
  - Body macro-angles use the same 'up' convention as evaluate_difference.py:
    MediaPipe body 'up' is -X (BODY_UP = [-1,0,0]). Trunk flexion is measured
    against that axis, consistent with your existing posture extraction, rather
    than assuming world-Z-up (which my earlier draft did and which produced a
    suspicious range).
  - Hand input is the FULL right-hand joint-angle set: every internal joint of
    every finger chain as a 3-point flexion angle (19 angles). Body input is
    the full articulated joint-angle set (trunk + elbows/shoulders/knees/hips,
    both sides). These raw angle sets are the fPCA INPUT, not the downstream
    features -- Stage 1 fPCA compresses each stream's full set to that
    domain's own component count (--n-components-pen/-body/-hand, default 3
    each, independently adjustable). Nothing in the raw set is hand-selected
    for importance. Because domains can now be given different component
    counts, the eigenvalue-mass block normalisation (Section 8) is what keeps
    a domain with more retained components from dominating the combined
    feature vector -- not equal component counts.
  - The pen data_quality / dropout column is deliberately NOT used to exclude
    samples, per instruction. Hand tracking is fully present within Place
    windows on the checked data; dropout occurs in idle periods between events.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import eigh

import fdasrsf as fs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
HEIGHTS = ["High", "Medium", "Low"]
VALID_HEIGHTS = ["High", "Medium", "Low"]    # the only height strata ever analysed
USE_ZFILT = True   # set from --no-zfilt in main(); redirects body reads to *_body_zfilt.csv

# Number of fPCA components each domain is compressed to. Per-domain (not one
# shared value) so hand/body/pen -- which differ a lot in raw dimensionality
# and articulation -- can each be tuned independently via CLI flags below.
N_COMPONENTS_DEFAULT = {"pen": 3, "body": 20, "hand": 15}

# 'up' convention copied from evaluate_difference.py so body angles are
# consistent with the existing posture pipeline.
BODY_UP = np.array([-1.0, 0.0, 0.0])   # MediaPipe body: 'up' is -X

# ---------------------------------------------------------------------------
# RAW JOINT-ANGLE DEFINITIONS (the full input set fed into fPCA).
# These are NOT the features used downstream -- they are the complete raw
# curve set that Stage 1 fPCA compresses into each domain's own component
# count (--n-components-pen/-body/-hand). Nothing here is hand-selected for
# importance; it is the full articulated skeleton expressed as joint angles.
# ---------------------------------------------------------------------------

# Hand: every internal joint of every finger chain, as a 3-point flexion angle
# formed by (parent, joint, child) along the chain. Right hand only.
HAND_FINGER_CHAINS = {
    "Index":  ["WristRoot", "Index0", "Index1", "Index2", "Index3", "IndexTip"],
    "Middle": ["WristRoot", "Middle0", "Middle1", "Middle2", "Middle3", "MiddleTip"],
    "Ring":   ["WristRoot", "Ring0", "Ring1", "Ring2", "Ring3", "RingTip"],
    "Pinky":  ["WristRoot", "Pinky0", "Pinky1", "Pinky2", "Pinky3", "PinkyTip"],
    "Thumb":  ["WristRoot", "Thumb1", "Thumb2", "Thumb3", "ThumbTip"],
}


def hand_angle_specs(side="Right"):
    """Return list of (feature_name, parent, joint, child) for every internal
    finger-chain joint angle."""
    specs = []
    for finger, chain in HAND_FINGER_CHAINS.items():
        joints = [f"{side}_Hand{j}" for j in chain]
        for i in range(1, len(joints) - 1):
            name = f"{finger.lower()}_{chain[i].lower()}_deg"
            specs.append((name, joints[i - 1], joints[i], joints[i + 1]))
    return specs


# Body: standard articulated three-point joint angles (both sides).
BODY_ANGLE_SPECS = [
    ("right_elbow_deg",    "RightShoulder", "RightElbow", "RightWrist"),
    ("left_elbow_deg",     "LeftShoulder",  "LeftElbow",  "LeftWrist"),
    ("right_shoulder_deg", "RightHip",      "RightShoulder", "RightElbow"),
    ("left_shoulder_deg",  "LeftHip",       "LeftShoulder",  "LeftElbow"),
    ("right_knee_deg",     "RightHip",      "RightKnee",  "RightAnkle"),
    ("left_knee_deg",      "LeftHip",       "LeftKnee",   "LeftAnkle"),
    ("right_hip_deg",      "RightShoulder", "RightHip",   "RightKnee"),
    ("left_hip_deg",       "LeftShoulder",  "LeftHip",    "LeftKnee"),
]

# Pen orientation angles derived from the quaternion (see pen_curves).
PEN_FEATURES = ["perp_deg", "updown_deg", "leftright_deg"]


# ============================================================================
# 1. Discovery (self-contained; mirrors evaluate_difference.py conventions)
# ============================================================================

def parse_participant_filter(s):
    if not s:
        return None
    return {p.strip() for p in s.split(",") if p.strip()}


def find_labelled_pen(trial_dir: Path):
    for pat in ("*_pen_flattened_labelled.csv", "*_pen_labelled.csv"):
        hits = sorted(trial_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def pen_stem(pen_path: Path) -> str:
    """Recover the trial stem from a labelled pen filename."""
    stem = pen_path.stem
    for suffix in ("_pen_flattened_labelled", "_pen_labelled"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_sibling(trial_dir: Path, stem: str, stream: str):
    """Prefer the median-filtered body (from clean_place_events.py) when present,
    then the labelled sibling, then unlabelled. Only body has a _zfilt variant."""
    names = []
    if USE_ZFILT and stream == "body":
        names.append(f"{stem}_body_zfilt.csv")
    names += [f"{stem}_{stream}_labelled.csv", f"{stem}_{stream}.csv"]
    for name in names:
        p = trial_dir / name
        if p.is_file():
            return p
    return None


def _event_key(participant, trial, height, place_index):
    """Match key against the cleaning manifest; place_index is per-height (== the
    manifest's place_index_in_height)."""
    try:
        pi = int(place_index)
    except (ValueError, TypeError):
        return None
    return (str(participant), str(trial), str(height), pi)


def load_excluded_events(exclude_csv):
    """Load excluded_place_events.csv (clean_place_events.py) as a set of
    (participant, trial, height, place_index) keys. Empty set if absent."""
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


def iter_trials_labelled(root: Path, participant_filter):
    """Yield (stem, pid, trial_dir) for every trial folder under
    <root>/<PID>/<trial>/ that has a labelled pen file."""
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if participant_filter and pid not in participant_filter:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            pen = find_labelled_pen(trial_dir)
            if pen is not None:
                yield pen_stem(pen), pid, trial_dir


def parse_factors_from_stem(stem: str) -> dict:
    """Length/Size/Weight/Angle from the trial stem. Weight left as None if it
    doesn't cleanly match, matching evaluate_difference.py's non-bucketing."""
    out = {k: None for k in PARAM_FACTORS}
    tokens = stem.split("_")
    joined = "_".join(tokens)
    # Weight has two levels only: Front_weighted and Not_weighted. The only
    # filename variation is the casing of the 'weighted' token
    # (Front_weighted / Front_Weighted / Not_weighted / Not_Weighted), so match
    # case-insensitively and canonicalise to the lowercase-'weighted' form.
    low = joined.lower()
    if "not_weighted" in low:
        out["Weight"] = "Not_weighted"
    elif "front_weighted" in low:
        out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:])
            break
    for tok in tokens:
        if tok in ("Long", "Short"):
            out["Length"] = tok
        elif tok in ("Large", "Small"):
            out["Size"] = tok
    return out


# ============================================================================
# 2. Place-event detection & height assignment (on the pen file)
# ============================================================================

def _truthy(v):
    return str(v).strip() in ("1", "1.0", "True", "true")


def detect_place_runs(df):
    runs, in_run, start, prev_t = [], False, None, None
    for _, r in df.iterrows():
        t, flag = r["t_s"], _truthy(r["Place"])
        if flag and not in_run:
            in_run, start = True, t
        elif not flag and in_run:
            in_run = False
            runs.append((start, prev_t))
        prev_t = t
    if in_run:
        runs.append((start, prev_t))
    return runs


def height_runs(df):
    out = {}
    for h in HEIGHTS:
        if h not in df.columns:
            continue
        runs, in_run, start, prev_t = [], False, None, None
        for _, r in df.iterrows():
            t, flag = r["t_s"], _truthy(r[h])
            if flag and not in_run:
                in_run, start = True, t
            elif not flag and in_run:
                in_run = False
                runs.append((start, prev_t))
            prev_t = t
        if in_run:
            runs.append((start, prev_t))
        out[h] = runs
    return out


def make_height_lookup(hruns):
    def get_height(t_mid):
        for h, runs in hruns.items():
            for s, e in runs:
                if s <= t_mid <= e:
                    return h
        return "Unknown"
    return get_height


# ============================================================================
# 3. Geometry helpers
# ============================================================================

def quat_rotate_local_z(qw, qx, qy, qz):
    vx = 2 * (qx * qz + qw * qy)
    vy = 2 * (qy * qz - qw * qx)
    vz = 1 - 2 * (qx * qx + qy * qy)
    return vx, vy, vz


def three_point_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def angle_to_axis(vecs, axis):
    """Angle (deg) between each row vector and a fixed axis."""
    n = np.linalg.norm(vecs, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = (vecs @ axis) / n
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def pts(df, name):
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].values


# ============================================================================
# 4. Domain feature curves
# ============================================================================

def pen_curves(df):
    vx, vy, vz = quat_rotate_local_z(df["qw"].values, df["qx"].values,
                                      df["qy"].values, df["qz"].values)
    perp = np.degrees(np.arccos(np.clip(vz, -1.0, 1.0)))
    updown = np.degrees(np.arctan2(vx, vz))
    leftright = np.degrees(np.arctan2(vy, vz))
    return {"perp_deg": perp, "updown_deg": updown, "leftright_deg": leftright}


def pen_speed(df):
    t = df["t_s"].values
    pos = df[["x_flat", "y_flat", "z_flat"]].values
    if len(t) < 3:
        return np.zeros(len(t))
    vel = np.gradient(pos, t, axis=0)
    return np.linalg.norm(vel, axis=1)


def body_curves(df):
    """Full articulated body joint-angle set (trunk + all specs)."""
    out = {}
    sh_mid = (pts(df, "LeftShoulder") + pts(df, "RightShoulder")) / 2
    hip_mid = (pts(df, "LeftHip") + pts(df, "RightHip")) / 2
    out["trunk_deg"] = angle_to_axis(sh_mid - hip_mid, BODY_UP)
    for name, a, b, c in BODY_ANGLE_SPECS:
        out[name] = three_point_angle(pts(df, a), pts(df, b), pts(df, c))
    return out


def hand_curves(df, side="Right"):
    """Full hand joint-angle set: every internal finger-chain joint as a
    3-point flexion angle."""
    out = {}
    for name, a, b, c in hand_angle_specs(side):
        out[name] = three_point_angle(pts(df, a), pts(df, b), pts(df, c))
    return out


def domain_feature_names():
    """Ordered feature name lists per domain (the full raw input set)."""
    return {
        "pen": list(PEN_FEATURES),
        "body": ["trunk_deg"] + [s[0] for s in BODY_ANGLE_SPECS],
        "hand": [s[0] for s in hand_angle_specs("Right")],
    }


# ============================================================================
# 5. Resampling to a common pre-registration grid
# ============================================================================

def resample_to_grid(t, y, n_grid):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2 or t[-1] == t[0]:
        return np.full(n_grid, y[0] if len(y) else np.nan)
    tn = (t - t[0]) / (t[-1] - t[0])
    tn, idx = np.unique(tn, return_index=True)
    y = y[idx]
    f = interp1d(tn, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
    return f(np.linspace(0.0, 1.0, n_grid))


# ============================================================================
# 6. Per-event curve assembly
# ============================================================================

def slice_window(df, s, e, pad=0.0):
    if df is None or df.empty or "t_s" not in df.columns:
        return None
    sub = df[(df["t_s"] >= s - pad) & (df["t_s"] <= e + pad)]
    return sub if len(sub) > 2 else None


def build_event_curves(df_pen, df_body, df_hand, s, e, n_grid):
    """Compute + resample all domain curves and the pen-speed reference for a
    single Place window. Returns None if the pen window is too short."""
    sub_p = slice_window(df_pen, s, e)
    if sub_p is None:
        return None
    t_pen = sub_p["t_s"].values
    ref = resample_to_grid(t_pen, pen_speed(sub_p), n_grid)

    curves = {}
    for feat, arr in pen_curves(sub_p).items():
        curves[("pen", feat)] = resample_to_grid(t_pen, arr, n_grid)

    feat_names = domain_feature_names()

    sub_b = slice_window(df_body, s, e, pad=0.1)
    if sub_b is not None:
        for feat, arr in body_curves(sub_b).items():
            curves[("body", feat)] = resample_to_grid(sub_b["t_s"].values, arr, n_grid)
    else:
        for feat in feat_names["body"]:
            curves[("body", feat)] = np.full(n_grid, np.nan)

    sub_h = slice_window(df_hand, s, e, pad=0.05)
    if sub_h is not None:
        for feat, arr in hand_curves(sub_h).items():
            curves[("hand", feat)] = resample_to_grid(sub_h["t_s"].values, arr, n_grid)
    else:
        for feat in feat_names["hand"]:
            curves[("hand", feat)] = np.full(n_grid, np.nan)

    return {"ref": ref, "curves": curves}


def collect_events(root, participant_filter, n_grid, exclude_keys=None):
    """Walk all trials, detect Place events, build per-event curves.
    Returns (events_meta DataFrame, event_data dict keyed by meta index)."""
    trials = list(iter_trials_labelled(root, participant_filter))
    if not trials:
        sys.exit(f"ERROR: no trial folders with a labelled pen file under {root}")
    print(f"Discovered {len(trials)} trial folder(s).")

    exclude_keys = exclude_keys or set()
    meta_rows = []
    event_data = {}
    eid = 0
    n_excluded = 0

    for stem, pid, trial_dir in trials:
        pen_path = find_labelled_pen(trial_dir)
        df_pen = pd.read_csv(pen_path)
        if "Place" not in df_pen.columns or "t_s" not in df_pen.columns:
            print(f"  [WARN] {stem}: no Place/t_s column; skipped.")
            continue

        places = detect_place_runs(df_pen)
        if not places:
            print(f"  [WARN] {stem}: 0 Place events.")
            continue

        get_height = make_height_lookup(height_runs(df_pen))
        factors = parse_factors_from_stem(stem)

        body_path = find_sibling(trial_dir, stem, "body")
        hand_path = find_sibling(trial_dir, stem, "hand")
        df_body = pd.read_csv(body_path) if body_path else pd.DataFrame()
        df_hand = pd.read_csv(hand_path) if hand_path else pd.DataFrame()

        heights_for_places = [get_height((s + e) / 2) for s, e in places]
        counters = {}
        for (s, e), h in zip(places, heights_for_places):
            counters[h] = counters.get(h, 0) + 1
            if exclude_keys and _event_key(pid, stem, h, counters[h]) in exclude_keys:
                n_excluded += 1
                continue
            built = build_event_curves(df_pen, df_body, df_hand, s, e, n_grid)
            if built is None:
                continue
            meta_rows.append({
                "participant": pid, "trial": stem, "place_index": counters[h],
                "height": h, "start_t_s": round(s, 4), "stop_t_s": round(e, 4),
                **factors,
            })
            event_data[eid] = built
            eid += 1
        print(f"  [{len(places):>4}] {stem}: {len(places)} Place event(s)")

    if exclude_keys:
        print(f"\n  [exclude] skipped {n_excluded} manifest-rejected place event(s).")

    meta = pd.DataFrame(meta_rows)
    if meta.empty:
        return meta, event_data

    # Quarantine events with no valid working-height label (midpoint fell outside
    # every High/Medium/Low window). Defensive: the HEIGHTS analysis loop already
    # ignores non-H/M/L strata, but this keeps the cache and place_events_meta.csv
    # clean and matters if the cache is ever used pooled / un-stratified.
    if "height" in meta.columns:
        meta["height"] = meta["height"].astype(str).str.strip()
        bad_h = ~meta["height"].isin(VALID_HEIGHTS)
        if bad_h.any():
            print(f"  [QUARANTINE] Dropping {int(bad_h.sum())} event(s) with no valid "
                  f"High/Medium/Low label (height='Unknown'/blank).")
            keep_ids = meta.index[~bad_h]
            event_data = {eid: event_data[eid] for eid in keep_ids if eid in event_data}
            meta = meta.loc[keep_ids].copy()

    # Quarantine any event whose prototype factors did not fully parse from
    # the filename (matches evaluate_difference.py's add_parameter_columns:
    # drop loudly rather than let a spurious 'Other' level reach the LDA and
    # leaderboard). meta's row index is the event id used as the event_data
    # key, so both are filtered in lockstep to stay aligned.
    bad_mask = meta[PARAM_FACTORS].isna().any(axis=1)
    if bad_mask.any():
        bad = meta.loc[bad_mask]
        print(f"\n  [QUARANTINE] Dropping {int(bad_mask.sum())} place event(s) from "
              f"{bad['trial'].nunique()} trial(s) whose Length/Size/Weight/Angle "
              f"could not be cleanly parsed from the filename:")
        for trial, grp in bad.groupby("trial"):
            missing = [f for f in PARAM_FACTORS if grp[f].isna().any()]
            print(f"      '{trial}'  (unparsed: {', '.join(missing)}; {len(grp)} event(s))")
        keep_ids = meta.index[~bad_mask]
        event_data = {eid: event_data[eid] for eid in keep_ids if eid in event_data}
        meta = meta.loc[keep_ids].copy()

    return meta, event_data


# ============================================================================
# 7. Registration (elastic, per height stratum)
# ============================================================================

def register_stratum(event_ids, event_data, n_grid):
    grid = np.linspace(0.0, 1.0, n_grid)
    F = np.array([event_data[eid]["ref"] for eid in event_ids]).T  # (n_grid, n_events)
    obj = fs.fdawarp(F, grid)
    obj.srsf_align(parallel=False, MaxItr=15, verbose=False)
    gam = obj.gam

    registered = {}
    for i, eid in enumerate(event_ids):
        gam_i = gam[:, i]
        reg = {}
        for key, curve in event_data[eid]["curves"].items():
            if np.isnan(curve).all():
                reg[key] = curve
                continue
            interp = interp1d(grid, curve, kind="linear", bounds_error=False,
                               fill_value=(curve[0], curve[-1]))
            reg[key] = interp(gam_i)
        registered[eid] = reg
    return registered


# ============================================================================
# 8. Domain fPCA + block normalisation
# ============================================================================

def domain_fpca(feature_matrix, n_components):
    """Compress a domain's full stacked registered curves to exactly
    n_components data-driven fPCA components. Returns
    (scores, eigenvalues, pca).

    Uses mean-centering only (NOT per-column unit-variance scaling). All
    curves within a domain are already in the same unit (degrees), so
    standardising each time-point column to unit variance would inflate
    low-variance / noisy segments to count equally with the segments that
    carry real between-event movement differences, spreading variance across
    more components. Measured on real body data, centre-only retains slightly
    more variance in the leading components than full standardisation and is
    the textbook fPCA choice."""
    # Impute any remaining NaNs (a fully-missing body/hand event) with the
    # column mean so a single bad stream doesn't drop an otherwise-good event.
    col_mean = np.nanmean(feature_matrix, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(feature_matrix))
    feature_matrix = feature_matrix.copy()
    feature_matrix[inds] = np.take(col_mean, inds[1])

    scaler = StandardScaler(with_std=False)  # centre only, do not rescale
    X = scaler.fit_transform(feature_matrix)
    max_comp = min(X.shape[0], X.shape[1])
    if max_comp < 2:
        return None, None, None
    k = min(n_components, max_comp)
    pca = PCA(n_components=k, random_state=42)
    scores = pca.fit_transform(X)
    return scores, pca.explained_variance_, pca


def normalise_block(scores, eigenvalues):
    total = eigenvalues.sum()
    return scores if total <= 0 else scores / total


# ============================================================================
# 9. LDA per factor + ranking
# ============================================================================

def _lda_eigenvalues(X, labels, ridge=1e-6):
    """All non-trivial LDA eigenvalues (descending) for a labelling of X, or
    None if degenerate. Shared by fit_lda_factor and the permutation test so
    both use identical scatter-matrix maths."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n, d = X.shape
    if len(classes) < 2 or n <= d + len(classes):
        return None
    grand = X.mean(axis=0)
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = X[labels == c]
        if len(Xc) == 0:
            continue
        mc = Xc.mean(axis=0)
        diff = (mc - grand).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)
        Sw += (Xc - mc).T @ (Xc - mc)
    Sw += ridge * np.eye(d)
    try:
        eigvals = eigh(Sb, Sw, eigvals_only=True)
    except np.linalg.LinAlgError:
        return None
    return np.sort(eigvals)[::-1]


def lda_permutation_pvalues(X, labels, n_perm=1000, ridge=1e-6, seed=42):
    """Permutation test for LDA separation, per discriminant axis.

    For each discriminant axis k (there are n_classes-1 of them: 1 for a
    binary factor, 2 for Angle), the observed eigenvalue eta_k measures how
    strongly the factor separates the prototypes along that axis. We shuffle
    the factor labels n_perm times, recompute the axis eigenvalues each time,
    and report the fraction of shuffles whose eta_k meets or exceeds the
    observed -- a valid non-parametric p-value that needs no distributional
    assumptions (appropriate here, where the combined feature vector is not
    a simple named metric). Returns a list of (axis_label, p_value) e.g.
    [('LDA1', 0.003)] for a binary factor, [('LDA1', .), ('LDA2', .)] for
    Angle, or None if the LDA is degenerate for this subset."""
    labels = np.asarray(labels, dtype=object)
    keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l))
                     for l in labels])
    Xv, lv = X[keep], labels[keep]
    obs = _lda_eigenvalues(Xv, lv, ridge)
    if obs is None:
        return None
    n_axes = len(np.unique(lv)) - 1
    obs = obs[:n_axes]

    rng = np.random.default_rng(seed)
    ge_counts = np.zeros(n_axes)
    valid_perms = 0
    for _ in range(n_perm):
        perm = rng.permutation(lv)
        pe = _lda_eigenvalues(Xv, perm, ridge)
        if pe is None:
            continue
        valid_perms += 1
        ge_counts += (pe[:n_axes] >= obs).astype(float)
    if valid_perms == 0:
        return None
    # +1 smoothing (observed counts as one arrangement) -> never reports p=0
    pvals = (ge_counts + 1.0) / (valid_perms + 1.0)
    return [(f"LDA{k+1}", float(pvals[k])) for k in range(n_axes)]


def fit_lda_factor(X, labels, ridge=1e-6):
    """Fisher's LDA via the generalised eigenproblem S_B w = eta S_W w.
    Returns None if fewer than 2 classes or too few samples relative to
    dimensionality; otherwise a dict with eta1 (leading eigenvalue), the
    leading discriminant direction, per-event canonical scores, per-class
    mean canonical scores, and used_index -- the positions (into the ORIGINAL
    X/labels passed in, before any NaN-label filtering) that canonical/
    labels correspond to, so callers can map scores back to event ids."""
    labels = np.asarray(labels, dtype=object)
    keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l))
                     for l in labels])
    used_index = np.where(keep)[0]
    if keep.sum() < len(labels):
        X = X[keep]
        labels = labels[keep]
    classes = np.unique(labels)
    n, d = X.shape
    if len(classes) < 2 or n <= d + len(classes):
        return None

    grand = X.mean(axis=0)
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = X[labels == c]
        if len(Xc) == 0:
            continue
        mc = Xc.mean(axis=0)
        diff = (mc - grand).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)
        Xc_c = Xc - mc
        Sw += Xc_c.T @ Xc_c

    Sw += ridge * np.eye(d)
    try:
        eigvals, eigvecs = eigh(Sb, Sw)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(eigvals)[::-1]
    eta1 = float(eigvals[order][0])
    w1 = eigvecs[:, order[0]]
    canonical = X @ w1
    class_means = {str(c): float(canonical[labels == c].mean()) for c in classes}
    return {"eta1": eta1, "w1": w1, "canonical": canonical, "used_index": used_index,
            "class_means": class_means, "classes": [str(c) for c in classes]}


def _factor_eta_sum(X, labels, ridge=1e-6):
    """Summed LDA eigenvalue for one factor: the sum of the (n_levels-1)
    discriminant-axis eigenvalues (so a 3-level factor like Angle is not
    penalised for spreading its separation across two axes). Scatter-normalised
    by construction (each eigenvalue is between/within scatter). Returns
    (eta_sum, n_levels)."""
    labels = np.asarray(labels, dtype=object)
    keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l)) for l in labels])
    Xv, lv = X[keep], labels[keep]
    classes = np.unique(lv)
    if len(classes) < 2:
        return np.nan, len(classes)
    eigs = _lda_eigenvalues(Xv, lv, ridge)
    if eigs is None:
        return np.nan, len(classes)
    n_axes = len(classes) - 1
    vals = np.clip(np.asarray(eigs[:n_axes], dtype=float), 0.0, None)
    vals = vals[np.isfinite(vals)]
    return (float(np.sum(vals)) if len(vals) else np.nan), len(classes)


def build_factor_leaderboard(X, sub_meta, height, n_perm, pvalue_rows):
    """Per-factor LDA-separation leaderboard for one height stratum.

    Data-driven weight = each factor's summed LDA eigenvalue normalised across
    the four factors -- 'how strongly, and how consistently, does this factor
    move people through the condensed feature space'. eigenvalue is scatter-
    normalised, so a factor scores high only if it separates the levels far
    RELATIVE to within-level spread, not merely along a high-variance direction.
    Also reports an equal-weight benchmark, each level's canonical mean on LDA1
    (which way the level sits), and the permutation p-value.

    Returns (level_rows_df, factor_rows_df, info_by_factor)."""
    info = {}
    for factor in PARAM_FACTORS:
        labels = sub_meta[factor].values
        result = fit_lda_factor(X, labels)
        eta_sum, n_levels = _factor_eta_sum(X, labels)
        perm = lda_permutation_pvalues(X, labels, n_perm=n_perm)
        p1 = np.nan
        if perm is not None:
            for axis_label, pval in perm:
                pvalue_rows.append({"stratum": height, "factor": factor,
                                    "metric": axis_label, "p_value": pval, "n": len(sub_meta)})
                if axis_label == "LDA1":
                    p1 = pval
        if result is None and not (eta_sum == eta_sum):
            continue
        sil = _factor_silhouette(X, labels)
        info[factor] = {"eta_sum": eta_sum, "eta1": (result["eta1"] if result else np.nan),
                        "n_levels": n_levels, "class_means": (result["class_means"] if result else {}),
                        "p_value": p1, "silhouette": sil, "result": result}
    if not info:
        return pd.DataFrame(), pd.DataFrame(), {}

    total = sum(v["eta_sum"] for v in info.values() if v["eta_sum"] == v["eta_sum"])
    n_valid = len(info)
    for v in info.values():
        v["data_weight"] = (v["eta_sum"] / total) if (total and v["eta_sum"] == v["eta_sum"]) else np.nan
        v["equal_weight"] = 1.0 / n_valid

    order = sorted(info, key=lambda k: -(info[k]["data_weight"] if info[k]["data_weight"] == info[k]["data_weight"] else -1))
    frows = []
    for f in order:
        v = info[f]
        frows.append({"height": height, "factor": f, "n_levels": v["n_levels"],
                      "lda_eigenvalue": v["eta_sum"], "data_weight": v["data_weight"],
                      "equal_weight": v["equal_weight"], "silhouette": v["silhouette"],
                      "p_value_lda1": v["p_value"]})
    fdf = pd.DataFrame(frows)
    fdf["rank"] = range(1, len(fdf) + 1)

    lrows = []
    for f in order:
        v = info[f]
        for lev, mean in (v["class_means"] or {}).items():
            lrows.append({"height": height, "factor": f, "level": lev,
                          "level_canonical_mean": mean, "lda_eigenvalue": v["eta_sum"],
                          "data_weight": v["data_weight"], "equal_weight": v["equal_weight"],
                          "silhouette": v["silhouette"], "p_value_lda1": v["p_value"]})
    ldf = pd.DataFrame(lrows)
    return ldf, fdf, info


def print_factor_leaderboard(fdf, height):
    print(f"\n{'='*74}\nFACTOR-INFLUENCE LEADERBOARD -- {height.upper()} "
          f"(weight = normalised summed LDA eigenvalue)\n{'='*74}")
    print(f"  {'Rk':<3} {'Factor':<8} {'Weight':>7} {'Eigenvalue':>11} {'EqualW':>7} {'Sil':>6} {'p(LDA1)':>8} {'Levels':>7}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*11} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")
    for _, r in fdf.iterrows():
        w = f"{r['data_weight']:.3f}" if pd.notna(r["data_weight"]) else "n/a"
        ev = f"{r['lda_eigenvalue']:.4f}" if pd.notna(r["lda_eigenvalue"]) else "n/a"
        sil = f"{r['silhouette']:+.2f}" if pd.notna(r.get("silhouette")) else "n/a"
        p = f"{r['p_value_lda1']:.3f}" if pd.notna(r["p_value_lda1"]) else "n/a"
        print(f"  {int(r['rank']):<3} {r['factor']:<8} {w:>7} {ev:>11} {r['equal_weight']:>7.3f} {sil:>6} {p:>8} {int(r['n_levels']):>7}")
    print("  *(Weight = factor's summed LDA eigenvalue / sum across factors -- how strongly it")
    print("    separates movement. Sil = silhouette of the level clusters in LDA space (higher =")
    print("    tighter/more separated). p = permutation test on LDA1 separation.)*")


def _lda_canonical(X, labels, n_axes=2, ridge=1e-6):
    """Project X onto the top LDA discriminant axes for `labels` (up to
    n_classes-1 of them). Returns (canonical [n_used x k], labels_used, k), or
    (None, None, 0) if the LDA is degenerate for this subset. Same scatter-matrix
    maths as fit_lda_factor, so the picture matches the reported eigenvalue."""
    labels = np.asarray(labels, dtype=object)
    keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l)) for l in labels])
    Xv, lv = X[keep], labels[keep]
    classes = np.unique(lv)
    n, d = Xv.shape
    if len(classes) < 2 or n <= d + len(classes):
        return None, None, 0
    grand = Xv.mean(axis=0)
    Sb = np.zeros((d, d)); Sw = np.zeros((d, d))
    for c in classes:
        Xc = Xv[lv == c]
        mc = Xc.mean(axis=0)
        diff = (mc - grand).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)
        Sw += (Xc - mc).T @ (Xc - mc)
    Sw += ridge * np.eye(d)
    try:
        eigvals, eigvecs = eigh(Sb, Sw)
    except np.linalg.LinAlgError:
        return None, None, 0
    order = np.argsort(eigvals)[::-1]
    k = min(n_axes, len(classes) - 1)
    return Xv @ eigvecs[:, order[:k]], lv, k


def _factor_silhouette(X, labels):
    """Silhouette of a factor's level clusters in its LDA discriminant space
    (how tight/separated the level groups are). Circular by nature -- LDA is fit
    to maximise this -- so read it as a relative measure across factors, with the
    permutation p-value as the honest significance test. NaN if degenerate."""
    can, lv, k = _lda_canonical(X, labels, n_axes=10)
    if can is None:
        return np.nan
    lv = lv.astype(str)
    uniq, counts = np.unique(lv, return_counts=True)
    if len(uniq) < 2 or (counts < 2).any() or len(lv) < 3:
        return np.nan
    try:
        return float(silhouette_score(can.reshape(len(can), -1), lv))
    except Exception:
        return np.nan


def plot_factor_clusters(X, sub_meta, info, verdict, height, out_dir):
    """2x2 grid, one panel per factor, in THAT factor's LDA discriminant space.
    Binary -> 1-D strip along LDA1 with level means; 3-level Angle -> LDA1 vs
    LDA2 scatter. Each level's cluster is annotated with its between-participant
    dispersion, and the title carries weight / eigenvalue / silhouette / p."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, factor in zip(axes.ravel(), PARAM_FACTORS):
        v = info.get(factor, {})
        wt = v.get("data_weight", np.nan); ev = v.get("eta_sum", np.nan)
        p = v.get("p_value", np.nan); sil = v.get("silhouette", np.nan)
        ld = (verdict.get(factor, {}) or {}).get("level_disp", {})
        title = (f"{factor}   w={wt:.2f}  eig={ev:.3f}  sil={sil:+.2f}  p={p:.3f}"
                 if wt == wt else f"{factor}   (LDA not estimable)")
        can, lv, k = _lda_canonical(X, sub_meta[factor].values, n_axes=2)
        if can is None:
            ax.text(0.5, 0.5, "LDA not estimable\n(too few events for dims)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=10); continue
        lv_str = lv.astype(str)
        for lev in [l for l in pd.unique(lv_str) if l != "nan"]:
            m = lv_str == lev
            disp = ld.get(lev, np.nan)
            dtag = f"  (σ={disp:.3f})" if disp == disp else ""
            if k >= 2:
                sc = ax.scatter(can[m, 0], can[m, 1], s=20, alpha=0.55, label=f"{lev}{dtag}")
                cx, cy = can[m, 0].mean(), can[m, 1].mean()
                ax.scatter(cx, cy, marker="X", s=170, edgecolor="black", linewidth=1.3, zorder=5, color=sc.get_facecolor())
            else:
                y = rng.normal(0.0, 0.05, int(m.sum()))
                sc = ax.scatter(can[m, 0], y, s=20, alpha=0.55, label=f"{lev}{dtag}")
                ax.axvline(can[m, 0].mean(), color=sc.get_facecolor()[0], linewidth=2.2, alpha=0.85, zorder=5)
        ax.set_xlabel("LDA1"); ax.set_ylabel("LDA2" if k >= 2 else "(jitter)")
        if k < 2:
            ax.set_yticks([])
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.25); ax.legend(fontsize=8, title="level (σ = between-ppt dispersion)")
    fig.suptitle(f"Factor separation in LDA discriminant space -- {height}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"factor_clusters_{height}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def plot_participant_clusters(X, sub_meta, info, verdict, height, out_dir):
    """2x2 grid, one panel per factor, in that factor's LDA discriminant space.
    DOTS = participants (coloured by participant, so a single person can be
    followed); black CROSSES = level centroids (one per option of that factor,
    labelled with the level name and its between-participant dispersion sigma).
    Reading how tightly the dots cluster around each level cross -- and whether a
    participant's dots shift at all between the level crosses -- is the visual of
    the consistency / 0.001 separation."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    parts_all = sub_meta["participant"].astype(str).values
    uparts = sorted(pd.unique(parts_all))
    cmap = plt.get_cmap("tab20" if len(uparts) > 10 else "tab10")
    pcolor = {p: cmap(i % cmap.N) for i, p in enumerate(uparts)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, factor in zip(axes.ravel(), PARAM_FACTORS):
        labels = sub_meta[factor].values
        can, lv, k = _lda_canonical(X, labels, n_axes=2)
        if can is None:
            ax.text(0.5, 0.5, "LDA not estimable", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{factor}", fontsize=10); continue
        keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l)) for l in labels])
        pk = parts_all[keep]
        lv_str = lv.astype(str)
        ld = (verdict.get(factor, {}) or {}).get("level_disp", {})
        # participant dots
        yj = rng.normal(0.0, 0.06, len(can)) if k < 2 else None
        for p in uparts:
            m = pk == p
            if not m.any():
                continue
            if k >= 2:
                ax.scatter(can[m, 0], can[m, 1], s=16, alpha=0.5, color=pcolor[p])
            else:
                ax.scatter(can[m, 0], yj[m], s=16, alpha=0.5, color=pcolor[p])
        # black crosses = level centroids (one per option)
        for lev in [l for l in pd.unique(lv_str) if l != "nan"]:
            mm = lv_str == lev
            disp = ld.get(lev, np.nan)
            lab = f"{lev}" + (f"  (σ={disp:.3f})" if disp == disp else "")
            cx = can[mm, 0].mean()
            cy = can[mm, 1].mean() if k >= 2 else 0.0
            ax.scatter(cx, cy, marker="X", s=230, color="black", edgecolor="white", linewidth=1.5, zorder=6)
            ax.annotate(lab, (cx, cy), textcoords="offset points",
                        xytext=(6, 8) if k >= 2 else (0, 12), ha="left" if k >= 2 else "center",
                        fontsize=8, fontweight="bold", zorder=7)
        ax.set_xlabel("LDA1"); ax.set_ylabel("LDA2" if k >= 2 else "(jitter)")
        if k < 2:
            ax.set_yticks([])
        ax.set_title(f"{factor}", fontsize=10); ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=pcolor[p], label=p) for p in uparts]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(uparts), 8), fontsize=8,
               title="participant (dot)   |   black X = level centroid (σ = between-participant dispersion)")
    fig.suptitle(f"Participants (dots) vs level centroids (X) in LDA space -- {height}", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = out_dir / f"participants_{height}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def plot_participant_drift(X, sub_meta, info, verdict, height, out_dir):
    """Option C: per-participant DRIFT across the levels of each factor. In each
    panel (one factor) the x-axis is the factor's levels and the y-axis is that
    participant's mean LDA1 centroid at each level; one line per participant. A
    steep line = the factor moves that person a lot; a flat line = it barely does.
    The vertical spread of the lines at each level IS the between-participant
    dispersion (sigma), so: lines piled flat and together = the consistency
    collapse (no drift, no spread); steep and parallel = a strong, consistent
    effect; steep but crossing = strong but inconsistent."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    parts_all = sub_meta["participant"].astype(str).values
    uparts = sorted(pd.unique(parts_all))
    cmap = plt.get_cmap("tab20" if len(uparts) > 10 else "tab10")
    pcolor = {p: cmap(i % cmap.N) for i, p in enumerate(uparts)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, factor in zip(axes.ravel(), PARAM_FACTORS):
        labels = sub_meta[factor].values
        can, lv, k = _lda_canonical(X, labels, n_axes=1)   # LDA1 carries the slope
        v = info.get(factor, {})
        wt = v.get("data_weight", np.nan); ev = v.get("eta_sum", np.nan); p = v.get("p_value", np.nan)
        title = (f"{factor}   w={wt:.2f}  eig={ev:.3f}  p={p:.3f}"
                 if wt == wt else f"{factor}   (LDA not estimable)")
        if can is None:
            ax.text(0.5, 0.5, "LDA not estimable", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=10); continue
        keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l)) for l in labels])
        pk = parts_all[keep]
        lv_str = lv.astype(str)
        levs = [l for l in pd.unique(lv_str) if l != "nan"]
        try:
            levs = sorted(levs, key=lambda s: float(s))
        except ValueError:
            levs = sorted(levs)
        xpos = {lev: i for i, lev in enumerate(levs)}
        for pp in uparts:
            xs, ys = [], []
            for lev in levs:
                mm = (pk == pp) & (lv_str == lev)
                if mm.any():
                    xs.append(xpos[lev]); ys.append(float(can[mm, 0].mean()))
            if xs:
                ax.plot(xs, ys, "-o", color=pcolor[pp], alpha=0.75, ms=5, linewidth=1.4)
        ld = (verdict.get(factor, {}) or {}).get("level_disp_lda1", {})
        ax.set_xticks(range(len(levs)))
        ax.set_xticklabels([f"{lev}\nσ={ld[lev]:.3f}" if ld.get(lev) == ld.get(lev) else lev for lev in levs], fontsize=8)
        ax.set_xlim(-0.3, len(levs) - 0.7)
        ax.set_xlabel(f"{factor} level"); ax.set_ylabel("LDA1 centroid (per participant)")
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="o", color=pcolor[p], label=p) for p in uparts]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(uparts), 8), fontsize=8,
               title="participant (one line = one participant's drift across levels)")
    fig.suptitle(f"Per-participant drift across factor levels (LDA1) -- {height}", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = out_dir / f"participant_drift_{height}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path



def recommend_consistency(X, sub_meta, info, height, min_participants=2, n_perm=200, alpha=0.05):
    """Per-height recommendation by CONSISTENCY (between-participant convergence),
    synthesised one factor at a time -- measured IN THAT FACTOR'S LDA DISCRIMINANT
    SPACE (the same projection the drift plot shows), NOT the block-normalised
    feature blocks. The normalised blocks divide every domain down to the same
    tiny eigenvalue-mass scale, which collapses every dispersion to ~0.001; the
    LDA axes are un-normalised and point along the direction the factor actually
    moves people, so the dispersion there matches the spread you can see by eye.

    For each factor, project events onto its LDA axes, then per level take each
    participant's centroid in that projection and measure the between-participant
    spread (RMS distance from the level centroid). Two versions are reported:
      - dispersion_lda1  : spread along LDA1 only (exactly the drift plot).
      - dispersion_full  : spread across all n_levels-1 axes (so 3-level Angle's
                           second discriminant axis is included).
    The most CONSISTENT level (lowest dispersion_full) is recommended per factor;
    winners combine into a config. Every participant sees every level (within-
    subject), so each estimate uses the full pool.

    CAVEAT: a participant's per-level centroid still pools over the other three
    factors, so a level's dispersion carries some of that; the balanced design
    largely averages it out. And 'most consistent' != 'best ergonomics' -- that
    equivalence is the calibration-free assumption, read alongside the eigenvalue
    weight (a tightly-agreed level of a low-weight factor barely matters)."""
    participants = sub_meta["participant"].values
    rows, verdict = [], {}
    for factor in PARAM_FACTORS:
        if factor not in sub_meta.columns:
            continue
        labels = sub_meta[factor].values
        can1, _, _ = _lda_canonical(X, labels, n_axes=1)
        canF, _, _ = _lda_canonical(X, labels, n_axes=10)
        keep = np.array([l is not None and not (isinstance(l, float) and np.isnan(l)) for l in labels])
        pk = participants[keep]
        lv_str = np.asarray(sub_meta[factor].astype(str).values)[keep]

        def spread(can, mask, pc, uniq):
            cents = np.array([can[mask][pc == p].mean(axis=0) for p in uniq])
            grand = cents.mean(axis=0)
            return float(np.sqrt(np.mean(np.sum((cents - grand) ** 2, axis=1))))

        level_d1, level_df = {}, {}
        for lev in [l for l in pd.unique(lv_str) if l != "nan"]:
            lev_mask = lv_str == lev
            pc = pk[lev_mask]
            uniq = pd.unique(pc)
            if can1 is None or len(uniq) < min_participants:
                level_d1[lev] = level_df[lev] = np.nan
                rows.append({"height": height, "factor": factor, "level": lev,
                             "dispersion_lda1": np.nan, "dispersion_full": np.nan,
                             "n_participants": len(uniq)})
                continue
            d1 = spread(can1, lev_mask, pc, uniq)
            df_ = spread(canF, lev_mask, pc, uniq) if canF is not None else d1
            level_d1[lev] = d1; level_df[lev] = df_
            rows.append({"height": height, "factor": factor, "level": lev,
                         "dispersion_lda1": d1, "dispersion_full": df_,
                         "n_participants": len(uniq)})
        valid = {k: v for k, v in level_df.items() if v == v}
        best = min(valid, key=valid.get) if valid else None

        # within-participant permutation test on the best-minus-worst full-LDA
        # dispersion gap (fixed LDA projection; shuffle level labels within each
        # participant). p < alpha => the levels really differ in convergence.
        def gap_from(lv_arr):
            ds = []
            for lv in [l for l in pd.unique(lv_arr) if l != "nan"]:
                mm = lv_arr == lv
                pc2 = pk[mm]; u2 = pd.unique(pc2)
                if canF is None or len(u2) < min_participants:
                    continue
                ds.append(spread(canF, mm, pc2, u2))
            return (max(ds) - min(ds)) if len(ds) >= 2 else np.nan

        pval = np.nan
        obs_gap = gap_from(lv_str)
        if best is not None and obs_gap == obs_gap:
            rng = np.random.default_rng(0)
            idx_by_p = [np.where(pk == pp)[0] for pp in pd.unique(pk)]
            ge = vv = 0
            for _ in range(n_perm):
                perm = lv_str.copy()
                for idx in idx_by_p:
                    perm[idx] = rng.permutation(lv_str[idx])
                g = gap_from(perm)
                if g == g:
                    vv += 1
                    if g >= obs_gap - 1e-12:
                        ge += 1
            pval = (ge + 1) / (vv + 1) if vv else np.nan
        sig = bool(pval == pval and pval < alpha)
        verdict[factor] = {"winner": best if sig else None, "best_level": best,
                           "level_disp": level_df, "level_disp_lda1": level_d1,
                           "p_value": pval, "significant": sig,
                           "weight": info.get(factor, {}).get("data_weight", np.nan)}
        for r in rows:
            if r["factor"] == factor:
                r.setdefault("p_value", pval)
                r.setdefault("significant", sig)

    synth = "_".join((verdict.get(f, {}).get("winner") or "?") for f in PARAM_FACTORS)
    return pd.DataFrame(rows), verdict, synth


def print_consistency_recommendation(verdict, synth, height):
    print(f"\n  RECOMMENDATION (consistency, measured in LDA space) -- {height}: {synth}")
    for f in PARAM_FACTORS:
        v = verdict.get(f, {})
        if not v or v.get("best_level") is None:
            print(f"      {f:<8}: no verdict (insufficient participants per level)")
            continue
        ordered = sorted(v["level_disp"].items(), key=lambda x: (x[1] if x[1] == x[1] else 9e99))
        ds = ", ".join(f"{lv}={d:.3f}" for lv, d in ordered)
        wt = v.get("weight", np.nan); pval = v.get("p_value", np.nan)
        wts = f"{wt:.2f}" if wt == wt else "n/a"
        ptxt = f"p={pval:.3f}" if pval == pval else "p=n/a"
        if v.get("significant"):
            print(f"      {f:<8}: {v['winner']:<14} weight={wts}  {ptxt}  [disp(full LDA) {ds}]")
        else:
            print(f"      {f:<8}: (n.s.) best={v['best_level']:<9} weight={wts}  {ptxt}  [disp {ds}]  -> no winner")


def add_prototype_label(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    def lab(row):
        parts = [str(row.get(f)) if pd.notna(row.get(f)) else "Other" for f in PARAM_FACTORS]
        return "_".join(parts)
    meta["Prototype_Config"] = meta.apply(lab, axis=1)
    return meta








def print_pvalue_matrix(pvalue_df, strata):
    """Print the factor x metric x stratum p-value matrix in the exact ASCII
    layout of evaluate_difference.py, with LDA discriminant axes (LDA1, LDA2)
    standing in for that script's named metrics."""
    print(f"\nFull p-value matrix (prototype factors x LDA components x stratum):")
    header_str = f"  {'Factor':<10} {'Metric':<35}"
    for s in strata:
        header_str += f" {s:>8}"
    print(header_str)
    sep_str = f"  {'-'*10} {'-'*35}"
    for _ in strata:
        sep_str += f" {'-'*8}"
    print(sep_str)

    metrics = ["LDA1", "LDA2"]
    for factor in PARAM_FACTORS:
        for metric in metrics:
            row_vals = []
            has_data = False
            for s in strata:
                match = pvalue_df[(pvalue_df["stratum"] == s) &
                                  (pvalue_df["factor"] == factor) &
                                  (pvalue_df["metric"] == metric)]
                if match.empty or pd.isna(match.iloc[0]["p_value"]):
                    row_vals.append("     n/a")
                else:
                    p = match.iloc[0]["p_value"]
                    has_data = True
                    marker = "*" if p < 0.05 else " "
                    row_vals.append(f"{p:>7.3f}{marker}")
            if has_data:
                row_str = f"  {factor:<10} {metric:<35}"
                for v in row_vals:
                    row_str += f" {v:>8}"
                print(row_str)

    print("\n  * = p < 0.05  (permutation test on LDA separation, "
          "per discriminant axis, within height stratum)")











def cache_paths(cache_dir: Path):
    return cache_dir / "meta.csv", cache_dir / "curves.npz", cache_dir / "cache_info.csv"


def save_event_cache(cache_dir: Path, meta: pd.DataFrame, event_data: dict, n_grid: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, curves_path, info_path = cache_paths(cache_dir)

    meta_out = meta.reset_index().rename(columns={"index": "event_id"})
    meta_out.to_csv(meta_path, index=False)

    arrays = {}
    for eid, d in event_data.items():
        arrays[f"e{eid}_ref"] = d["ref"]
        for (domain, feat), arr in d["curves"].items():
            arrays[f"e{eid}_{domain}__{feat}"] = arr
    np.savez_compressed(curves_path, **arrays)

    pd.DataFrame([{"n_grid": n_grid}]).to_csv(info_path, index=False)


def load_event_cache(cache_dir: Path, requested_n_grid):
    meta_path, curves_path, info_path = cache_paths(cache_dir)
    if not (meta_path.exists() and curves_path.exists() and info_path.exists()):
        sys.exit(f"ERROR: no cached extraction found at {cache_dir}.\n"
                 f"Run with --mode extract (or --mode all) first.")

    info = pd.read_csv(info_path)
    cached_n_grid = int(info.iloc[0]["n_grid"])
    if requested_n_grid is not None and requested_n_grid != cached_n_grid:
        sys.exit(f"ERROR: cache at {cache_dir} was built with --n-grid {cached_n_grid}, "
                 f"but --n-grid {requested_n_grid} was requested.\n"
                 f"Re-run --mode extract with --n-grid {requested_n_grid}, or omit "
                 f"--n-grid to use the cached value.")

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

    return meta, event_data, cached_n_grid


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["extract", "compare", "all"], default="compare",
                    help="'extract': walk trial folders, build+cache per-event curves, "
                         "then stop. 'compare' (default): load the cache and run "
                         "registration/fPCA/LDA/ranking -- fails with a clear message "
                         "if no cache exists yet. 'all': both, in one run.")
    ap.add_argument("--landmarks-root", type=Path, required=True,
                    help="Root directory containing <PID>/<trial>/ labelled CSVs")
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated participant IDs to include (extract only)")
    ap.add_argument("--n-grid", type=int, default=None,
                    help="Points in the common registration grid (default 100 when "
                         "extracting; when comparing, defaults to whatever the cache "
                         "was built with)")
    ap.add_argument("--n-components-pen", type=int, default=N_COMPONENTS_DEFAULT["pen"],
                    help=f"fPCA components the pen stream is compressed to "
                         f"(default {N_COMPONENTS_DEFAULT['pen']})")
    ap.add_argument("--n-components-body", type=int, default=N_COMPONENTS_DEFAULT["body"],
                    help=f"fPCA components the body stream is compressed to "
                         f"(default {N_COMPONENTS_DEFAULT['body']})")
    ap.add_argument("--n-components-hand", type=int, default=N_COMPONENTS_DEFAULT["hand"],
                    help=f"fPCA components the hand stream is compressed to "
                         f"(default {N_COMPONENTS_DEFAULT['hand']})")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <landmarks-root>/metrics/fpca_results)")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="Permutations for the LDA separation p-value test (default 1000)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Cache dir (default: <landmarks-root>/metrics/fpca_cache)")
    ap.add_argument("--no-zfilt", action="store_true",
                    help="Use raw *_body.csv even when *_body_zfilt.csv exist (extract only)")
    ap.add_argument("--no-exclude", action="store_true",
                    help="Keep events even if in the cleaning manifest (extract only)")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="excluded_place_events.csv (default <root>/metrics/cleaning/excluded_place_events.csv)")
    args = ap.parse_args()

    if not args.landmarks_root.is_dir():
        sys.exit(f"ERROR: {args.landmarks_root} is not a directory")

    out_dir = args.out or (args.landmarks_root / "metrics" / "fpca_results")
    cache_dir = args.cache_dir or (args.landmarks_root / "metrics" / "fpca_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, event_data, n_grid = None, None, None

    if args.mode in ("extract", "all"):
        n_grid = args.n_grid or 100
        pfilter = parse_participant_filter(args.participants)
        global USE_ZFILT
        USE_ZFILT = not args.no_zfilt
        exclude_keys = set()
        if not args.no_exclude:
            exclude_csv = args.exclude_csv or (args.landmarks_root / "metrics" / "cleaning" / "excluded_place_events.csv")
            exclude_keys = load_excluded_events(exclude_csv)
        print(f"{'='*70}\nCOLLECTING PLACE EVENTS (extract)\n{'='*70}")
        print(f"  body source: {'median-filtered (_zfilt) when present' if USE_ZFILT else 'raw (_zfilt disabled)'}")
        meta, event_data = collect_events(args.landmarks_root, pfilter, n_grid, exclude_keys=exclude_keys)
        if meta.empty:
            sys.exit("No Place events collected.")
        meta = add_prototype_label(meta)
        save_event_cache(cache_dir, meta, event_data, n_grid)
        print(f"\nCached {len(meta)} place events -> {cache_dir}")
        print("Per height:", meta["height"].value_counts().to_dict())
        if args.mode == "extract":
            print("\nDone (extract only). Run --mode compare to analyse the cache.")
            return

    if args.mode == "compare":
        meta, event_data, n_grid = load_event_cache(cache_dir, args.n_grid)
        print(f"{'='*70}\nLOADED CACHED PLACE EVENTS\n{'='*70}")
        print(f"{len(meta)} place events from {cache_dir} (n_grid={n_grid})")
        print("Per height:", meta["height"].value_counts().to_dict())

    fpca_summary_rows = []
    factor_level_rows = []
    factor_weight_rows = []
    recommendation_rows = []
    pvalue_rows = []

    for height in HEIGHTS:
        ids = meta.index[meta["height"] == height].tolist()
        if len(ids) < 4:
            print(f"\n[{height}] Skipping -- only {len(ids)} events (need >=4).")
            continue

        print(f"\n{'='*70}\nHEIGHT: {height}  ({len(ids)} events)\n{'='*70}")

        registered = register_stratum(ids, event_data, n_grid)
        print("  Registration complete.")

        feat_names = domain_feature_names()
        n_components_by_domain = {
            "pen": args.n_components_pen,
            "body": args.n_components_body,
            "hand": args.n_components_hand,
        }
        block_scores, block_eigs = {}, {}
        for domain, feats in feat_names.items():
            stacked = np.array([
                np.concatenate([registered[eid][(domain, f)] for f in feats])
                for eid in ids
            ])
            scores, eigvals, pca = domain_fpca(stacked, n_components_by_domain[domain])
            if scores is None:
                print(f"  [{domain}] fPCA skipped (insufficient events).")
                continue
            block_scores[domain] = scores
            block_eigs[domain] = eigvals
            var_pct = np.sum(pca.explained_variance_ratio_) * 100
            fpca_summary_rows.append({
                "height": height, "domain": domain, "input_curves": len(feats),
                "input_dims": stacked.shape[1],
                "components_retained": scores.shape[1],
                "variance_explained_pct": round(var_pct, 1), "n_events": len(ids),
            })
            print(f"  [{domain}] {len(feats)} raw angles ({stacked.shape[1]} grid dims) -> "
                  f"{scores.shape[1]} components ({var_pct:.1f}% variance)")

        if len(block_scores) < 2:
            print(f"  Skipping LDA for {height} -- fewer than 2 usable domains.")
            continue

        normalised = {d: normalise_block(block_scores[d], block_eigs[d]) for d in block_scores}
        X = np.hstack([normalised[d] for d in sorted(normalised)])
        print(f"  Combined feature vector: {X.shape[1]} dims "
              f"({', '.join(f'{d}:{normalised[d].shape[1]}' for d in sorted(normalised))})")

        sub_meta = meta.loc[ids]

        # --- FACTOR-INFLUENCE leaderboard: LDA per factor, ranked by data-driven
        #     weight = normalised summed LDA eigenvalue; permutation p-value for
        #     significance; 2x2 cluster plot as the visual. (Replaces the old
        #     desirability/distance-from-mean config leaderboard.) ---
        ldf, fdf, info = build_factor_leaderboard(X, sub_meta, height, args.n_perm, pvalue_rows)
        if fdf.empty:
            print(f"  No estimable LDA separation for any factor in {height}.")
        else:
            print_factor_leaderboard(fdf, height)
            factor_level_rows.append(ldf)
            factor_weight_rows.append(fdf)

            # Per-height recommendation by CONSISTENCY: per factor, the level with
            # the lowest between-participant dispersion (most convergent), combined
            # into a config; each factor's LDA eigenvalue weight shown alongside.
            rec_rows, rec_verdict, synth = recommend_consistency(X, sub_meta, info, height, n_perm=args.n_perm)
            if not rec_rows.empty:
                print_consistency_recommendation(rec_verdict, synth, height)
                recommendation_rows.append(rec_rows.assign(recommended_config=synth))

            png = plot_factor_clusters(X, sub_meta, info, rec_verdict, height, out_dir)
            if png:
                print(f"  Wrote {png}")
            png_p = plot_participant_clusters(X, sub_meta, info, rec_verdict, height, out_dir)
            if png_p:
                print(f"  Wrote {png_p}")
            png_d = plot_participant_drift(X, sub_meta, info, rec_verdict, height, out_dir)
            if png_d:
                print(f"  Wrote {png_d}")

    # --- outputs ---
    if pvalue_rows:
        pvalue_df = pd.DataFrame(pvalue_rows)
        strata_present = [h for h in HEIGHTS if h in set(pvalue_df["stratum"])]
        print(f"\n{'='*70}\nLDA SEPARATION SIGNIFICANCE (permutation test)\n{'='*70}")
        print_pvalue_matrix(pvalue_df, strata_present)
        pvalue_df.to_csv(out_dir / "stratified_lda_tests.csv", index=False)
        print(f"\nWrote {out_dir / 'stratified_lda_tests.csv'}")

    pd.DataFrame(fpca_summary_rows).to_csv(out_dir / "domain_fpca_summary.csv", index=False)
    print(f"Wrote {out_dir / 'domain_fpca_summary.csv'}")
    if factor_weight_rows:
        pd.concat(factor_weight_rows, ignore_index=True).to_csv(out_dir / "lda_factor_weights.csv", index=False)
        print(f"Wrote {out_dir / 'lda_factor_weights.csv'}")
    if factor_level_rows:
        pd.concat(factor_level_rows, ignore_index=True).to_csv(out_dir / "lda_factor_leaderboard.csv", index=False)
        print(f"Wrote {out_dir / 'lda_factor_leaderboard.csv'}")
    if recommendation_rows:
        allrec = pd.concat(recommendation_rows, ignore_index=True)
        allrec.to_csv(out_dir / "recommended_config_by_height.csv", index=False)
        print(f"Wrote {out_dir / 'recommended_config_by_height.csv'}")
        print(f"\n{'='*64}\nRECOMMENDED CONFIG PER HEIGHT (consistency-synthesised)\n{'='*64}")
        for h in HEIGHTS:
            sub = allrec[allrec["height"] == h]
            if not sub.empty:
                print(f"  {h:<7}: {sub['recommended_config'].iloc[0]}")
    meta.to_csv(out_dir / "place_events_meta.csv", index=False)
    print(f"Wrote {out_dir / 'place_events_meta.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()