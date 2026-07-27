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

  Ranking: design factors weighted two ways -- fixed (equal) and data-driven
    (proportional to each factor's discriminant eigenvalue) -- reported side
    by side.

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

warnings.simplefilter("ignore")

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
HEIGHTS = ["High", "Medium", "Low"]

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
    """Prefer labelled sibling, fall back to unlabelled."""
    for name in (f"{stem}_{stream}_labelled.csv", f"{stem}_{stream}.csv"):
        p = trial_dir / name
        if p.is_file():
            return p
    return None


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


def collect_events(root, participant_filter, n_grid):
    """Walk all trials, detect Place events, build per-event curves.
    Returns (events_meta DataFrame, event_data dict keyed by meta index)."""
    trials = list(iter_trials_labelled(root, participant_filter))
    if not trials:
        sys.exit(f"ERROR: no trial folders with a labelled pen file under {root}")
    print(f"Discovered {len(trials)} trial folder(s).")

    meta_rows = []
    event_data = {}
    eid = 0

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

    meta = pd.DataFrame(meta_rows)
    if meta.empty:
        return meta, event_data

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


def build_ranking(lda_results):
    valid = [f for f, r in lda_results.items() if r is not None]
    if not valid:
        return None, None
    etas = {f: lda_results[f]["eta1"] for f in valid}
    total = sum(etas.values())
    fixed_w = {f: 1.0 / len(valid) for f in valid}
    data_w = {f: (etas[f] / total if total > 0 else fixed_w[f]) for f in valid}
    rows = []
    for f in valid:
        for level, mean_score in lda_results[f]["class_means"].items():
            rows.append({"factor": f, "level": level, "mean_canonical_score": mean_score,
                         "eta1": etas[f], "fixed_weight": fixed_w[f], "data_weight": data_w[f]})
    return pd.DataFrame(rows), {"fixed": fixed_w, "data_driven": data_w}


# ============================================================================
# 10. Prototype leaderboard: distance-from-mean scoring on LDA canonical scores
#
# This is the only thing LDA canonical scores can support: how far a
# prototype's movement sits from the grand mean (0) along each factor's
# discriminant axis. There is no notion of "which direction is better" here
# (unlike rank_prototypes.py's METRIC_REGISTRY, which knows e.g. lower
# duration is better) -- only how atypical a prototype's movement is. The
# working assumption, made explicit rather than silent, is that movement
# closer to the population-average profile is the more ergonomic one, so
# 100 = closest to the mean, 0 = most extreme deviation, same convention as
# the desirability scores in rank_prototypes.py.
# ============================================================================

def add_prototype_label(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    def lab(row):
        parts = [str(row.get(f)) if pd.notna(row.get(f)) else "Other" for f in PARAM_FACTORS]
        return "_".join(parts)
    meta["Prototype_Config"] = meta.apply(lab, axis=1)
    return meta


def build_deviation_table(ids, lda_results):
    """For every event id (in the same order as `ids`), and every factor with
    a valid LDA fit, compute a 0-100 desirability score: 100 = that event's
    canonical score is closest to 0 (the grand mean) among events in this
    height stratum; 0 = furthest. NaN where the event had no usable label for
    that factor. Returns a DataFrame indexed positionally to match `ids`."""
    n = len(ids)
    id_pos = {eid: i for i, eid in enumerate(ids)}
    data = {}
    for factor, result in lda_results.items():
        col = np.full(n, np.nan)
        if result is not None:
            dev = np.abs(result["canonical"])
            v_min, v_max = dev.min(), dev.max()
            desirability = (np.full(len(dev), 100.0) if (v_max - v_min) < 1e-9
                            else 100.0 * (v_max - dev) / (v_max - v_min))
            # result['used_index'] are positions into the sub_meta/ids array
            # passed to fit_lda_factor, in the same order as `ids`.
            for pos, orig_pos in enumerate(result["used_index"]):
                col[orig_pos] = desirability[pos]
        data[f"{factor}_score"] = col
    return pd.DataFrame(data)


def weighted_mean_ignore_nan(row, weights):
    """Weighted average over available (non-NaN) factor scores only, with
    weights renormalised over whichever factors are actually present for this
    event -- an event missing one factor's label still gets a meaningful
    Grand_Score from the factors it does have, rather than being silently
    dragged down by a zero-fill."""
    vals, ws = [], []
    for f, w in weights.items():
        v = row.get(f"{f}_score")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
            ws.append(w)
    if not ws:
        return np.nan
    vals, ws = np.array(vals), np.array(ws)
    return float(np.sum(vals * ws) / np.sum(ws))


def score_and_rank_lda(dev_df, sub_meta, weights):
    """Two-stage aggregation mirroring rank_prototypes.py's score_and_rank:
    event -> (Prototype_Config, participant) mean -> Prototype_Config mean.
    This prevents a participant who happened to contribute more place events
    to a given config from pulling its score toward their own results."""
    df = dev_df.copy()
    df["participant"] = sub_meta["participant"].values
    df["Prototype_Config"] = sub_meta["Prototype_Config"].values
    df["Grand_Score"] = df.apply(lambda r: weighted_mean_ignore_nan(r, weights), axis=1)
    factor_cols = [f"{f}_score" for f in weights]

    stage1 = df.groupby(["Prototype_Config", "participant"], dropna=False).agg(
        Grand_Score=("Grand_Score", "mean"),
        N_Events=("Grand_Score", "size"),
        **{c: (c, "mean") for c in factor_cols},
    ).reset_index()

    stage2 = stage1.groupby("Prototype_Config", dropna=False).agg(
        Grand_Score=("Grand_Score", "mean"),
        Score_SD=("Grand_Score", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        N_Participants=("participant", "nunique"),
        N_Events=("N_Events", "sum"),
        **{c: (c, "mean") for c in factor_cols},
    ).reset_index()

    stage2 = stage2.sort_values("Grand_Score", ascending=False).reset_index(drop=True)
    stage2["Rank"] = stage2.index + 1
    cols = ["Rank", "Prototype_Config", "Grand_Score", "Score_SD",
            "N_Participants", "N_Events"] + factor_cols
    return stage2[cols]


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


def print_ascii_leaderboard(rankings, title, top_n=10):
    """Verbatim layout of rank_prototypes.py's print_ascii_leaderboard so the
    two scripts' leaderboards are directly comparable side by side. The
    post-bar columns here are the four prototype-factor distance-from-mean
    scores (Length/Size/Weight/Angle), occupying the same slot rank_prototypes
    uses for its ergonomic domains."""
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    domains = [c for c in rankings.columns
               if c not in ("Rank", "Prototype_Config", "Grand_Score",
                            "Score_SD", "N_Participants", "N_Events")]
    domain_headers = "  ".join([f"{d.replace('_score',''):>10}" for d in domains])

    print(f" {'Rk':<3} {'Prototype Configuration':<32} {'Grand':>6} {'(SD)':>6} {'Np':>3} {'Ne':>4} | {domain_headers}")
    print(f" {'-'*3} {'-'*32} {'-'*6} {'-'*6} {'-'*3} {'-'*4}-+-{'-'*len(domain_headers)}")

    for _, r in rankings.head(top_n).iterrows():
        domain_vals = "  ".join([f"{r[d]:>10.1f}" if pd.notna(r[d]) else f"{'n/a':>10}" for d in domains])
        np_val = r.get("N_Participants", np.nan)
        np_str = f"{int(np_val):>3}" if pd.notna(np_val) else "  ?"
        print(f" {int(r['Rank']):<3} {r['Prototype_Config']:<32} {r['Grand_Score']:>6.1f} ({r['Score_SD']:>4.1f}) {np_str} {int(r['N_Events']):>4} | {domain_vals}")

    if len(rankings) > top_n:
        print(f" ... and {len(rankings) - top_n} more configurations.")
    print(f" *(Scores 0-100; Np = distinct participants contributing to this config -- typically 2-4")
    print(f"   of 10, given the incomplete-block allocation. Individual config ranks therefore rest on")
    print(f"   substantially weaker evidence than the factor-level verdicts above.)*")


# ============================================================================
# 11. Extraction cache (avoids re-walking/re-parsing raw trial CSVs on every
# run -- mirrors evaluate_difference.py's extract/compare split). The costly
# step is collect_events(): reading every trial's labelled CSVs, computing
# every joint-angle curve, and resampling to the grid. Registration/fPCA/LDA/
# ranking are comparatively cheap and depend on CLI parameters, so only the
# pre-registration per-event curves + metadata are cached.
# ============================================================================

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
        print(f"{'='*70}\nCOLLECTING PLACE EVENTS (extract)\n{'='*70}")
        meta, event_data = collect_events(args.landmarks_root, pfilter, n_grid)
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
    ranking_tables = {}
    leaderboards = {}
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

        print("\n  [OUTLIER DETECTION] Scanning fPCA space for extreme kinematic anomalies (Z > 3.0)...")
        sub_meta = meta.loc[ids]
        
        for domain, scores in block_scores.items():
            # Calculate absolute Z-score for every event on every component
            z_scores = np.abs(scores) / np.std(scores, axis=0)
            
            # Find row indices where ANY component exceeds 3 standard deviations
            glitch_indices = np.where(np.any(z_scores > 3.0, axis=1))[0]
            
            if len(glitch_indices) > 0:
                print(f"    -> {domain.upper()} domain: {len(glitch_indices)} anomaly(s) flagged.")
                for idx in glitch_indices:
                    eid = ids[idx]
                    row = sub_meta.loc[eid]
                    # Find which specific fPCA component triggered the flag
                    max_comp = np.argmax(z_scores[idx]) 
                    max_z = z_scores[idx, max_comp]
                    print(f"       {row['participant']} | Trial: {row['trial']} | "
                          f"PC{max_comp+1} Z-Score: {max_z:.1f}")
            else:
                print(f"    -> {domain.upper()} domain: Clean (no Z > 3 anomalies).")
        print("  " + "-"*68)

        normalised = {d: normalise_block(block_scores[d], block_eigs[d]) for d in block_scores}
        X = np.hstack([normalised[d] for d in sorted(normalised)])
        print(f"  Combined feature vector: {X.shape[1]} dims "
              f"({', '.join(f'{d}:{normalised[d].shape[1]}' for d in sorted(normalised))})")

        sub_meta = meta.loc[ids]
        lda_results = {}
        for factor in PARAM_FACTORS:
            result = fit_lda_factor(X, sub_meta[factor].values)
            lda_results[factor] = result
            if result is None:
                print(f"  [LDA:{factor}] skipped (insufficient levels/samples).")
            else:
                print(f"  [LDA:{factor}] eta1={result['eta1']:.4f}, classes={result['classes']}")
            # Permutation p-value per discriminant axis (LDA1, LDA2, ...)
            perm = lda_permutation_pvalues(X, sub_meta[factor].values, n_perm=args.n_perm)
            if perm is not None:
                for axis_label, pval in perm:
                    pvalue_rows.append({"stratum": height, "factor": factor,
                                        "metric": axis_label, "p_value": pval,
                                        "n": len(ids)})

        ranking_df, weights = build_ranking(lda_results)
        if ranking_df is not None:
            ranking_tables[height] = ranking_df
            print("\n  Factor weights (fixed vs data-driven):")
            for f in weights["fixed"]:
                print(f"    {f:8s}  fixed={weights['fixed'][f]:.3f}  "
                      f"data-driven={weights['data_driven'][f]:.3f}")

            dev_df = build_deviation_table(ids, lda_results)
            for scheme_name, scheme_weights in weights.items():
                lb = score_and_rank_lda(dev_df, sub_meta, scheme_weights)
                leaderboards[(height, scheme_name)] = lb
                title = f"PROTOTYPE LEADERBOARD -- {height.upper()} ({scheme_name.replace('_',' ').upper()} WEIGHTS)"
                print_ascii_leaderboard(lb, title, top_n=10)
                lb.to_csv(out_dir / f"leaderboard_{height}_{scheme_name}.csv", index=False)

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
    if ranking_tables:
        allr = pd.concat([df.assign(height=h) for h, df in ranking_tables.items()],
                         ignore_index=True)
        allr.to_csv(out_dir / "lda_factor_scores.csv", index=False)
        print(f"Wrote {out_dir / 'lda_factor_scores.csv'}")
    meta.to_csv(out_dir / "place_events_meta.csv", index=False)
    print(f"Wrote {out_dir / 'place_events_meta.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()