#!/usr/bin/env python3
"""
Functional PCA for prototype clustering analysis.

Applies fPCA to raw time-series data from all three tracking streams
(body landmarks, hand skeleton, pen) during Place events, to ask:
  Do different prototype configurations produce distinct movement signatures?

Approach (two-stage):
  1. Per-signal fPCA: each of the 25 signals is treated as a function over
     normalised placement time [0,1]. fPCA extracts the principal modes of
     variation in curve shape (not just means/SDs). Each Place event gets a
     score on each functional PC.
  2. Cross-signal PCA: the per-signal FPC scores are concatenated and a final
     PCA reduces them to a small number of components capturing the dominant
     multi-stream variation patterns.

Two analyses are produced:
  A. STRATIFIED: fPCA run separately within each height stratum (High/Medium/Low).
     Height is removed by design. Any structure in PC space reflects prototype
     or participant differences.
  B. FULL-DATA: fPCA on all events with height partialled out (residualised)
     before PCA. Keeps more data but requires the height-removal step.

Signals extracted per-frame during each Place event:
  Body (8): trunk_flex, neck_flex, right_upperarm_flex, right_elbow_flex,
            right_upperarm_abduct, wrist_neutral_dev, reach_ratio, wrist_elevation_m
  Hand (10): wrist_flex, wrist_ulnar_dev, aperture, finger flexion ×4,
             wrist position x/y/z (world, captures gross trajectory)
  Pen  (7):  position x/y/z in calibration plane, orientation qw/qx/qy/qz

Each signal is resampled to N_GRID=30 normalised time points, then smoothed
with a B-spline basis before fPCA.

Outputs (under <root>/metrics/fpca/):
  fpca_scores.csv                   FPC scores per Place event (all analyses)
  scree_<analysis>.png              Variance explained per component
  biplot_<analysis>_<colour>.png    PC1 vs PC2 coloured by factor
  loadings_<analysis>.png           FPC loading curves (what PC1/PC2 represent)
  fpca_summary.txt                  Which PCs discriminate which factors

Dependencies:
    pip install scikit-fda scikit-learn scipy matplotlib pandas numpy

Usage:
    python 12_fpca_prototype_clustering.py --landmarks-root "A:\\Automated_chain_BETA\\Participant_Landmarks"
    python 12_fpca_prototype_clustering.py \\
        --landmarks-root ... --participants P003,P004 --n-grid 30
"""

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import skfda
    from skfda.representation.basis import BSplineBasis
    from skfda.preprocessing.dim_reduction import FPCA
    HAVE_SKFDA = True
except ImportError:
    HAVE_SKFDA = False
    print("WARNING: scikit-fda not found. Install with: pip install scikit-fda")
    print("         Falling back to standard PCA on resampled curves.")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_GRID       = 30      # normalised time points per event
N_FPCS       = 3       # fPCs to extract per signal
N_FINAL_PCS  = 10      # final PCs after cross-signal reduction
MIN_FRAMES   = 10      # minimum frames in a Place event to include
CONF_MIN     = 0.3
BODY_UP      = np.array([-1.0, 0.0, 0.0])  # MediaPipe: up = -X

PROTO_FACTORS = ["Length", "Size", "Weight", "Angle"]
HEIGHT_ORDER  = {"High": 0, "Medium": 1, "Low": 2}
HEIGHT_COLS   = {"High": "#e66100", "Medium": "#5d3a9b", "Low": "#1a85ff"}

SIGNAL_NAMES = {
    "body": ["trunk_flex", "neck_flex", "r_ua_flex", "r_elbow_flex",
             "r_ua_abduct", "wrist_neutral_dev", "reach_ratio", "wrist_elev"],
    "hand_left":  [f"L_{s}" for s in ["wrist_flex", "wrist_ulnar_dev", "aperture",
                                        "idx_flex", "mid_flex", "ring_flex", "pinky_flex",
                                        "wr_x", "wr_y", "wr_z"]],
    "hand_right": [f"R_{s}" for s in ["wrist_flex", "wrist_ulnar_dev", "aperture",
                                        "idx_flex", "mid_flex", "ring_flex", "pinky_flex",
                                        "wr_x", "wr_y", "wr_z"]],
    "pen":  ["pen_x", "pen_y", "pen_z",
             "pen_qw", "pen_qx", "pen_qy", "pen_qz"],
}
ALL_SIGNALS = (SIGNAL_NAMES["body"]
               + SIGNAL_NAMES["hand_left"]
               + SIGNAL_NAMES["hand_right"]
               + SIGNAL_NAMES["pen"])


# --------------------------------------------------------------------------- #
# Parameter parsing
# --------------------------------------------------------------------------- #
def parse_params(trial: str) -> dict:
    out = {k: None for k in PROTO_FACTORS}
    tokens = trial.split("_"); joined = "_".join(tokens)
    if "Not_weighted"   in joined: out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined: out["Weight"] = "Front_weighted"
    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:]); break
    for tok in tokens:
        if tok in ("Long", "Short"):    out["Length"] = tok
        elif tok in ("Large", "Small"): out["Size"]   = tok
    return out


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def parse_float(s):
    try: return float(s)
    except: return None

def read_csv_t(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        delim = "\t" if first.count("\t") >= first.count(",") else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        fields = list(reader.fieldnames or [])
        for r in reader: rows.append(r)
    return rows, fields

def place_runs(rows, fields):
    if "Place" not in fields or "t_s" not in fields: return []
    runs = []; in_run = False; t0 = None; prev_t = None
    for r in rows:
        t = parse_float(r.get("t_s")); flag = str(r.get("Place","")).strip() in ("1","1.0")
        if t is None: continue
        if flag and not in_run:  in_run = True;  t0 = t
        elif not flag and in_run: in_run = False; runs.append((t0, prev_t))
        prev_t = t
    if in_run: runs.append((t0, prev_t))
    return runs

def height_at(t_mid, rows):
    """Find the height label at t_mid by scanning rows for the nearest time."""
    best_t = None; best_h = "Unknown"
    for r in rows:
        t = parse_float(r.get("t_s"))
        if t is None: continue
        if best_t is None or abs(t - t_mid) < abs(best_t - t_mid):
            for h in ("High", "Medium", "Low"):
                if str(r.get(h, "")).strip() in ("1", "1.0"):
                    best_t = t; best_h = h; break
    return best_h


# --------------------------------------------------------------------------- #
# Geometry (same as 07_extract_posture_features.py)
# --------------------------------------------------------------------------- #
def ang(v1, v2):
    n1,n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-9 or n2<1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)))

def finger_curl(pts):
    if any(p is None for p in pts): return np.nan
    total = 0.0
    for k in range(1, len(pts)-1):
        a = ang(pts[k-1]-pts[k], pts[k+1]-pts[k])
        if not np.isnan(a): total += 180.0 - a
    return total


# --------------------------------------------------------------------------- #
# Per-frame signal extraction
# --------------------------------------------------------------------------- #
def body_signals_at(row, conf_fields):
    """Extract 8 body signals from a single body CSV row."""
    up = BODY_UP
    def gp(name):
        try:
            p = np.array([float(row[f"{name}_x"]),
                          float(row[f"{name}_y"]),
                          float(row[f"{name}_z"])])
            if conf_fields:
                c = float(row.get(f"{name}_conf", 1.0))
                if c < CONF_MIN: return None
            return p
        except: return None

    ls,rs = gp("LeftShoulder"), gp("RightShoulder")
    lh,rh = gp("LeftHip"),      gp("RightHip")
    le,re = gp("LeftEar"),      gp("RightEar")
    sh  = gp("RightShoulder"); el = gp("RightElbow"); wr = gp("RightWrist")
    idx = gp("RightIndex")

    out = [np.nan]*8

    # 0 trunk_flex
    if all(v is not None for v in (ls,rs,lh,rh)):
        sh_m = (ls+rs)/2; hip_m = (lh+rh)/2
        out[0] = ang(sh_m-hip_m, up)
    # 1 neck_flex
    if all(v is not None for v in (ls,rs,le,re,lh,rh)):
        sh_m=(ls+rs)/2; ear_m=(le+re)/2
        nfv = ang(ear_m-sh_m, up)
        tfv = ang(sh_m-(lh+rh)/2, up)
        out[1] = nfv - tfv if not (np.isnan(nfv) or np.isnan(tfv)) else np.nan
    # 2,3,4 right arm
    if all(v is not None for v in (sh,el)):
        tv = ((ls+rs)/2-(lh+rh)/2) if all(v is not None for v in (ls,rs,lh,rh)) else up
        out[2] = ang(el-sh, -tv)
        sa = (rs-ls) if all(v is not None for v in (ls,rs)) else np.array([0.,0.,1.])
        out[4] = abs(90 - ang(el-sh, sa))
    if all(v is not None for v in (sh,el,wr)):
        out[3] = ang(sh-el, wr-el)
    # 5 wrist_neutral_dev
    if all(v is not None for v in (sh,el,wr,idx)):
        fore = wr-el; fnt = np.linalg.norm(fore)
        hv   = idx-wr; hn  = np.linalg.norm(hv)
        if fnt>1e-6 and hn>1e-6:
            lat = np.cross(fore/fnt, up); ln = np.linalg.norm(lat)
            if ln>1e-6:
                out[5] = np.degrees(np.arcsin(np.clip(abs(np.dot(hv/hn,lat/ln)),0,1)))
    # 6 reach_ratio
    if all(v is not None for v in (sh,el,wr)):
        arm = np.linalg.norm(el-sh)+np.linalg.norm(wr-el)
        out[6] = float(np.linalg.norm(wr-sh)/arm) if arm>1e-6 else np.nan
    # 7 wrist_elevation_m
    if all(v is not None for v in (sh,wr)):
        out[7] = float(np.dot(sh-wr, up))
    return out


def hand_signals_at(row, side):
    """Extract 10 hand signals from a single hand CSV row for one side."""
    def hp(j):
        try:
            return np.array([float(row[f"{side}_{j}_x"]),
                             float(row[f"{side}_{j}_y"]),
                             float(row[f"{side}_{j}_z"])])
        except: return None

    wr = hp("HandWristRoot"); st = hp("HandStart"); m0 = hp("HandMiddle0")
    p0 = hp("HandPinky0")
    tt = hp("HandThumbTip"); it = hp("HandIndexTip")
    chains = {
        "idx":  [hp(f"HandIndex{k}") for k in range(1,4)] + [hp("HandIndexTip")],
        "mid":  [hp(f"HandMiddle{k}") for k in range(1,4)] + [hp("HandMiddleTip")],
        "ring": [hp(f"HandRing{k}")   for k in range(1,4)] + [hp("HandRingTip")],
        "pinky":[hp(f"HandPinky{k}")  for k in range(1,4)] + [hp("HandPinkyTip")],
    }
    out = [np.nan]*10
    if all(v is not None for v in (st,wr,m0)):
        a = ang(wr-st, m0-wr)
        if not np.isnan(a): out[0] = 180.0-a
    if all(v is not None for v in (wr,m0,p0,st)):
        fore=wr-st; kn=m0-p0
        fn=np.linalg.norm(fore); knn=np.linalg.norm(kn)
        if fn>1e-6 and knn>1e-6:
            out[1] = np.degrees(np.arcsin(np.clip(np.dot(kn/knn,fore/fn),-1,1)))
    if all(v is not None for v in (tt,it)):
        out[2] = float(np.linalg.norm(tt-it)*1000)
    for k,(name,chain) in enumerate(chains.items()):
        out[3+k] = finger_curl(chain)
    if wr is not None:
        out[7],out[8],out[9] = float(wr[0]),float(wr[1]),float(wr[2])
    return out


def pen_signals_at(row):
    """Extract 7 pen signals (position + quaternion) from a pen row."""
    out = [np.nan]*7
    for k,(col,idx) in enumerate([("x",0),("y",1),("z",2),
                                    ("qw",3),("qx",4),("qy",5),("qz",6)]):
        v = parse_float(row.get(col))
        if v is not None: out[idx] = v
    return out


# --------------------------------------------------------------------------- #
# Event extraction — build matrix (N_GRID × N_SIGNALS) per Place event
# --------------------------------------------------------------------------- #
def extract_event(t_start, t_end, body_rows, hand_rows, pen_rows,
                  body_fields, hand_fields, n_grid):
    """Resample all signals for one Place event to n_grid normalised time pts."""

    def rows_in_window(rows):
        return [(parse_float(r.get("t_s")), r)
                for r in rows if parse_float(r.get("t_s")) is not None
                and t_start <= parse_float(r.get("t_s")) <= t_end]

    body_w = rows_in_window(body_rows) if body_rows else []
    hand_w = rows_in_window(hand_rows) if hand_rows else []
    pen_w  = rows_in_window(pen_rows)  if pen_rows  else []

    if len(pen_w) < MIN_FRAMES and len(hand_w) < MIN_FRAMES:
        return None

    def resample_stream(tw, sig_fn, n_sigs):
        if len(tw) < 2:
            return np.full((n_grid, n_sigs), np.nan)
        ts  = np.array([t for t,_ in tw])
        t_n = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-12)   # normalise to [0,1]
        t_g = np.linspace(0, 1, n_grid)
        mat = np.full((n_grid, n_sigs), np.nan)
        sigs_raw = np.array([sig_fn(r) for _,r in tw], dtype=float)
        for k in range(n_sigs):
            col = sigs_raw[:, k]
            valid = ~np.isnan(col)
            if valid.sum() < 2: continue
            f = interp1d(t_n[valid], col[valid], kind="linear",
                         bounds_error=False, fill_value=(col[valid][0], col[valid][-1]))
            mat[:, k] = f(t_g)
        return mat

    conf_f = any(f.endswith("_conf") for f in body_fields) if body_fields else False
    body_m = resample_stream(body_w, lambda r: body_signals_at(r, conf_f), 8)

    # Both hand sides — each produces 10 signals, stacked as (n_grid, 20)
    hand_left_m  = resample_stream(hand_w, lambda r: hand_signals_at(r, "Left"),  10)
    hand_right_m = resample_stream(hand_w, lambda r: hand_signals_at(r, "Right"), 10)

    pen_m  = resample_stream(pen_w, pen_signals_at, 7)

    return np.hstack([body_m, hand_left_m, hand_right_m, pen_m])  # (n_grid, 35)


# --------------------------------------------------------------------------- #
# fPCA core
# --------------------------------------------------------------------------- #
def run_fpca(X, signal_names, n_fpcs=N_FPCS, n_grid=N_GRID):
    """
    Two-stage fPCA:
      1. Run fPCA independently on each signal's curves across events.
      2. Concatenate per-signal FPC scores; run a final PCA.

    X: (n_events, n_grid, n_signals)
    Returns (final_scores, explained_ratios, per_signal_loadings)
    """
    t_grid = np.linspace(0, 1, n_grid)
    all_scores   = []
    all_var_exp  = []
    per_sig_info = []

    for k, sname in enumerate(signal_names):
        curves = X[:, :, k].copy()   # (n_events, n_grid)

        # ---- robust NaN handling ----
        # 1. Per-event: if an event has <50% valid points, fill with global median
        #    per time-point; if >50% valid, interpolate within the event.
        for i in range(curves.shape[0]):
            row = curves[i]
            valid = ~np.isnan(row)
            if valid.sum() == 0:
                continue   # handled below
            if valid.sum() < n_grid * 0.5:
                # too sparse — fill with column medians (computed after)
                pass
            else:
                # interpolate within the event
                t_idx = np.arange(n_grid)
                f = interp1d(t_idx[valid], row[valid], kind="linear",
                             bounds_error=False,
                             fill_value=(row[valid][0], row[valid][-1]))
                curves[i] = f(t_idx)

        # 2. Column-wise median across events (ignore remaining NaNs)
        col_med = np.nanmedian(curves, axis=0)   # (n_grid,)

        # 3. If the whole signal is NaN (e.g. body stream absent),
        #    skip this signal entirely.
        if np.all(np.isnan(col_med)):
            print(f"    [skip] signal '{sname}' — all NaN, stream unavailable")
            # Insert zero scores as placeholder so array shapes stay consistent
            placeholder = np.zeros((curves.shape[0], min(n_fpcs, 1)))
            all_scores.append(placeholder)
            all_var_exp.append(np.array([0.0]))
            per_sig_info.append({"name": sname + " [N/A]",
                                  "loadings": np.zeros((1, n_grid)),
                                  "var_exp": np.array([0.0]),
                                  "t_grid": t_grid})
            continue

        # 4. Fill any remaining NaNs with column medians
        for j in range(curves.shape[1]):
            mask = np.isnan(curves[:, j])
            if mask.any():
                fill = col_med[j] if not np.isnan(col_med[j]) else 0.0
                curves[mask, j] = fill

        # Final safety check
        if np.any(np.isnan(curves)) or np.any(np.isinf(curves)):
            curves = np.nan_to_num(curves, nan=0.0, posinf=0.0, neginf=0.0)

        if HAVE_SKFDA:
            fd = skfda.FDataGrid(data_matrix=curves, grid_points=t_grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n_comp = min(n_fpcs, len(curves) - 1, curves.shape[1])
                if n_comp < 1:
                    all_scores.append(np.zeros((len(curves), 1)))
                    all_var_exp.append(np.array([0.0]))
                    per_sig_info.append({"name": sname,
                                         "loadings": np.zeros((1, n_grid)),
                                         "var_exp": np.array([0.0]),
                                         "t_grid": t_grid})
                    continue
                fpca = FPCA(n_components=n_comp)
                scores  = fpca.fit_transform(fd)
                loadings = np.array([fpca.components_.data_matrix[i, :, 0]
                                     for i in range(fpca.n_components)])
                var_exp  = fpca.explained_variance_ratio_
        else:
            scaler = StandardScaler()
            C = scaler.fit_transform(curves)
            n_comp = min(n_fpcs, len(curves) - 1, curves.shape[1])
            pca = PCA(n_components=max(1, n_comp))
            scores   = pca.fit_transform(C)
            loadings = pca.components_.reshape(pca.n_components_, n_grid)
            var_exp  = pca.explained_variance_ratio_

        all_scores.append(scores)
        all_var_exp.append(var_exp)
        per_sig_info.append({"name": sname, "loadings": loadings,
                              "var_exp": var_exp, "t_grid": t_grid})

    # Concatenate all per-signal FPC scores
    scores_cat = np.hstack(all_scores)

    # Final safety imputation before second-stage PCA
    if np.any(np.isnan(scores_cat)):
        col_med2 = np.nanmedian(scores_cat, axis=0)
        col_med2 = np.where(np.isnan(col_med2), 0.0, col_med2)
        idx_nan  = np.where(np.isnan(scores_cat))
        scores_cat[idx_nan] = np.take(col_med2, idx_nan[1])

    # Final PCA to reduce further and orthogonalise
    n_final = min(N_FINAL_PCS, scores_cat.shape[0] - 1, scores_cat.shape[1])
    scaler2 = StandardScaler()
    scores_sc  = scaler2.fit_transform(scores_cat)
    pca_final  = PCA(n_components=max(1, n_final))
    final_scores = pca_final.fit_transform(scores_sc)

    return final_scores, pca_final.explained_variance_ratio_, per_sig_info


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
MARKER_MAP = {"Long":"o","Short":"s",
              "Large":"^","Small":"v",
              "Front_weighted":"D","Not_weighted":"o",
              90:"o",135:"s",180:"^"}

def scatter_pcs(scores, meta_df, colour_col, title, out_path,
                pc_x=0, pc_y=1, expl=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = sorted(meta_df[colour_col].dropna().unique(), key=str)
    palette = cm.get_cmap("tab10").resampled(max(len(levels),1))
    colour_map = {lv: palette(i) for i,lv in enumerate(levels)}

    for lv in levels:
        mask = meta_df[colour_col] == lv
        ax.scatter(scores[mask, pc_x], scores[mask, pc_y],
                   color=colour_map[lv], label=str(lv),
                   alpha=0.65, s=40, edgecolors="none")

    xl = f"PC{pc_x+1}" + (f" ({expl[pc_x]*100:.1f}%)" if expl is not None else "")
    yl = f"PC{pc_y+1}" + (f" ({expl[pc_y]*100:.1f}%)" if expl is not None else "")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title); ax.legend(title=colour_col, fontsize=8, markerscale=1.2)
    ax.axhline(0, color="black", lw=0.4, ls="--")
    ax.axvline(0, color="black", lw=0.4, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def loading_curves_plot(per_sig_info, out_path, n_pcs=2):
    n_sigs = len(per_sig_info)
    fig, axes = plt.subplots(n_pcs, n_sigs,
                              figsize=(max(16, n_sigs*2.5), n_pcs*2.5),
                              sharey=False)
    if n_pcs == 1: axes = axes[np.newaxis, :]

    colours = ["#1f77b4", "#d62728", "#2ca02c"]
    for col, info in enumerate(per_sig_info):
        for row in range(n_pcs):
            ax = axes[row, col]
            if row < len(info["loadings"]):
                ax.plot(info["t_grid"], info["loadings"][row],
                        color=colours[row % len(colours)], lw=1.5)
                ax.axhline(0, color="black", lw=0.4, ls="--")
                ve = info["var_exp"][row]*100 if row < len(info["var_exp"]) else 0
                ax.set_title(f"{info['name']}\nFPC{row+1} ({ve:.0f}%)",
                             fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0: ax.set_ylabel(f"FPC{row+1} loading", fontsize=7)
            if row == n_pcs-1: ax.set_xlabel("norm. time", fontsize=7)
    fig.suptitle("fPCA loading curves per signal\n(shape of principal modes of variation)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def scree_plot(var_exp, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = np.cumsum(var_exp)*100
    ax.bar(range(1, len(var_exp)+1), var_exp*100, color="#1f77b4", alpha=0.7)
    ax.plot(range(1, len(var_exp)+1), cs, "ro-", markersize=4, label="Cumulative")
    ax.axhline(80, color="gray", ls="--", lw=0.8, label="80%")
    ax.set_xlabel("Component"); ax.set_ylabel("Variance explained (%)")
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# MANOVA-style permutation test on PC scores
# --------------------------------------------------------------------------- #
def permanova(scores, labels, n_perm=999):
    """Permutation test: does group label explain variance in PC scores?
    Returns F-statistic and p-value."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(scores)
    labels = np.array(labels)
    groups = np.unique(labels)
    n = len(labels)

    def f_stat(D, labels):
        groups = np.unique(labels)
        ss_w = 0.0
        for g in groups:
            idx = np.where(labels==g)[0]
            if len(idx) < 2: continue
            ss_w += D[np.ix_(idx,idx)].sum() / (2*len(idx))
        ss_t = D.sum() / (2*n)
        ss_b = ss_t - ss_w
        k = len(groups)
        return (ss_b/(k-1)) / (ss_w/(n-k)) if n-k > 0 else np.nan

    f_obs = f_stat(D, labels)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(labels)
        if f_stat(D, perm) >= f_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(f_obs), float(p)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--participants", default=None)
    ap.add_argument("--n-grid", type=int, default=N_GRID,
                    help=f"Normalised time points per event (default {N_GRID})")
    ap.add_argument("--n-fpcs", type=int, default=N_FPCS,
                    help=f"fPCs to extract per signal (default {N_FPCS})")
    ap.add_argument("--no-permanova", action="store_true",
                    help="Skip permutation tests (faster)")
    args = ap.parse_args()

    root   = args.landmarks_root
    if not root.is_dir(): sys.exit(f"Not a directory: {root}")
    pfilter = ({p.strip() for p in args.participants.split(",") if p.strip()}
               if args.participants else None)

    out_dir = root / "metrics" / "fpca"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Discover trials and extract event matrices
    # ------------------------------------------------------------------ #
    all_events = []   # list of dicts with metadata + matrix

    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pid == "metrics": continue
        if pfilter and pid not in pfilter: continue

        for tdir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            stem = tdir.name

            # Find labelled pen file
            pen_path = None
            for name in (f"{stem}_pen_flattened_labelled.csv",
                         f"{stem}_pen_labelled.csv"):
                if (tdir / name).is_file():
                    pen_path = tdir / name; break
            if pen_path is None: continue

            body_path = tdir / f"{stem}_body.csv"
            hand_path = tdir / f"{stem}_hand.csv"

            pen_rows,  pen_fields  = read_csv_t(pen_path)
            body_rows, body_fields = (read_csv_t(body_path)
                                      if body_path.is_file() else ([], []))
            hand_rows, hand_fields = (read_csv_t(hand_path)
                                      if hand_path.is_file() else ([], []))

            runs = place_runs(pen_rows, pen_fields)
            params = parse_params(stem)

            for ev_idx, (t0, t1) in enumerate(runs):
                h = height_at((t0+t1)/2, pen_rows)
                mat = extract_event(t0, t1, body_rows, hand_rows, pen_rows,
                                    body_fields, hand_fields,
                                    args.n_grid)
                if mat is None: continue
                all_events.append({
                    "participant": pid, "trial": stem,
                    "event_idx": ev_idx+1,
                    "height": h, "t_start": t0, "t_end": t1,
                    **params,
                    "matrix": mat,   # (n_grid, 25)
                })

    n = len(all_events)
    print(f"Extracted {n} Place event matrices  "
          f"({len(set(e['participant'] for e in all_events))} participants)")
    if n < 10:
        sys.exit("Too few events — check trial folder structure.")

    meta_df = pd.DataFrame([{k:v for k,v in e.items() if k!="matrix"}
                             for e in all_events]).reset_index(drop=True)
    X = np.stack([e["matrix"] for e in all_events])   # (n, n_grid, 25)

    # ------------------------------------------------------------------ #
    # Analysis A: STRATIFIED by height
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("ANALYSIS A — Stratified fPCA (within each height stratum)")
    print("="*60)

    summary_lines = ["FPCA PROTOTYPE CLUSTERING SUMMARY", "="*60, ""]
    all_scores_strat = np.full((n, N_FINAL_PCS), np.nan)

    for height in ("High", "Medium", "Low"):
        mask = meta_df["height"] == height
        if mask.sum() < 10:
            print(f"  [{height}] too few events ({mask.sum()}) — skipping")
            continue

        Xh = X[mask.values]
        print(f"\n  [{height}] {mask.sum()} events")

        scores, var_exp, per_sig = run_fpca(Xh, ALL_SIGNALS,
                                             n_fpcs=args.n_fpcs,
                                             n_grid=args.n_grid)

        # Store scores back into the full array
        idx = np.where(mask.values)[0]
        n_fill = min(scores.shape[1], N_FINAL_PCS)
        all_scores_strat[np.ix_(idx, range(n_fill))] = scores[:, :n_fill]

        mh = meta_df[mask].reset_index(drop=True)

        # Scree
        scree_plot(var_exp, f"fPCA scree — {height}",
                   out_dir / f"scree_stratified_{height}.png")

        # Scatter coloured by each prototype factor + participant
        for factor in PROTO_FACTORS + ["participant"]:
            if mh[factor].notna().any():
                scatter_pcs(scores, mh, factor,
                            f"{height}: PC1 vs PC2 by {factor}",
                            out_dir / f"biplot_strat_{height}_{factor}.png",
                            expl=var_exp)

        # Loading curves
        loading_curves_plot(per_sig,
                            out_dir / f"loadings_strat_{height}.png")

        # Permanova
        if not args.no_permanova:
            summary_lines.append(f"[{height}] Stratified fPCA")
            for factor in PROTO_FACTORS:
                labels = mh[factor].astype(str).values
                if len(set(labels)) < 2: continue
                F, p = permanova(scores, labels)
                star = "*" if p < 0.05 else " "
                line = (f"  PERMANOVA {factor:<10}: F={F:.3f}  p={p:.3f}{star}")
                print(line); summary_lines.append(line)
            summary_lines.append("")

        print(f"  Variance explained (top 5): "
              f"{np.cumsum(var_exp[:5])*100}")

    # ------------------------------------------------------------------ #
    # Analysis B: FULL DATA — partial out height
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("ANALYSIS B — Full-data fPCA (height residualised)")
    print("="*60)

    # Residualise height: subtract the within-height mean for each signal
    # at each time point, then run fPCA on the residuals
    X_resid = X.copy()
    for height in ("High", "Medium", "Low"):
        mask = (meta_df["height"] == height).values
        if mask.sum() == 0: continue
        height_mean = X[mask].mean(axis=0, keepdims=True)   # (1, n_grid, 25)
        X_resid[mask] -= height_mean

    print(f"\n  Full data: {n} events after height residualisation")

    scores_full, var_exp_full, per_sig_full = run_fpca(
        X_resid, ALL_SIGNALS, n_fpcs=args.n_fpcs, n_grid=args.n_grid)

    scree_plot(var_exp_full, "fPCA scree — full data (height residualised)",
               out_dir / "scree_full.png")

    for factor in PROTO_FACTORS + ["participant", "height"]:
        if meta_df[factor].notna().any():
            scatter_pcs(scores_full, meta_df, factor,
                        f"Full data: PC1 vs PC2 by {factor}",
                        out_dir / f"biplot_full_{factor}.png",
                        expl=var_exp_full)

    loading_curves_plot(per_sig_full, out_dir / "loadings_full.png")

    if not args.no_permanova:
        summary_lines.append("[Full data — height residualised]")
        for factor in PROTO_FACTORS:
            labels = meta_df[factor].astype(str).values
            if len(set(labels)) < 2: continue
            F, p = permanova(scores_full, labels)
            star = "*" if p < 0.05 else " "
            line = f"  PERMANOVA {factor:<10}: F={F:.3f}  p={p:.3f}{star}"
            print(line); summary_lines.append(line)
        summary_lines.append("")

    # ------------------------------------------------------------------ #
    # Save scores CSV
    # ------------------------------------------------------------------ #
    score_cols_strat = [f"strat_PC{i+1}" for i in range(N_FINAL_PCS)]
    score_cols_full  = [f"full_PC{i+1}"  for i in range(scores_full.shape[1])]

    scores_df = meta_df.copy()
    for i, col in enumerate(score_cols_strat):
        scores_df[col] = all_scores_strat[:, i]
    for i, col in enumerate(score_cols_full):
        scores_df[col] = scores_full[:, i]

    csv_out = out_dir / "fpca_scores.csv"
    scores_df.to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out}  ({len(scores_df)} rows)")

    # Summary text
    summary_lines += [
        "",
        "INTERPRETATION GUIDE",
        "-"*40,
        "Stratified: run within each height — prototype differences are the",
        "  only systematic factor remaining. Clustering = distinct posture signature.",
        "Full data: height residualised before PCA — preserves all events but",
        "  relies on mean-subtraction to remove height effect.",
        "",
        "PERMANOVA * = p<0.05 that the factor explains significant variance",
        "  in the full PC space (permutation test, 999 shuffles).",
        "  Non-significant means prototype configurations do NOT produce",
        "  detectably distinct multi-stream movement signatures.",
        "",
        "Loading curves: show what each PC represents as a function of",
        "  normalised placement time. A PC that loads positively early and",
        "  negatively late captures 'approach vs hold' differences.",
    ]
    (out_dir / "fpca_summary.txt").write_text("\n".join(summary_lines),
                                               encoding="utf-8")

    print(f"\nOutputs written to {out_dir}/")
    print("  scree_stratified_<Height>.png")
    print("  biplot_strat_<Height>_<factor>.png")
    print("  loadings_strat_<Height>.png")
    print("  biplot_full_<factor>.png")
    print("  loadings_full.png")
    print("  fpca_scores.csv")
    print("  fpca_summary.txt")


if __name__ == "__main__":
    main()