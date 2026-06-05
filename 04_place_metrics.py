#!/usr/bin/env python3
"""
Task-performance metrics for Place events, from pen tracking data.

For every Place event (START->STOP interval labelled 'Place') this computes,
over the hold:
  - duration (s)
  - perpendicularity angle of the pen to the calibration plane (deg):
        mean and variance. 0 deg = pen shaft perpendicular to the plane.
  - left/right tilt (deg): signed deviation along the plane's in-plane x axis,
        mean and variance.
  - up/down tilt (deg): signed deviation along the plane's in-plane y axis,
        mean and variance.
  - positional jitter (mm): RMS distance of the tip from its mean position.
  - angular jitter (deg): RMS angular deviation of the pen axis from its mean.
  - height condition (High / Medium / Low) active during the Place.

Geometry (validated against the data):
  - The pen's shaft / pointing axis is the stylus local +Y axis, rotated into
    world coordinates by the per-sample quaternion.
  - The plane normal & centroid come from the flatten step's quality log; the
    in-plane axes are reconstructed with the same transform flatten used, so
    left/right & up/down match the flattened coordinate frame.

Inputs per trial folder <root>/<PID>/<stem>/ :
    <stem>_pen.csv            (t_s, qw,qx,qy,qz, x,y,z, data_quality)
    <stem>_boris_synced.csv   (Behavior, Behavior type, t_s_synced)
Plus the shared plane log:
    <root>/plane_quality_log.csv   (normal & centroid per trial_stem)

Outputs:
    <root>/metrics/<PID>_place_metrics.csv     (per participant)
    <root>/metrics/place_metrics_combined.csv  (all participants)
    <root>/metrics/*.png                        (graphs grouped by height)

Usage:
    python 04_place_metrics.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python 04_place_metrics.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks --participants P003,P004
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import parse_float, read_table
from utils.discovery import find_labelled_pen, iter_trials_labelled
from utils.params import parse_participant_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEIGHT_LABELS = ["High", "Medium", "Low"]
PEN_LOCAL_AXIS = np.array([0.0, 0.0, 1.0])   # local +Z = pen shaft (validated)
# Note: world Y is vertical (headset frame); the calibration plane is vertical,
# so its normal is horizontal and the pen shaft (+Z) points along that normal
# when placing. The flattened frame's v-axis aligns with world Y (up/down) and
# its u-axis is the horizontal in-plane direction (left/right).


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_plane_log(log_path: Path) -> dict:
    """Map trial_stem -> dict with normal, centroid, and (if logged) the
    in-plane axes u/v and the normal axis n. If the axes aren't in the log
    (older runs), only normal+centroid are returned and the frame is rebuilt."""
    out = {}
    if not log_path.is_file():
        return out
    rows, fields = read_table(log_path)
    have_axes = all(c in fields for c in
                    ("axis_u_x", "axis_v_x", "axis_n_x"))
    for r in rows:
        stem = r.get("trial_stem")
        try:
            normal = np.array([float(r["normal_x"]), float(r["normal_y"]),
                               float(r["normal_z"])])
            centroid = np.array([float(r["centroid_x"]), float(r["centroid_y"]),
                                 float(r["centroid_z"])])
        except (TypeError, ValueError, KeyError):
            continue
        entry = {"normal": normal, "centroid": centroid,
                 "u": None, "v": None, "n": None}
        if have_axes:
            try:
                entry["u"] = np.array([float(r["axis_u_x"]), float(r["axis_u_y"]),
                                       float(r["axis_u_z"])])
                entry["v"] = np.array([float(r["axis_v_x"]), float(r["axis_v_y"]),
                                       float(r["axis_v_z"])])
                entry["n"] = np.array([float(r["axis_n_x"]), float(r["axis_n_y"]),
                                       float(r["axis_n_z"])])
            except (TypeError, ValueError, KeyError):
                pass
        out[stem] = entry   # last wins if duplicated
    return out


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
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
    """Return (n, u, v): unit normal plus two orthonormal in-plane axes.

    Mirrors build_flattening_transform's convention: the normal is flipped to
    have positive world-z, then in-plane axes are derived consistently so that
    'u' ~ plane x (left/right) and 'v' ~ plane y (up/down).
    """
    n = normal / np.linalg.norm(normal)
    if n[2] < 0:
        n = -n
    # Pick a reference not parallel to n to build the first in-plane axis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def pen_axis_world(qw, qx, qy, qz):
    """World-space direction of the pen shaft (PEN_LOCAL_AXIS, = local +Z)."""
    R = rotmat(qw, qx, qy, qz)
    a = R @ PEN_LOCAL_AXIS
    return a / np.linalg.norm(a)


def best_shaft_axis(pen, place_mask, normal):
    """Diagnostic: during Place samples, return (best_axis_index, alignments)
    where alignment = mean |dot(local_axis_world, normal)|. The shaft axis
    should be the one closest to 1.0."""
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


# --------------------------------------------------------------------------- #
# Place event extraction
# --------------------------------------------------------------------------- #
def get_place_intervals(boris_rows):
    """Return list of (start_t_s, stop_t_s) for Place state events, paired in
    time order."""
    evs = []
    for r in boris_rows:
        beh = (r.get("Behavior") or "").strip()
        if beh != "Place":
            continue
        btype = (r.get("Behavior type") or "").strip().upper()
        t = parse_float(r.get("t_s_synced"))
        if t is None:
            continue
        evs.append((t, btype))
    evs.sort()
    intervals, open_start = [], None
    for t, typ in evs:
        if typ == "START":
            open_start = t
        elif typ == "STOP" and open_start is not None:
            intervals.append((open_start, t))
            open_start = None
    return intervals


def get_height_intervals(boris_rows):
    """Return list of (label, start, stop) for High/Medium/Low state events."""
    by_label = defaultdict(list)
    for r in boris_rows:
        beh = (r.get("Behavior") or "").strip()
        if beh not in HEIGHT_LABELS:
            continue
        btype = (r.get("Behavior type") or "").strip().upper()
        t = parse_float(r.get("t_s_synced"))
        if t is None:
            continue
        by_label[beh].append((t, btype))
    intervals = []
    for label, evs in by_label.items():
        evs.sort()
        open_start = None
        for t, typ in evs:
            if typ == "START":
                open_start = t
            elif typ == "STOP" and open_start is not None:
                intervals.append((label, open_start, t))
                open_start = None
    return intervals


def height_for(t_mid, height_intervals):
    """Which height condition is active at t_mid (or 'Unknown')."""
    for label, s, e in height_intervals:
        if s <= t_mid <= e:
            return label
    return "Unknown"


# --------------------------------------------------------------------------- #
# Metrics for one Place hold
# --------------------------------------------------------------------------- #
def compute_place_metrics(pen, start, stop, normal, u, v):
    """pen: structured array-ish dict of arrays. Returns a metrics dict or None."""
    t = pen["t_s"]
    mask = (t >= start) & (t <= stop)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None

    pos = np.column_stack([pen["x"][idx], pen["y"][idx], pen["z"][idx]])
    quats = np.column_stack([pen["qw"][idx], pen["qx"][idx],
                             pen["qy"][idx], pen["qz"][idx]])

    # Pen axis per sample
    axes = np.array([pen_axis_world(*q) for q in quats])

    # Perpendicularity: angle between pen axis and plane normal.
    # Use absolute alignment so a flipped axis convention doesn't matter.
    dots = np.clip(np.abs(axes @ normal), -1, 1)
    perp_angle = np.degrees(np.arccos(dots))   # 0 = perpendicular to plane

    # Decompose the tilt direction: project the pen axis onto the plane,
    # measure signed angle components along u (left/right) and v (up/down).
    # Component of axis along in-plane axes vs along normal:
    a_n = axes @ normal
    a_u = axes @ u
    a_v = axes @ v
    # Signed tilt angles (deg): how far the axis leans toward u / v from normal
    lr_angle = np.degrees(np.arctan2(a_u, np.abs(a_n)))   # left/right
    ud_angle = np.degrees(np.arctan2(a_v, np.abs(a_n)))   # up/down

    # Positional jitter: RMS distance of tip from its mean position (mm)
    mean_pos = pos.mean(axis=0)
    dists = np.linalg.norm(pos - mean_pos, axis=1)
    pos_jitter_mm = float(np.sqrt(np.mean(dists**2)) * 1000.0)

    # Angular jitter: RMS angular deviation of the pen axis from its mean axis (deg)
    mean_axis = axes.mean(axis=0)
    mean_axis /= np.linalg.norm(mean_axis)
    ang_dev = np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1)))
    ang_jitter_deg = float(np.sqrt(np.mean(ang_dev**2)))

    return {
        "duration_s": float(stop - start),
        "n_samples": int(len(idx)),
        "perp_mean_deg": float(perp_angle.mean()),
        "perp_var_deg2": float(perp_angle.var()),
        "leftright_mean_deg": float(lr_angle.mean()),
        "leftright_var_deg2": float(lr_angle.var()),
        "updown_mean_deg": float(ud_angle.mean()),
        "updown_var_deg2": float(ud_angle.var()),
        "pos_jitter_mm": pos_jitter_mm,
        "ang_jitter_deg": ang_jitter_deg,
    }


# --------------------------------------------------------------------------- #
# Per-trial processing
# --------------------------------------------------------------------------- #
def load_labelled(path: Path):
    """Read a labelled pen CSV. Returns (cols dict of arrays, behaviour_names).
    Requires t_s + quaternion + x,y,z. Behaviour columns are any extra 0/1
    columns after the known tracking columns."""
    rows, fields = read_table(path)
    need = ("t_s", "qw", "qx", "qy", "qz", "x", "y", "z")
    for c in need:
        if c not in fields:
            raise RuntimeError(f"Labelled pen file missing '{c}': {path.name}")
    known = set(need) | {"data_quality", "x_flat", "y_flat", "z_flat",
                         "is_calibration"}
    beh_names = [c for c in fields if c not in known]

    numeric = list(need)
    cols = {c: [] for c in numeric}
    beh = {b: [] for b in beh_names}
    for r in rows:
        vals = {c: parse_float(r.get(c)) for c in numeric}
        if any(v is None for v in vals.values()):
            continue
        for c in numeric:
            cols[c].append(vals[c])
        for b in beh_names:
            v = r.get(b)
            beh[b].append(1 if str(v).strip() in ("1", "1.0", "True", "true")
                          else 0)
    out = {c: np.asarray(cols[c]) for c in numeric}
    out_beh = {b: np.asarray(beh[b], dtype=int) for b in beh_names}
    return out, out_beh, beh_names


def runs_from_flag(t_s: np.ndarray, flag: np.ndarray):
    """Return list of (start_t_s, stop_t_s) for each contiguous run of 1s."""
    intervals = []
    in_run = False
    start = None
    for i, v in enumerate(flag):
        if v and not in_run:
            in_run = True
            start = t_s[i]
        elif not v and in_run:
            in_run = False
            intervals.append((start, t_s[i - 1]))
    if in_run:
        intervals.append((start, t_s[-1]))
    return intervals


HEIGHT_COLS = ["High", "Medium", "Low"]


def height_at_index_range(beh, lo_i, hi_i):
    """Majority height label active over rows [lo_i..hi_i], or 'Unknown'."""
    best, best_count = "Unknown", 0
    for h in HEIGHT_COLS:
        if h not in beh:
            continue
        count = int(beh[h][lo_i:hi_i + 1].sum())
        if count > best_count:
            best, best_count = h, count
    return best


def process_trial(stem, pid, trial_dir, plane_log):
    pen_path = find_labelled_pen(trial_dir, stem)
    if pen_path is None:
        return [], f"no labelled pen file in {stem}"
    if stem not in plane_log:
        return [], f"no plane in quality log for {stem} (run flatten first)"

    entry = plane_log[stem]
    if entry["u"] is not None:
        n, u, v = entry["n"], entry["u"], entry["v"]
        n = n / np.linalg.norm(n)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
    else:
        # Older log without axes: rebuild from the normal
        n, u, v = plane_frame(entry["normal"])

    pen, beh, beh_names = load_labelled(pen_path)
    if "Place" not in beh:
        return [], f"no 'Place' column in {pen_path.name}"

    t_s = pen["t_s"]
    place_runs = runs_from_flag(t_s, beh["Place"])

    # Sanity check: confirm the configured shaft axis (+Z) is the one best
    # aligned with the plane normal during Place. Warn if not.
    axis_warning = None
    best_ax, aligns = best_shaft_axis(pen, beh["Place"].astype(bool), n)
    if best_ax is not None:
        configured = int(np.argmax(np.abs(PEN_LOCAL_AXIS)))
        if best_ax != configured and aligns[best_ax] - aligns[configured] > 0.15:
            names = {0: "+X", 1: "+Y", 2: "+Z"}
            axis_warning = (
                f"{stem}: configured shaft axis {names[configured]} "
                f"(align {aligns[configured]:.2f}) but {names[best_ax]} aligns "
                f"better ({aligns[best_ax]:.2f}) with the plane normal during "
                f"Place. Angles may be off for this trial.")

    out = []
    for i, (s, e) in enumerate(place_runs, 1):
        m = compute_place_metrics(pen, s, e, n, u, v)
        if m is None:
            continue
        # height: majority over the same row range
        lo_i = int(np.searchsorted(t_s, s, side="left"))
        hi_i = int(np.searchsorted(t_s, e, side="right")) - 1
        m.update({
            "participant": pid,
            "trial": stem,
            "place_index": i,
            "start_t_s": round(float(s), 4),
            "stop_t_s": round(float(e), 4),
            "height": height_at_index_range(beh, lo_i, hi_i),
        })
        out.append(m)
    return out, axis_warning


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CSV + graphs
# --------------------------------------------------------------------------- #
COL_ORDER = [
    "participant", "trial", "place_index", "height",
    "start_t_s", "stop_t_s", "duration_s", "n_samples",
    "perp_mean_deg", "perp_var_deg2",
    "leftright_mean_deg", "leftright_var_deg2",
    "updown_mean_deg", "updown_var_deg2",
    "pos_jitter_mm", "ang_jitter_deg",
]


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COL_ORDER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COL_ORDER})


def _round_fmt(v, nd=3):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


def make_graphs(rows, out_dir):
    """Graphs grouped by height condition, showing trial-to-trial variation."""
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # Metrics to plot: (column, label, unit)
    metrics = [
        ("duration_s", "Duration", "s"),
        ("perp_mean_deg", "Perpendicularity (mean)", "deg"),
        ("pos_jitter_mm", "Positional jitter", "mm"),
        ("ang_jitter_deg", "Angular jitter", "deg"),
        ("leftright_mean_deg", "Left/right tilt (mean)", "deg"),
        ("updown_mean_deg", "Up/down tilt (mean)", "deg"),
    ]
    heights_present = [h for h in HEIGHT_LABELS
                       if any(r["height"] == h for r in rows)]
    if not heights_present:
        heights_present = sorted({r["height"] for r in rows})

    # --- 1) Box plots by height for each metric ---
    for col, label, unit in metrics:
        data = []
        for h in heights_present:
            vals = [r[col] for r in rows
                    if r["height"] == h and isinstance(r.get(col), (int, float))]
            data.append(vals)
        if not any(data):
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, tick_labels=heights_present, showmeans=True)
        # overlay individual points (trial-to-trial spread)
        for i, vals in enumerate(data, 1):
            if vals:
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           alpha=0.5, s=20, color="#1f77b4", zorder=3)
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_xlabel("Height condition")
        ax.set_title(f"{label} by height condition")
        ax.grid(axis="y", alpha=0.3)
        p = out_dir / f"by_height_{col}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # --- 2) Per-trial scatter (trial-to-trial variation), coloured by height ---
    colour = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#1f77b4",
              "Unknown": "#888888"}
    for col, label, unit in metrics:
        # x axis: trial label (ordered), y: metric, marker colour: height
        trials = sorted({r["trial"] for r in rows})
        tindex = {t: i for i, t in enumerate(trials)}
        fig, ax = plt.subplots(figsize=(max(7, len(trials) * 0.8), 5))
        for h in heights_present:
            xs = [tindex[r["trial"]] for r in rows
                  if r["height"] == h and isinstance(r.get(col), (int, float))]
            ys = [r[col] for r in rows
                  if r["height"] == h and isinstance(r.get(col), (int, float))]
            if xs:
                jitter = (np.random.rand(len(xs)) - 0.5) * 0.25
                ax.scatter(np.array(xs) + jitter, ys, label=h,
                           color=colour.get(h, "#333"), alpha=0.7, s=30)
        ax.set_xticks(range(len(trials)))
        ax.set_xticklabels(trials, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(f"{label}: trial-to-trial variation")
        ax.legend(title="Height")
        ax.grid(axis="y", alpha=0.3)
        p = out_dir / f"by_trial_{col}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    return made


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True,
                    help="Root of Participant_Landmarks")
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated PIDs to restrict to (e.g. P003,P004)")
    ap.add_argument("--no-graphs", action="store_true",
                    help="Skip graph generation (tables only)")
    args = ap.parse_args()

    root = args.landmarks_root
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    pfilter = parse_participant_filter(args.participants)

    plane_log = load_plane_log(root / "plane_quality_log.csv")
    if not plane_log:
        print("WARNING: no plane_quality_log.csv found or it's empty. "
              "Run flatten_pen_to_plane.py first; metrics need the plane.")

    trials = list(iter_trials_labelled(root, pfilter))
    if not trials:
        sys.exit("No trial folders with a labelled pen file found.")

    print(f"Processing {len(trials)} trial(s)\n")

    all_rows = []
    by_participant = defaultdict(list)
    warnings = []
    for stem, pid, trial_dir in trials:
        rows, msg = process_trial(stem, pid, trial_dir, plane_log)
        if not rows:
            # No rows -> msg (if any) is a real error/skip reason
            if msg:
                warnings.append(msg)
                print(f"  [skip] {stem}: {msg}")
            else:
                print(f"  [   0] {stem}: no Place events")
            continue
        # Rows present; msg (if any) is a non-fatal warning
        if msg:
            warnings.append(msg)
        note = "  (!) axis warning" if msg else ""
        print(f"  [{len(rows):>4}] {stem}: {len(rows)} Place event(s){note}")
        # round numeric fields for output
        for r in rows:
            for k in list(r.keys()):
                if isinstance(r[k], float):
                    r[k] = _round_fmt(r[k])
        all_rows.extend(rows)
        by_participant[pid].extend(rows)

    if not all_rows:
        sys.exit("\nNo Place events found in any trial; nothing to write.")

    out_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-participant tables
    for pid, rows in by_participant.items():
        p = out_dir / f"{pid}_place_metrics.csv"
        write_metrics_csv(p, rows)
        print(f"\nWrote {p}  ({len(rows)} rows)")

    # Combined table
    combined = out_dir / "place_metrics_combined.csv"
    write_metrics_csv(combined, all_rows)
    print(f"Wrote {combined}  ({len(all_rows)} rows)")

    # Graphs
    if not args.no_graphs:
        made = make_graphs(all_rows, out_dir)
        print(f"\nWrote {len(made)} graph(s) to {out_dir}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
