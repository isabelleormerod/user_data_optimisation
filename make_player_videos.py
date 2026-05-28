#!/usr/bin/env python3
"""
Render MP4 sanity-check videos from tracking data.

Produces three videos (hand, body, pen) from either the raw JSON or the
synced CSVs. The point is visual sense-checking: do the streams move
sensibly? Are there obvious sync issues? Does the data line up with what
you remember seeing during the recording?

Visualization design:
  - Hand: 3D scatter of joints, with skeleton lines auto-detected from
          joint names (HandThumb1..2..3..Tip, HandIndex0..1..2..3..Tip, etc).
  - Body: 3D scatter of MediaPipe pose landmarks, with the standard
          skeleton edges. View rotates slowly during playback.
  - Pen:  Tip position as a sphere, with a short line drawn from tip in the
          pen's pointing direction (derived from quaternion). Optional trail.

Data source (--source):
  - 'raw'    -> read directly from <recording>.json (native rates)
  - 'synced' -> read from <recording>_pen.csv, _body.csv, _hand.csv

If --source synced, points are colour-coded by data_quality:
  real           green
  interpolated   blue
  filled_dropout orange
  extrapolated   red

Usage:
    python make_player_videos.py <recording.json>
    python make_player_videos.py <recording.json> --source synced
    python make_player_videos.py <recording.json> --streams hand pen
    python make_player_videos.py <recording.json> --speed 0.5 --trail-seconds 2.0

Requires: matplotlib, numpy, ffmpeg on PATH.
"""

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ----------------------------------------------------------------------------
# Constants: skeleton connections
# ----------------------------------------------------------------------------

# Body pose skeleton (MediaPipe 33-landmark)
BODY_EDGES = [
    # Face
    ("LeftEar", "LeftEyeOuter"), ("LeftEyeOuter", "LeftEye"),
    ("LeftEye", "LeftEyeInner"), ("LeftEyeInner", "Nose"),
    ("Nose", "RightEyeInner"), ("RightEyeInner", "RightEye"),
    ("RightEye", "RightEyeOuter"), ("RightEyeOuter", "RightEar"),
    ("MouthLeft", "MouthRight"),
    # Torso
    ("LeftShoulder", "RightShoulder"),
    ("LeftShoulder", "LeftHip"),
    ("RightShoulder", "RightHip"),
    ("LeftHip", "RightHip"),
    # Left arm
    ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist"),
    ("LeftWrist", "LeftPinky"), ("LeftWrist", "LeftIndex"),
    ("LeftWrist", "LeftThumb"), ("LeftIndex", "LeftPinky"),
    # Right arm
    ("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"),
    ("RightWrist", "RightPinky"), ("RightWrist", "RightIndex"),
    ("RightWrist", "RightThumb"), ("RightIndex", "RightPinky"),
    # Left leg
    ("LeftHip", "LeftKnee"), ("LeftKnee", "LeftAnkle"),
    ("LeftAnkle", "LeftHeel"), ("LeftHeel", "LeftFootIndex"),
    ("LeftAnkle", "LeftFootIndex"),
    # Right leg
    ("RightHip", "RightKnee"), ("RightKnee", "RightAnkle"),
    ("RightAnkle", "RightHeel"), ("RightHeel", "RightFootIndex"),
    ("RightAnkle", "RightFootIndex"),
]

# Colours by data_quality
QUALITY_COLOURS = {
    "real": "#2ca02c",
    "interpolated": "#1f77b4",
    "filled_dropout": "#ff7f0e",
    "extrapolated": "#d62728",
    "missing": "#777777",
    "no_hand_data": "#777777",
}


def detect_hand_edges(joint_ids: list) -> list:
    """Auto-detect hand skeleton edges from joint name patterns.

    Looks for chains like HandThumb1 -> HandThumb2 -> ... -> HandThumbTip.
    Connects each finger's base joint back to the wrist if present.
    """
    edges = []
    # Group joint IDs by finger prefix (HandThumb, HandIndex, ...)
    fingers = defaultdict(list)
    for jid in joint_ids:
        # E.g., HandThumb1, HandIndex0, HandPinkyTip
        for finger_name in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            prefix = f"Hand{finger_name}"
            if jid.startswith(prefix):
                # Extract suffix: 0, 1, 2, 3, Tip
                suffix = jid[len(prefix):]
                # Order: numeric first (0..3), then Tip
                order = (0, int(suffix)) if suffix.isdigit() else (1, 0)
                fingers[finger_name].append((order, jid))
                break

    # Within each finger, sort by order and connect adjacent
    for finger_name, items in fingers.items():
        items.sort()
        sorted_ids = [jid for _, jid in items]
        for a, b in zip(sorted_ids[:-1], sorted_ids[1:]):
            edges.append((a, b))

    # Connect each finger's first joint to the wrist root if present
    wrist = "HandWristRoot"
    if wrist in joint_ids:
        for finger_name, items in fingers.items():
            if items:
                _, base = items[0]
                edges.append((wrist, base))

    return edges


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_raw_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_csv_folder(json_path: Path) -> tuple:
    """Locate the per-trial folder containing the synced CSVs.

    Layout assumed:
        <root>/<json_stem>.csv.json                 (or <json_stem>.json)
        <root>/<participant>/<csv_stem>/<csv_stem>_pen.csv  etc.

    Where:
        json_stem  = the JSON filename without extensions (strip .csv.json or .json)
        participant = the first token of the stem (e.g. 'P003' from 'P003_...')
        csv_stem   = same as json_stem (the trial folder is named identically)

    Returns (csv_folder, csv_stem). csv_folder may not exist; caller checks.
    """
    name = json_path.name
    # Strip '.csv.json' or '.json'
    if name.endswith(".csv.json"):
        stem = name[:-len(".csv.json")]
    elif name.endswith(".json"):
        stem = name[:-len(".json")]
    else:
        stem = json_path.stem

    participant = stem.split("_", 1)[0]
    csv_folder = json_path.parent / participant / stem
    return csv_folder, stem


def load_synced_csv(path: Path) -> tuple:
    """Return (rows_as_list_of_dicts, fieldnames)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def parse_float(s):
    """Tolerant float parse: return float or None."""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Extract per-frame data structures suitable for animation
# ----------------------------------------------------------------------------

def extract_hand_frames_raw(data: dict) -> list:
    """Return list of dicts: {t_s, hands: [{type, joints: {jid: (x,y,z)}}]}.

    t_s is seconds since the first hand sample.
    """
    frames = data.get("handTracking", {}).get("frames", []) or []
    if not frames:
        return []
    t0 = frames[0]["timestamp"]
    out = []
    for f in frames:
        entry = {"t_s": f["timestamp"] - t0, "quality": "real", "hands": []}
        for hand in f.get("hands", []) or []:
            joints = {}
            for j in hand.get("joints", []) or []:
                pos = j.get("position", {}) or {}
                joints[j["jointId"]] = (pos.get("x"), pos.get("y"), pos.get("z"))
            entry["hands"].append({"type": hand.get("handType", "?"), "joints": joints})
        out.append(entry)
    return out


def extract_body_frames_raw(data: dict) -> list:
    frames = data.get("bodyTracking", {}).get("frames", []) or []
    if not frames:
        return []
    t0 = frames[0]["timestamp"]
    out = []
    for f in frames:
        entry = {"t_s": (f["timestamp"] - t0) / 1000.0, "quality": "real", "landmarks": {}}
        poses = f.get("poses") or []
        if poses:
            for lm in poses[0].get("landmarks", []) or []:
                entry["landmarks"][lm["name"]] = (lm.get("x"), lm.get("y"), lm.get("z"))
        out.append(entry)
    return out


def extract_pen_frames_raw(data: dict) -> list:
    frames = data.get("penTracking", {}).get("frames", []) or []
    if not frames:
        return []
    t0 = frames[0]["timestamp"]
    out = []
    for f in frames:
        pos = f.get("position", {}) or {}
        rot = f.get("rotation", {}) or {}
        out.append({
            "t_s": (f["timestamp"] - t0) / 1000.0,
            "quality": "real",
            "pos": (pos.get("x"), pos.get("y"), pos.get("z")),
            "rot": (rot.get("x"), rot.get("y"), rot.get("z"), rot.get("w")),
        })
    return out


def extract_hand_frames_csv(rows: list, fields: list) -> list:
    """rows from <stem>_hand.csv. Each row has t_s, *_x/y/z per joint, data_quality."""
    # Find joint names by grouping columns
    joint_columns = defaultdict(dict)  # joint_name -> {axis: column}
    for f in fields:
        if f in ("t_s", "data_quality", "frame_idx"):
            continue
        # Expect format: <HandType>_<JointId>_<axis>
        parts = f.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("x", "y", "z", "qx", "qy", "qz", "qw"):
            joint_columns[parts[0]][parts[1]] = f

    out = []
    for r in rows:
        entry = {
            "t_s": parse_float(r.get("t_s")),
            "quality": r.get("data_quality", "real"),
            "hands": [],
        }
        # Group joints by hand type (prefix before first _)
        hand_groups = defaultdict(dict)  # hand_type -> {jid: (x,y,z)}
        for full_name, axes in joint_columns.items():
            # Skip if any xyz missing for this joint in this row
            x = parse_float(r.get(axes.get("x", "")))
            y = parse_float(r.get(axes.get("y", "")))
            z = parse_float(r.get(axes.get("z", "")))
            if x is None or y is None or z is None:
                continue
            # full_name like "Left_HandThumb1" -> split first '_'
            if "_" in full_name:
                hand_type, jid = full_name.split("_", 1)
            else:
                hand_type, jid = "Unknown", full_name
            hand_groups[hand_type][jid] = (x, y, z)
        for hand_type, joints in hand_groups.items():
            if joints:
                entry["hands"].append({"type": hand_type, "joints": joints})
        out.append(entry)
    return out


def extract_body_frames_csv(rows: list, fields: list) -> list:
    landmark_columns = defaultdict(dict)
    for f in fields:
        if f in ("t_s", "data_quality"):
            continue
        # Format: <LandmarkName>_<axis>  (axis in {x,y,z,conf})
        parts = f.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("x", "y", "z", "conf"):
            landmark_columns[parts[0]][parts[1]] = f

    out = []
    for r in rows:
        entry = {
            "t_s": parse_float(r.get("t_s")),
            "quality": r.get("data_quality", "real"),
            "landmarks": {},
        }
        for name, axes in landmark_columns.items():
            x = parse_float(r.get(axes.get("x", "")))
            y = parse_float(r.get(axes.get("y", "")))
            z = parse_float(r.get(axes.get("z", "")))
            if x is None or y is None or z is None:
                continue
            entry["landmarks"][name] = (x, y, z)
        out.append(entry)
    return out


def extract_pen_frames_csv(rows: list, fields: list) -> list:
    out = []
    for r in rows:
        x = parse_float(r.get("x"))
        y = parse_float(r.get("y"))
        z = parse_float(r.get("z"))
        qx = parse_float(r.get("qx")) or 0
        qy = parse_float(r.get("qy")) or 0
        qz = parse_float(r.get("qz")) or 0
        qw = parse_float(r.get("qw")) or 1
        if x is None or y is None or z is None:
            continue
        out.append({
            "t_s": parse_float(r.get("t_s")),
            "quality": r.get("data_quality", "real"),
            "pos": (x, y, z),
            "rot": (qx, qy, qz, qw),
        })
    return out


# ----------------------------------------------------------------------------
# Bounding box helpers
# ----------------------------------------------------------------------------

def compute_bbox_hand(frames: list) -> tuple:
    pts = []
    for f in frames:
        for h in f["hands"]:
            for xyz in h["joints"].values():
                if all(v is not None for v in xyz):
                    pts.append(xyz)
    return _bbox_from_points(pts)


def compute_bbox_body(frames: list) -> tuple:
    pts = []
    for f in frames:
        for xyz in f["landmarks"].values():
            if all(v is not None for v in xyz):
                pts.append(xyz)
    return _bbox_from_points(pts)


def compute_bbox_pen(frames: list) -> tuple:
    pts = []
    for f in frames:
        if all(v is not None for v in f["pos"]):
            pts.append(f["pos"])
    return _bbox_from_points(pts)


def _bbox_from_points(pts):
    if not pts:
        return ((-1, 1), (-1, 1), (-1, 1))
    arr = np.array(pts)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    pad = max((mx - mn).max() * 0.1, 0.05)
    return tuple((mn[i] - pad, mx[i] + pad) for i in range(3))


# ----------------------------------------------------------------------------
# Anatomical 3D view setup (shared by all streams)
# ----------------------------------------------------------------------------
#
# Coordinate convention in the data:
#   data_x = vertical, but INCREASING DOWNWARD (head has smaller x than feet)
#   data_y = anterior-posterior (front/back); sagittal plane parallel to x-y
#   data_z = medial-lateral (left/right); floor is the z-y plane
#
# matplotlib's 3D vertical axis is always plot-z. To make data_x the true
# on-screen vertical we REMAP every plotted point:
#   plot_x = data_y,  plot_y = data_z,  plot_z = data_x
# and invert the plot_z limits so the head (small data_x) is at the top.

BODY_VIEW_ELEV = 15    # head-up three-quarter view
BODY_VIEW_AZIM = -60


def remap_point(p):
    """Remap (data_x, data_y, data_z) -> (plot_x, plot_y, plot_z) so that
    data_x is drawn on matplotlib's vertical (plot-z) axis."""
    dx, dy, dz = p
    return (dy, dz, dx)


def setup_anatomical_axes(ax, bbox, equal_aspect=True, elev=None, azim=None):
    """Configure a 3D axis so data_x is the on-screen vertical (head up).

    `bbox` is in DATA coords: ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi)).
    Points must be plotted through remap_point() to line up with this.
    """
    if elev is None:
        elev = BODY_VIEW_ELEV
    if azim is None:
        azim = BODY_VIEW_AZIM
    xlim, ylim, zlim = bbox  # data x, y, z ranges

    # plot_x = data_y, plot_y = data_z, plot_z = data_x (inverted -> head up)
    ax.set_xlim(ylim)
    ax.set_ylim(zlim)
    ax.set_zlim(xlim[1], xlim[0])
    ax.set_xlabel("y (front/back)")
    ax.set_ylabel("z (left/right)")
    ax.set_zlabel("x (up/down)")

    if equal_aspect:
        ranges = [abs(ylim[1] - ylim[0]),
                  abs(zlim[1] - zlim[0]),
                  abs(xlim[1] - xlim[0])]
        ranges = [r if r > 1e-9 else 1.0 for r in ranges]
        try:
            ax.set_box_aspect(ranges)
        except Exception:
            pass  # older matplotlib without set_box_aspect

    ax.view_init(elev=elev, azim=azim)




# ----------------------------------------------------------------------------
# Quaternion -> direction vector
# ----------------------------------------------------------------------------

def quat_forward(qx, qy, qz, qw):
    """Return the local 'forward' vector (0,0,1) rotated by the quaternion.

    This represents the pen's pointing direction. If the pen's axis is
    different, you can adjust which local axis to rotate.
    """
    # Rotate (0,0,1) by quaternion (x,y,z,w):
    # v' = q * v * q^-1, simplified for v=(0,0,1)
    x, y, z, w = qx, qy, qz, qw
    fx = 2 * (x * z + w * y)
    fy = 2 * (y * z - w * x)
    fz = 1 - 2 * (x * x + y * y)
    return (fx, fy, fz)


# ----------------------------------------------------------------------------
# Frame subsampling (to keep output FPS reasonable)
# ----------------------------------------------------------------------------

def subsample_to_target_fps(frames: list, target_fps: float) -> list:
    """Keep frames closest to each 1/target_fps tick of t_s.

    Frames with t_s == None are skipped.
    """
    if not frames:
        return frames
    valid = [f for f in frames if f.get("t_s") is not None]
    if not valid:
        return []
    t_start = valid[0]["t_s"]
    t_end = valid[-1]["t_s"]
    dt = 1.0 / target_fps
    n = int((t_end - t_start) / dt) + 1
    out = []
    f_idx = 0
    for k in range(n):
        target_t = t_start + k * dt
        # Advance f_idx until next frame is past target
        while f_idx < len(valid) - 1 and valid[f_idx + 1]["t_s"] <= target_t:
            f_idx += 1
        # Pick the closer of valid[f_idx] and valid[f_idx+1]
        f = valid[f_idx]
        if f_idx + 1 < len(valid):
            if abs(valid[f_idx + 1]["t_s"] - target_t) < abs(valid[f_idx]["t_s"] - target_t):
                f = valid[f_idx + 1]
        out.append(f)
    return out


# ----------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------

def _make_progress_callback(label: str, total: int):
    """Return a callback for FuncAnimation.save that prints a single, updating
    line with percentage. Falls back to a quiet no-op if total is zero.
    """
    if total <= 0:
        return None
    import time
    state = {"last_print": 0.0, "start": time.time()}

    def cb(current_frame: int, total_frames: int):
        now = time.time()
        # Throttle to ~5 updates per second to avoid spamming the terminal
        if now - state["last_print"] < 0.2 and current_frame + 1 < total_frames:
            return
        state["last_print"] = now
        pct = (current_frame + 1) / total_frames * 100
        elapsed = now - state["start"]
        fps_so_far = (current_frame + 1) / elapsed if elapsed > 0 else 0
        eta = (total_frames - current_frame - 1) / fps_so_far if fps_so_far > 0 else 0
        msg = (f"\r  [{label}] frame {current_frame + 1}/{total_frames} "
               f"({pct:5.1f}%) - {fps_so_far:.1f} fps - eta {eta:5.1f}s")
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current_frame + 1 >= total_frames:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return cb


def render_hand_video(frames: list, output_path: Path, output_fps: float,
                      colour_by_quality: bool, view_elev=None, view_azim=None):
    if not frames:
        print(f"  No hand frames to render; skipping {output_path.name}")
        return

    # Auto-detect skeleton edges from joint names
    all_joints = set()
    for f in frames:
        for h in f["hands"]:
            all_joints.update(h["joints"].keys())
    edges = detect_hand_edges(sorted(all_joints))

    bbox = compute_bbox_hand(frames)

    # Hand-type colours. Anything else falls back to a neutral grey.
    HAND_COLOURS = {"Left": "#1f77b4", "Right": "#d62728"}
    DEFAULT_HAND_COLOUR = "#7f7f7f"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_anatomical_axes(ax, bbox, equal_aspect=True,
                          elev=view_elev, azim=view_azim)
    title = ax.set_title("")

    # Static legend so the user can tell which hand is which colour
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Left",
               markerfacecolor=HAND_COLOURS["Left"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Right",
               markerfacecolor=HAND_COLOURS["Right"], markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    scatter_artists = []
    line_artists = []

    def update(i):
        f = frames[i]
        # Clear previous artists
        for a in scatter_artists + line_artists:
            a.remove()
        scatter_artists.clear()
        line_artists.clear()

        for hand in f["hands"]:
            colour = HAND_COLOURS.get(hand["type"], DEFAULT_HAND_COLOUR)
            joints = hand["joints"]
            # Points (remapped so data_x is vertical)
            pts = [remap_point(v) for v in joints.values()
                   if all(c is not None for c in v)]
            if pts:
                xs, ys, zs = zip(*pts)
                sc = ax.scatter(xs, ys, zs, color=colour, s=20)
                scatter_artists.append(sc)
            # Edges
            for a, b in edges:
                if a in joints and b in joints:
                    pa, pb = joints[a], joints[b]
                    if all(c is not None for c in pa) and all(c is not None for c in pb):
                        pa, pb = remap_point(pa), remap_point(pb)
                        ln, = ax.plot(
                            [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                            color=colour, linewidth=1.5,
                        )
                        line_artists.append(ln)

        title.set_text(f"Hand  t={f['t_s']:.2f}s  quality={f['quality']}")
        return scatter_artists + line_artists + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / output_fps,
                         blit=False, repeat=False)
    writer = FFMpegWriter(fps=output_fps, bitrate=2000)
    anim.save(str(output_path), writer=writer,
              progress_callback=_make_progress_callback("hand", len(frames)))
    plt.close(fig)


def render_body_video(frames: list, output_path: Path, output_fps: float,
                      colour_by_quality: bool, view_elev=None, view_azim=None):
    if not frames:
        print(f"  No body frames to render; skipping {output_path.name}")
        return

    bbox = compute_bbox_body(frames)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_anatomical_axes(ax, bbox, equal_aspect=True,
                          elev=view_elev, azim=view_azim)
    title = ax.set_title("")

    scatter_artists = []
    line_artists = []

    def update(i):
        f = frames[i]
        for a in scatter_artists + line_artists:
            a.remove()
        scatter_artists.clear()
        line_artists.clear()

        colour = QUALITY_COLOURS.get(f["quality"], "#000000") if colour_by_quality else "#1f77b4"

        lms = f["landmarks"]
        pts = [remap_point(v) for v in lms.values()
               if all(c is not None for c in v)]
        if pts:
            xs, ys, zs = zip(*pts)
            sc = ax.scatter(xs, ys, zs, color=colour, s=15)
            scatter_artists.append(sc)
        for a, b in BODY_EDGES:
            if a in lms and b in lms:
                pa, pb = lms[a], lms[b]
                if all(c is not None for c in pa) and all(c is not None for c in pb):
                    pa, pb = remap_point(pa), remap_point(pb)
                    ln, = ax.plot(
                        [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        color=colour, linewidth=2,
                    )
                    line_artists.append(ln)

        title.set_text(f"Body  t={f['t_s']:.2f}s  quality={f['quality']}")
        return scatter_artists + line_artists + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / output_fps,
                         blit=False, repeat=False)
    writer = FFMpegWriter(fps=output_fps, bitrate=2000)
    anim.save(str(output_path), writer=writer,
              progress_callback=_make_progress_callback("body", len(frames)))
    plt.close(fig)


def render_pen_video(frames: list, output_path: Path, output_fps: float,
                     colour_by_quality: bool, trail_seconds: float,
                     direction_length: float = None,
                     view_elev=None, view_azim=None):
    if not frames:
        print(f"  No pen frames to render; skipping {output_path.name}")
        return

    bbox = compute_bbox_pen(frames)
    # Length of the direction line: ~10% of the largest axis range
    if direction_length is None:
        ranges = [hi - lo for (lo, hi) in bbox]
        direction_length = max(ranges) * 0.08 if max(ranges) > 0 else 0.05

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_anatomical_axes(ax, bbox, equal_aspect=True,
                          elev=view_elev, azim=view_azim)
    title = ax.set_title("")

    tip_artists = []
    dir_artists = []
    trail_artists = []

    def update(i):
        f = frames[i]
        for a in tip_artists + dir_artists + trail_artists:
            a.remove()
        tip_artists.clear()
        dir_artists.clear()
        trail_artists.clear()

        colour = QUALITY_COLOURS.get(f["quality"], "#000000") if colour_by_quality else "#1f77b4"
        tip = f["pos"]
        # Direction end in data space, then remap both tip and end
        fwd = quat_forward(*f["rot"])
        end = (tip[0] + direction_length * fwd[0],
               tip[1] + direction_length * fwd[1],
               tip[2] + direction_length * fwd[2])
        tip_r = remap_point(tip)
        end_r = remap_point(end)
        # Tip
        sc = ax.scatter([tip_r[0]], [tip_r[1]], [tip_r[2]], color=colour, s=80)
        tip_artists.append(sc)
        # Direction line
        ln, = ax.plot(
            [tip_r[0], end_r[0]], [tip_r[1], end_r[1]], [tip_r[2], end_r[2]],
            color=colour, linewidth=3,
        )
        dir_artists.append(ln)

        # Trail
        if trail_seconds > 0:
            t_now = f["t_s"]
            trail_pts = []
            for k in range(i - 1, -1, -1):
                tk = frames[k]["t_s"]
                if tk is None or t_now - tk > trail_seconds:
                    break
                trail_pts.append(remap_point(frames[k]["pos"]))
            if len(trail_pts) >= 2:
                xs, ys, zs = zip(*trail_pts)
                ln2, = ax.plot(xs, ys, zs, color="#888888", linewidth=1, alpha=0.6)
                trail_artists.append(ln2)

        title.set_text(f"Pen  t={f['t_s']:.2f}s  quality={f['quality']}")
        return tip_artists + dir_artists + trail_artists + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / output_fps,
                         blit=False, repeat=False)
    writer = FFMpegWriter(fps=output_fps, bitrate=2000)
    anim.save(str(output_path), writer=writer,
              progress_callback=_make_progress_callback("pen", len(frames)))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path,
                    help="Path to recording JSON")
    ap.add_argument("--source", choices=["raw", "synced"], default="raw",
                    help="Read raw JSON or synced CSVs (default: raw)")
    ap.add_argument("--streams", nargs="+", choices=["hand", "body", "pen"],
                    default=["hand", "body", "pen"],
                    help="Which streams to render (default: all)")
    ap.add_argument("--output-folder", type=Path, default=None,
                    help="Where to write MP4 files (default: same as JSON for "
                         "raw, or the synced CSV folder for synced)")
    ap.add_argument("--csv-folder", type=Path, default=None,
                    help="Override the auto-detected synced CSV folder. "
                         "Only used when --source synced.")
    ap.add_argument("--output-fps", type=float, default=30.0,
                    help="MP4 frame rate (default: 30)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed (1.0 = real-time, 0.5 = half-speed)")
    ap.add_argument("--trail-seconds", type=float, default=0.0,
                    help="Pen trail length in seconds (0 = no trail, default: 0)")
    ap.add_argument("--view-elev", type=float, default=None,
                    help=f"3D view elevation angle (default: {BODY_VIEW_ELEV})")
    ap.add_argument("--view-azim", type=float, default=None,
                    help=f"3D view azimuth angle (default: {BODY_VIEW_AZIM})")
    args = ap.parse_args()

    if not args.json_path.is_file():
        sys.exit(f"ERROR: {args.json_path} not found")
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH")

    json_path = args.json_path
    stem = json_path.stem
    folder = args.output_folder or json_path.parent
    folder.mkdir(parents=True, exist_ok=True)

    # Effective fps after speed adjustment: at speed=0.5, we need 2x as many frames
    # in the output to show the same span -> sample raw frames at output_fps/speed
    sample_fps = args.output_fps / args.speed

    # Load data
    if args.source == "raw":
        data = load_raw_json(json_path)
        hand_frames = extract_hand_frames_raw(data) if "hand" in args.streams else []
        body_frames = extract_body_frames_raw(data) if "body" in args.streams else []
        pen_frames = extract_pen_frames_raw(data) if "pen" in args.streams else []
        colour_by_quality = False
    else:  # synced
        # Find the per-trial folder containing the synced CSVs
        csv_folder, csv_stem = find_csv_folder(json_path)
        # Allow override of the auto-detected folder
        if args.csv_folder is not None:
            csv_folder = args.csv_folder
        if not csv_folder.is_dir():
            sys.exit(
                f"ERROR: expected per-trial CSV folder at {csv_folder}, but it "
                f"doesn't exist. Layout expected:\n"
                f"    <root>/<stem>.csv.json\n"
                f"    <root>/<participant>/<stem>/<stem>_pen.csv  etc.\n"
                f"You can override with --csv-folder."
            )
        print(f"Looking for synced CSVs in: {csv_folder}")

        hand_frames = body_frames = pen_frames = []
        if "hand" in args.streams:
            p = csv_folder / f"{csv_stem}_hand.csv"
            if p.is_file():
                rows, fields = load_synced_csv(p)
                hand_frames = extract_hand_frames_csv(rows, fields)
            else:
                print(f"WARNING: {p.name} not found; skipping hand")
        if "body" in args.streams:
            p = csv_folder / f"{csv_stem}_body.csv"
            if p.is_file():
                rows, fields = load_synced_csv(p)
                body_frames = extract_body_frames_csv(rows, fields)
            else:
                print(f"WARNING: {p.name} not found; skipping body")
        if "pen" in args.streams:
            p = csv_folder / f"{csv_stem}_pen.csv"
            if p.is_file():
                rows, fields = load_synced_csv(p)
                pen_frames = extract_pen_frames_csv(rows, fields)
            else:
                print(f"WARNING: {p.name} not found; skipping pen")
        colour_by_quality = True

        # If user didn't override output folder, default to the CSV folder
        # (keeps outputs alongside the synced CSVs)
        if args.output_folder is None:
            folder = csv_folder
            folder.mkdir(parents=True, exist_ok=True)
        # Use csv_stem for output filenames in synced mode (matches CSV naming)
        stem = csv_stem

    # Subsample to keep output reasonable
    hand_frames = subsample_to_target_fps(hand_frames, sample_fps)
    body_frames = subsample_to_target_fps(body_frames, sample_fps)
    pen_frames  = subsample_to_target_fps(pen_frames,  sample_fps)

    print(f"After subsampling at {sample_fps:.1f} Hz:")
    print(f"  hand frames: {len(hand_frames)}")
    print(f"  body frames: {len(body_frames)}")
    print(f"  pen frames:  {len(pen_frames)}")

    # Render
    src_tag = "_synced" if args.source == "synced" else "_raw"
    if "hand" in args.streams and hand_frames:
        out = folder / f"{stem}_hand{src_tag}.mp4"
        print(f"\nRendering hand -> {out.name}")
        render_hand_video(hand_frames, out, args.output_fps, colour_by_quality,
                          view_elev=args.view_elev, view_azim=args.view_azim)
    if "body" in args.streams and body_frames:
        out = folder / f"{stem}_body{src_tag}.mp4"
        print(f"\nRendering body -> {out.name}")
        render_body_video(body_frames, out, args.output_fps, colour_by_quality,
                          view_elev=args.view_elev, view_azim=args.view_azim)
    if "pen" in args.streams and pen_frames:
        out = folder / f"{stem}_pen{src_tag}.mp4"
        print(f"\nRendering pen -> {out.name}")
        render_pen_video(pen_frames, out, args.output_fps, colour_by_quality,
                         args.trail_seconds,
                         view_elev=args.view_elev, view_azim=args.view_azim)

    print("\nDone.")


if __name__ == "__main__":
    main()
