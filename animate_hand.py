#!/usr/bin/env python3
"""
Animate hand posture and dynamic ergonomic scoring from labelled hand CSV data.

Fixes applied over the original version:
  - Wrist flexion on a secondary Y-axis (right side), aperture on primary (left)
  - Correct 3D skeleton lines: ax.plot([x0,x1],[y0,y1],[z0,z1]) not ([pa,pb] x3)
  - Strange diagonal line removed (was caused by the broken plot call above)
  - Palm cross-connections added (Index0-Middle0-Ring0-Pinky0) for a clear skeleton
  - Place events shaded in green, Insert events shaded in orange on the score plot
  - Active event banner on the 3D panel

Usage:
    python animate_hand.py --csv path/to/_hand_labelled.csv --side Left
    python animate_hand.py --csv path/to/_hand_labelled.csv --side Right --duration 30
    python animate_hand.py --root "A:\\...\\Participant_Landmarks" \\
        --participant P007 --trial P007_Long_Large_Front_weighted_A135
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, total=None, desc=None, unit=None):
        total = total or (len(it) if hasattr(it, '__len__') else None)
        if desc:
            print(desc)
        for i, x in enumerate(it, 1):
            yield x
            if total:
                print(f"\r{desc or ''}: {i}/{total} ({100*i/total:.0f}%)",
                      end='', flush=True)
        print()


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------
def build_edges():
    """Return all skeleton edges as (joint_a, joint_b) pairs."""
    edges = []

    # Forearm anchor → wrist
    edges.append(("HandStart", "HandWristRoot"))

    # Wrist → finger bases (MCP knuckles)
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        edges.append(("HandWristRoot", f"Hand{finger}0"))
    edges.append(("HandWristRoot", "HandThumb1"))   # thumb has no *0 joint

    # Finger chains
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        base = f"Hand{finger}0"
        for seg in (1, 2, 3):
            edges.append((f"Hand{finger}{seg-1}", f"Hand{finger}{seg}"))
        edges.append((f"Hand{finger}3", f"Hand{finger}Tip"))

    # Thumb chain
    for seg in (1, 2, 3):
        edges.append((f"HandThumb{seg}", f"HandThumb{seg+1}" if seg < 3 else "HandThumbTip"))

    # Palm cross-connections (knuckle row)
    for a, b in (("HandIndex0", "HandMiddle0"),
                 ("HandMiddle0", "HandRing0"),
                 ("HandRing0",   "HandPinky0")):
        edges.append((a, b))

    return edges


EDGES = build_edges()


# ---------------------------------------------------------------------------
# Colour coding
# ---------------------------------------------------------------------------
HEIGHT_COLOURS = {"High": "#e66100", "Medium": "#5d3a9b", "Low": "#1a85ff", "": "#888888"}
PLACE_ALPHA  = 0.18    # shading on score plot
INSERT_ALPHA = 0.22


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def remap(p):
    """data (x,y,z) → plot axes so the hand reads naturally.
    data-X is the headset up-axis; we map it to plot-Z (vertical),
    data-Y → plot-X (depth), data-Z → plot-Y (lateral)."""
    return (p[1], p[2], p[0])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_frames(csv_path, side, max_duration=None):
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)

    t0 = None
    frames = []

    for _, row in df.iterrows():
        t = row.get("t_s")
        if pd.isna(t):
            continue
        if t0 is None:
            t0 = t
        if max_duration is not None and (t - t0) > max_duration:
            break

        wrc = f"{side}_HandWristRoot_x"
        if wrc not in row or pd.isna(row[wrc]):
            continue

        wr = np.array([row[f"{side}_HandWristRoot_x"],
                       row[f"{side}_HandWristRoot_y"],
                       row[f"{side}_HandWristRoot_z"]])

        joints = {}
        for col in df.columns:
            if col.startswith(f"{side}_") and col.endswith("_x"):
                j = col[len(f"{side}_"):-2]
                xv = row.get(f"{side}_{j}_x")
                yv = row.get(f"{side}_{j}_y")
                zv = row.get(f"{side}_{j}_z")
                if not any(pd.isna(v) for v in (xv, yv, zv)):
                    joints[j] = np.array([xv, yv, zv]) - wr   # wrist-relative

        # Wrist flexion (HandStart→WristRoot→HandMiddle0)
        flex = np.nan
        if all(k in joints for k in ("HandStart", "HandWristRoot", "HandMiddle0")):
            fore = joints["HandWristRoot"] - joints["HandStart"]
            hand = joints["HandMiddle0"]   - joints["HandWristRoot"]
            a = angle_between(fore, hand)
            if not np.isnan(a):
                flex = 180.0 - a

        # Thumb-index aperture (mm)
        aperture = np.nan
        if all(k in joints for k in ("HandThumbTip", "HandIndexTip")):
            aperture = np.linalg.norm(
                joints["HandThumbTip"] - joints["HandIndexTip"]) * 1000.0

        height = next((h for h in ("High", "Medium", "Low")
                       if row.get(h, 0) == 1), "")

        frames.append({
            "t_s":     t,
            "joints":  joints,
            "flex":    flex,
            "ap":      aperture,
            "place":   row.get("Place",  0) == 1,
            "insert":  row.get("Insert", 0) == 1,
            "height":  height,
        })

    print(f"  {len(frames)} frames  "
          f"t=[{frames[0]['t_s']:.1f}, {frames[-1]['t_s']:.1f}]s")
    return frames


# ---------------------------------------------------------------------------
# Event interval extraction (for background shading)
# ---------------------------------------------------------------------------
def extract_intervals(frames, key):
    """Return list of (t_start, t_end) where frames[key] is True."""
    ivs = []
    in_ev = False
    t_start = None
    for f in frames:
        active = f.get(key, False)
        if active and not in_ev:
            in_ev = True; t_start = f["t_s"]
        elif not active and in_ev:
            in_ev = False; ivs.append((t_start, f["t_s"]))
    if in_ev:
        ivs.append((t_start, frames[-1]["t_s"]))
    return ivs


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(frames, output_path, fps=30):
    times = [f["t_s"]  for f in frames]
    flexs = [f["flex"] for f in frames]
    aps   = [f["ap"]   for f in frames]

    place_ivs  = extract_intervals(frames, "place")
    insert_ivs = extract_intervals(frames, "insert")

    # ---- figure layout ----
    fig = plt.figure(figsize=(15, 7), facecolor="white")
    ax3d = fig.add_subplot(121, projection="3d")
    ax_ap  = fig.add_subplot(122)           # aperture — primary left axis
    ax_fl  = ax_ap.twinx()                  # wrist flexion — secondary right axis

    fig.suptitle("Hand Posture Dashboard", color="black", fontsize=13, y=0.98)

    # ---- 3D panel ----
    pad = 0.18
    for ax in (ax3d,):
        ax.set_facecolor("white")
        ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
        ax.set_xlabel("depth",  color="black", fontsize=8)
        ax.set_ylabel("lateral",color="black", fontsize=8)
        ax.set_zlabel("up",     color="black", fontsize=8)
        ax.tick_params(colors="black", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#cccccc")
        ax.yaxis.pane.set_edgecolor("#cccccc")
        ax.zaxis.pane.set_edgecolor("#cccccc")
    ax3d.view_init(elev=20, azim=-55)

    # ---- score panel ----
    ax_ap.set_facecolor("white")
    for spine in ax_ap.spines.values():
        spine.set_color("#cccccc")
    ax_fl.spines["right"].set_color("#d62728")

    # Background shading: Place = green, Insert = orange
    # Also stamp a small label at the top of each shaded region
    y_label = ax_ap.get_ylim()[1] if ax_ap.get_ylim()[1] != 0 else 1   # updated below
    for t0, t1 in place_ivs:
        ax_ap.axvspan(t0, t1, color="#00aa44", alpha=PLACE_ALPHA, zorder=0)
    for t0, t1 in insert_ivs:
        ax_ap.axvspan(t0, t1, color="#ff8800", alpha=INSERT_ALPHA, zorder=0)

    # Add invisible patches to legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#00aa44", alpha=0.5, label="Place event"),
        Patch(color="#ff8800", alpha=0.5, label="Insert event"),
    ]

    # Plot full time-series (static background lines)
    ax_ap.plot(times, aps,  color="#1f77b4", lw=1.2, alpha=0.5, label="Aperture (mm)")
    ax_fl.plot(times, flexs, color="#d62728", lw=1.2, alpha=0.5, label="Wrist flexion (°)")
    ax_fl.axhline(15, color="#d62728", lw=0.8, ls="--", alpha=0.35, label="REBA threshold 15°")

    ax_ap.set_xlim(times[0], times[-1])
    ap_clean  = [v for v in aps   if not np.isnan(v)]
    fl_clean  = [v for v in flexs if not np.isnan(v)]
    ap_max = max(ap_clean) * 1.15 + 5  if ap_clean  else 120
    fl_max = max(fl_clean) * 1.15 + 3  if fl_clean  else 30
    ax_ap.set_ylim(0, ap_max)
    ax_fl.set_ylim(0, fl_max)

    # Static text labels above each event band — drawn once, always visible
    label_y = ax_ap.transData.inverted().transform(
        ax_ap.transAxes.transform([[0, 0.97]]))[0][1]   # 97% of axis height
    for t0, t1 in place_ivs:
        ax_ap.text((t0 + t1) / 2, label_y, "PLACE",
                   ha="center", va="top", fontsize=7, fontweight="bold",
                   color="#006622", clip_on=True)
    for t0, t1 in insert_ivs:
        ax_ap.text((t0 + t1) / 2, label_y, "INSERT",
                   ha="center", va="top", fontsize=7, fontweight="bold",
                   color="#994400", clip_on=True)

    ax_ap.set_xlabel("Time (s)",           color="black")
    ax_ap.set_ylabel("Aperture (mm)",      color="#1f77b4")
    ax_fl.set_ylabel("Wrist flexion (°)",  color="#d62728")
    ax_ap.tick_params(colors="black")
    ax_fl.tick_params(axis="y", colors="#d62728")
    ax_ap.set_title("Ergonomic Scores", color="black", pad=8)

    # Combined legend
    lines_ap, labels_ap = ax_ap.get_legend_handles_labels()
    lines_fl, labels_fl = ax_fl.get_legend_handles_labels()
    ax_ap.legend(lines_ap + lines_fl + legend_patches,
                 labels_ap + labels_fl + [p.get_label() for p in legend_patches],
                 loc="upper left", fontsize=7,
                 facecolor="white", edgecolor="#cccccc", labelcolor="black")

    # ---- animated elements ----
    time_line,    = ax_ap.plot([], [], color="black", lw=1.5, zorder=5)
    score_text    = ax_ap.text(
        0.02, 0.97, "", transform=ax_ap.transAxes,
        fontsize=9, va="top", ha="left", color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.9))
    ax3d_title    = ax3d.set_title("", color="black", fontsize=10, pad=8)

    # Event banner: sits clearly above the 3D panel, large and bold
    # Position 0.02 (left) and y=0.96 keeps it inside the left half of the figure
    event_banner  = fig.text(
        0.26, 0.97, "", ha="center", va="top",
        fontsize=13, fontweight="bold", color="white",
        zorder=10, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#007700",
                  edgecolor="none", alpha=0.0))

    skel_lines = []
    joint_dots  = [None]

    def update(i):
        f = frames[i]
        joints = f["joints"]
        h = f["height"]

        # ---- clear previous 3D artists ----
        for ln in skel_lines:
            ln.remove()
        skel_lines.clear()
        if joint_dots[0] is not None:
            joint_dots[0].remove()
            joint_dots[0] = None

        # ---- joint dots ----
        pts = [remap(v) for v in joints.values()]
        if pts:
            xs, ys, zs = zip(*pts)
            joint_dots[0] = ax3d.scatter(
                xs, ys, zs, color="#333333", s=22, depthshade=True, zorder=3)

        # ---- skeleton edges ----
        wrist_ok = not np.isnan(f["flex"])
        for a, b in EDGES:
            if a not in joints or b not in joints:
                continue
            pa = remap(joints[a]); pb = remap(joints[b])

            # colour by segment type
            if a == "HandStart" or b == "HandStart":
                col = "#555555"           # forearm anchor — dark grey
            elif "Thumb" in a or "Thumb" in b:
                col = "#f0a500"           # thumb = amber
            elif a in ("HandIndex0","HandMiddle0","HandRing0","HandPinky0") \
              or b in ("HandIndex0","HandMiddle0","HandRing0","HandPinky0"):
                # palm row
                col = HEIGHT_COLOURS.get(h, "#888888")
            elif "WristRoot" in (a, b):
                col = ("#d62728" if wrist_ok and f["flex"] > 15
                       else "#2ca02c")    # red = REBA caution, green = OK
            else:
                col = "#5599ee"           # generic finger bone

            ln, = ax3d.plot(
                [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color=col, lw=2.0, zorder=2)
            skel_lines.append(ln)

        # ---- time cursor ----
        t = f["t_s"]
        time_line.set_data([t, t], [0, ap_max])
        time_line.set_xdata([t, t])

        # ---- score readout ----
        fl_str = f"{f['flex']:.1f}°" if not np.isnan(f["flex"]) else "n/a"
        ap_str = f"{f['ap']:.0f} mm" if not np.isnan(f["ap"])   else "n/a"
        score_text.set_text(
            f"t = {t:.2f} s\n"
            f"Wrist flex : {fl_str}\n"
            f"Aperture   : {ap_str}\n"
            f"Height     : {h or '—'}")

        # ---- event banner ----
        if f["place"] and f["insert"]:
            event_banner.set_text("● PLACE + INSERT")
            event_banner.get_bbox_patch().set(facecolor="#cc4400", alpha=0.9)
            event_banner.set_visible(True)
        elif f["place"]:
            event_banner.set_text("● PLACE")
            event_banner.get_bbox_patch().set(facecolor="#007700", alpha=0.9)
            event_banner.set_visible(True)
        elif f["insert"]:
            event_banner.set_text("● INSERT")
            event_banner.get_bbox_patch().set(facecolor="#cc6600", alpha=0.9)
            event_banner.set_visible(True)
        else:
            event_banner.set_visible(False)

        # ---- 3D title with height ----
        ax3d_title.set_text(
            f"{f['t_s']:.2f} s  |  {h or '—'}" if h else f"{f['t_s']:.2f} s")
        ax3d_title.set_color(HEIGHT_COLOURS.get(h, "black"))

        return skel_lines + [joint_dots[0], time_line, score_text,
                              event_banner, ax3d_title]

    print(f"Rendering {len(frames)} frames → {output_path}")
    writer = FFMpegWriter(fps=fps, bitrate=3000,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(output_path), dpi=110):
        for i in tqdm(range(len(frames)), desc="Rendering", unit="fr"):
            update(i)
            writer.grab_frame()
    plt.close(fig)
    print(f"Done → {output_path}")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def find_csv(csv_path=None, root=None, participant=None, trial=None):
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            return p
        raise FileNotFoundError(p)
    if not root:
        raise ValueError("Provide --csv or --root")
    root = Path(root)
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir(): continue
        if participant and pdir.name.lower() != participant.lower(): continue
        for tdir in sorted(pdir.iterdir()):
            if not tdir.is_dir(): continue
            if trial and trial.lower() not in tdir.name.lower(): continue
            for name in (f"{tdir.name}_hand_labelled.csv",
                         f"{tdir.name}_hand.csv"):
                p = tdir / name
                if p.exists():
                    return p
    raise FileNotFoundError(f"No hand CSV under {root} p={participant} t={trial}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",         default=None)
    ap.add_argument("--root",        default=r"A:\Automated_chain_BETA\Participant_Landmarks")
    ap.add_argument("--participant", default=None)
    ap.add_argument("--trial",       default=None)
    ap.add_argument("--side",        default="Left",  help="Left or Right")
    ap.add_argument("--output",      default=None)
    ap.add_argument("--duration",    type=float, default=None,
                    help="Limit to this many seconds (for quick tests)")
    ap.add_argument("--fps",         type=int,   default=30)
    args = ap.parse_args()

    csv_path = find_csv(args.csv, args.root, args.participant, args.trial)
    print(f"Input : {csv_path}")

    if args.output:
        out = Path(args.output)
    else:
        out = csv_path.parent / f"{csv_path.stem.replace('_hand_labelled','')}_hand_viz.mp4"
    print(f"Output: {out}")

    frames = load_frames(csv_path, args.side, args.duration)
    render(frames, out, fps=args.fps)