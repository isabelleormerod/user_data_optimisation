#!/usr/bin/env python3
"""
plot_landmark_verification.py

Generates a side-by-side pre-processing verification figure:
  Panel A: Raw Experimental Screenshot (Loaded from image file)
  Panel B: Reconstructed 3D Wireframe (Body, Right Hand, and Pen from CSVs)
  Panel C: Synchronized Kinematic Time-Series (Pen Speed with frame marker)

Usage:
  python plot_landmark_verification.py --trial P001_Long_Large_Front_weighted_A90 --frame-idx 150 --screenshot path/to/shot.png
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# Define anatomical connections (wireframe bones)
BODY_BONES = [
    ("LeftShoulder", "RightShoulder"),
    ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist"),
    ("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"),
    ("LeftShoulder", "LeftHip"), ("RightShoulder", "RightHip"),
    ("LeftHip", "RightHip")
]

HAND_BONES = [
    ("HandWristRoot", "HandThumb1"), ("HandThumb1", "HandThumb2"), ("HandThumb2", "HandThumb3"), ("HandThumb3", "HandThumbTip"),
    ("HandWristRoot", "HandIndex0"), ("HandIndex0", "HandIndex1"), ("HandIndex1", "HandIndex2"), ("HandIndex2", "HandIndex3"), ("HandIndex3", "HandIndexTip"),
    ("HandWristRoot", "HandMiddle0"), ("HandMiddle0", "HandMiddle1"), ("HandMiddle1", "HandMiddle2"), ("HandMiddle2", "HandMiddle3"), ("HandMiddle3", "HandMiddleTip"),
]

def get_coords(df, idx, prefix, jname):
    """Safely extracts X, Y, Z coordinates for a joint at a specific row index."""
    cols = [f"{prefix}{jname}_{ax}" for ax in ("x", "y", "z")]
    if not all(c in df.columns for c in cols):
        return None
    vals = df.loc[idx, cols].values.astype(float)
    return vals if not np.any(np.isnan(vals)) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pen-csv", type=Path, required=True, help="Path to *_pen_labelled.csv")
    ap.add_argument("--body-csv", type=Path, required=True, help="Path to *_body.csv")
    ap.add_argument("--hand-csv", type=Path, required=True, help="Path to *_hand.csv")
    ap.add_argument("--screenshot", type=Path, default=None, help="Path to raw video/VR screenshot PNG")
    ap.add_argument("--frame-idx", type=int, default=100, help="Row index representing the snapshot timestamp")
    ap.add_argument("--out", type=Path, default=Path("verification_figure.png"))
    args = ap.parse_args()

    df_pen = pd.read_csv(args.pen_csv)
    df_body = pd.read_csv(args.body-csv)
    df_hand = pd.read_csv(args.hand-csv)

    idx = min(args.frame_idx, len(df_pen) - 1)
    target_t = df_pen.loc[idx, "t_s"]

    # Setup 3-Panel Figure
    fig = plt.figure(figsize=(15, 5))
    
    # --- PANEL A: Raw Experimental Screenshot ---
    ax1 = fig.add_subplot(1, 3, 1)
    if args.screenshot and args.screenshot.is_file():
        img = mpimg.imread(args.screenshot)
        ax1.imshow(img)
    else:
        ax1.text(0.5, 0.5, "[Insert Raw VR Screenshot\nat t = {:.2f}s]".format(target_t),
                 ha="center", va="center", fontsize=12, bbox=dict(boxstyle="square", fc="white", ec="black"))
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("A. Raw Experimental Recording", fontweight="bold", pad=10)

    # --- PANEL B: Reconstructed 3D Wireframe ---
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    
    # Draw Body Skeleton (Blue)
    for j1, j2 in BODY_BONES:
        p1 = get_coords(df_body, idx, "", j1)
        p2 = get_coords(df_body, idx, "", j2)
        if p1 is not None and p2 is not None:
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="#2b5c8f", lw=2.5, marker="o", ms=4)

    # Draw Right Hand Skeleton (Green)
    for j1, j2 in HAND_BONES:
        p1 = get_coords(df_hand, idx, "Right_", j1)
        p2 = get_coords(df_hand, idx, "Right_", j2)
        if p1 is not None and p2 is not None:
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="#1b9e77", lw=1.5, marker=".", ms=3)

    # Draw Pen Vector (Orange Cylinder approximation)
    if all(c in df_pen.columns for c in ("x_flat", "y_flat", "z_flat")):
        px, py, pz = df_pen.loc[idx, ["x_flat", "y_flat", "z_flat"]].values
        ax2.scatter(px, py, pz, c="#d95f02", s=60, marker="^", label="Pen Tip")

    ax2.set_title("B. Reconstructed Kinematic Space", fontweight="bold", pad=10)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    ax2.view_init(elev=20, azim=-60) # Adjust viewing angle to match camera

    # --- PANEL C: Synchronized Kinematic Trace ---
    ax3 = fig.add_subplot(1, 3, 3)
    pos = df_pen[["x_flat", "y_flat", "z_flat"]].values
    vel = np.gradient(pos, df_pen["t_s"].values, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    
    ax3.plot(df_pen["t_s"], speed, c="#7570b3", lw=1.5, label="Pen Speed")
    ax3.axvline(target_t, color="red", linestyle="--", lw=2, label="Snapshot Frame ($t_0$)")
    ax3.fill_between(df_pen["t_s"], 0, speed, where=(df_pen["Place"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])),
                     color="#1b9e77", alpha=0.2, label="Detected Place Event")
    
    ax3.set_title("C. Kinematic Trace Alignment", fontweight="bold", pad=10)
    ax3.set_xlabel("Trial Time (s)"); ax3.set_ylabel("Pen Speed (m/s)")
    ax3.legend(loc="upper right", fontsize=9, frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Verification figure saved to: {args.out}")

if __name__ == "__main__":
    main()