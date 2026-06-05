#!/usr/bin/env python3
"""
Animate hand posture and dynamic ergonomic scoring from labelled hand CSV data.

Usage examples:
    python animate_hand.py --csv "A:\\Automated_chain_BETA\\Participant_Landmarks\\P007\\P007_Long_Large_Front_weighted_A135\\P007_Long_Large_Front_weighted_A135_hand_labelled.csv" \
        --output P007_hand.mp4 --side Left --duration 5

    python animate_hand.py --root "A:\\Automated_chain_BETA\\Participant_Landmarks" 
        --participant P007 --trial P007_Long_Large_Front_weighted_A135 --duration 5

    python animate_hand.py --csv path/to/_hand_labelled.csv --side Right

The rendered dashboard shows a 3D hand skeleton plus live score plots for wrist flexion and aperture.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None, unit=None):
        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)
        if desc:
            print(desc)
        for index, item in enumerate(iterable, start=1):
            yield item
            if total:
                pct = index / total * 100
                print(f"\r{desc or 'Progress'}: {index}/{total} {pct:.1f}%", end="", flush=True)
        if total:
            print()

# ----------------------------------------------------------------------------
# Math from 07_extract_posture_features.py
# ----------------------------------------------------------------------------
def angle_between(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return np.nan
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))

def remap_point(p):
    """Remap (data_x, data_y, data_z) -> (plot_x, plot_y, plot_z) 
    so data_x is vertical (head up)."""
    dx, dy, dz = p
    return (dy, dz, dx)


def find_hand_csv(csv_path=None, root=None, participant=None, trial=None):
    """Resolve a hand CSV path from command-line arguments."""
    if csv_path is not None:
        path = Path(csv_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Input CSV not found: {path}")

    if root is None:
        raise ValueError("Either --csv or --root must be provided")

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    candidates = []
    participant_lower = participant.lower() if participant else None
    trial_lower = trial.lower() if trial else None

    for participant_dir in sorted(root.iterdir()):
        if not participant_dir.is_dir():
            continue
        if participant_lower and participant_dir.name.lower() != participant_lower:
            continue
        for trial_dir in sorted(participant_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if trial_lower and trial_lower not in trial_dir.name.lower():
                continue
            labelled = trial_dir / f"{trial_dir.name}_hand_labelled.csv"
            raw_hand = trial_dir / f"{trial_dir.name}_hand.csv"
            output_path = trial_dir / f"{trial_dir.name}_hand.mp4"
            if labelled.exists():
                candidates.append(labelled)
            elif raw_hand.exists():
                candidates.append(raw_hand)

    if not candidates:
        raise FileNotFoundError(
            f"No hand CSV found under {root} for participant {participant} trial={trial}")
    if len(candidates) > 1:
        print("Warning: multiple matching hand CSVs found, using the first one:")
        for c in candidates:
            print(f"  {c}")
    return candidates[0]

# ----------------------------------------------------------------------------
# Main Process
# ----------------------------------------------------------------------------
def process_data(csv_path, target_side="Left", max_duration=None):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Hand skeleton edges
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    edges = [("HandStart", "HandWristRoot")]
    for f in fingers:
        edges.append(("HandWristRoot", f"Hand{f}1" if f == "Thumb" else f"Hand{f}0"))
        if f == "Thumb":
            edges.extend([("HandThumb1", "HandThumb2"), ("HandThumb2", "HandThumb3"), ("HandThumb3", "HandThumbTip")])
        else:
            edges.extend([(f"Hand{f}0", f"Hand{f}1"), (f"Hand{f}1", f"Hand{f}2"), (f"Hand{f}2", f"Hand{f}3"), (f"Hand{f}3", f"Hand{f}Tip")])

    frames = []
    times = []
    flex_history = []
    aperture_history = []
    first_time = None
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading frames", unit="row"):
        t = row.get('t_s')
        if pd.isna(t):
            continue
        if first_time is None:
            first_time = t
        if max_duration is not None and (t - first_time) > max_duration:
            break
        
        wr_col_x = f'{target_side}_HandWristRoot_x'
        if wr_col_x not in row or pd.isna(row[wr_col_x]): 
            continue
            
        # Get Wrist position to use as 0,0,0 anchor
        wr_root = np.array([row[f'{target_side}_HandWristRoot_x'], 
                            row[f'{target_side}_HandWristRoot_y'], 
                            row[f'{target_side}_HandWristRoot_z']])
        
        joints = {}
        for col in df.columns:
            if col.startswith(f'{target_side}_') and col.endswith('_x'):
                j_name = col[len(f'{target_side}_'):-2]
                x, y, z = row[f'{target_side}_{j_name}_x'], row[f'{target_side}_{j_name}_y'], row[f'{target_side}_{j_name}_z']
                if not pd.isna(x):
                    # Center joint relative to wrist root
                    joints[j_name] = np.array([x, y, z]) - wr_root
        
        # Calculate Ergonomic Scores
        flexion = np.nan
        aperture = np.nan
        
        if all(k in joints for k in ['HandStart', 'HandMiddle0', 'HandWristRoot']):
            st = joints['HandStart']
            wr2 = joints['HandWristRoot'] # This is exactly 0,0,0
            m0 = joints['HandMiddle0']
            fore = wr2 - st
            hand_axis = m0 - wr2
            a = angle_between(fore, hand_axis)
            if not np.isnan(a):
                flexion = 180.0 - a
                
        if all(k in joints for k in ['HandThumbTip', 'HandIndexTip']):
            aperture = np.linalg.norm(joints['HandThumbTip'] - joints['HandIndexTip']) * 1000.0 # to mm
            
        times.append(t)
        flex_history.append(flexion if not np.isnan(flexion) else 0)
        aperture_history.append(aperture if not np.isnan(aperture) else 0)
        
        frames.append({
            't_s': t,
            'joints': joints,
            'flexion': flexion,
            'aperture': aperture
        })
        
    return frames, times, flex_history, aperture_history, edges

def render_dashboard(csv_path, output_path, target_side="Left", output_fps=30, max_duration=None):
    frames, times, flex_hist, ap_hist, edges = process_data(csv_path, target_side, max_duration=max_duration)
    if not frames:
        print("No valid frames found.")
        return

    # Setup Figure (Split Screen)
    fig = plt.figure(figsize=(14, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_metrics = fig.add_subplot(122)
    
    # Configure 3D axis (Locked around 0,0,0 with 15cm padding)
    pad = 0.15 
    ax_3d.set_xlim(-pad, pad)
    ax_3d.set_ylim(-pad, pad)
    ax_3d.set_zlim(-pad, pad)
    ax_3d.set_xlabel("y (front/back)")
    ax_3d.set_ylabel("z (left/right)")
    ax_3d.set_zlabel("x (up/down)")
    ax_3d.set_title(f"{target_side} Hand (Wrist Centered)")
    ax_3d.view_init(elev=15, azim=-60)
    
    # Configure Metrics Axis
    ax_metrics.plot(times, flex_hist, label="Wrist Flexion (°)", color="#d62728")
    ax_metrics.plot(times, ap_hist, label="Aperture (mm)", color="#1f77b4")
    ax_metrics.axhline(15, color='r', linestyle='--', alpha=0.3, label="REBA Flexion Threshold")
    ax_metrics.set_xlim(times[0], times[-1])
    ax_metrics.set_ylim([0, max(max(flex_hist), max(ap_hist)) + 20])
    ax_metrics.set_xlabel("Time (s)")
    ax_metrics.set_ylabel("Measurement")
    ax_metrics.legend(loc="upper right")
    ax_metrics.set_title("Live Ergonomic Scores")
    
    time_line = ax_metrics.axvline(x=times[0], color='k', linewidth=2)
    score_text = ax_metrics.text(0.02, 0.95, "", transform=ax_metrics.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    
    scatter_artists = []
    line_artists = []

    def update(i):
        f = frames[i]
        
        # Clear previous 3D artists
        for a in scatter_artists + line_artists: a.remove()
        scatter_artists.clear()
        line_artists.clear()
        
        joints = f['joints']
        
        # Plot Joints
        pts = [remap_point(v) for v in joints.values()]
        if pts:
            xs, ys, zs = zip(*pts)
            sc = ax_3d.scatter(xs, ys, zs, color="#7f7f7f", s=30)
            scatter_artists.append(sc)
            
        # Plot Edges
        for a, b in edges:
            if a in joints and b in joints:
                pa, pb = remap_point(joints[a]), remap_point(joints[b])
                
                # Option 2 applied: Color code the wrist joint based on REBA rules
                color = "#7f7f7f"
                if "HandWristRoot" in [a, b] and not np.isnan(f['flexion']):
                    color = "#d62728" if f['flexion'] > 15 else "#2ca02c"
                
                ln, = ax_3d.plot([pa, pb], [pa, pb], [pa, pb], color=color, linewidth=2.5)
                line_artists.append(ln)

        # Update Playhead on Metrics Chart
        time_line.set_xdata([f['t_s'], f['t_s']])
        score_text.set_text(
            f"Time: {f['t_s']:.2f}s\nFlexion: {f['flexion']:.1f}°\nAperture: {f['aperture']:.1f} mm")
        
        return scatter_artists + line_artists + [time_line, score_text]
    
    print(f"Rendering video to {output_path}...")
    writer = FFMpegWriter(fps=output_fps, bitrate=2000)
    with writer.saving(fig, output_path, dpi=100):
        for i in tqdm(range(len(frames)), desc="Rendering frames", unit="frame"):
            update(i)
            writer.grab_frame()
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a hand posture dashboard video from labelled landmark CSV data.")
    parser.add_argument("--csv", default=None,
                        help="Path to the input hand CSV file (labelled or raw). If omitted, --root and --trial are used.")
    parser.add_argument("--root", default="A:\\Automated_chain_BETA\\Participant_Landmarks",
                        help="Root folder containing participant trial folders")
    parser.add_argument("--participant", default=None,
                        help="Participant ID to target (e.g. P007)")
    parser.add_argument("--trial", default=None,
                        help="Trial folder name or unique suffix to target")
    parser.add_argument("--side", default="Left", help="Hand side to render (Left or Right)")
    parser.add_argument("--output", default=None,
                        help="Optional output MP4 path. Defaults to trial-based name under the trial folder.")
    parser.add_argument("--duration", type=float, default=None, help="Limit recording to this many seconds for testing")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate")
    args = parser.parse_args()

    csv_path = find_hand_csv(csv_path=args.csv, root=args.root,
                             participant=args.participant, trial=args.trial)
    print(f"Using CSV: {csv_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        trial_name = csv_path.parent.name
        output_path = csv_path.parent / f"{trial_name}_hand.mp4"

    print(f"Output path: {output_path}")
    render_dashboard(csv_path, output_path=output_path, target_side=args.side,
                     output_fps=args.fps, max_duration=args.duration)