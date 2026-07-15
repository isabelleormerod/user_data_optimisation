#!/usr/bin/env python3
"""
Animate body posture and dynamic ergonomic scoring from labelled body CSV data.

Usage examples:
    python animate_body.py --csv "A:\\Automated_chain_BETA\\Participant_Landmarks\\P007\\P007_Long_Large_Front_weighted_A135\\P007_Long_Large_Front_weighted_A135_body_labelled.csv" \
        --output P007_body.mp4 --fps 30 --duration 5



    python animate_body.py --csv path/to/_body_labelled.csv

The rendered dashboard shows a 3D body skeleton plus live trunk/neck angle scores and active events.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

DEFAULT_ROOT = "A:\\Automated_chain_BETA\\Participant_Landmarks"
DEFAULT_FPS = 30

BODY_EDGES = [
    ("LeftEar", "LeftEyeOuter"), ("LeftEyeOuter", "LeftEye"),
    ("LeftEye", "LeftEyeInner"), ("LeftEyeInner", "Nose"),
    ("Nose", "RightEyeInner"), ("RightEyeInner", "RightEye"),
    ("RightEye", "RightEyeOuter"), ("RightEyeOuter", "RightEar"),
    ("MouthLeft", "MouthRight"), ("LeftShoulder", "RightShoulder"), 
    ("LeftShoulder", "LeftHip"), ("RightShoulder", "RightHip"), 
    ("LeftHip", "RightHip"), ("LeftShoulder", "LeftElbow"), 
    ("LeftElbow", "LeftWrist"), ("LeftWrist", "LeftPinky"), 
    ("LeftWrist", "LeftIndex"), ("LeftWrist", "LeftThumb"), 
    ("LeftIndex", "LeftPinky"), ("RightShoulder", "RightElbow"), 
    ("RightElbow", "RightWrist"), ("RightWrist", "RightPinky"), 
    ("RightWrist", "RightIndex"), ("RightWrist", "RightThumb"), 
    ("RightIndex", "RightPinky"), ("LeftHip", "LeftKnee"), 
    ("LeftKnee", "LeftAnkle"), ("LeftAnkle", "LeftHeel"), 
    ("LeftHeel", "LeftFootIndex"), ("LeftAnkle", "LeftFootIndex"),
    ("RightHip", "RightKnee"), ("RightKnee", "RightAnkle"),
    ("RightAnkle", "RightHeel"), ("RightHeel", "RightFootIndex"),
    ("RightAnkle", "RightFootIndex"),
]

# The labels found in your body CSV
EVENT_COLS = ['High', 'Insert', 'Low', 'Medium', 'Place', 'Point_1', 'Point_2', 'Point_3', 'Point_4', 'Point_5', 'Point_6']


def find_body_csv(csv_path=None, root=None, participant=None, trial=None):
    """Resolve a body CSV path from command-line arguments."""
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
            labelled = trial_dir / f"{trial_dir.name}_body_labelled.csv"
            raw_body = trial_dir / f"{trial_dir.name}_body.csv"
            if labelled.exists():
                candidates.append(labelled)
            elif raw_body.exists():
                candidates.append(raw_body)

    if not candidates:
        raise FileNotFoundError(
            f"No body CSV found under {root} for participant={participant} trial={trial}")
    if len(candidates) > 1:
        print("Warning: multiple matching body CSVs found, using the first one:")
        for c in candidates:
            print(f"  {c}")
    return candidates[0]


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

def main(csv_path, output_path, fps, max_duration=None):
    print(f"Loading {csv_path}...")
    body_df = pd.read_csv(csv_path).sort_values('t_s')
    
    # Find the lowest foot coordinate to use as the floor (MediaPipe X is down)
    foot_cols = [c for c in body_df.columns if 'Ankle_x' in c or 'Heel_x' in c or 'FootIndex_x' in c]
    floor_x = body_df[foot_cols].max().max()

    t_start = body_df['t_s'].min()
    t_end = body_df['t_s'].max()
    if max_duration is not None:
        t_end = min(t_start + max_duration, t_end)

    # Subsample to the target framerate
    time_ticks = np.arange(t_start, t_end, 1.0 / fps)
    merged = pd.merge_asof(pd.DataFrame({'t_s': time_ticks}), body_df, on='t_s', direction='nearest')

    # Setup the Figure
    fig = plt.figure(figsize=(14, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_metrics = fig.add_subplot(122)

    # Configure 3D View
    ax_3d.set_zlim(0, 1.8)
    ax_3d.view_init(elev=15, azim=-60)
    ax_3d.set_title("Body Posture (World Frame)")

    # Pre-calculate the Metrics
    times = merged['t_s'].values
    trunk_flex, neck_flex = [], []
    
    for _, row in merged.iterrows():
        # Up vector in raw data is -X
        up = np.array([-1, 0, 0])
        try:
            ls = np.array([row['LeftShoulder_x'], row['LeftShoulder_y'], row['LeftShoulder_z']])
            rs = np.array([row['RightShoulder_x'], row['RightShoulder_y'], row['RightShoulder_z']])
            lh = np.array([row['LeftHip_x'], row['LeftHip_y'], row['LeftHip_z']])
            rh = np.array([row['RightHip_x'], row['RightHip_y'], row['RightHip_z']])
            nose = np.array([row['Nose_x'], row['Nose_y'], row['Nose_z']])
            
            mid_shoulder = (ls + rs) / 2
            mid_hip = (lh + rh) / 2
            
            trunk_vec = mid_shoulder - mid_hip
            neck_vec = nose - mid_shoulder
            
            trunk_flex.append(angle_between(trunk_vec, up))
            neck_flex.append(angle_between(neck_vec, trunk_vec))
        except:
            trunk_flex.append(np.nan)
            neck_flex.append(np.nan)

    # Configure 2D Chart
    ax_metrics.plot(times, trunk_flex, label="Trunk Flexion (°)", color="blue")
    ax_metrics.plot(times, neck_flex, label="Neck Flexion (°)", color="orange")
    ax_metrics.set_xlim(times[0], times[-1])
    ax_metrics.set_ylim(0, 90)
    ax_metrics.set_xlabel("Time (s)")
    ax_metrics.set_ylabel("Degrees (°)")
    ax_metrics.legend(loc="upper right")
    ax_metrics.set_title("Live Ergonomic Scores")
    
    time_line = ax_metrics.axvline(x=times[0], color='k', linewidth=2)
    
    # Text box for active labels
    event_text = ax_metrics.text(0.05, 0.95, "", transform=ax_metrics.transAxes, fontsize=14, 
                                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    available_landmarks = {c[:-2] for c in body_df.columns if c.endswith('_x')}
    pbar = tqdm(total=len(merged), desc="Rendering Body", unit="frames")
    artists = []

    def update(i):
        for a in artists: a.remove()
        artists.clear()
        row = merged.iloc[i]

        # Read joint coordinates
        joints = {}
        for lm in available_landmarks:
            x, y, z = row.get(f"{lm}_x"), row.get(f"{lm}_y"), row.get(f"{lm}_z")
            if pd.notna(x): 
                # ROTATE: MediaPipe +X is Down. Plot Z becomes Up.
                joints[lm] = (y, z, floor_x - x)

        if joints:
            # Draw joints
            xs, ys, zs = zip(*joints.values())
            artists.append(ax_3d.scatter(xs, ys, zs, color="#1f77b4", s=15))
            
            # Draw bones
            for a, b in BODY_EDGES:
                if a in joints and b in joints:
                    pa, pb = joints[a], joints[b]
                    # Highlight torso/spine links in red, limbs in green
                    color = "#d62728" if "Shoulder" in a or "Hip" in a else "#2ca02c"
                    ln, = ax_3d.plot([pa, pb], [pa, pb], [pa, pb], color=color, linewidth=2)
                    artists.append(ln)

        # Update event label text
        active_events = [col for col in EVENT_COLS if row.get(col) == 1]
        evt_str = "Active Event: " + (", ".join(active_events) if active_events else "None")
        event_text.set_text(evt_str)
        
        # Update playhead
        time_line.set_xdata([row['t_s'], row['t_s']])
        
        pbar.update(1)
        return artists + [time_line, event_text]

    anim = FuncAnimation(fig, update, frames=len(merged), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    pbar.close()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a body posture dashboard video from labelled body CSV data.")
    parser.add_argument("--csv", default=None,
                        help="Path to the input body CSV file (labelled or raw). If omitted, --root and --trial are used.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Root folder containing participant trial folders")
    parser.add_argument("--participant", default=None,
                        help="Participant ID to target (e.g. P007)")
    parser.add_argument("--trial", default=None,
                        help="Trial folder name or unique suffix to target")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path. Defaults to <trial>_Body_Metrics.mp4")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Output frame rate")
    parser.add_argument("--duration", type=float, default=None,
                        help="Limit recording to this many seconds for testing")
    args = parser.parse_args()

    csv_path = find_body_csv(csv_path=args.csv, root=args.root,
                              participant=args.participant, trial=args.trial)
    output_path = Path(args.output) if args.output else Path(csv_path).parent / f"{Path(csv_path).parent.name}_Body_Metrics.mp4"
    print(f"Using CSV: {csv_path}")
    print(f"Saving output to: {output_path}")
    main(csv_path, output_path, fps=args.fps, max_duration=args.duration)