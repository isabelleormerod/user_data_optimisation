#!/usr/bin/env python3
"""
Animate pen movement and dynamic speed/events from labelled pen CSV data.

Usage examples:
    python animate_pen.py --csv "A:\\Automated_chain_BETA\\Participant_Landmarks\\P007\\P007_Long_Large_Front_weighted_A135\\P007_Long_Large_Front_weighted_A135_pen_flattened_labelled.csv" \
        --output P007_Pen_Metrics.mp4 --fps 30 --duration 5

    python animate_pen.py --root "A:\\Automated_chain_BETA\\Participant_Landmarks" --participant P007 --trial P007_Long_Large_Front_weighted_A135 --duration 5

    python animate_pen.py --csv path/to/_pen_flattened_labelled.csv

The rendered dashboard shows the 3D pen trajectory plus live speed and active event labels.
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
TARGET_FPS = 30
DEFAULT_FPS = 30
DEFAULT_OUTPUT_SUFFIX = "_Pen_Metrics.mp4"

EVENT_COLS = ['High', 'Insert', 'Low', 'Medium', 'Place', 'Point_1', 'Point_2', 'Point_3', 'Point_4', 'Point_5', 'Point_6']

def quat_forward(qx, qy, qz, qw):
    fx = 2 * (qx * qz + qw * qy)
    fy = 2 * (qy * qz - qw * qx)
    fz = 1 - 2 * (qx * qx + qy * qy)
    return np.array([fx, fy, fz])

def find_pen_csv(csv_path=None, root=None, participant=None, trial=None):
    """Resolve a pen CSV path from command-line arguments."""
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
            labelled = trial_dir / f"{trial_dir.name}_pen_flattened_labelled.csv"
            raw_pen = trial_dir / f"{trial_dir.name}_pen_flattened.csv"
            if labelled.exists():
                candidates.append(labelled)
            elif raw_pen.exists():
                candidates.append(raw_pen)

    if not candidates:
        raise FileNotFoundError(
            f"No pen CSV found under {root} for participant={participant} trial={trial}")
    if len(candidates) > 1:
        print("Warning: multiple matching pen CSVs found, using the first one:")
        for c in candidates:
            print(f"  {c}")
    return candidates[0]

def main(csv_path, output_path, fps, max_duration=None):
    df = pd.read_csv(csv_path).sort_values('t_s')
    
    t_start, t_end = df['t_s'].min(), df['t_s'].max()
    if max_duration is not None:
        t_end = min(t_start + max_duration, t_end)
    
    time_ticks = np.arange(t_start, t_end, 1.0 / fps)
    merged = pd.merge_asof(pd.DataFrame({'t_s': time_ticks}), df, on='t_s', direction='nearest')

    # Calculate Speed
    dx = np.diff(merged['x'], prepend=merged['x'].iloc[0])
    dy = np.diff(merged['y'], prepend=merged['y'].iloc[0])
    dz = np.diff(merged['z'], prepend=merged['z'].iloc[0])
    dt = 1.0 / fps
    speeds = np.sqrt(dx**2 + dy**2 + dz**2) / dt

    fig = plt.figure(figsize=(14, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_metrics = fig.add_subplot(122)

    ax_3d.set_xlim(merged['x'].min(), merged['x'].max())
    ax_3d.set_ylim(merged['z'].min(), merged['z'].max())
    ax_3d.set_zlim(0, merged['y'].max())
    ax_3d.view_init(elev=15, azim=-60)
    ax_3d.set_title("Pen Trajectory (World Frame)")

    times = merged['t_s'].values
    ax_metrics.plot(times, speeds, label="Speed (m/s)", color="purple")
    ax_metrics.set_xlim(times[0], times[-1])
    ax_metrics.set_ylim(0, np.percentile(speeds, 99) * 1.2) # Cap at 99th percentile to ignore noise spikes
    ax_metrics.legend(loc="upper right")
    
    time_line = ax_metrics.axvline(x=times[0], color='k', linewidth=2)
    event_text = ax_metrics.text(0.05, 0.95, "", transform=ax_metrics.transAxes, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    trail_x, trail_y, trail_z = [], [], []
    artists = []
    pbar = tqdm(total=len(merged), desc="Rendering Pen", unit="frames")

    def update(i):
        for a in artists: a.remove()
        artists.clear()
        row = merged.iloc[i]

        px, py, pz = row.get('x'), row.get('y'), row.get('z')
        if pd.notna(px):
            # Y is up in quest
            trail_x.append(px); trail_y.append(pz); trail_z.append(py)
            if len(trail_x) > 30: 
                trail_x.pop(0); trail_y.pop(0); trail_z.pop(0)

            trail, = ax_3d.plot(trail_x, trail_y, trail_z, color="#7f7f7f", alpha=0.5, linewidth=2)
            pt = ax_3d.scatter([px], [pz], [py], color="#d62728", s=40)
            artists.extend([trail, pt])

            qw, qx, qy, qz = row.get('qw'), row.get('qx'), row.get('qy'), row.get('qz')
            if pd.notna(qw):
                fwd = quat_forward(qx, qy, qz, qw) * 0.1
                ln, = ax_3d.plot([px, px + fwd[0]], [pz, pz + fwd[1]], [py, py + fwd[2]], color="blue", linewidth=3)
                artists.append(ln)

        # Labels
        active_events = [col for col in EVENT_COLS if row.get(col) == 1]
        evt_str = "Active Event: " + (", ".join(active_events) if active_events else "None")
        event_text.set_text(evt_str)
        time_line.set_xdata([row['t_s'], row['t_s']])
        
        pbar.update(1)
        return artists + [time_line, event_text]

    anim = FuncAnimation(fig, update, frames=len(merged), blit=False)
    anim.save(str(output_path), writer=FFMpegWriter(fps=fps, bitrate=2000))
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a pen trajectory dashboard video from labelled pen CSV data.")
    parser.add_argument("--csv", default=None,
                        help="Path to the input pen CSV file (labelled or raw). If omitted, --root and --trial are used.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Root folder containing participant trial folders")
    parser.add_argument("--participant", default=None,
                        help="Participant ID to target (e.g. P007)")
    parser.add_argument("--trial", default=None,
                        help="Trial folder name or unique suffix to target")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path. Defaults to <trial>_Pen_Metrics.mp4")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Output frame rate")
    parser.add_argument("--duration", type=float, default=None,
                        help="Limit recording to this many seconds for testing")
    args = parser.parse_args()

    csv_path = find_pen_csv(csv_path=args.csv, root=args.root,
                             participant=args.participant, trial=args.trial)
    output_path = Path(args.output) if args.output else Path(csv_path).parent / f"{Path(csv_path).parent.name}{DEFAULT_OUTPUT_SUFFIX}"
    print(f"Using CSV: {csv_path}")
    print(f"Saving output to: {output_path}")
    main(csv_path, output_path, fps=args.fps, max_duration=args.duration)