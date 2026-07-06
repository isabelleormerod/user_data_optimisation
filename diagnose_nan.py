#!/usr/bin/env python3
"""
Diagnose NaN / missing data across the three tracking streams (body, hand, pen)
for every Place event in every trial.

Reports:
  - Which trials are missing a body / hand / pen file entirely
  - For each Place event: % valid frames per stream and per signal group
  - Signals that are ALL-NaN for specific trials (stream present but signal
    always untrackable, e.g. occluded arm)
  - A summary CSV for further inspection

Usage:
    python diagnose_nan.py --landmarks-root "A:\\...\\Participant_Landmarks"
    python diagnose_nan.py --landmarks-root ... --participants P003,P004
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

CONF_MIN = 0.3
BODY_UP  = np.array([-1.0, 0.0, 0.0])

BODY_SIGNALS = ["trunk_flex", "neck_flex", "r_ua_flex", "r_elbow_flex",
                "r_ua_abduct", "wrist_neutral_dev", "reach_ratio", "wrist_elev"]
HAND_SIGNALS = ["wrist_flex", "wrist_ulnar_dev", "aperture",
                "idx_flex", "mid_flex", "ring_flex", "pinky_flex",
                "wr_x", "wr_y", "wr_z"]
PEN_SIGNALS  = ["pen_x", "pen_y", "pen_z",
                "pen_qw", "pen_qx", "pen_qy", "pen_qz"]


# --------------------------------------------------------------------------- #
# IO
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
        t = parse_float(r.get("t_s"))
        flag = str(r.get("Place","")).strip() in ("1","1.0")
        if t is None: continue
        if flag and not in_run:  in_run = True; t0 = t
        elif not flag and in_run: in_run = False; runs.append((t0, prev_t))
        prev_t = t
    if in_run: runs.append((t0, prev_t))
    return runs

def height_at(t_mid, rows):
    best_t = None; best_h = "Unknown"
    for r in rows:
        t = parse_float(r.get("t_s"))
        if t is None: continue
        if best_t is None or abs(t - t_mid) < abs(best_t - t_mid):
            for h in ("High","Medium","Low"):
                if str(r.get(h,"")).strip() in ("1","1.0"):
                    best_t = t; best_h = h; break
    return best_h


# --------------------------------------------------------------------------- #
# Per-stream validity check
# --------------------------------------------------------------------------- #
def check_body_event(body_rows, t0, t1, body_fields):
    """Returns dict of signal_name -> fraction of frames that are non-NaN."""
    conf_fields = any(f.endswith("_conf") for f in body_fields)
    window = [(parse_float(r.get("t_s")), r) for r in body_rows
              if t0 <= (parse_float(r.get("t_s")) or -1) <= t1]
    if not window:
        return {s: 0.0 for s in BODY_SIGNALS}, 0

    n = len(window)
    # Check each signal's key joints
    joint_groups = {
        "trunk_flex":        ["LeftShoulder","RightShoulder","LeftHip","RightHip"],
        "neck_flex":         ["LeftEar","RightEar","LeftShoulder","RightShoulder"],
        "r_ua_flex":         ["RightShoulder","RightElbow"],
        "r_elbow_flex":      ["RightShoulder","RightElbow","RightWrist"],
        "r_ua_abduct":       ["RightShoulder","RightElbow","LeftShoulder"],
        "wrist_neutral_dev": ["RightElbow","RightWrist","RightIndex"],
        "reach_ratio":       ["RightShoulder","RightElbow","RightWrist"],
        "wrist_elev":        ["RightShoulder","RightWrist"],
    }
    fractions = {}
    for sig, joints in joint_groups.items():
        valid = 0
        for _, r in window:
            ok = True
            for j in joints:
                x = parse_float(r.get(f"{j}_x"))
                if x is None or (np.isnan(x) if isinstance(x, float) else False):
                    ok = False; break
                if conf_fields:
                    c = parse_float(r.get(f"{j}_conf", "1"))
                    if c is not None and c < CONF_MIN:
                        ok = False; break
            if ok: valid += 1
        fractions[sig] = valid / n if n > 0 else 0.0
    return fractions, n


def check_hand_event(hand_rows, t0, t1, sides):
    """Check validity for all requested sides. Returns flat dict
    prefixed with side name, e.g. left_wrist_flex, right_aperture."""
    window = [(parse_float(r.get("t_s")), r) for r in hand_rows
              if t0 <= (parse_float(r.get("t_s")) or -1) <= t1]

    if not window:
        return {f"{side.lower()}_{s}": 0.0
                for side in sides for s in HAND_SIGNALS}, 0

    n = len(window)
    result = {}

    for sl in sides:
        key = {
            "wrist_flex":     [f"{sl}_HandStart",f"{sl}_HandWristRoot",f"{sl}_HandMiddle0"],
            "wrist_ulnar_dev":[f"{sl}_HandWristRoot",f"{sl}_HandMiddle0",
                               f"{sl}_HandPinky0",f"{sl}_HandStart"],
            "aperture":       [f"{sl}_HandThumbTip",f"{sl}_HandIndexTip"],
            "idx_flex":       [f"{sl}_HandIndex1",f"{sl}_HandIndex2",
                               f"{sl}_HandIndex3",f"{sl}_HandIndexTip"],
            "mid_flex":       [f"{sl}_HandMiddle1",f"{sl}_HandMiddle2",
                               f"{sl}_HandMiddle3",f"{sl}_HandMiddleTip"],
            "ring_flex":      [f"{sl}_HandRing1",f"{sl}_HandRing2",
                               f"{sl}_HandRing3",f"{sl}_HandRingTip"],
            "pinky_flex":     [f"{sl}_HandPinky1",f"{sl}_HandPinky2",
                               f"{sl}_HandPinky3",f"{sl}_HandPinkyTip"],
            "wr_x":           [f"{sl}_HandWristRoot"],
            "wr_y":           [f"{sl}_HandWristRoot"],
            "wr_z":           [f"{sl}_HandWristRoot"],
        }
        for sig, cols in key.items():
            valid = sum(
                1 for _, r in window
                if all(parse_float(r.get(f"{c}_x"
                                         if not c.endswith(("_x","_y","_z"))
                                         else c)) is not None
                       for c in cols))
            result[f"{sl.lower()}_{sig}"] = valid / n

    return result, n


def detect_hand_side(hand_fields, hand_rows):
    """Return list of sides that have actual data in the file."""
    sides_present = []
    if any(f.startswith("Left_")  for f in hand_fields): sides_present.append("Left")
    if any(f.startswith("Right_") for f in hand_fields): sides_present.append("Right")

    sides_with_data = []
    for side in sides_present:
        col = f"{side}_HandWristRoot_x"
        if col in hand_fields:
            valid = sum(1 for r in hand_rows[:100]
                        if parse_float(r.get(col)) is not None)
            if valid > 5:
                sides_with_data.append(side)

    return sides_with_data if sides_with_data else sides_present
    window = [(parse_float(r.get("t_s")), r) for r in pen_rows
              if t0 <= (parse_float(r.get("t_s")) or -1) <= t1]
    if not window:
        return {s: 0.0 for s in PEN_SIGNALS}, 0
    n = len(window)
    col_map = {"pen_x":"x","pen_y":"y","pen_z":"z",
               "pen_qw":"qw","pen_qx":"qx","pen_qy":"qy","pen_qz":"qz"}
    fractions = {}
    for sig, col in col_map.items():
        valid = sum(1 for _, r in window if parse_float(r.get(col)) is not None)
        fractions[sig] = valid / n
    return fractions, n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def check_pen_event(pen_rows, t0, t1):
    """Returns dict of signal_name -> fraction valid."""
    window = [(parse_float(r.get("t_s")), r) for r in pen_rows
              if t0 <= (parse_float(r.get("t_s")) or -1) <= t1]
    if not window:
        return {s: 0.0 for s in PEN_SIGNALS}, 0
    n = len(window)
    col_map = {"pen_x": "x", "pen_y": "y", "pen_z": "z",
               "pen_qw": "qw", "pen_qx": "qx", "pen_qy": "qy", "pen_qz": "qz"}
    fractions = {}
    for sig, col in col_map.items():
        valid = sum(1 for _, r in window if parse_float(r.get(col)) is not None)
        fractions[sig] = valid / n
    return fractions, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--participants", default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Fraction below which a signal is flagged (default 0.5)")
    args = ap.parse_args()

    root    = args.landmarks_root
    pfilter = ({p.strip() for p in args.participants.split(",") if p.strip()}
               if args.participants else None)

    out_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []    # one row per Place event
    trial_summary = []  # one row per trial

    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pid == "metrics": continue
        if pfilter and pid not in pfilter: continue

        for tdir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            stem = tdir.name

            # Find labelled pen
            pen_path = None
            for name in (f"{stem}_pen_flattened_labelled.csv",
                         f"{stem}_pen_labelled.csv"):
                if (tdir / name).is_file():
                    pen_path = tdir / name; break
            if pen_path is None:
                trial_summary.append({
                    "participant": pid, "trial": stem,
                    "has_pen": False, "has_body": False, "has_hand": False,
                    "n_place_events": 0, "issue": "no labelled pen file"})
                continue

            body_path = tdir / f"{stem}_body.csv"
            hand_path = tdir / f"{stem}_hand.csv"
            has_body  = body_path.is_file()
            has_hand  = hand_path.is_file()

            pen_rows,  pen_fields  = read_csv_t(pen_path)
            body_rows, body_fields = read_csv_t(body_path) if has_body else ([], [])
            hand_rows, hand_fields = (read_csv_t(hand_path) if has_hand else ([], []))

            # Detect hand side — check both Left and Right
            sides_with_data = detect_hand_side(hand_fields, hand_rows)
            hand_side_str = "/".join(sides_with_data) if sides_with_data else "none"

            runs = place_runs(pen_rows, pen_fields)

            trial_summary.append({
                "participant": pid, "trial": stem,
                "has_pen": True, "has_body": has_body, "has_hand": has_hand,
                "hand_sides_tracked": hand_side_str,
                "n_place_events": len(runs),
                "issue": ("no body" if not has_body else "") +
                         (" no hand" if not has_hand else "") +
                         (" no hand data" if has_hand and not sides_with_data else "")})

            for ev_idx, (t0, t1) in enumerate(runs):
                h = height_at((t0+t1)/2, pen_rows)
                dur = round(t1-t0, 3) if t1 is not None else None

                body_frac, n_body = (check_body_event(body_rows, t0, t1, body_fields)
                                     if has_body
                                     else ({s: np.nan for s in BODY_SIGNALS}, 0))

                if has_hand and sides_with_data:
                    hand_frac, n_hand = check_hand_event(
                        hand_rows, t0, t1, sides_with_data)
                else:
                    hand_frac = {f"{sl.lower()}_{s}": np.nan
                                 for sl in ["Left","Right"]
                                 for s in HAND_SIGNALS}
                    n_hand = 0

                pen_frac, n_pen = check_pen_event(pen_rows, t0, t1)

                # Flag any signal below threshold
                flags = []
                thr = args.threshold
                for s, v in body_frac.items():
                    if not np.isnan(v) and v < thr: flags.append(f"body:{s}={v:.0%}")
                for s, v in hand_frac.items():
                    if not np.isnan(v) and v < thr: flags.append(f"hand:{s}={v:.0%}")
                for s, v in pen_frac.items():
                    if not np.isnan(v) and v < thr: flags.append(f"pen:{s}={v:.0%}")

                rec = {
                    "participant": pid, "trial": stem,
                    "event_idx": ev_idx+1, "height": h,
                    "t_start": round(t0, 3),
                    "t_end": round(t1, 3) if t1 else None,
                    "duration_s": dur,
                    "hand_sides_tracked": hand_side_str,
                    "n_body_frames": n_body,
                    "n_hand_frames": n_hand,
                    "n_pen_frames":  n_pen,
                    "has_body": has_body, "has_hand": has_hand,
                    "flagged": len(flags) > 0,
                    "flags": " | ".join(flags) if flags else "",
                }
                for s, v in body_frac.items():
                    rec[f"body_{s}"] = round(v, 3) if not np.isnan(v) else np.nan
                for s, v in hand_frac.items():
                    rec[f"hand_{s}"] = round(v, 3) if not np.isnan(v) else np.nan
                for s, v in pen_frac.items():
                    rec[f"pen_{s}"]  = round(v, 3) if not np.isnan(v) else np.nan
                records.append(rec)

    df = pd.DataFrame(records)
    ts = pd.DataFrame(trial_summary)

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    df.to_csv(out_dir / "nan_diagnostic_events.csv", index=False)
    ts.to_csv(out_dir / "nan_diagnostic_trials.csv", index=False)

    # ------------------------------------------------------------------ #
    # Console report
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print(f"  NaN DIAGNOSTIC REPORT")
    print(f"{'='*65}")
    print(f"  Trials scanned : {len(ts)}")
    print(f"  Place events   : {len(df)}")
    print(f"  Flag threshold : {args.threshold:.0%} valid frames\n")

    if ts.empty:
        print("  No trials found. Check --landmarks-root points to the correct")
        print("  directory and that trial folders contain labelled pen files.")
        return

    # Hand side summary
    if "hand_sides_tracked" in ts.columns:
        print("  HAND SIDES TRACKED PER TRIAL:")
        side_counts = ts[ts["has_hand"]]["hand_sides_tracked"].value_counts()
        for side, n in side_counts.items():
            print(f"    {side:<20} {n} trial(s)")
        print(f"  (Both Left and Right are used in analysis when present)\n")

    # Missing files
    missing_body = ts[~ts["has_body"] & ts["has_pen"]] if "has_body" in ts.columns else pd.DataFrame()
    missing_hand = ts[~ts["has_hand"] & ts["has_pen"]] if "has_hand" in ts.columns else pd.DataFrame()
    if len(missing_body):
        print(f"  MISSING BODY CSV ({len(missing_body)} trials):")
        for _, r in missing_body.iterrows():
            print(f"    {r['participant']} / {r['trial']}")
    if len(missing_hand):
        print(f"\n  MISSING HAND CSV ({len(missing_hand)} trials):")
        for _, r in missing_hand.iterrows():
            print(f"    {r['participant']} / {r['trial']}")

    # Flagged events
    flagged = df[df["flagged"]]
    print(f"\n  FLAGGED EVENTS below {args.threshold:.0%} valid: "
          f"{len(flagged)} / {len(df)} ({100*len(flagged)/max(len(df),1):.1f}%)")

    if len(flagged):
        # Group by trial to find the worst offenders
        by_trial = (flagged.groupby(["participant","trial"])
                    .agg(n_flagged=("flagged","sum"),
                         example_flags=("flags", "first"))
                    .reset_index()
                    .sort_values("n_flagged", ascending=False))
        print(f"\n  Worst trials (most flagged events):")
        for _, r in by_trial.head(15).iterrows():
            print(f"    {r['participant']:6} / {r['trial']:<45} "
                  f"{int(r['n_flagged'])} events  |  {r['example_flags'][:80]}")

    # Per-signal NaN summary across all events
    print(f"\n  SIGNAL VALIDITY (mean fraction of frames that are non-NaN):")
    print(f"  {'Signal':<30} {'Mean valid':>12}  {'% events <50%':>14}")
    all_sig_cols = [c for c in df.columns
                    if c.startswith(("body_","hand_","pen_"))
                    and c not in ("body_frames","hand_frames","pen_frames")]
    for col in all_sig_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not len(vals): continue
        mean_v = vals.mean()
        pct_bad = (vals < 0.5).mean() * 100
        bar = "█" * int(mean_v * 20)
        flag = "  ← CHECK" if mean_v < 0.5 else ""
        print(f"  {col:<30} {bar:<20} {mean_v:5.1%}  {pct_bad:6.1f}%{flag}")

    print(f"\n  Wrote: {out_dir / 'nan_diagnostic_events.csv'}")
    print(f"  Wrote: {out_dir / 'nan_diagnostic_trials.csv'}")
    print(f"\n  Use nan_diagnostic_events.csv to filter events in 12_fpca_prototype_clustering.py")
    print(f"  Use nan_diagnostic_trials.csv to identify trials to re-process")


if __name__ == "__main__":
    main()