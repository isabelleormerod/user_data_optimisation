#!/usr/bin/env python3
"""
Label tracking CSVs (pen / body / hand) with BORIS behaviours.

Takes a trial folder containing:
    <id>_pen.csv, <id>_body.csv, <id>_hand.csv   (tracking, each with a t_s column)
    <id>_boris_synced.csv                         (events with t_s_synced)

and writes labelled copies:
    <id>_pen_labelled.csv, <id>_body_labelled.csv, <id>_hand_labelled.csv

Each labelled CSV gets one extra binary column per behaviour (wide format).
A row's column is 1 when that behaviour is active at the row's t_s.

Behaviour handling:
  - STATE events (Behavior type START / STOP): paired in time order per
    behaviour. Rows with start_t_s <= t_s <= stop_t_s are marked 1.
  - POINT events (Behavior type POINT): the single nearest tracking row is
    marked 1 (within --point-window-s seconds; default 0.1s). If no row is
    within the window, the closest row is still marked and a note is printed.

Behaviours from different categories (e.g. High/Medium/Low vs Insert/Place)
can overlap; each has its own column, so multiple columns can be 1 at once.

Column naming: behaviour names are sanitised to valid-ish column names
(spaces -> underscores). A 'beh_' prefix avoids clashing with tracking columns.

Usage:
    python label_tracking_with_boris.py /path/to/trial_folder
    python label_tracking_with_boris.py /path/to/trial_folder --point-window-s 0.2
    python label_tracking_with_boris.py --batch /path/to/Participant_Landmarks

Per-participant usage (process all trials for one or more participants):
    # Process all trials for a single participant P003
    python label_tracking_with_boris.py A:\Automated_chain_BETA\Participant_Landmarks --participants P003
s
    # Process multiple participants (comma-separated)
    python 03_label_tracking_with_boris.py A:\Automated_chain_BETA\Participant_Landmarks --participants P003,P004
"""

import argparse
import re
import sys
from bisect import bisect_left
from pathlib import Path

from utils.io import parse_float, read_table, write_table
from utils.discovery import iter_trial_folders
from utils.params import parse_participant_filter


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def sanitise(name: str) -> str:
    """Turn a behaviour name into a safe column suffix."""
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "", s)
    return s or "unnamed"




# ----------------------------------------------------------------------------
# Build intervals and points from BORIS events
# ----------------------------------------------------------------------------

def build_label_spec(boris_rows: list):
    """Return (intervals, points, warnings).

    intervals: list of (behaviour, start_t_s, stop_t_s)
    points:    list of (behaviour, t_s)
    """
    # Group events by behaviour, keep only those with a valid t_s_synced
    state_events = {}   # behaviour -> list of (t_s, type)
    points = []
    warnings = []

    for r in boris_rows:
        beh = (r.get("Behavior") or "").strip()
        btype = (r.get("Behavior type") or "").strip().upper()
        t_s = parse_float(r.get("t_s_synced"))
        if not beh or t_s is None:
            continue
        if btype == "POINT":
            points.append((beh, t_s))
        elif btype in ("START", "STATE_START"):
            state_events.setdefault(beh, []).append((t_s, "START"))
        elif btype in ("STOP", "STATE_STOP"):
            state_events.setdefault(beh, []).append((t_s, "STOP"))
        else:
            # Unknown type; treat as a point so it's not lost
            points.append((beh, t_s))
            warnings.append(f"Unknown Behavior type '{btype}' for '{beh}'; "
                            f"treated as a point.")

    intervals = []
    for beh, evs in state_events.items():
        evs.sort(key=lambda e: e[0])
        # Pair START with the next STOP, in time order
        open_start = None
        for t_s, typ in evs:
            if typ == "START":
                if open_start is not None:
                    warnings.append(f"'{beh}': START at {open_start:.3f}s "
                                    f"without a STOP before next START at {t_s:.3f}s.")
                open_start = t_s
            else:  # STOP
                if open_start is None:
                    warnings.append(f"'{beh}': STOP at {t_s:.3f}s without a "
                                    f"matching START; ignored.")
                else:
                    intervals.append((beh, open_start, t_s))
                    open_start = None
        if open_start is not None:
            warnings.append(f"'{beh}': START at {open_start:.3f}s never STOPped; "
                            f"interval left open (not labelled to end).")

    return intervals, points, warnings


# ----------------------------------------------------------------------------
# Apply labels to a tracking CSV
# ----------------------------------------------------------------------------

def label_tracking(rows, fields, intervals, points, point_window_s):
    """Add one binary column per behaviour. Returns (rows, new_fields, behaviours)."""
    # Collect the full behaviour set (so every file gets identical columns)
    behaviours = sorted({b for b, _, _ in intervals} | {b for b, _ in points})
    col_for = {b: f"{sanitise(b)}" for b in behaviours}

    # Pre-extract row t_s values
    t_list = [parse_float(r.get("t_s")) for r in rows]

    # Initialise all behaviour columns to 0
    for r in rows:
        for b in behaviours:
            r[col_for[b]] = 0

    # State intervals: mark rows with start <= t_s <= stop
    for beh, start, stop in intervals:
        lo, hi = (start, stop) if start <= stop else (stop, start)
        col = col_for[beh]
        for r, t in zip(rows, t_list):
            if t is not None and lo <= t <= hi:
                r[col] = 1

    # Point events: mark the nearest row (within window if possible)
    valid_idx = [i for i, t in enumerate(t_list) if t is not None]
    valid_times = [t_list[i] for i in valid_idx]
    for beh, pt in points:
        if not valid_times:
            break
        j = bisect_left(valid_times, pt)
        # candidate nearest indices
        cands = []
        if j < len(valid_times):
            cands.append(j)
        if j > 0:
            cands.append(j - 1)
        best = min(cands, key=lambda k: abs(valid_times[k] - pt))
        # Mark it (even if outside the window — closest available)
        rows[valid_idx[best]][col_for[beh]] = 1

    new_fields = list(fields) + [col_for[b] for b in behaviours]
    return rows, new_fields, behaviours


# ----------------------------------------------------------------------------
# Process one trial folder
# ----------------------------------------------------------------------------

def find_files(trial_folder: Path):
    """Locate the boris_synced CSV and the three tracking CSVs by suffix."""
    boris = list(trial_folder.glob("*_boris_synced.csv"))
    pen = list(trial_folder.glob("*_pen_flattened.csv"))
    body = list(trial_folder.glob("*_body.csv"))
    hand = list(trial_folder.glob("*_hand.csv"))
    # Exclude already-labelled files
    def clean(lst):
        return [p for p in lst if not p.stem.endswith("_labelled")]
    return {
        "boris": clean(boris),
        "pen": clean(pen),
        "body": clean(body),
        "hand": clean(hand),
    }


def process_folder(trial_folder: Path, point_window_s: float) -> dict:
    found = find_files(trial_folder)
    result = {"folder": trial_folder.name, "status": "pending",
              "labelled": [], "warnings": [], "behaviours": []}

    if not found["boris"]:
        result["status"] = "skipped_no_boris"
        return result
    if len(found["boris"]) > 1:
        result["warnings"].append(
            f"Multiple boris_synced files found; using {found['boris'][0].name}")
    boris_path = found["boris"][0]

    boris_rows, _ = read_table(boris_path)
    intervals, points, warns = build_label_spec(boris_rows)
    result["warnings"].extend(warns)
    behaviours = sorted({b for b, _, _ in intervals} | {b for b, _ in points})
    result["behaviours"] = behaviours

    if not behaviours:
        result["status"] = "no_behaviours"
        return result

    any_done = False
    for stream in ("pen", "body", "hand"):
        if not found[stream]:
            result["warnings"].append(f"No {stream} CSV found.")
            continue
        track_path = found[stream][0]
        rows, fields = read_table(track_path)
        rows, new_fields, _ = label_tracking(
            rows, fields, intervals, points, point_window_s)
        out_path = track_path.with_name(f"{track_path.stem}_labelled.csv")
        write_table(out_path, rows, new_fields)
        result["labelled"].append(out_path.name)
        any_done = True

    result["status"] = "ok" if any_done else "no_tracking_files"
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="A single trial folder, or (with --batch) a root to scan")
    ap.add_argument("--batch", action="store_true",
                    help="Treat 'path' as a root and process all trial folders "
                         "(<root>/<participant>/<trial>/) that have a "
                         "boris_synced CSV.")
    ap.add_argument(
        "--participants", type=str, default=None,
        help="Comma-separated participant IDs to restrict a batch run to "
             "(e.g. 'P003,P004'). Implies batch mode over those participants.")
    ap.add_argument("--point-window-s", type=float, default=0.1,
                    help="Window for snapping POINT events to the nearest "
                         "tracking row (seconds; default 0.1)")
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"ERROR: {args.path} not found")

    participant_filter = parse_participant_filter(args.participants)

    batch_mode = args.batch or (participant_filter is not None)

    if batch_mode:
        folders = list(iter_trial_folders(args.path, participant_filter))
        if not folders:
            which = (f"participants {', '.join(sorted(participant_filter))}"
                     if participant_filter else "any participant")
            sys.exit(f"No trial folders with a boris_synced CSV under {args.path} for {which}")
        print(f"Found {len(folders)} trial folder(s) to label\n")
        if participant_filter:
            print(f"  (restricted to: {', '.join(sorted(participant_filter))})")
        print()
        results = []
        for i, fol in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}] {fol.name}")
            r = process_folder(fol, args.point_window_s)
            results.append(r)
            print(f"   status: {r['status']}")
            if r["behaviours"]:
                print(f"   behaviours: {', '.join(r['behaviours'])}")
            for w in r["warnings"]:
                print(f"   warning: {w}")
        print()
        from collections import Counter
        print("Summary:", dict(Counter(r["status"] for r in results)))
    else:
        r = process_folder(args.path, args.point_window_s)
        print(f"Folder: {r['folder']}")
        print(f"Status: {r['status']}")
        if r["behaviours"]:
            print(f"Behaviours ({len(r['behaviours'])}): {', '.join(r['behaviours'])}")
        if r["labelled"]:
            print("Labelled files:")
            for f in r["labelled"]:
                print(f"  {f}")
        for w in r["warnings"]:
            print(f"warning: {w}")


if __name__ == "__main__":
    main()
