#!/usr/bin/env python3
"""
Convert BORIS event labels from video-frame indices into the same `t_s`
timeline used by the sync_to_pen_time.py outputs.

For each BORIS event row:
  - Read `Image index` (the true video frame number)
  - Look up pen_samples[Image_index] from the JSON to get the honest UTC ms
  - Convert to t_s using the same anchor: t=0 := body[0]'s wall-clock moment

Result: a new CSV (BORIS layout preserved, plus a `t_s_synced` column) where
each event's time matches the t_s column in the synced tracking CSVs.

This handles the variable frame rate correctly because each pen sample carries
its own honest UTC ms — no average fps approximation needed.

Usage:
    python convert_boris_to_synced_time.py path/to/recording.json path/to/boris.tsv
    python convert_boris_to_synced_time.py path/to/recording.json path/to/boris.tsv --output custom.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_pen_samples(data: dict) -> list:
    """Return list of pen sample timestamps in UTC ms (index = pen sample index)."""
    frames = data.get("penTracking", {}).get("frames", []) or []

    clean_timestamps = []
    for f in frames:
        ts = f["timestamp"]
        # If the timestamp is trapped in a list, extract it
        if isinstance(ts, list):
            ts = ts[0]
        clean_timestamps.append(ts)

    return clean_timestamps


def get_body_first_timestamp_ms(data: dict) -> int:
    """Return body.timestamp (ms since pen recording started)."""
    frames = data.get("bodyTracking", {}).get("frames", []) or []
    if not frames:
        raise RuntimeError("No body samples in JSON; can't anchor t=0.")

    # Get the FIRST frame from the list
    first_frame = frames[0]

    # Then extract the timestamp from that frame
    ts = first_frame["timestamp"]

    # If the timestamp is trapped in a list, extract it
    if isinstance(ts, list):
        ts = ts[0]

    return ts

def read_boris(path: Path) -> tuple:
    """Read a BORIS export. Returns (rows, fieldnames, delimiter).

    Supports tab-separated (.tsv) and comma-separated (.csv) automatically.
    Forces tab for .tsv files, since csv.Sniffer can mis-detect on
    files where commas appear inside cells.
    """
    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    elif path.suffix.lower() == ".csv":
        delimiter = ","
    else:
        # Sniff if extension is ambiguous
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames, delimiter


def write_boris_with_sync(path: Path, rows: list, fieldnames: list,
                          delimiter: str = ",") -> None:
    """Write rows with the original BORIS columns plus the synced columns."""
    extra = ["t_s_synced", "synced_quality"]
    out_fields = list(fieldnames) + [c for c in extra if c not in fieldnames]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def convert_frame_to_t_s(frame_idx: int, pen_timestamps: list,
                         body_first_utc_ms: int) -> tuple:
    """Convert a video frame index to t_s using pen[frame_idx].

    Returns (t_s, quality) where quality is:
      - 'ok'              : pen sample at this index exists
      - 'out_of_range'    : frame index >= number of pen samples
      - 'before_body'     : frame is before body started (t_s < 0)
    """
    if frame_idx < 0:
        return (None, "negative_frame")
    if frame_idx >= len(pen_timestamps):
        return (None, "out_of_range")
    utc = pen_timestamps[frame_idx]
    t_s = (utc - body_first_utc_ms) / 1000.0
    quality = "before_body" if t_s < 0 else "ok"
    return (t_s, quality)


def process(json_path: Path, boris_path: Path, output_path: Path = None) -> dict:
    # 1. READ BORIS FIRST (Lightweight validation check)
    rows, fieldnames, delimiter = read_boris(boris_path)

    if "Image index" not in fieldnames:
        raise RuntimeError(
            f"BORIS file does not have an 'Image index' column. "
            f"Found columns: {fieldnames}"
        )

    # 2. THEN LOAD JSON (Heavy lifting, only runs if BORIS is valid)
    data = load_json(json_path)
    pen_timestamps = get_pen_samples(data)
    if len(pen_timestamps) < 2:
        raise RuntimeError("Need at least 2 pen samples in JSON.")
    
    body_first_ts_ms = get_body_first_timestamp_ms(data)
    pen_first_utc_ms = pen_timestamps[0]
    body_first_utc_ms = pen_first_utc_ms + body_first_ts_ms

    # 3. PROCESS THE DATA
    quality_counts = {}
    for r in rows:
        try:
            frame_idx = int(r["Image index"])
        except (TypeError, ValueError):
            r["t_s_synced"] = ""
            r["synced_quality"] = "invalid_frame"
            quality_counts["invalid_frame"] = quality_counts.get("invalid_frame", 0) + 1
            continue
        
        t_s, quality = convert_frame_to_t_s(
            frame_idx, pen_timestamps, body_first_utc_ms
        )
        r["t_s_synced"] = f"{t_s:.4f}" if t_s is not None else ""
        r["synced_quality"] = quality
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    # Default output: <boris stem>_synced.tsv next to the BORIS file
    if output_path is None:
        output_path = boris_path.with_name(f"{boris_path.stem}_synced.tsv")

    write_boris_with_sync(output_path, rows, fieldnames, delimiter)

    return {
        "output_path": output_path,
        "n_rows": len(rows),
        "n_pen_samples": len(pen_timestamps),
        "body_first_utc_ms": body_first_utc_ms,
        "pen_first_utc_ms": pen_first_utc_ms,
        "quality_counts": quality_counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path,
                    help="Path to the master recording JSON (must contain "
                         "penTracking and bodyTracking)")
    ap.add_argument("boris_path", type=Path,
                    help="Path to the BORIS export (.tsv or .csv) with an "
                         "'Image index' column")
    ap.add_argument("--output", type=Path, default=None,
                    help="Where to write the result (default: <boris>_synced.tsv)")
    args = ap.parse_args()

    if not args.json_path.is_file():
        sys.exit(f"ERROR: {args.json_path} not found")
    if not args.boris_path.is_file():
        sys.exit(f"ERROR: {args.boris_path} not found")

    print(f"JSON:  {args.json_path.name}")
    print(f"BORIS: {args.boris_path.name}")
    print()

    result = process(args.json_path, args.boris_path, args.output)

    print(f"Wrote: {result['output_path']}")
    print()
    print(f"Anchor: t=0 = body's first sample UTC = "
          f"{result['body_first_utc_ms']} ms")
    print(f"Pen samples available: {result['n_pen_samples']}")
    print(f"BORIS rows processed:  {result['n_rows']}")
    print()
    print(f"Quality breakdown:")
    for q, c in sorted(result['quality_counts'].items(), key=lambda x: -x[1]):
        print(f"  {q}: {c}")


if __name__ == "__main__":
    main()
