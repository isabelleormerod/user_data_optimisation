#!/usr/bin/env python3
"""
Convert BORIS event labels from video-frame indices into the same `t_s`
timeline used by the sync_to_pen_time.py outputs.

For each BORIS event row:
  - Read `Image index` (the true video frame number)
  - Look up pen_samples[Image_index] from the JSON to get the honest UTC ms
  - Convert to t_s using the same anchor: t=0 := body[0]'s wall-clock moment

Result: a new CSV (BORIS layout preserved, plus a `t_s_synced` column) where
each event's time matches the t_s column in the synced tracking CSVs. Written
to the per-trial folder <json_parent>/<participant>/<stem>/<stem>_boris_synced.csv,
alongside the tracking CSVs (unless --output overrides it).

This handles the variable frame rate correctly because each pen sample carries
its own honest UTC ms — no average fps approximation needed.

Usage:
    python convert_boris_to_synced_time.py path/to/recording.json path/to/boris.tsv
    python convert_boris_to_synced_time.py path/to/recording.json path/to/boris.tsv --output custom.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import PurePosixPath, PureWindowsPath, Path


# Video filename suffixes to strip to recover the JSON stem
_CAM_SUFFIX_RE = re.compile(r'_Cam\d+\.(mp4|avi|mov|mkv|m4v)$', re.IGNORECASE)

# Trailing date/time suffix on observation IDs, e.g. _20260330_115539
_OBSERVATION_TIMESTAMP_RE = re.compile(r'_(\d{8}_\d{6}|\d{14})$')


def video_name_from_path(media_value: str) -> str:
    """Extract the bare video filename from a BORIS media path.

    Handles:
      - 'player #1:A:/path/to/video.mp4'  (the Source column form)
      - 'A:/path/to/video.mp4'            (the Media file name column form)
      - Windows ('A:\\...') or POSIX ('/...') separators
    """
    if media_value is None:
        return ""
    s = media_value.strip()
    # Strip a leading 'player #N:' prefix if present
    if ":" in s and s.lower().startswith("player"):
        s = s.split(":", 1)[1]
    # Now s is a path; take the final component, handling both separators
    # Try Windows first (handles 'A:/...' and 'A:\...'), then POSIX.
    name = PureWindowsPath(s).name
    if not name or name == s:
        name = PurePosixPath(s).name
    return name


def json_stem_from_video(video_filename: str) -> str:
    """Derive the JSON stem from a video filename.

    'P003_..._112241_Cam1.mp4' -> 'P003_..._112241'
    Falls back to stripping just the extension if no _CamN suffix.
    """
    stem = _CAM_SUFFIX_RE.sub("", video_filename)
    if stem == video_filename:
        # No _CamN match; strip a plain extension
        stem = Path(video_filename).stem
    return stem


def get_boris_observation_id(boris_path: Path) -> str:
    """Read the Observation id from the first data row (used as the trial
    folder name). Falls back to the BORIS filename stem if absent."""
    rows, fieldnames, _ = read_boris(boris_path)
    if rows and "Observation id" in fieldnames and rows[0].get("Observation id"):
        return rows[0]["Observation id"].strip()
    return boris_path.stem


def normalize_observation_id(observation_id: str) -> str:
    """Strip a trailing date/time stamp from the observation ID."""
    return _OBSERVATION_TIMESTAMP_RE.sub("", observation_id)


def get_boris_media_stem(boris_path: Path) -> str:
    """Read the first data row's media filename and derive the JSON stem.

    Prefers the 'Media file name' column; falls back to 'Source'.
    """
    rows, fieldnames, _ = read_boris(boris_path)
    if not rows:
        raise RuntimeError(f"{boris_path.name}: no data rows.")
    media = None
    for col in ("Media file name", "Source"):
        if col in fieldnames and rows[0].get(col):
            media = rows[0][col]
            break
    if not media:
        raise RuntimeError(
            f"{boris_path.name}: no 'Media file name' or 'Source' column found."
        )
    video_fn = video_name_from_path(media)
    return json_stem_from_video(video_fn)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_pen_samples(data: dict) -> list:
    """Return list of pen sample timestamps in UTC ms (index = pen sample index)."""
    frames = data.get("penTracking", {}).get("frames", []) or []
    return [f["timestamp"] for f in frames]


def get_body_first_timestamp_ms(data: dict) -> int:
    """Return body[0].timestamp (ms since pen recording started)."""
    frames = data.get("bodyTracking", {}).get("frames", []) or []
    if not frames:
        raise RuntimeError("No body samples in JSON; can't anchor t=0.")
    return frames[0]["timestamp"]


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


def video_stem_folder(json_path: Path) -> tuple:
    """Where the canonical tracking CSVs live (matches sync_to_pen_time.py):
        <root>/<participant>/<video_stem>/
    Returns (folder, video_stem). Does NOT create it.
    """
    name = json_path.name
    if name.endswith(".csv.json"):
        stem = name[:-len(".csv.json")]
    elif name.endswith(".json"):
        stem = name[:-len(".json")]
    else:
        stem = json_path.stem
    participant = stem.split("_", 1)[0]
    folder = json_path.parent / participant / stem
    return folder, stem


def trial_folder_from_observation(json_path: Path, observation_id: str) -> Path:
    """Per-BORIS-trial folder, named by the BORIS Observation id:
        <root>/<participant>/<observation_id>/
    Participant is taken from the observation id's first token. Created here.
    """
    participant = observation_id.split("_", 1)[0]
    folder = json_path.parent / participant / observation_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def process(json_path: Path, boris_path: Path, output_path: Path = None) -> dict:
    import shutil

    data = load_json(json_path)
    pen_timestamps = get_pen_samples(data)
    if len(pen_timestamps) < 2:
        raise RuntimeError("Need at least 2 pen samples in JSON.")
    body_first_ts_ms = get_body_first_timestamp_ms(data)
    pen_first_utc_ms = pen_timestamps[0]
    body_first_utc_ms = pen_first_utc_ms + body_first_ts_ms

    rows, fieldnames, delimiter = read_boris(boris_path)

    if "Image index" not in fieldnames:
        raise RuntimeError(
            f"BORIS file does not have an 'Image index' column. "
            f"Found columns: {fieldnames}"
        )

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

    # Folder named by the BORIS Observation id (one per BORIS trial)
    observation_id = get_boris_observation_id(boris_path)
    trial_folder = trial_folder_from_observation(json_path, observation_id)
    output_id = normalize_observation_id(observation_id)

    # Write the BORIS synced CSV without the trailing date/time stamp.
    if output_path is None:
        output_path = trial_folder / f"{output_id}_boris_synced.csv"
    write_boris_with_sync(output_path, rows, fieldnames, delimiter)

    # Copy the canonical tracking CSVs into this trial folder, renamed with
    # the normalized observation ID. Source: the video-stem folder produced by the sync.
    src_folder, video_stem = video_stem_folder(json_path)
    copied = []
    missing = []
    for suffix in ("pen.csv", "body.csv", "hand.csv", "sync.json"):
        src = src_folder / f"{video_stem}_{suffix}"
        if src.is_file():
            dst = trial_folder / f"{output_id}_{suffix}"
            # If the BORIS observation ID matches the video stem, the
            # source and destination can be identical. In that case, skip
            # the copy instead of crashing.
            if src.resolve() == dst.resolve():
                copied.append(dst.name)
                continue
            shutil.copyfile(src, dst)
            copied.append(dst.name)
        else:
            missing.append(src.name)

    return {
        "output_path": output_path,
        "observation_id": observation_id,
        "trial_folder": trial_folder,
        "copied_tracking": copied,
        "missing_tracking": missing,
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
                    help="Where to write the result (default: <boris>_synced.csv)")
    args = ap.parse_args()

    if not args.json_path.is_file():
        sys.exit(f"ERROR: {args.json_path} not found")
    if not args.boris_path.is_file():
        sys.exit(f"ERROR: {args.boris_path} not found")

    print(f"JSON:  {args.json_path.name}")
    print(f"BORIS: {args.boris_path.name}")
    print()

    result = process(args.json_path, args.boris_path, args.output)

    print(f"Observation id (trial folder): {result['observation_id']}")
    print(f"Trial folder: {result['trial_folder']}")
    print(f"Wrote BORIS synced: {result['output_path'].name}")
    if result['copied_tracking']:
        print(f"Copied tracking CSVs: {', '.join(result['copied_tracking'])}")
    if result['missing_tracking']:
        print(f"WARNING - tracking CSVs not found (run sync first?): "
              f"{', '.join(result['missing_tracking'])}")
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
