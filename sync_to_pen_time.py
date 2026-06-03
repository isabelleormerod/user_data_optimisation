#!/usr/bin/env python3
"""
Sync hand / body / pen tracking JSONs onto pen's UTC timeline.

Anchor model
------------
All three streams come from the same Unity Update() loop, so per-sample
timestamps are honest real-time (with their own units):

  - Pen:  UTC epoch ms in `timestamp`
  - Body: ms-since-pen-start in `timestamp` (Unity's real-time clock)
  - Hand: frame index in `frame` (matches pen sample index on the same Update tick)

Strategy
--------
1. t = 0 := body's first sample's wall-clock moment
   In UTC: body_first_utc_ms = pen[0].timestamp + body[0].timestamp
2. t_end := body's last sample's wall-clock moment
3. Pen samples before t=0 are dropped (pen warms up earlier than body).
4. Pen samples after t_end are dropped (pen keeps logging after body stops).
5. Hand: each hand sample carries a `frame` index that matches the pen sample
   it was recorded alongside. So hand's wall-clock UTC = pen[hand.frame].timestamp.
   Hand samples before/after the body window are dropped.
6. Output: three CSVs sharing the same time column (t_s = seconds since body[0]).
   Pen rows are at pen's native UTC timestamps within the window.
   Body and hand are linearly interpolated at each pen UTC.

Outputs into a per-trial folder <json_parent>/<participant>/<stem>/ :
  <stem>_pen.csv    -- one row per kept pen sample
  <stem>_body.csv   -- one row per kept pen sample (body landmarks interpolated)
  <stem>_hand.csv   -- one row per kept pen sample (hand joints interpolated)
  <stem>_sync.json  -- sync parameters and diagnostics

where <stem> is the JSON filename with '.csv.json' (or '.json') stripped, and
<participant> is the first token of the stem (e.g. 'P003'). The folder is
created if needed.

Usage:
    python sync_to_pen_time.py path/to/recording.json
"""

import argparse
import csv
import json
import re
import sys
from bisect import bisect_left
from pathlib import Path


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Pen
# ----------------------------------------------------------------------------

def extract_pen_samples(data: dict) -> list:
    """Each pen sample has:
       - utc_ms: original UTC epoch ms
       - position xyz, rotation xyzw
    """
    frames = data.get("penTracking", {}).get("frames", []) or []
    samples = []
    for f in frames:
        pos = f.get("position", {}) or {}
        rot = f.get("rotation", {}) or {}
        samples.append({
            "utc_ms": f["timestamp"],
            "x": pos.get("x"), "y": pos.get("y"), "z": pos.get("z"),
            "qx": rot.get("x"), "qy": rot.get("y"),
            "qz": rot.get("z"), "qw": rot.get("w"),
        })
    return samples


# ----------------------------------------------------------------------------
# Body
# ----------------------------------------------------------------------------

def extract_body_samples(data: dict, pen_first_utc_ms: int) -> list:
    """Body's `timestamp` is ms since pen recording started.
    Convert to UTC ms by adding pen_first_utc_ms. Flatten landmarks.
    """
    frames = data.get("bodyTracking", {}).get("frames", []) or []
    samples = []
    for f in frames:
        sample = {"utc_ms": pen_first_utc_ms + f["timestamp"]}
        poses = f.get("poses") or []
        if poses:
            for lm in poses[0].get("landmarks", []) or []:
                name = lm.get("name", "unknown")
                sample[f"{name}_x"] = lm.get("x")
                sample[f"{name}_y"] = lm.get("y")
                sample[f"{name}_z"] = lm.get("z")
                sample[f"{name}_conf"] = lm.get("conf")
        samples.append(sample)
    return samples


# ----------------------------------------------------------------------------
# Hand
# ----------------------------------------------------------------------------

def extract_hand_samples(data: dict, pen_samples: list) -> list:
    """Hand's `frame` index matches the pen sample index from the same Update
    tick. So hand sample wall-clock = pen[frame_index].utc_ms.

    Flatten all joints. Mark samples whose frame index falls outside pen's
    valid range (rare but possible if pen logging stopped earlier than hand).
    """
    frames = data.get("handTracking", {}).get("frames", []) or []
    samples = []
    n_pen = len(pen_samples)
    for f in frames:
        frame_idx = f.get("frame")
        if frame_idx is None or frame_idx < 0 or frame_idx >= n_pen:
            continue  # cannot anchor this hand sample to wall-clock
        sample = {"utc_ms": pen_samples[frame_idx]["utc_ms"],
                  "frame_idx": frame_idx}
        for hand in f.get("hands", []) or []:
            ht_type = hand.get("handType", "Unknown")
            for joint in hand.get("joints", []) or []:
                jid = joint.get("jointId", "unknown")
                prefix = f"{ht_type}_{jid}"
                pos = joint.get("position", {}) or {}
                rot = joint.get("rotation", {}) or {}
                sample[f"{prefix}_x"] = pos.get("x")
                sample[f"{prefix}_y"] = pos.get("y")
                sample[f"{prefix}_z"] = pos.get("z")
                sample[f"{prefix}_qx"] = rot.get("x")
                sample[f"{prefix}_qy"] = rot.get("y")
                sample[f"{prefix}_qz"] = rot.get("z")
                sample[f"{prefix}_qw"] = rot.get("w")
        samples.append(sample)
    return samples


# ----------------------------------------------------------------------------
# Window + interpolation
# ----------------------------------------------------------------------------

def trim_to_window(samples: list, start_utc_ms: int, end_utc_ms: int,
                   time_key: str = "utc_ms") -> list:
    """Return samples whose utc_ms is in [start, end] inclusive."""
    return [s for s in samples if start_utc_ms <= s[time_key] <= end_utc_ms]


def interpolate_at(samples: list, target_utc_ms: int,
                   time_key: str = "utc_ms",
                   dropout_threshold_ms: float = 100.0,
                   data_fields: list = None) -> dict:
    """Linearly interpolate sample fields at target_utc_ms.

    Assumes samples are sorted by time_key (ascending).
    `data_fields`: explicit list of fields to interpolate. If None, falls back
    to the union of keys across ALL samples (not just the first) so that fields
    present in only some samples (e.g. a second hand that isn't always visible)
    are not silently dropped.
    `data_quality` values:
      - 'real':           target_utc_ms matches a sample's time exactly
      - 'interpolated':   linearly interpolated; bracketing gap < dropout_threshold_ms
      - 'filled_dropout': linearly interpolated; bracketing gap >= dropout_threshold_ms
                          (values exist but are unreliable — gap too big to trust)
      - 'extrapolated':   target outside the sample range
    """
    if not samples:
        return None
    times = [s[time_key] for s in samples]
    if data_fields is None:
        # Union of all keys across all samples (robust to fields that appear
        # only in some samples).
        field_set = set()
        for s in samples:
            field_set.update(s.keys())
        field_set.discard(time_key)
        field_set.discard("data_quality")
        field_set.discard("frame_idx")
        data_fields = sorted(field_set)

    if target_utc_ms <= times[0]:
        out = {k: samples[0].get(k) for k in data_fields}
        out["data_quality"] = "real" if target_utc_ms == times[0] else "extrapolated"
        return out
    if target_utc_ms >= times[-1]:
        out = {k: samples[-1].get(k) for k in data_fields}
        out["data_quality"] = "real" if target_utc_ms == times[-1] else "extrapolated"
        return out

    i = bisect_left(times, target_utc_ms)
    if times[i] == target_utc_ms:
        out = {k: samples[i].get(k) for k in data_fields}
        out["data_quality"] = "real"
        return out

    t0, t1 = times[i - 1], times[i]
    gap_ms = t1 - t0
    frac = (target_utc_ms - t0) / gap_ms
    out = {}
    for k in data_fields:
        v0 = samples[i - 1].get(k)
        v1 = samples[i].get(k)
        if v0 is None and v1 is None:
            out[k] = None
        elif v0 is None:
            out[k] = v1
        elif v1 is None:
            out[k] = v0
        elif isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            out[k] = v0 + frac * (v1 - v0)
        else:
            out[k] = v0
    out["data_quality"] = ("filled_dropout" if gap_ms >= dropout_threshold_ms
                           else "interpolated")
    return out


# ----------------------------------------------------------------------------
# CSV writing
# ----------------------------------------------------------------------------

def write_csv(path: Path, rows: list) -> None:
    """Write rows; 't_s' first, 'data_quality' last, other fields sorted between."""
    if not rows:
        path.write_text("t_s,data_quality\n")
        return

    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys.discard("t_s")
    keys.discard("data_quality")
    fieldnames = ["t_s"] + sorted(keys) + ["data_quality"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

_TIMESTAMP_SUFFIX_RE = re.compile(r'_\d{8}_\d{6}$')


def strip_timestamp_suffix(stem: str) -> str:
    """Strip a trailing YYYYMMDD_HHMMSS timestamp from a stem."""
    return _TIMESTAMP_SUFFIX_RE.sub("", stem)


def resolve_output(json_path: Path) -> tuple:
    """Determine the per-trial output folder and file stem.

    Layout:
        <root>/<stem>.csv.json   (master JSON; may also be plain <stem>.json)
        ->  <root>/<participant>/<stem_without_timestamp>/   (per-trial output folder)

    Where:
        stem        = JSON filename with '.csv.json' or '.json' stripped
        participant = first '_'-separated token of stem (e.g. 'P003')

    The folder is created if it doesn't exist. Returns (folder, stem).
    """
    name = json_path.name
    if name.endswith(".csv.json"):
        stem = name[:-len(".csv.json")]
    elif name.endswith(".json"):
        stem = name[:-len(".json")]
    else:
        stem = json_path.stem

    participant = stem.split("_", 1)[0]
    folder_stem = strip_timestamp_suffix(stem)
    folder = json_path.parent / participant / folder_stem
    folder.mkdir(parents=True, exist_ok=True)
    return folder, stem


def process(json_path: Path,
            hand_dropout_threshold_ms: float = 100.0,
            body_dropout_threshold_ms: float = 200.0) -> dict:
    data = load_json(json_path)
    folder, stem = resolve_output(json_path)

    pen_samples = extract_pen_samples(data)
    if len(pen_samples) < 2:
        raise RuntimeError("Need at least 2 pen samples; cannot anchor wall-clock.")

    pen_first_utc_ms = pen_samples[0]["utc_ms"]
    pen_last_utc_ms = pen_samples[-1]["utc_ms"]

    body_samples = extract_body_samples(data, pen_first_utc_ms)
    if len(body_samples) < 2:
        raise RuntimeError("Need at least 2 body samples; body defines the sync window.")

    hand_samples = extract_hand_samples(data, pen_samples)

    # t = 0 is body's first sample; t_end is body's last sample
    window_start_utc = body_samples[0]["utc_ms"]
    window_end_utc = body_samples[-1]["utc_ms"]

    # Trim pen to the body window. Pen's UTCs *within* the window define the
    # output timeline (one row per kept pen sample).
    pen_in_window = trim_to_window(pen_samples, window_start_utc, window_end_utc)
    if not pen_in_window:
        raise RuntimeError("No pen samples fall within the body window.")

    # Build output rows
    pen_rows = []
    body_rows = []
    hand_rows = []

    # Precompute the union of data fields across ALL samples for body and hand,
    # so fields present in only some samples (e.g. a hand not always visible)
    # are never dropped. Computing once here is much faster than per-row.
    def union_fields(samples):
        fs = set()
        for s in samples:
            fs.update(s.keys())
        fs.discard("utc_ms")
        fs.discard("data_quality")
        fs.discard("frame_idx")
        return sorted(fs)

    body_fields = union_fields(body_samples)
    hand_fields = union_fields(hand_samples) if hand_samples else []

    for p in pen_in_window:
        utc = p["utc_ms"]
        t_s = (utc - window_start_utc) / 1000.0

        # Pen row: always 'real' (pen sample IS at this UTC)
        pen_rows.append({
            "t_s": t_s,
            "x": p.get("x"), "y": p.get("y"), "z": p.get("z"),
            "qx": p.get("qx"), "qy": p.get("qy"),
            "qz": p.get("qz"), "qw": p.get("qw"),
            "data_quality": "real",
        })

        # Body row: interpolated to this UTC
        b = interpolate_at(body_samples, utc,
                           dropout_threshold_ms=body_dropout_threshold_ms,
                           data_fields=body_fields)
        if b is None:
            b = {"data_quality": "missing"}
        b["t_s"] = t_s
        body_rows.append(b)

        # Hand row: interpolated to this UTC (or empty if no hand data)
        if hand_samples:
            h = interpolate_at(hand_samples, utc,
                               dropout_threshold_ms=hand_dropout_threshold_ms,
                               data_fields=hand_fields)
            if h is None:
                h = {"data_quality": "missing"}
        else:
            h = {"data_quality": "no_hand_data"}
        h["t_s"] = t_s
        hand_rows.append(h)

    # Write CSVs
    pen_path = folder / f"{stem}_pen.csv"
    body_path = folder / f"{stem}_body.csv"
    hand_path = folder / f"{stem}_hand.csv"
    write_csv(pen_path, pen_rows)
    write_csv(body_path, body_rows)
    write_csv(hand_path, hand_rows)

    # Sidecar
    # Count quality flags
    def count_quality(rows):
        counts = {}
        for r in rows:
            q = r.get("data_quality", "unknown")
            counts[q] = counts.get(q, 0) + 1
        return counts

    sidecar = {
        "sourceFile": json_path.name,
        "sessionID": data.get("sessionID"),
        "fileNameIdentifier": data.get("fileNameIdentifier"),
        "headerTimestamp": data.get("timestamp"),
        "thresholds": {
            "hand_dropout_threshold_ms": hand_dropout_threshold_ms,
            "body_dropout_threshold_ms": body_dropout_threshold_ms,
        },
        "timeline": {
            "anchor": "t=0 := body's first sample UTC",
            "window_start_utc_ms": window_start_utc,
            "window_end_utc_ms": window_end_utc,
            "window_duration_s": (window_end_utc - window_start_utc) / 1000.0,
            "n_output_rows": len(pen_rows),
        },
        "quality_counts": {
            "pen": count_quality(pen_rows),
            "body": count_quality(body_rows),
            "hand": count_quality(hand_rows),
        },
        "raw_streams": {
            "pen": {
                "n_total": len(pen_samples),
                "n_in_window": len(pen_in_window),
                "n_dropped_pre": sum(1 for p in pen_samples
                                    if p["utc_ms"] < window_start_utc),
                "n_dropped_post": sum(1 for p in pen_samples
                                     if p["utc_ms"] > window_end_utc),
                "first_utc_ms": pen_first_utc_ms,
                "last_utc_ms": pen_last_utc_ms,
            },
            "body": {
                "n_total": len(body_samples),
                "first_utc_ms": body_samples[0]["utc_ms"],
                "last_utc_ms": body_samples[-1]["utc_ms"],
                "anchor_method": "body.timestamp + pen_first_utc = body.utc",
            },
            "hand": {
                "n_total": len(hand_samples),
                "first_frame_idx": (hand_samples[0]["frame_idx"]
                                    if hand_samples else None),
                "last_frame_idx": (hand_samples[-1]["frame_idx"]
                                   if hand_samples else None),
                "anchor_method": ("hand.frame -> pen[frame].utc_ms"
                                  if hand_samples else "no_hand_data"),
            },
        },
        "warnings": _gather_warnings(data, pen_samples, body_samples,
                                     hand_samples, window_start_utc,
                                     window_end_utc),
    }
    sidecar_path = folder / f"{stem}_sync.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str),
                            encoding="utf-8")

    return {
        "pen_csv": pen_path, "body_csv": body_path, "hand_csv": hand_path,
        "sidecar": sidecar_path, "summary": sidecar,
    }


def _gather_warnings(data, pen_samples, body_samples, hand_samples,
                     start_utc, end_utc):
    warns = []
    if not hand_samples:
        warns.append("No hand tracking data in this recording.")
    else:
        n_before = sum(1 for h in hand_samples if h["utc_ms"] < start_utc)
        n_after = sum(1 for h in hand_samples if h["utc_ms"] > end_utc)
        if n_before:
            warns.append(f"{n_before} hand samples occur before body window starts.")
        if n_after:
            warns.append(f"{n_after} hand samples occur after body window ends.")

    # Check body sample coverage vs pen rate (gaps in body coverage matter)
    if len(body_samples) >= 2:
        body_dur = (body_samples[-1]["utc_ms"] - body_samples[0]["utc_ms"]) / 1000.0
        body_rate = (len(body_samples) - 1) / body_dur if body_dur > 0 else 0
        if body_rate < 10:
            warns.append(
                f"Body sample rate is low ({body_rate:.1f} Hz)."
            )
        # Look for any body gap > 0.5s
        max_gap_ms = 0
        for i in range(1, len(body_samples)):
            gap = body_samples[i]["utc_ms"] - body_samples[i - 1]["utc_ms"]
            if gap > max_gap_ms:
                max_gap_ms = gap
        if max_gap_ms > 500:
            warns.append(f"Body has a dropout of {max_gap_ms} ms; "
                         "interpolated rows across that gap will be unreliable.")

    return warns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path,
                    help="Path to the master recording JSON")
    ap.add_argument("--hand-dropout-ms", type=float, default=100.0,
                    help="Hand gaps >= this are flagged 'filled_dropout' "
                         "instead of 'interpolated' (default: 100ms)")
    ap.add_argument("--body-dropout-ms", type=float, default=200.0,
                    help="Body gaps >= this are flagged 'filled_dropout' "
                         "instead of 'interpolated' (default: 200ms)")
    args = ap.parse_args()

    if not args.json_path.is_file():
        sys.exit(f"ERROR: {args.json_path} not found")

    print(f"Processing: {args.json_path.name}")
    result = process(args.json_path,
                     hand_dropout_threshold_ms=args.hand_dropout_ms,
                     body_dropout_threshold_ms=args.body_dropout_ms)
    s = result["summary"]

    print()
    print(f"Wrote to: {result['pen_csv'].parent}")
    print(f"  {result['pen_csv'].name}")
    print(f"  {result['body_csv'].name}")
    print(f"  {result['hand_csv'].name}")
    print(f"  {result['sidecar'].name}")
    print()
    print(f"Sync window (anchored on body):")
    print(f"  Duration: {s['timeline']['window_duration_s']:.3f} s")
    print(f"  Output rows: {s['timeline']['n_output_rows']}")
    print()
    print(f"Stream coverage:")
    print(f"  Pen:  {s['raw_streams']['pen']['n_in_window']} of "
          f"{s['raw_streams']['pen']['n_total']} samples in window "
          f"(dropped {s['raw_streams']['pen']['n_dropped_pre']} pre, "
          f"{s['raw_streams']['pen']['n_dropped_post']} post)")
    print(f"  Body: {s['raw_streams']['body']['n_total']} raw samples")
    print(f"  Hand: {s['raw_streams']['hand']['n_total']} raw samples")
    print()
    print(f"Quality counts (rows per stream):")
    for stream in ("pen", "body", "hand"):
        qc = s['quality_counts'][stream]
        print(f"  {stream}: {dict(qc)}")
    if s['warnings']:
        print()
        print(f"Warnings:")
        for w in s['warnings']:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
