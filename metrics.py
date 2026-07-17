#!/usr/bin/env python3
"""
metrics.py -- hand-derived ergonomic metrics, one row per PLACE EVENT,
              batched across participants into one combined table.

Mirrors the pipeline's combined-table logic: walk a landmarks root, compute
per-place-event metrics for every trial file-set, parse the trial stem into
prototype factors, and concatenate into a single place_metrics_combined.csv
(one row per Place event, all participants). Metrics are computed INSIDE place
events only and reported per trial height.

Metrics implemented so far (calibration-based; see references below):
  duration_s          length of the place event (s)
  perp_mean_deg       total pen tilt off perpendicular insertion (deg; 0 = perp)
  updown_mean_deg     vertical (y-z) component of that tilt
  leftright_mean_deg  horizontal (x-z) component of that tilt
Still to add: pos_jitter_mm, ang_jitter_deg (SPARC), aperture_mm + comfort, rula.

Trial stem parsing (same vocabulary as 05_compare_performance.py):
  <PID>_<Length>_<Size>_<Weight>_A<Angle>   e.g. P007_Long_Small_Not_weighted_A135
  Length: Long|Short   Size: Large|Small   Weight: Not_weighted|Front_weighted
  Angle:  A<digits>    ('Front'/'Not' belong to Weight, not a separate factor)

CALIBRATION: the pen file carries x_flat/y_flat/z_flat, a rigid transform of raw
pen position into a frame where the six calibration points lie on z_flat = 0 (the
insertion plane); perpendicular insertion = pointing along +z_flat. Pen ORIENTATION
is in the raw frame, so we recover raw->flat (Kabsch on the six points) and rotate
the pen's forward axis into the flat frame. The forward (tip) axis is auto-detected
as the local axis pointing most toward the board during Place.

References: RULA McAtamney & Corlett (1993) Appl Ergon 24(2):91-99;
SPARC Balasubramanian et al. (2015) J NeuroEng Rehabil 12:112;
grip Kong & Lowe (2005) Int J Ind Ergon 35(6):495-507.

Usage:
  python metrics.py --landmarks-root <ROOT> [--participants P007 P008] [--out combined.csv]
  python metrics.py --pen <one_pen_flattened_labelled.csv> [--out one.csv]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEIGHTS = ["High", "Medium", "Low"]
CALIB_POINTS = ["Point_1", "Point_2", "Point_3", "Point_4", "Point_5", "Point_6"]
PEN_SUFFIX = "_pen_flattened_labelled.csv"
PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]


# --------------------------------------------------------------------------- #
# Trial-stem parsing (same vocabulary/logic as the comparison script)
# --------------------------------------------------------------------------- #
def parse_stem(stem: str) -> dict:
    """Participant + prototype factors from the trial stem, by vocabulary match
    (robust to extra tokens such as a date-time in the filename)."""
    out = {"participant": None, "trial": stem, **{k: None for k in PARAM_FACTORS}}
    m = re.match(r"(P\d+)", stem)
    out["participant"] = m.group(1) if m else None

    if "Not_weighted" in stem:
        out["Weight"] = "Not_weighted"
    elif "Front_weighted" in stem:
        out["Weight"] = "Front_weighted"

    m = re.search(r"_A(\d+)", stem)
    out["Angle"] = int(m.group(1)) if m else None

    tokens = stem.split("_")
    for tok in tokens:
        if tok in ("Long", "Short"):
            out["Length"] = tok
        elif tok in ("Large", "Small"):
            out["Size"] = tok
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def quat_to_R(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = np.array([w, x, y, z]) / np.sqrt(n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def find_runs(mask):
    m = np.asarray(mask, dtype=int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def calibration_transform(p):
    raw = np.array([p.loc[p[c] == 1, ["x", "y", "z"]].values[0] for c in CALIB_POINTS])
    flat = np.array([p.loc[p[c] == 1, ["x_flat", "y_flat", "z_flat"]].values[0] for c in CALIB_POINTS])
    return kabsch(raw, flat)


def pen_forward_axis(p, Rrf):
    q = p[["qw", "qx", "qy", "qz"]].values
    plc = np.where(p["Place"] == 1)[0]
    meanz = np.zeros(3)
    for i in plc:
        meanz += (Rrf @ quat_to_R(q[i]))[2, :]
    meanz /= max(len(plc), 1)
    fwd = int(np.argmax(np.abs(meanz)))
    return fwd, float(np.sign(meanz[fwd]))


def pen_angles(p, Rrf, fwd, sign):
    q = p[["qw", "qx", "qy", "qz"]].values
    F = np.array([sign * (Rrf @ quat_to_R(qq))[:, fwd] for qq in q])
    Fz = np.clip(F[:, 2], -1, 1)
    perp = np.degrees(np.arccos(Fz))                 # total tilt off perpendicular
    updown = np.degrees(np.arctan2(F[:, 1], F[:, 2]))
    side = np.degrees(np.arctan2(F[:, 0], F[:, 2]))
    return perp, updown, side


# --------------------------------------------------------------------------- #
# Per-file metrics
# --------------------------------------------------------------------------- #
def metrics_for_pen(pen_path: Path) -> pd.DataFrame:
    p = pd.read_csv(pen_path)
    p.columns = [c.strip() for c in p.columns]
    t = p["t_s"].values
    Rrf = calibration_transform(p)
    fwd, sign = pen_forward_axis(p, Rrf)
    perp, updown, side = pen_angles(p, Rrf, fwd, sign)

    stem = pen_path.name[: -len(PEN_SUFFIX)]
    meta = parse_stem(stem)

    rows = []
    for s, e in find_runs(p["Place"] == 1):
        height = max(HEIGHTS, key=lambda H: p[H].iloc[s:e].sum())
        rows.append({
            **meta,
            "height": height,
            "t_start": round(float(t[s]), 2),
            "duration_s": round(float(t[e - 1] - t[s]), 3),
            "perp_mean_deg": round(float(np.nanmean(perp[s:e])), 2),
            "updown_mean_deg": round(float(np.nanmean(updown[s:e])), 2),
            "leftright_mean_deg": round(float(np.nanmean(side[s:e])), 2),
        })
    df = pd.DataFrame(rows).sort_values("t_start").reset_index(drop=True)
    if not df.empty:
        # NOTE: named 'place_index' (not 'trial_num') to match the column name
        # used by evaluate_difference.py's posture extraction -- these two
        # scripts' outputs get merged downstream on
        # (participant, trial, place_index, height), and a name mismatch here
        # previously caused that key to silently degrade to
        # (participant, trial, height), which is NOT unique (~5 place events
        # share it), producing a ~5x cartesian-product merge blowup that
        # corrupted every downstream statistic. Keep this name in sync with
        # extract_posture_features()'s "place_index" in evaluate_difference.py.
        df["place_index"] = df.groupby("height").cumcount() + 1
    return df


# --------------------------------------------------------------------------- #
# Discovery + batch
# --------------------------------------------------------------------------- #
def discover_pens(root: Path, participants=None):
    pens = sorted(root.rglob(f"*{PEN_SUFFIX}"))
    if participants:
        keep = set(participants)
        pens = [p for p in pens if parse_stem(p.name[: -len(PEN_SUFFIX)])["participant"] in keep]
    return pens


def run_batch(pens):
    frames, skipped = [], []
    for pen in pens:
        stem = pen.name[: -len(PEN_SUFFIX)]
        try:
            df = metrics_for_pen(pen)
            if df.empty:
                skipped.append((stem, "no place events"))
            else:
                frames.append(df)
                print(f"  {stem}: {len(df)} place events")
        except Exception as ex:
            skipped.append((stem, str(ex)))
            print(f"  {stem}: SKIPPED ({ex})", file=sys.stderr)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None,
                    help="Root searched recursively for *_pen_flattened_labelled.csv")
    ap.add_argument("--pen", type=Path, default=None, help="A single pen file")
    ap.add_argument("--participants", nargs="+", default=None,
                    help="Restrict to these PIDs (e.g. P007 P008)")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = ap.parse_args()

    if args.pen:
        pens = [args.pen]
        out = args.out or args.pen.with_name("place_metrics.csv")
    elif args.landmarks_root:
        pens = discover_pens(args.landmarks_root, args.participants)
        out = args.out or (args.landmarks_root / "metrics" / "place_metrics_combined.csv")
    else:
        sys.exit("Provide --landmarks-root or --pen")

    if not pens:
        sys.exit("No pen files found.")
    print(f"Found {len(pens)} trial file-set(s):")
    combined, skipped = run_batch(pens)
    if combined.empty:
        sys.exit("No place events extracted.")

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nWrote {len(combined)} place events "
          f"({combined['participant'].nunique()} participant(s), "
          f"{combined['trial'].nunique()} trial(s)) -> {out}")
    if skipped:
        print(f"Skipped {len(skipped)}: " + "; ".join(f"{s} ({r})" for s, r in skipped))


if __name__ == "__main__":
    main()