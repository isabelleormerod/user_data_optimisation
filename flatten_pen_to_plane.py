#!/usr/bin/env python3
"""
Flatten pen data onto a calibration plane.

Resolves all file paths automatically from a single trial stem.  You only
need to supply the stem and (once) the root paths for your data.

Stem format:  <PID>_<trial_descriptor>
Example:      P003_Short_Large_Front_weighted_A135

File layout assumed:
  Pen CSV:
    <landmarks_root>/<PID>/<stem>_<datetime>/<stem>_<datetime>_pen.csv

  Synced BORIS:
    <boris_root>/<stem>_synced.tsv

  Flattened output (written here):
    <landmarks_root>/<PID>/<stem>_<datetime>/<stem>_<datetime>_pen_flattened.csv

  Quality log (appended, shared across all participants):
    <landmarks_root>/plane_quality_log.csv

Default roots (edit these two lines to match your machine, then you never
need to pass them on the command line again):
    LANDMARKS_ROOT = A:/Automated_chain_BETA/Participant_Landmarks
    BORIS_ROOT     = A:/Automated_chain_BETA/BORIS_csvs

Usage:
    python flatten_pen_to_plane.py P003_Short_Large_Front_weighted_A135
    python flatten_pen_to_plane.py P003_Short_Large_Front_weighted_A135 \\
        --landmarks-root "D:/MyData/Participant_Landmarks" \\
        --boris-root "D:/MyData/BORIS_csvs"
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ #
# *** EDIT THESE TWO LINES TO MATCH YOUR MACHINE ***
DEFAULT_LANDMARKS_ROOT = Path("A:/Automated_chain_BETA/Participant_Landmarks")
DEFAULT_BORIS_ROOT     = Path("A:/Automated_chain_BETA/BORIS_csvs")
# ------------------------------------------------------------------ #


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def resolve_paths(stem: str,
                  landmarks_root: Path,
                  boris_root: Path) -> dict:
    """
    Given a trial stem (e.g. 'P003_Short_Large_Front_weighted_A135'), find:
      - participant ID  (first token, e.g. 'P003')
      - datetime-stamped subfolder  (<stem>_YYYYMMDD_HHMMSS)
      - pen CSV, synced BORIS, intended output path, quality log path

    Raises FileNotFoundError with a clear message if anything is missing.
    """
    pid = stem.split("_")[0]                         # e.g. P003
    participant_dir = landmarks_root / pid

    if not participant_dir.is_dir():
        raise FileNotFoundError(
            f"Participant folder not found: {participant_dir}\n"
            f"  (landmarks_root = {landmarks_root})")

    # Find the datetime-stamped subfolder: must start with stem + '_'
    # and end with an 8-digit date + 6-digit time.
    matches = [
        d for d in participant_dir.iterdir()
        if d.is_dir() and d.name.startswith(stem + "_")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No subfolder starting with '{stem}_' found inside:\n"
            f"  {participant_dir}\n"
            f"  Available folders: {[d.name for d in participant_dir.iterdir() if d.is_dir()]}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple subfolders match '{stem}_' inside {participant_dir}:\n"
            f"  {[d.name for d in matches]}\n"
            f"  Please ensure only one datetime-stamped folder exists per stem.")

    trial_dir   = matches[0]                         # e.g. …/P003/P003_…_113934/
    trial_stamp = trial_dir.name                     # e.g. P003_…_113934

    pen_path    = trial_dir / f"{trial_stamp}_pen.csv"
    output_path = trial_dir / f"{trial_stamp}_pen_flattened.csv"
    boris_path  = boris_root / f"{stem}_synced.tsv"
    log_path    = landmarks_root / "plane_quality_log.csv"

    missing = []
    if not pen_path.is_file():
        missing.append(f"  Pen CSV:      {pen_path}")
    if not boris_path.is_file():
        missing.append(f"  Synced BORIS: {boris_path}")
    if missing:
        raise FileNotFoundError(
            "Expected file(s) not found:\n" + "\n".join(missing))

    return {
        "pen_path":    pen_path,
        "boris_path":  boris_path,
        "output_path": output_path,
        "log_path":    log_path,
        "trial_stamp": trial_stamp,
        "pid":         pid,
    }


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        delim = "\t"
    elif suffix == ".csv":
        delim = ","
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delim = ","
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def load_pen(path: Path):
    rows, fields = read_table(path)
    for col in ("t_s", "x", "y", "z"):
        if col not in fields:
            raise RuntimeError(
                f"Pen CSV missing '{col}' column.\n"
                f"  Found: {fields}\n  File: {path}")
    t_s, xyz, kept = [], [], []
    for r in rows:
        try:
            t_s.append(float(r["t_s"]))
            xyz.append((float(r["x"]), float(r["y"]), float(r["z"])))
            kept.append(r)
        except (TypeError, ValueError):
            continue
    if len(t_s) < 2:
        raise RuntimeError(f"Fewer than 2 valid rows in pen CSV: {path}")
    return np.asarray(t_s), np.asarray(xyz), kept


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def fit_plane(points: np.ndarray):
    """Least-squares plane via PCA. Returns (centroid, unit normal)."""
    if points.shape[0] < 3:
        raise RuntimeError(
            f"Need >=3 calibration points to fit a plane, got {points.shape[0]}.")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return centroid, normal


def build_flattening_transform(centroid: np.ndarray, normal: np.ndarray,
                               recenter_xy: bool = False):
    """Return (R, t) so that p_flat = R @ p + t lays the plane on z = 0."""
    n = normal / np.linalg.norm(normal)
    if n[2] < 0:
        n = -n
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c = float(np.dot(n, target))
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],     0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    t = -R @ centroid if recenter_xy else np.array([0., 0., -(R @ centroid)[2]])
    return R, t


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def process(pen_path: Path, boris_path: Path, output_path: Path,
            calib_prefix: str = "Point",
            max_dt: float = None,
            recenter_xy: bool = False,
            outlier_factor: float = 3.0) -> dict:

    t_s_arr, xyz_arr, pen_rows = load_pen(pen_path)
    boris_rows, boris_fields  = read_table(boris_path)

    if "t_s_synced" not in boris_fields:
        raise RuntimeError(
            f"Synced BORIS file has no 't_s_synced' column.\n"
            f"  File: {boris_path}\n"
            f"  Found columns: {boris_fields}\n"
            f"  Has this file been processed by convert_boris_to_synced_time.py?")
    if "Behavior" not in boris_fields:
        raise RuntimeError(
            f"Synced BORIS file has no 'Behavior' column.\n"
            f"  File: {boris_path}\n"
            f"  Found columns: {boris_fields}")

    # Startup placeholder: the frozen pose emitted before the stylus is tracked.
    # Any calibration point matching this value is silently not-yet-tracking.
    placeholder = xyz_arr[0].copy()
    first_move_t = None
    moved = np.where(np.any(np.abs(xyz_arr - placeholder) > 1e-5, axis=1))[0]
    if len(moved):
        first_move_t = float(t_s_arr[moved[0]])

    # ---- gather calibration candidates ---------------------------------- #
    calib_entries, calib_meta = [], []

    for r in boris_rows:
        beh = (r.get("Behavior") or "").strip()
        if not beh.startswith(calib_prefix):
            continue
        raw = (r.get("t_s_synced") or "").strip()
        if not raw:
            calib_meta.append((beh, None, "DROPPED — no t_s_synced value", None))
            continue

        t_ev = float(raw)
        idx  = int(np.argmin(np.abs(t_s_arr - t_ev)))
        dt   = float(abs(t_s_arr[idx] - t_ev))

        if max_dt is not None and dt > max_dt:
            calib_meta.append(
                (beh, idx,
                 f"DROPPED — nearest pen row is {dt*1000:.0f} ms away "
                 f"(limit {max_dt*1000:.0f} ms)", dt))
            continue

        pos = xyz_arr[idx]
        if np.allclose(pos, placeholder, atol=1e-5):
            note = "DROPPED — pen not yet tracking at this timestamp"
            if first_move_t is not None:
                note += f" (tracking starts ~t_s={first_move_t:.2f}s)"
            calib_meta.append((beh, idx, note, dt))
            continue

        calib_entries.append({"beh": beh, "idx": idx, "pos": pos,
                               "t": t_s_arr[idx], "dt": dt})

    if len(calib_entries) < 3:
        detail = "\n".join(f"  {b}: {n}" for b, _, n, _ in calib_meta)
        raise RuntimeError(
            f"Only {len(calib_entries)} usable calibration point(s) after "
            f"filtering — need at least 3.\n"
            f"  Prefix searched: '{calib_prefix}'\n"
            f"  Per-point detail:\n{detail}")

    # ---- fit, then outlier-reject and refit ----------------------------- #
    def _fit_resid(entries):
        P = np.array([e["pos"] for e in entries])
        c, n = fit_plane(P)
        return c, n, (P - c) @ n     # signed distances

    centroid, normal, resid = _fit_resid(calib_entries)

    if len(calib_entries) >= 5:
        med = np.median(np.abs(resid))
        thresh = outlier_factor * med if med > 0 else np.inf
        survivors, pruned = [], []
        for e, d in zip(calib_entries, resid):
            if abs(d) > thresh and abs(d) > 0.02:
                pruned.append((e, d))
            else:
                survivors.append((e, d))
        if pruned and len(survivors) >= 3:
            for e, d in pruned:
                calib_meta.append(
                    (e["beh"], e["idx"],
                     f"DROPPED — outlier: {d*1000:.1f} mm from plane "
                     f"(>{outlier_factor:.0f}× median)", e["dt"]))
            calib_entries = [e for e, _ in survivors]
            centroid, normal, resid = _fit_resid(calib_entries)

    for e, d in zip(calib_entries, resid):
        calib_meta.append(
            (e["beh"], e["idx"],
             f"USED — dt={e['dt']*1000:.1f} ms  resid={d*1000:.1f} mm",
             e["dt"]))

    calib     = np.array([e["pos"] for e in calib_entries])
    calib_t   = [e["t"] for e in calib_entries]

    # ---- build transform + write flattened CSV -------------------------- #
    R, t_vec  = build_flattening_transform(centroid, normal, recenter_xy)
    flat_cal  = (R @ calib.T).T + t_vec
    rms_z     = float(np.sqrt(np.mean(flat_cal[:, 2] ** 2)))
    max_z     = float(np.abs(flat_cal[:, 2]).max())

    calib_t_set = set(np.round(calib_t, 6).tolist())
    flat_all    = (R @ xyz_arr.T).T + t_vec

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x", "y", "z",
                    "x_flat", "y_flat", "z_flat",
                    "data_quality", "is_calibration"])
        for i, row in enumerate(pen_rows):
            p, pf = xyz_arr[i], flat_all[i]
            is_cal = 1 if round(float(t_s_arr[i]), 6) in calib_t_set else 0
            w.writerow([f"{t_s_arr[i]:.4f}",
                        f"{p[0]:.6f}",  f"{p[1]:.6f}",  f"{p[2]:.6f}",
                        f"{pf[0]:.6f}", f"{pf[1]:.6f}", f"{pf[2]:.6f}",
                        row.get("data_quality", ""), is_cal])

    return {
        "output_path": output_path,
        "n_calib":     len(calib_entries),
        "calib_meta":  calib_meta,
        "centroid":    centroid,
        "normal":      normal,
        "rms_z":       rms_z,
        "max_z":       max_z,
        "n_written":   len(pen_rows),
    }


# --------------------------------------------------------------------------- #
# Quality log
# --------------------------------------------------------------------------- #
_LOG_FIELDS = [
    "run_timestamp", "pid", "trial_stem",
    "pen_file", "boris_file", "output_file",
    "n_pen_rows", "n_calib_used", "n_calib_dropped", "dropped_labels",
    "rms_z_mm", "max_z_mm",
    "normal_x", "normal_y", "normal_z",
    "centroid_x", "centroid_y", "centroid_z",
    "per_point_detail",
]


def write_quality_log(log_path: Path, res: dict, stem: str, pid: str,
                      pen_path: Path, boris_path: Path) -> None:
    dropped = [(b, n) for b, _, n, _ in res["calib_meta"] if "DROPPED" in n]
    dropped_labels = "; ".join(
        f"{b} ({n.split('DROPPED')[1].strip()})" for b, n in dropped
    ) if dropped else ""
    per_point = "; ".join(
        f"{b}:{n}" for b, _, n, _ in res["calib_meta"]
    )
    row = {
        "run_timestamp":   datetime.now().isoformat(timespec="seconds"),
        "pid":             pid,
        "trial_stem":      stem,
        "pen_file":        pen_path.name,
        "boris_file":      boris_path.name,
        "output_file":     Path(res["output_path"]).name,
        "n_pen_rows":      res["n_written"],
        "n_calib_used":    res["n_calib"],
        "n_calib_dropped": len(dropped),
        "dropped_labels":  dropped_labels,
        "rms_z_mm":        f"{res['rms_z']*1000:.3f}",
        "max_z_mm":        f"{res['max_z']*1000:.3f}",
        "normal_x":        f"{res['normal'][0]:+.6f}",
        "normal_y":        f"{res['normal'][1]:+.6f}",
        "normal_z":        f"{res['normal'][2]:+.6f}",
        "centroid_x":      f"{res['centroid'][0]:+.6f}",
        "centroid_y":      f"{res['centroid'][1]:+.6f}",
        "centroid_z":      f"{res['centroid'][2]:+.6f}",
        "per_point_detail": per_point,
    }
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument(
        "stem",
        help="Trial stem, e.g. P003_Short_Large_Front_weighted_A135")

    ap.add_argument(
        "--landmarks-root", type=Path,
        default=DEFAULT_LANDMARKS_ROOT,
        help=f"Root of Participant_Landmarks folder "
             f"(default: {DEFAULT_LANDMARKS_ROOT})")
    ap.add_argument(
        "--boris-root", type=Path,
        default=DEFAULT_BORIS_ROOT,
        help=f"Folder containing *_synced.tsv files "
             f"(default: {DEFAULT_BORIS_ROOT})")

    ap.add_argument(
        "--calib-prefix", default="Point",
        help="Behavior label prefix for calibration rows (default: 'Point')")
    ap.add_argument(
        "--max-dt", type=float, default=None, metavar="SECONDS",
        help="Reject a calibration match if the nearest pen row is more than "
             "this many seconds away (e.g. 0.05)")
    ap.add_argument(
        "--outlier-factor", type=float, default=3.0,
        help="Drop a calibration point whose plane residual exceeds this "
             "multiple of the median (and >2 cm), then refit. "
             "Set very high to disable. (default: 3.0)")
    ap.add_argument(
        "--recenter-xy", action="store_true",
        help="Translate so the calibration centroid becomes (0, 0, 0); "
             "default keeps the original x/y position")

    args = ap.parse_args()

    # ---- resolve paths -------------------------------------------------- #
    try:
        paths = resolve_paths(args.stem, args.landmarks_root, args.boris_root)
    except FileNotFoundError as e:
        sys.exit(f"ERROR — could not find required files:\n{e}")

    print(f"Trial:        {args.stem}")
    print(f"Pen CSV:      {paths['pen_path']}")
    print(f"Synced BORIS: {paths['boris_path']}")
    print(f"Output:       {paths['output_path']}")
    print()

    # ---- process -------------------------------------------------------- #
    try:
        res = process(
            paths["pen_path"], paths["boris_path"], paths["output_path"],
            calib_prefix  = args.calib_prefix,
            max_dt        = args.max_dt,
            recenter_xy   = args.recenter_xy,
            outlier_factor= args.outlier_factor,
        )
    except RuntimeError as e:
        sys.exit(f"ERROR — processing failed:\n{e}")

    # ---- console summary ------------------------------------------------ #
    print(f"Calibration points ({res['n_calib']} used):")
    for beh, idx, note, _ in sorted(res["calib_meta"],
                                    key=lambda x: (x[0] or "")):
        tag = "  [DROPPED]" if "DROPPED" in note else "  [used]  "
        print(f"{tag} {beh:<12}  pen_row={str(idx):<6}  {note}")

    print()
    print(f"Plane normal:    [{res['normal'][0]:+.4f},  "
          f"{res['normal'][1]:+.4f},  {res['normal'][2]:+.4f}]")
    print(f"Plane centroid:  [{res['centroid'][0]:+.4f},  "
          f"{res['centroid'][1]:+.4f},  {res['centroid'][2]:+.4f}]")
    print(f"Fit quality  —   RMS: {res['rms_z']*1000:.2f} mm   "
          f"max: {res['max_z']*1000:.2f} mm")
    print(f"Pen rows written: {res['n_written']}")
    print(f"Output:  {res['output_path']}")

    # ---- quality log ---------------------------------------------------- #
    log_path = paths["log_path"]
    write_quality_log(log_path, res, args.stem, paths["pid"],
                      paths["pen_path"], paths["boris_path"])
    print(f"Quality log: {log_path}")


if __name__ == "__main__":
    main()
