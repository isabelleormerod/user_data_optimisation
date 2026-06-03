#!/usr/bin/env python3
"""
Flatten pen data onto a calibration plane.

Resolves all file paths automatically from a single trial identifier (the BORIS
observation id, which is also the trial folder name).

Folder layout assumed (timestamps already stripped by the pipeline; the synced
BORIS file lives inside the trial folder):
  <landmarks_root>/<PID>/<stem>/<stem>_pen.csv
  <landmarks_root>/<PID>/<stem>/<stem>_boris_synced.csv
  <landmarks_root>/<PID>/<stem>/<stem>_pen_flattened.csv   (output, written here)
  <landmarks_root>/plane_quality_log.csv                   (shared quality log)

where <stem> is the trial / observation id, e.g.
  P003_Long_Large_Front_weighted_A180
and <PID> is its first token, e.g. P003.

Default root (edit to match your machine, then you never need to pass it):
    LANDMARKS_ROOT = A:/Automated_chain_BETA/Participant_Landmarks

Usage:
    python flatten_pen_to_plane.py P003_Long_Large_Front_weighted_A180
    python flatten_pen_to_plane.py P003_Long_Large_Front_weighted_A180 \\
        --landmarks-root "D:/MyData/Participant_Landmarks"

Per-participant usage (process all trials for one or more participants):
    # Process all trials for a single participant P003
    python flatten_pen_to_plane.py --participants P003

    # Process multiple participants (comma-separated)
    python flatten_pen_to_plane.py --participants P003,P004

    # Dry-run to list which trials would be processed for participant P003
    python flatten_pen_to_plane.py --participants P003 --dry-run
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ #
# *** EDIT THIS LINE TO MATCH YOUR MACHINE ***
DEFAULT_LANDMARKS_ROOT = Path("A:/Automated_chain_BETA/Participant_Landmarks")
# Kept for CLI backwards-compatibility; no longer used for resolution.
DEFAULT_BORIS_ROOT     = Path("A:/Automated_chain_BETA/BORIS_csvs")
# ------------------------------------------------------------------ #


def iter_trial_stems(landmarks_root: Path, participants: set = None):
    """Yield (stem, participant) for every trial folder under the root that
    contains both a *_pen.csv and a *_boris_synced.csv.

    Layout: <root>/<PID>/<stem>/<stem>_pen.csv etc.
    If `participants` is given (a set of PIDs), only those are yielded.
    """
    if not landmarks_root.is_dir():
        return
    for pid_dir in sorted(p for p in landmarks_root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if participants is not None and pid not in participants:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            stem = trial_dir.name
            pen = trial_dir / f"{stem}_pen.csv"
            boris = trial_dir / f"{stem}_boris_synced.csv"
            if pen.is_file() and boris.is_file():
                yield stem, pid


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def resolve_paths(stem: str,
                  landmarks_root: Path,
                  boris_root: Path = None) -> dict:
    """
    Given a trial identifier (the BORIS observation id, which is also the trial
    folder name, e.g. 'P003_Long_Large_Front_weighted_A180'), find:
      - participant ID  (first token, e.g. 'P003')
      - the trial folder  <root>/<PID>/<stem>/
      - pen CSV, synced BORIS, output path, quality log path

    `boris_root` is accepted but unused now (kept for CLI backwards-compat).

    Raises FileNotFoundError with a clear message if anything is missing.
    """
    pid = stem.split("_")[0]                         # e.g. P003
    participant_dir = landmarks_root / pid

    if not participant_dir.is_dir():
        raise FileNotFoundError(
            f"Participant folder not found: {participant_dir}\n"
            f"  (landmarks_root = {landmarks_root})")

    trial_dir = participant_dir / stem
    if not trial_dir.is_dir():
        available = [d.name for d in participant_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Trial folder not found: {trial_dir}\n"
            f"  Expected a folder named exactly '{stem}' inside {participant_dir}\n"
            f"  Available folders: {available}")

    pen_path    = trial_dir / f"{stem}_pen.csv"
    output_path = trial_dir / f"{stem}_pen_flattened.csv"
    boris_path  = trial_dir / f"{stem}_boris_synced.csv"
    log_path    = landmarks_root / "plane_quality_log.csv"

    missing = []
    if not pen_path.is_file():
        missing.append(f"  Pen CSV:      {pen_path}")
    if not boris_path.is_file():
        missing.append(f"  Synced BORIS: {boris_path}")
    if missing:
        present = [p.name for p in trial_dir.iterdir() if p.is_file()]
        raise FileNotFoundError(
            "Expected file(s) not found:\n" + "\n".join(missing)
            + f"\n  Files present in {trial_dir.name}: {present}")

    return {
        "pen_path":    pen_path,
        "boris_path":  boris_path,
        "output_path": output_path,
        "log_path":    log_path,
        "trial_stamp": stem,
        "pid":         pid,
    }


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_table(path: Path):
    """Read a delimited table, detecting tab vs comma from the actual content
    (not the file extension). The synced BORIS file keeps the delimiter of its
    source, so a .csv may in fact be tab-delimited."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
    # Decide delimiter from the header line: prefer whichever splits it into
    # more fields. Tab wins ties because BORIS exports are usually TSV.
    n_tab = first_line.count("\t")
    n_comma = first_line.count(",")
    if n_tab >= n_comma and n_tab > 0:
        delim = "\t"
    elif n_comma > 0:
        delim = ","
    else:
        # Single-column or unknown; fall back to extension then comma
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
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
# Run a single trial (shared by single + batch modes)
# --------------------------------------------------------------------------- #
def run_one(stem: str, landmarks_root: Path, boris_root: Path,
            calib_prefix: str, max_dt, recenter_xy: bool,
            outlier_factor: float, verbose: bool = True) -> dict:
    """Resolve, process, and log one trial. Returns a status dict.
    Does not raise on the expected FileNotFound / Runtime errors; instead
    captures them in the returned dict so batch runs continue."""
    status = {"stem": stem, "status": "pending", "error": None,
              "n_calib": None, "rms_z_mm": None}
    try:
        paths = resolve_paths(stem, landmarks_root, boris_root)
    except FileNotFoundError as e:
        status["status"] = "missing_files"
        status["error"] = str(e)
        return status

    if verbose:
        print(f"Trial:        {stem}")
        print(f"Pen CSV:      {paths['pen_path']}")
        print(f"Synced BORIS: {paths['boris_path']}")
        print(f"Output:       {paths['output_path']}")
        print()

    try:
        res = process(
            paths["pen_path"], paths["boris_path"], paths["output_path"],
            calib_prefix=calib_prefix, max_dt=max_dt,
            recenter_xy=recenter_xy, outlier_factor=outlier_factor,
        )
    except RuntimeError as e:
        status["status"] = "failed"
        status["error"] = str(e)
        return status

    write_quality_log(paths["log_path"], res, stem, paths["pid"],
                      paths["pen_path"], paths["boris_path"])

    status["status"] = "ok"
    status["n_calib"] = res["n_calib"]
    status["rms_z_mm"] = res["rms_z"] * 1000
    status["n_dropped"] = sum(1 for _, _, n, _ in res["calib_meta"]
                              if "DROPPED" in n)

    if verbose:
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
        print(f"Quality log: {paths['log_path']}")

    return status


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument(
        "stem", nargs="?", default=None,
        help="Trial / observation id (= trial folder name) for a single run, "
             "e.g. P003_Long_Large_Front_weighted_A180. Omit when using "
             "--batch or --participants.")

    ap.add_argument(
        "--landmarks-root", type=Path,
        default=DEFAULT_LANDMARKS_ROOT,
        help=f"Root of Participant_Landmarks folder "
             f"(default: {DEFAULT_LANDMARKS_ROOT})")
    ap.add_argument(
        "--boris-root", type=Path,
        default=DEFAULT_BORIS_ROOT,
        help="(Deprecated/unused — BORIS synced file is now read from inside "
             "the trial folder. Accepted for backwards-compatibility.)")

    ap.add_argument(
        "--batch", action="store_true",
        help="Process every trial folder under the landmarks root that has "
             "both a pen CSV and a boris_synced CSV.")
    ap.add_argument(
        "--participants", type=str, default=None,
        help="Comma-separated participant IDs to restrict a batch run to "
             "(e.g. 'P003,P004'). Implies batch mode over those participants.")

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
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List the trials that would be processed, without doing it "
             "(batch modes only).")

    args = ap.parse_args()

    # Parse participant filter
    participant_filter = None
    if args.participants:
        participant_filter = {p.strip() for p in args.participants.split(",")
                              if p.strip()}

    batch_mode = args.batch or participant_filter is not None

    # ---- single-trial mode --------------------------------------------- #
    if not batch_mode:
        if not args.stem:
            sys.exit("ERROR: provide a trial stem, or use --batch / "
                     "--participants for a batch run.")
        status = run_one(
            args.stem, args.landmarks_root, args.boris_root,
            args.calib_prefix, args.max_dt, args.recenter_xy,
            args.outlier_factor, verbose=True)
        if status["status"] != "ok":
            sys.exit(f"\nERROR — {status['status']}:\n{status['error']}")
        return

    # ---- batch mode ----------------------------------------------------- #
    trials = list(iter_trial_stems(args.landmarks_root, participant_filter))
    if not trials:
        which = (f"participants {', '.join(sorted(participant_filter))}"
                 if participant_filter else "any participant")
        sys.exit(f"No trial folders (with pen + boris_synced CSVs) found for "
                 f"{which} under {args.landmarks_root}")

    print(f"Found {len(trials)} trial(s) to process")
    if participant_filter:
        print(f"  (restricted to: {', '.join(sorted(participant_filter))})")
    print()

    if args.dry_run:
        print("DRY RUN - would process:")
        for stem, pid in trials:
            print(f"  [{pid}] {stem}")
        return

    results = []
    for i, (stem, pid) in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {stem}")
        status = run_one(
            stem, args.landmarks_root, args.boris_root,
            args.calib_prefix, args.max_dt, args.recenter_xy,
            args.outlier_factor, verbose=False)
        # One-line per-trial summary
        if status["status"] == "ok":
            drop = status.get("n_dropped", 0)
            extra = f", {drop} dropped" if drop else ""
            print(f"   ok — {status['n_calib']} calib points{extra}, "
                  f"RMS {status['rms_z_mm']:.2f} mm")
        else:
            print(f"   {status['status']}: "
                  f"{(status['error'] or '').splitlines()[0]}")
        results.append(status)

    # ---- summary -------------------------------------------------------- #
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import Counter
    counts = Counter(r["status"] for r in results)
    print(f"Total trials: {len(results)} | {dict(counts)}")

    oks = [r for r in results if r["status"] == "ok"]
    if oks:
        rms_vals = [r["rms_z_mm"] for r in oks]
        print(f"Fit RMS across {len(oks)} successful trials: "
              f"min {min(rms_vals):.2f}  max {max(rms_vals):.2f}  "
              f"mean {sum(rms_vals)/len(rms_vals):.2f} mm")

    problems = [r for r in results if r["status"] != "ok"]
    if problems:
        print()
        print("Trials with issues:")
        for r in problems:
            first_line = (r["error"] or "").splitlines()[0] if r["error"] else ""
            print(f"  [{r['status']}] {r['stem']}: {first_line}")


if __name__ == "__main__":
    main()
