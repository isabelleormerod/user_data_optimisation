#!/usr/bin/env python3
r"""
clean_place_events.py - Dataset Cleaner (run BEFORE MCDA/fPCA)

One pre-analysis pass that does two independent jobs:

  (1) COVERAGE FILTER (per Place event). Flags -- does NOT delete -- events where
      any present source (pen/body/hand) has coverage below a threshold (default
      0.70, --min-coverage). Filtering at event granularity keeps far more data
      than dropping whole height blocks. Coverage per source = the mean fraction
      of that source's POSITION markers present (finite AND non-zero) across the
      event's frames; whole-source dropout drives it to 0, a single flaky joint
      barely moves it. Orientation (quaternion) markers are not counted, and
      noise does NOT reduce coverage (a noisy-but-present value is still present).

  (2) BODY-z MEDIAN FILTER (raw position). The single side-camera MediaPipe view
      has a systematically noisy DEPTH (z) axis while x/y are clean; on slow body
      movement that noise is high-frequency jitter a short MEDIAN filter removes
      without smearing the slow signal. Only body _z columns are filtered (x/y and
      pen/hand -- Quest depth is real -- are untouched). Written non-destructively
      to a sibling {stem}_body_zfilt.csv. Disable with --no-smooth; tune --window.

ORDERING NOTE: coverage is measured on the RAW body, because presence is not
changed by median-filtering z. The two steps are independent; bundling them here
is a convenience, not a dependency.

NON-DESTRUCTIVE. Originals are never modified. Writes:
  event_quality.csv          one row per Place event: coverage per source + keep/drop
  excluded_place_events.csv  the drop manifest (the extractors can skip these)
  cleaning_report.txt        human-readable, grouped by trial:
                             "Trial X (P0xx): dropped events 2, 7 (hand 41%, hand 38%)"
  zfilter_report.csv         per-marker z before/after roughness + raw-vs-filtered corr
  {stem}_body_zfilt.csv      filtered body, beside each source body file

EVENT IDENTITY: events are numbered 1..N in TEMPORAL order within each trial
(event_num -- what the report lists), and also carry height + the per-height index
the extractor uses (place_index_in_height), so the manifest matches downstream.

USAGE:
    python clean_place_events.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python clean_place_events.py --landmarks-root ... --min-coverage 0.75 --window 7
    python clean_place_events.py --landmarks-root ... --coverage-stat min --no-smooth
    python clean_place_events.py --landmarks-root ... --participants P002,P003 --out .\cleaning

  Optional:
    --min-coverage F        Drop event if any present source's coverage < F (default 0.70)
    --coverage-stat S       'mean' (default) over a source's markers, or 'min' = worst marker
    --require-all-sources   Treat a missing body/hand file as a failure too (default: ignore absent)
    --no-smooth             Skip the body-z median filter (coverage filtering only)
    --window N              Median window in frames, forced odd (default 5; larger = smoother)
    --participants ...      Comma-separated PIDs
    --trials SUBSTR         Only trials whose stem contains any given substring (comma-separated)
    --max-trials N          Cap number of trials
    --out PATH              Report dir (DEFAULT <landmarks-root>/metrics/cleaning). The _zfilt.csv
                            files are always written next to their source body file.

NOTES / ASSUMPTIONS TO VERIFY:
  - "Present" = finite AND not all-zero (Quest/MediaPipe zero-fill a lost marker).
    If loss is flagged by a confidence/visibility column instead, tell me.
  - Coverage windows are the exact Place spans (no pad), matching the events the
    extractor forms from contiguous Place==1 runs in the pen file.
  - Median filter: centred rolling window (min_periods=1 at edges), so no frames
    are dropped and no NaNs introduced. Zero-filled frames are included; median is
    robust and body loss is minor. Check corr_raw_filt ~1.0 (noise removed, slow
    movement kept); if it dips, shrink --window.
  - Filtering z changes body joint ANGLES slightly (intended). Re-extract after.
  - To honour outputs downstream: (a) prefer {stem}_body_zfilt.csv in the
    extractor's find_sibling; (b) load excluded_place_events.csv and skip matching
    (participant, trial, height, place_index) rows in collect_events. This script
    produces both; it does not modify your analysis scripts.

Self-contained: numpy + pandas only, no project/utils imports.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEIGHTS = ["High", "Medium", "Low"]


# ------------------------------------------------------------------ discovery
def find_labelled_pen(trial_dir):
    for pat in ("*_pen_flattened_labelled.csv", "*_pen_labelled.csv"):
        hits = sorted(trial_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def pen_stem(pen_path):
    stem = pen_path.stem
    for suffix in ("_pen_flattened_labelled", "_pen_labelled"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def find_sibling(trial_dir, stem, stream):
    for name in (f"{stem}_{stream}_labelled.csv", f"{stem}_{stream}.csv"):
        p = trial_dir / name
        if p.is_file():
            return p
    return None


def iter_trials(root, pfilter, trial_substrs):
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pfilter and pid not in pfilter:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            pen = find_labelled_pen(trial_dir)
            if pen is None:
                continue
            stem = pen_stem(pen)
            if trial_substrs and not any(s in stem for s in trial_substrs):
                continue
            yield stem, pid, trial_dir


# ------------------------------------------------------------ markers & runs
def detect_xyz_markers(df, stream):
    """Position markers only: {name: [x_col, y_col, z_col]}."""
    if df is None:
        return {}
    cols = set(df.columns)
    markers = {}
    if stream == "pen":
        for tag, trip in (("tip (flattened)", ("x_flat", "y_flat", "z_flat")),
                          ("tip (raw)", ("x", "y", "z"))):
            if all(c in cols for c in trip):
                markers[tag] = list(trip)
    for c in df.columns:
        if c.endswith("_x"):
            base = c[:-2]
            trip = (f"{base}_x", f"{base}_y", f"{base}_z")
            if all(t in cols for t in trip):
                markers[base] = list(trip)
    return markers


def contiguous_runs(mask):
    mask = np.asarray(mask)
    if not mask.any():
        return
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, stops):
        yield int(s), int(e)


def place_runs(df):
    t = df["t_s"].to_numpy(float)
    flag = df["Place"].astype(str).str.strip().isin(["1", "1.0", "True", "true"]).to_numpy()
    return [(float(t[s]), float(t[e])) for s, e in contiguous_runs(flag)]


def height_lookup(df):
    runs = {}
    for h in HEIGHTS:
        if h in df.columns:
            t = df["t_s"].to_numpy(float)
            flag = df[h].astype(str).str.strip().isin(["1", "1.0", "True", "true"]).to_numpy()
            runs[h] = [(float(t[s]), float(t[e])) for s, e in contiguous_runs(flag)]
    def get(tmid):
        for h, rr in runs.items():
            for s, e in rr:
                if s <= tmid <= e:
                    return h
        return "Unknown"
    return get


# ------------------------------------------------------------ coverage & filter
def source_coverage(df, markers, s, e, stat):
    if df is None or df.empty or "t_s" not in df.columns or not markers:
        return None
    sub = df[(df["t_s"] >= s) & (df["t_s"] <= e)]
    if len(sub) < 1:
        return None
    per_marker = {}
    for name, cols in markers.items():
        arr = sub[cols].to_numpy(float)
        valid = np.isfinite(arr).all(axis=1) & ~(np.abs(arr) < 1e-9).all(axis=1)
        per_marker[name] = float(valid.mean())
    worst = min(per_marker, key=per_marker.get)
    cov = per_marker[worst] if stat == "min" else float(np.mean(list(per_marker.values())))
    return cov, len(sub)


def roughness(a):
    a = np.asarray(a, float)
    d = np.abs(np.diff(a))
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    ap.add_argument("--coverage-stat", choices=["mean", "min"], default="mean")
    ap.add_argument("--require-all-sources", action="store_true")
    ap.add_argument("--no-smooth", action="store_true", help="Skip the body-z median filter")
    ap.add_argument("--window", type=int, default=5, help="Median window (frames, forced odd, default 5)")
    ap.add_argument("--participants", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None)
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.landmarks_root.is_dir():
        sys.exit(f"ERROR: {args.landmarks_root} is not a directory")

    win = args.window if args.window % 2 == 1 else args.window + 1
    out_dir = args.out or (args.landmarks_root / "metrics" / "cleaning")
    out_dir.mkdir(parents=True, exist_ok=True)

    pfilter = {p.strip() for p in args.participants.split(",")} if args.participants else None
    trial_substrs = [s.strip() for s in args.trials.split(",")] if args.trials else None
    trials = list(iter_trials(args.landmarks_root, pfilter, trial_substrs))
    if args.max_trials:
        trials = trials[:args.max_trials]
    if not trials:
        sys.exit("No matching trials found.")

    smooth_txt = "off" if args.no_smooth else f"on (window={win})"
    print(f"Found {len(trials)} trial(s). Coverage: any source < {args.min_coverage:.0%} "
          f"({args.coverage_stat}) -> drop.  Body-z median filter: {smooth_txt}.\n")

    thr = args.min_coverage
    ev_rows, z_rows, n_zfiles = [], [], 0

    for stem, pid, trial_dir in trials:
        pen = pd.read_csv(find_labelled_pen(trial_dir))
        if "Place" not in pen.columns or "t_s" not in pen.columns:
            print(f"  [warn] {pid}/{stem}: no Place/t_s; skipped.")
            continue
        events = place_runs(pen)
        if not events:
            print(f"  [warn] {pid}/{stem}: 0 Place events.")
            continue
        get_h = height_lookup(pen)

        body_p = find_sibling(trial_dir, stem, "body")
        hand_p = find_sibling(trial_dir, stem, "hand")
        body = pd.read_csv(body_p) if body_p else None
        hand = pd.read_csv(hand_p) if hand_p else None
        src_df = {"pen": pen, "body": body, "hand": hand}
        src_mk = {"pen": detect_xyz_markers(pen, "pen"),
                  "body": detect_xyz_markers(body, "body"),
                  "hand": detect_xyz_markers(hand, "hand")}

        # ---- (1) coverage filter, on RAW body ----
        counters = {}
        n_drop = 0
        for i, (s, e) in enumerate(events, start=1):
            h = get_h((s + e) / 2)
            counters[h] = counters.get(h, 0) + 1
            row = {"participant": pid, "trial": stem, "event_num": i, "height": h,
                   "place_index_in_height": counters[h],
                   "start_t_s": round(s, 4), "stop_t_s": round(e, 4)}
            present, fails = {}, []
            for src in ("pen", "body", "hand"):
                res = source_coverage(src_df[src], src_mk[src], s, e, args.coverage_stat)
                if res is None:
                    row[f"cov_{src}"] = np.nan
                    row[f"n_{src}"] = 0
                    if args.require_all_sources:
                        fails.append(f"{src} absent")
                    continue
                cov, nfr = res
                row[f"cov_{src}"] = round(cov, 4)
                row[f"n_{src}"] = nfr
                present[src] = cov
                if cov < thr:
                    fails.append(f"{src} {cov*100:.0f}%")
            row["min_source_cov"] = round(min(present.values()), 4) if present else np.nan
            row["decision"] = "drop" if fails else "keep"
            row["failing_sources"] = "; ".join(fails)
            n_drop += bool(fails)
            ev_rows.append(row)

        # ---- (2) body-z median filter, from RAW body ----
        n_zc = 0
        if not args.no_smooth and body is not None and src_mk["body"]:
            out = body.copy()
            for name, cols in src_mk["body"].items():
                zc = cols[2]
                raw = body[zc].to_numpy(float)
                filt = pd.Series(raw).rolling(win, center=True, min_periods=1).median().to_numpy()
                out[zc] = filt
                rb, ra = roughness(raw), roughness(filt)
                corr = float(np.corrcoef(raw, filt)[0, 1]) if np.isfinite(raw).all() else np.nan
                z_rows.append({"participant": pid, "trial": stem, "marker": name,
                               "roughness_before": round(rb, 6), "roughness_after": round(ra, 6),
                               "pct_reduction": (round((1 - ra / rb) * 100, 1) if rb else np.nan),
                               "corr_raw_filt": round(corr, 4)})
                n_zc += 1
            out.to_csv(trial_dir / f"{stem}_body_zfilt.csv", index=False)
            n_zfiles += 1

        print(f"  [ok]  {pid}/{stem}: {len(events)} events, {n_drop} dropped"
              + ("" if args.no_smooth else f"; z-filtered {n_zc} col(s)"))

    # ---- outputs ----
    df = pd.DataFrame(ev_rows)
    if df.empty:
        sys.exit("No Place events measured.")
    cols = ["participant", "trial", "event_num", "height", "place_index_in_height",
            "start_t_s", "stop_t_s", "n_pen", "n_body", "n_hand",
            "cov_pen", "cov_body", "cov_hand", "min_source_cov", "decision", "failing_sources"]
    df = df[cols]
    df.to_csv(out_dir / "event_quality.csv", index=False)
    df[df.decision == "drop"].to_csv(out_dir / "excluded_place_events.csv", index=False)

    lines = [f"Coverage cleaning report  (threshold: any source < {thr:.0%}, {args.coverage_stat})",
             "=" * 72]
    for (pid, stem), grp in df.groupby(["participant", "trial"]):
        d = grp[grp.decision == "drop"]
        if d.empty:
            lines.append(f"{pid} / {stem}: {len(grp)} events, all kept")
        else:
            detail = ", ".join(f"#{int(r.event_num)} ({r.height}: {r.failing_sources})" for r in d.itertuples())
            lines.append(f"{pid} / {stem}: {len(grp)} events, dropped {len(d)} -> {detail}")
    (out_dir / "cleaning_report.txt").write_text("\n".join(lines), encoding="utf-8")

    z_df = pd.DataFrame(z_rows)
    if not z_df.empty:
        z_df.to_csv(out_dir / "zfilter_report.csv", index=False)

    # ---- console summary ----
    total, ndrop = len(df), int((df.decision == "drop").sum())
    print(f"\n{'='*66}")
    print(f"COVERAGE  events: {total}  kept: {total-ndrop}  dropped: {ndrop} "
          f"({ndrop/total*100:.1f}%)  retained: {(total-ndrop)/total*100:.1f}%")
    if ndrop:
        by_src = {}
        for fs in df.loc[df.decision == "drop", "failing_sources"]:
            for tok in fs.split("; "):
                by_src[tok.split()[0]] = by_src.get(tok.split()[0], 0) + 1
        print("  dropped by failing source:", by_src)
        print("  dropped by height:", df[df.decision == "drop"]["height"].value_counts().to_dict())
    if not z_df.empty:
        print(f"BODY-z    filtered {n_zfiles} file(s); roughness -{z_df['pct_reduction'].mean():.1f}% mean, "
              f"corr {z_df['corr_raw_filt'].mean():.4f} (near 1.0 = slow movement kept)")
    print(f"{'='*66}")
    print("OUTPUT WRITTEN TO:")
    for f in ("event_quality.csv", "excluded_place_events.csv", "cleaning_report.txt",
              *(("zfilter_report.csv",) if not z_df.empty else ())):
        print(f"   {(out_dir / f).resolve()}")
    if not args.no_smooth:
        print("   *_body_zfilt.csv  (next to each source body file)")
    print("   (reports under --landmarks-root by default; use --out to change)")
    print("=" * 66)


if __name__ == "__main__":
    main()