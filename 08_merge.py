#!/usr/bin/env python3
"""
Merge the per-Place-event PERFORMANCE metrics and POSTURE features into one
wide table, ready for association modelling (Stage 1) and optimisation.

Inputs (under <root>/metrics/):
    place_metrics_combined.csv        (from 04 — performance per Place event)
    posture_features_combined.csv     (from 07 — posture per Place event)

Join keys: participant, trial, place_index, height
Shared timing columns (start_t_s, stop_t_s, duration_s) are taken from the
performance table and not duplicated.

Output:
    <root>/metrics/combined_all.csv   (one row per Place event: keys + all
                                       performance metrics + all posture features)

A short report prints how many rows matched, and flags any events present in
one table but not the other (e.g. a trial whose body/hand data was missing).

Usage:
    python 09_merge.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python 09_merge.py --perf path/to/place_metrics_combined.csv \\
                       --posture path/to/posture_features_combined.csv \\
                       --out path/to/combined_all.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


KEYS = ["participant", "trial", "place_index", "height"]
SHARED = ["start_t_s", "stop_t_s", "duration_s"]   # keep from performance only


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--perf", type=Path, default=None,
                    help="place_metrics_combined.csv (overrides --landmarks-root)")
    ap.add_argument("--posture", type=Path, default=None,
                    help="posture_features_combined.csv (overrides --landmarks-root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <root>/metrics/combined_all.csv)")
    ap.add_argument("--how", choices=["inner", "outer", "left"], default="inner",
                    help="Join type: inner (only events in BOTH, default), "
                         "outer (all events), left (all performance events)")
    args = ap.parse_args()

    if args.perf and args.posture:
        perf_path, post_path = args.perf, args.posture
        out_path = args.out or (perf_path.parent / "combined_all.csv")
    elif args.landmarks_root:
        m = args.landmarks_root / "metrics"
        perf_path = m / "place_metrics_combined.csv"
        post_path = m / "posture_features_combined.csv"
        out_path = args.out or (m / "combined_all.csv")
    else:
        sys.exit("Provide --landmarks-root, or both --perf and --posture")

    for p, label in ((perf_path, "performance"), (post_path, "posture")):
        if not p.is_file():
            sys.exit(f"{label} table not found: {p}\n"
                     f"  (run the relevant earlier script first)")

    perf = pd.read_csv(perf_path)
    post = pd.read_csv(post_path)

    for name, dfx in (("performance", perf), ("posture", post)):
        missing = [k for k in KEYS if k not in dfx.columns]
        if missing:
            sys.exit(f"{name} table is missing join key(s): {missing}")

    # Normalise key dtypes so the join matches (place_index int, others str)
    for dfx in (perf, post):
        dfx["place_index"] = pd.to_numeric(dfx["place_index"], errors="coerce").astype("Int64")
        for k in ("participant", "trial", "height"):
            dfx[k] = dfx[k].astype(str)

    # Drop the shared timing columns from the posture side to avoid _x/_y suffixes
    post_trimmed = post.drop(columns=[c for c in SHARED if c in post.columns],
                             errors="ignore")

    merged = perf.merge(post_trimmed, on=KEYS, how=args.how,
                        suffixes=("", "_posture"), indicator=True)

    n_both = int((merged["_merge"] == "both").sum())
    n_perf_only = int((merged["_merge"] == "left_only").sum())
    n_post_only = int((merged["_merge"] == "right_only").sum())

    print(f"Performance events: {len(perf)}")
    print(f"Posture events:     {len(post)}")
    print(f"Join ({args.how}): {len(merged)} rows  "
          f"[both={n_both}, perf-only={n_perf_only}, posture-only={n_post_only}]")

    # Report mismatches so missing body/hand data is visible
    if n_perf_only:
        ex = merged[merged["_merge"] == "left_only"][KEYS].head(8)
        print(f"\n{n_perf_only} event(s) have performance but NO posture "
              f"(missing/short body or hand data?). Examples:")
        for _, r in ex.iterrows():
            print(f"  {r['participant']} / {r['trial']} / place {r['place_index']} / {r['height']}")
    if n_post_only:
        ex = merged[merged["_merge"] == "right_only"][KEYS].head(8)
        print(f"\n{n_post_only} event(s) have posture but NO performance. Examples:")
        for _, r in ex.iterrows():
            print(f"  {r['participant']} / {r['trial']} / place {r['place_index']} / {r['height']}")

    merged = merged.drop(columns=["_merge"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(merged)} rows, {len(merged.columns)} columns)")


if __name__ == "__main__":
    main()
