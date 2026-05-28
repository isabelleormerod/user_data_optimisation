#!/usr/bin/env python3
"""
Batch wrapper around sync_to_pen_time.py and convert_boris_to_synced_time.py.

Phase 1: run the tracking sync on every JSON in --json-folder.
Phase 2: for every BORIS file in --boris-folder, read the video filename
         recorded INSIDE the BORIS file (the 'Media file name' column),
         derive the JSON stem from it (stripping the _CamN.mp4 suffix), and
         match it to the corresponding JSON. This allows multiple BORIS files
         that reference the same video (e.g. two trials recorded in one video)
         to each be matched and converted.

Outputs go into the per-trial folder <json_parent>/<participant>/<stem>/.
BORIS outputs include the BORIS file's own stem in the name so multiple BORIS
files for one video don't overwrite each other.

Errors on individual files are logged and processing continues. A summary is
printed at the end.

Usage:
    python batch_sync.py --json-folder /path/to/jsons --boris-folder /path/to/boris
    python batch_sync.py --json-folder /path/to/jsons   # sync only, no BORIS
    python batch_sync.py --json-folder /path/to/jsons --participants 3,4,5
    python batch_sync.py --json-folder /path/to/jsons --participants P003,P004
    # Skip the tracking sync and only convert BORIS files:
    python batch_sync.py --json-folder /path/to/jsons --boris-folder /path/to/boris --skip-sync
    python batch_sync.py --json-folder /path/to/jsons --boris-folder /path/to/boris --participants 3 --skip-sync
    # Note: participants may be numeric (e.g. '3') or full IDs (e.g. 'P003').
"""

import argparse
import sys
import traceback
from pathlib import Path

# Import the two pipeline modules (both must be alongside this script)
sys.path.insert(0, str(Path(__file__).parent))
import sync_to_pen_time  # noqa: E402
import convert_boris_to_synced_time as convert_boris  # noqa: E402


def json_stem(json_path: Path) -> str:
    """Stem of a master JSON, stripping '.csv.json' or '.json'."""
    name = json_path.name
    if name.endswith(".csv.json"):
        return name[:-len(".csv.json")]
    if name.endswith(".json"):
        return name[:-len(".json")]
    return json_path.stem


def participant_of(stem: str) -> str:
    """First '_'-separated token of a stem, e.g. 'P003' from 'P003_Short_...'."""
    return stem.split("_", 1)[0]


def build_json_index(json_folder: Path) -> dict:
    """Map JSON stem -> JSON path for all JSONs in the folder."""
    index = {}
    for p in json_folder.iterdir():
        if p.suffix.lower() == ".json":
            index[json_stem(p)] = p
    return index


def list_boris_files(boris_folder: Path) -> list:
    """All .tsv/.csv files in the BORIS folder."""
    if boris_folder is None:
        return []
    return sorted(p for p in boris_folder.iterdir()
                  if p.suffix.lower() in (".tsv", ".csv"))


def sync_one(json_path: Path, hand_dropout_ms: float,
             body_dropout_ms: float) -> dict:
    """Run the tracking sync for one JSON. Returns a status dict."""
    result = {"json": json_path.name, "sync_status": "pending", "sync_error": None}
    try:
        sync_to_pen_time.process(
            json_path,
            hand_dropout_threshold_ms=hand_dropout_ms,
            body_dropout_threshold_ms=body_dropout_ms,
        )
        result["sync_status"] = "ok"
    except Exception as e:
        result["sync_status"] = "failed"
        result["sync_error"] = str(e)
    return result


def convert_one_boris(boris_path: Path, json_index: dict) -> dict:
    """Match a BORIS file to its JSON (via embedded media name) and convert."""
    result = {
        "boris": boris_path.name,
        "boris_status": "pending",
        "boris_error": None,
        "matched_json": None,
        "media_stem": None,
    }
    # Derive the video/JSON stem from the media name inside the BORIS file
    try:
        media_stem = convert_boris.get_boris_media_stem(boris_path)
        result["media_stem"] = media_stem
    except Exception as e:
        result["boris_status"] = "failed_reading_media"
        result["boris_error"] = str(e)
        return result

    json_path = json_index.get(media_stem)
    if json_path is None:
        result["boris_status"] = "skipped_no_json_match"
        return result

    result["matched_json"] = json_path.name
    try:
        convert_boris.process(json_path, boris_path)
        result["boris_status"] = "ok"
    except Exception as e:
        result["boris_status"] = "failed"
        result["boris_error"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json-folder", type=Path, required=True,
                    help="Folder containing recording JSONs")
    ap.add_argument("--boris-folder", type=Path, default=None,
                    help="Folder containing BORIS exports (.tsv/.csv). "
                         "Optional - if omitted, only tracking sync runs.")
    ap.add_argument("--hand-dropout-ms", type=float, default=100.0,
                    help="Hand gap threshold for 'filled_dropout' tag "
                         "(default: 100ms)")
    ap.add_argument("--body-dropout-ms", type=float, default=200.0,
                    help="Body gap threshold for 'filled_dropout' tag "
                         "(default: 200ms)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be processed without doing it")
    ap.add_argument("--skip-sync", action="store_true",
                    help="Skip the tracking sync phase; only convert BORIS files")
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated participant IDs to restrict to "
                         "(e.g. 'P003,P004'). Matches the first '_'-token of "
                         "each file's stem. If omitted, all participants run.")
    args = ap.parse_args()

    if not args.json_folder.is_dir():
        sys.exit(f"ERROR: {args.json_folder} is not a directory")
    if args.boris_folder is not None and not args.boris_folder.is_dir():
        sys.exit(f"ERROR: {args.boris_folder} is not a directory")

    # Parse the participants filter if provided. Accept numeric IDs (e.g. 3)
    # or full IDs (e.g. P003). Accepts comma- or whitespace-separated tokens.
    participant_filter = None
    if args.participants:
        try:
            tokens = [t for part in args.participants.split(",") for t in part.split()]
            ids = []
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                if tok.upper().startswith("P"):
                    ids.append(tok.upper())
                else:
                    ids.append(f"P{int(tok):03d}")
            participant_filter = set(ids)
        except ValueError as e:
            sys.exit(f"ERROR: Invalid participant IDs: {e}")

    json_files = sorted(p for p in args.json_folder.iterdir()
                        if p.suffix.lower() == ".json")
    if not json_files:
        sys.exit(f"No .json files found in {args.json_folder}")

    # Apply participant filter if given (filter JSON files first)
    if participant_filter is not None:
        json_files = [j for j in json_files
                      if participant_of(json_stem(j)) in participant_filter]
        if not json_files:
            sys.exit("No JSON files match the participant filter; nothing to do.")

    # Build the JSON index from only the JSON files we're going to process
    # (this enforces per-participant filtering for BORIS matching too).
    json_index = {json_stem(p): p for p in json_files}

    boris_files = list_boris_files(args.boris_folder)
    # Filter BORIS files by participant if specified. If we can't read a
    # BORIS file's media stem, keep it so the error surfaces during phase 2.
    if participant_filter is not None and boris_files:
        filtered_boris = []
        for b in boris_files:
            try:
                ms = convert_boris.get_boris_media_stem(b)
            except Exception:
                filtered_boris.append(b)
                continue
            if participant_of(ms) in participant_filter:
                filtered_boris.append(b)
        boris_files = filtered_boris

    print(f"Found {len(json_files)} JSON file(s) in {args.json_folder}")
    if participant_filter is not None:
        print(f"  (filtered to participants: {', '.join(sorted(participant_filter))})")
    if args.boris_folder:
        print(f"Found {len(boris_files)} BORIS file(s) in {args.boris_folder}")
    else:
        print("No BORIS folder given - tracking sync only")
    print()

    if args.dry_run:
        print("DRY RUN - tracking sync would run on:")
        for j in json_files:
            print(f"  {j.name}")
        if boris_files:
            print()
            print("DRY RUN - BORIS files would match (by embedded media name):")
            for b in boris_files:
                try:
                    ms = convert_boris.get_boris_media_stem(b)
                except Exception as e:
                    print(f"  {b.name}  -> ERROR reading media: {e}")
                    continue
                jp = json_index.get(ms)
                if jp:
                    print(f"  {b.name}  -> {jp.name}  (media stem '{ms}')")
                else:
                    print(f"  {b.name}  -> NO JSON for media stem '{ms}'")
        return

    # Phase 1: sync every JSON (can be skipped)
    sync_results = []
    if args.skip_sync:
        print("=== Phase 1: tracking sync ===")
        print("SKIPPED (--skip-sync)")
    else:
        print("=== Phase 1: tracking sync ===")
        for i, j in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] {j.name}")
            try:
                r = sync_one(j, args.hand_dropout_ms, args.body_dropout_ms)
            except Exception as e:
                r = {"json": j.name, "sync_status": "crashed",
                     "sync_error": f"{type(e).__name__}: {e}"}
                print(f"   CRASH: {e}")
                traceback.print_exc()
            else:
                print(f"   sync: {r['sync_status']}"
                      + (f" ({r['sync_error']})" if r['sync_error'] else ""))
            sync_results.append(r)

    # Phase 2: convert every BORIS file, matched by embedded media name
    boris_results = []
    if boris_files:
        print()
        print("=== Phase 2: BORIS conversion (matched by media name) ===")
        for i, b in enumerate(boris_files, 1):
            print(f"[{i}/{len(boris_files)}] {b.name}")
            try:
                r = convert_one_boris(b, json_index)
            except Exception as e:
                r = {"boris": b.name, "boris_status": "crashed",
                     "boris_error": f"{type(e).__name__}: {e}",
                     "matched_json": None, "media_stem": None}
                print(f"   CRASH: {e}")
                traceback.print_exc()
            else:
                line = f"   boris: {r['boris_status']}"
                if r["matched_json"]:
                    line += f" -> {r['matched_json']}"
                elif r["media_stem"]:
                    line += f" (media stem '{r['media_stem']}')"
                if r["boris_error"]:
                    line += f" ({r['boris_error']})"
                print(line)
            boris_results.append(r)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import Counter
    sync_counts = Counter(r["sync_status"] for r in sync_results)
    print(f"JSONs: {len(sync_results)} | sync: {dict(sync_counts)}")
    if boris_results:
        boris_counts = Counter(r["boris_status"] for r in boris_results)
        print(f"BORIS files: {len(boris_results)} | {dict(boris_counts)}")

    # List failures explicitly
    sync_failures = [r for r in sync_results
                     if r["sync_status"] in ("failed", "crashed")]
    boris_failures = [r for r in boris_results
                      if r["boris_status"] not in ("ok",)]
    if sync_failures:
        print()
        print("Sync errors:")
        for r in sync_failures:
            print(f"  {r['json']}: {r.get('sync_error')}")
    if boris_failures:
        print()
        print("BORIS issues:")
        for r in boris_failures:
            detail = r.get("boris_error") or r["boris_status"]
            print(f"  {r['boris']}: {detail}"
                  + (f" (media stem '{r['media_stem']}')" if r.get("media_stem") else ""))


if __name__ == "__main__":
    main()
