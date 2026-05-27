#!/usr/bin/env python3
"""
Batch wrapper around sync_to_pen_time.py and convert_boris_to_synced_time.py.

For every JSON in --json-folder:
  1. Run the sync pipeline -> three CSVs + sidecar JSON next to the input
  2. If a matching BORIS file (by exact stem) exists in --boris-folder,
     run the BORIS converter -> _synced.csv next to the BORIS file

Files are matched by exact stem: <name>.json <-> <name>.tsv or <name>.csv.

Errors on individual files are logged and processing continues. A summary
of successes, skips, and failures is printed at the end.

Usage:
    python batch_sync.py --json-folder /path/to/jsons --boris-folder /path/to/boris
    python batch_sync.py --json-folder /path/to/jsons   # no BORIS conversion
    python batch_sync.py --json-folder /path/to/jsons --participants 1,2,5-7
    python batch_sync.py --json-folder /path/to/jsons --boris-folder /path/to/boris --participants 3
"""

import argparse
import sys
import traceback
from pathlib import Path
import re

# Import the two pipeline modules (both must be alongside this script)
sys.path.insert(0, str(Path(__file__).parent))
import sync_to_pen_time  # noqa: E402
import convert_boris_to_synced_time as convert_boris  # noqa: E402


def find_boris_match(json_path: Path, boris_folder: Path) -> Path | None:
    """Find matching .tsv or .csv in boris_folder by stripping timestamps and extensions."""
    if boris_folder is None:
        return None
        
    # 1. Chop off the timestamp (e.g., _20260330_113934) and anything after it
    clean_stem = re.sub(r"_\d{8}_\d{6}.*$", "", json_path.name)
    
    # 2. Fallback: if a file doesn't have a timestamp, just remove the extensions
    if clean_stem == json_path.name:
        clean_stem = json_path.name.split('.')
        
    # 3. Look for the exact match with the newly cleaned stem
    for ext in (".tsv", ".csv"):
        candidate = boris_folder / f"{clean_stem}{ext}"
        if candidate.is_file():
            return candidate
            
    return None


def parse_participants(spec: str) -> set[int]:
    """Parse a participant spec like '1,2,5-7' into a set of ints."""
    parts: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            parts.update(range(int(a), int(b) + 1))
        else:
            parts.add(int(token))
    return parts


def file_has_participant(json_path: Path, participants: set[int]) -> bool:
    """Return True if any digit-group in the file stem matches one of participants.

    This is flexible to cope with names like 'P001_session.json' or 'session_1.json'.
    """
    for m in re.findall(r"\d+", json_path.stem):
        try:
            if int(m) in participants:
                return True
        except ValueError:
            continue
    return False


def process_one(json_path: Path, boris_folder: Path | None,
                hand_dropout_ms: float, body_dropout_ms: float) -> dict:
    """Process one JSON file. Returns a result dict with status and details."""
    result = {
        "json": json_path.name,
        "sync_status": "pending",
        "boris_status": "pending",
        "sync_error": None,
        "boris_error": None,
        "boris_match": None,
    }

    # --- NEW PRE-CHECK ---
    # If a BORIS folder is provided, find the match BEFORE running the heavy sync
    boris_path = None
    if boris_folder is not None:
        boris_path = find_boris_match(json_path, boris_folder)
        if boris_path is None:
            # Abort processing completely for this file
            result["sync_status"] = "skipped_no_boris_match"
            result["boris_status"] = "skipped_no_match"
            return result
        result["boris_match"] = boris_path.name

    # Step 1: tracking sync (Only runs if boris_match was found, or if no boris_folder was provided)
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
        result["boris_status"] = "skipped_sync_failed"
        return result

    # Step 2: BORIS conversion
    if boris_folder is None:
        result["boris_status"] = "skipped_no_boris_folder"
        return result

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
    ap.add_argument("--participants", type=str, default=None,
                    help="Comma-separated participant numbers or ranges, e.g. '1,2,5-7'."
                        " Only JSONs whose filename contains a matching number are processed.")
    ap.add_argument("--hand-dropout-ms", type=float, default=100.0,
                    help="Hand gap threshold for 'filled_dropout' tag "
                         "(default: 100ms)")
    ap.add_argument("--body-dropout-ms", type=float, default=200.0,
                    help="Body gap threshold for 'filled_dropout' tag "
                         "(default: 200ms)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be processed without doing it")
    args = ap.parse_args()

    if not args.json_folder.is_dir():
        sys.exit(f"ERROR: {args.json_folder} is not a directory")
    if args.boris_folder is not None and not args.boris_folder.is_dir():
        sys.exit(f"ERROR: {args.boris_folder} is not a directory")

    json_files = sorted(p for p in args.json_folder.iterdir()
                        if p.suffix.lower() == ".json")
    if not json_files:
        sys.exit(f"No .json files found in {args.json_folder}")

    # If participant filtering specified, keep only matching JSONs.
    if args.participants:
        try:
            parts = parse_participants(args.participants)
        except Exception as e:
            sys.exit(f"ERROR parsing --participants: {e}")
        orig_count = len(json_files)
        json_files = [p for p in json_files if file_has_participant(p, parts)]
        if not json_files:
            sys.exit(f"No .json files matching participants {args.participants} in {args.json_folder}")
        print(f"Filtering for participants: {args.participants} ({len(json_files)} of {orig_count} files)")

    print(f"Found {len(json_files)} JSON file(s) in {args.json_folder}")
    if args.boris_folder:
        print(f"Looking for BORIS files in {args.boris_folder}")
    else:
        print("No BORIS folder given - tracking sync only")
    print()

    if args.dry_run:
        print("DRY RUN - listing what would be processed:")
        for j in json_files:
            match = find_boris_match(j, args.boris_folder)
            marker = f"-> {match.name}" if match else "(no BORIS match)"
            print(f"  {j.name}  {marker}")
        return

    results = []
    for i, j in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] {j.name}")
        try:
            r = process_one(j, args.boris_folder,
                           args.hand_dropout_ms, args.body_dropout_ms)
        except Exception as e:
            # Catch anything that bubbled up unexpectedly
            r = {
                "json": j.name,
                "sync_status": "crashed",
                "sync_error": f"{type(e).__name__}: {e}",
                "boris_status": "skipped_crash",
                "boris_error": None,
                "boris_match": None,
            }
            print(f"   CRASH: {e}")
            traceback.print_exc()
        else:
            print(f"   sync: {r['sync_status']}", end="")
            if r["sync_error"]:
                print(f" ({r['sync_error']})", end="")
            print()
            print(f"   boris: {r['boris_status']}", end="")
            if r["boris_match"]:
                print(f" [{r['boris_match']}]", end="")
            if r["boris_error"]:
                print(f" ({r['boris_error']})", end="")
            print()
        results.append(r)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import Counter
    sync_counts = Counter(r["sync_status"] for r in results)
    boris_counts = Counter(r["boris_status"] for r in results)
    print(f"Total files: {len(results)}")
    print(f"Sync: {dict(sync_counts)}")
    print(f"BORIS: {dict(boris_counts)}")

    # List failures explicitly
    failures = [r for r in results
                if r["sync_status"] in ("failed", "crashed")
                or r["boris_status"] == "failed"]
    if failures:
        print()
        print("Files with errors:")
        for r in failures:
            print(f"  {r['json']}")
            if r["sync_error"]:
                print(f"    sync: {r['sync_error']}")
            if r["boris_error"]:
                print(f"    boris: {r['boris_error']}")


if __name__ == "__main__":
    main()
