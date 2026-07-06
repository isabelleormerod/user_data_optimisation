"""Shared I/O helpers used across the pipeline scripts."""

import csv
import json
from pathlib import Path


def parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict:
    with open(path, "rb") as f:
        # Skip leading null bytes (OS pre-allocation artefact)
        chunk_size = 65536
        offset = 0
        start = None
        while start is None:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"{path.name}: file contains only null bytes")
            i = next((i for i, b in enumerate(chunk) if b != 0), None)
            if i is not None:
                start = offset + i
            offset += len(chunk)
        f.seek(start)
        raw = f.read()

    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return json.loads(text)


def read_table(path: Path) -> tuple:
    """Auto-detect tab vs comma delimiter. Returns (rows, fieldnames)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
    n_tab = first_line.count("\t")
    n_comma = first_line.count(",")
    if n_tab > 0 and n_tab >= n_comma:
        delim = "\t"
    elif n_comma > 0:
        delim = ","
    else:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_table(path: Path, rows: list, fieldnames: list) -> None:
    """Write a CSV with the given fieldnames."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
