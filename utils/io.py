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
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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
