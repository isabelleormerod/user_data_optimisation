"""Trial parameter parsing shared across pipeline scripts."""

import re

_TIMESTAMP_SUFFIX_RE = re.compile(r'_\d{8}_\d{6}$')


def strip_timestamp_suffix(stem: str) -> str:
    return _TIMESTAMP_SUFFIX_RE.sub("", stem)


def participant_of(stem: str) -> str:
    """First '_'-separated token, e.g. 'P003' from 'P003_Short_...'."""
    return stem.split("_", 1)[0]


def parse_params(trial: str) -> dict:
    """Extract Length, Size, Weight, Angle from a trial stem.

    Weight is two-token ('Front_weighted' / 'Not_weighted').
    Angle is encoded as A<digits> (stored as int).
    """
    out = {"Length": None, "Size": None, "Weight": None, "Angle": None}
    tokens = trial.split("_")
    joined = "_".join(tokens)

    if "Not_weighted" in joined:
        out["Weight"] = "Not_weighted"
    elif "Front_weighted" in joined:
        out["Weight"] = "Front_weighted"

    for tok in tokens:
        if tok and tok[0].upper() == "A" and tok[1:].isdigit():
            out["Angle"] = int(tok[1:])
            break

    for tok in tokens:
        if tok in ("Long", "Short"):
            out["Length"] = tok
        elif tok in ("Large", "Small"):
            out["Size"] = tok

    return out


def parse_participant_filter(participants_str) -> set:
    """Parse a comma-separated --participants argument. Returns a set of PID
    strings (e.g. {'P003', 'P004'}), or None if participants_str is falsy."""
    if not participants_str:
        return None
    return {p.strip() for p in participants_str.split(",") if p.strip()}
