"""Trial-folder discovery helpers shared across pipeline scripts."""

from pathlib import Path


def find_labelled_pen(trial_dir: Path, stem: str):
    """Return the labelled pen CSV path for a trial, or None if absent.

    Preference order: flattened+labelled > labelled > glob fallback.
    """
    candidates = [
        trial_dir / f"{stem}_pen_flattened_labelled.csv",
        trial_dir / f"{stem}_pen_labelled.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    globbed = sorted(trial_dir.glob("*_pen*labelled*.csv"))
    return globbed[0] if globbed else None


def iter_trial_stems(landmarks_root: Path, participants=None):
    """Yield (stem, pid) for every trial folder that has both
    a *_pen.csv and a *_boris_synced.csv.

    Layout: <root>/<PID>/<stem>/
    """
    if not landmarks_root.is_dir():
        return
    for pid_dir in sorted(p for p in landmarks_root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pid == "metrics":
            continue
        if participants is not None and pid not in participants:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            stem = trial_dir.name
            if ((trial_dir / f"{stem}_pen.csv").is_file()
                    and (trial_dir / f"{stem}_boris_synced.csv").is_file()):
                yield stem, pid


def iter_trial_folders(root: Path, participants=None):
    """Yield trial folder Paths that contain a *_boris_synced.csv.

    Layout: <root>/<PID>/<trial>/
    """
    if not root.is_dir():
        return
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pid == "metrics":
            continue
        if participants is not None and pid not in participants:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            if list(trial_dir.glob("*_boris_synced.csv")):
                yield trial_dir


def iter_trials_labelled(root: Path, participants=None):
    """Yield (stem, pid, trial_dir) for every trial folder that has a
    labelled pen CSV (as found by find_labelled_pen).

    Layout: <root>/<PID>/<stem>/
    """
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pid == "metrics":
            continue
        if participants and pid not in participants:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            stem = trial_dir.name
            if find_labelled_pen(trial_dir, stem) is not None:
                yield stem, pid, trial_dir
