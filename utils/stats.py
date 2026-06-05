"""Statistical and model-output helpers shared across pipeline scripts."""

import re

import numpy as np


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values aligned to input."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return q
    ps = p[idx]
    order = np.argsort(ps)
    ranked = ps[order]
    n = len(ranked)
    qr = ranked * n / (np.arange(1, n + 1))
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.clip(qr, 0, 1)
    qfull = np.empty(n)
    qfull[order] = qr
    q[idx] = qfull
    return q


def zscore(s):
    """Z-score a pandas Series; returns None if SD is zero or NaN."""
    import pandas as pd
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return None
    return (s - s.mean()) / sd


def summarise(arr):
    """Return (mean, std) of arr, ignoring None and NaN values."""
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)],
                   dtype=float)
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)


def eta_squared_oneway(values, groups):
    """One-way eta^2 = SS_between / SS_total. Returns (eta2, p_anova)."""
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    if df["g"].nunique() < 2 or len(df) < 3:
        return np.nan, np.nan
    grand = df["y"].mean()
    ss_total = ((df["y"] - grand) ** 2).sum()
    if ss_total <= 0:
        return np.nan, np.nan
    ss_between = 0.0
    group_arrays = []
    for _, sub in df.groupby("g"):
        ss_between += len(sub) * (sub["y"].mean() - grand) ** 2
        group_arrays.append(sub["y"].values)
    eta2 = ss_between / ss_total
    try:
        _, p = stats.f_oneway(*group_arrays)
    except Exception:
        p = np.nan
    return float(eta2), float(p) if p == p else np.nan


def tidy_term_name(name: str) -> str:
    """Shorten statsmodels categorical term names.

    'C(Weight, Treatment(...))[T.Front_weighted]' -> 'Weight[Front_weighted]'
    """
    return re.sub(r"C\(([a-zA-Z0-9_]+)[^\)]*\)\[T\.([^\]]+)\]", r"\1[\2]", name)
