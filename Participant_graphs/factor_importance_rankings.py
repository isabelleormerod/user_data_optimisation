#!/usr/bin/env python3
"""
factor_importance_rankings.py

Participants' stated importance of the four prototype factors at each
workstation height, shown in the same format as the pipelines' factor-weight
figures (05_rank_prototypes.py, run_hierarchical_fpca.py): weights from 0 to 1,
grouped by workstation, one bar per factor, same factor colours.

Weights are Borda scores: each participant gives 3 points to the factor they
ranked 1st (most important), 2 to the 2nd, 1 to the 3rd and 0 to the 4th; a
factor's weight is its share of all points awarded at that height, so the four
weights sum to 1.

Kendall's W (agreement between participants) and n are given under each
workstation.

Input: factor_importance.tsv, in the layout of the ranking sheet:
    Participant Name | Factor | Low | Medium | High
with the participant's name on the first of their four rows only.
Participants without a complete, valid ranking (each of 1-4 used exactly once)
at a given height are excluded from that height and reported in the console.

Usage:  python factor_importance_rankings.py [path/to/factor_importance.tsv]
Outputs: factor_importance_rankings.pdf / .png, factor_importance_weights.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HEIGHTS = ["High", "Medium", "Low"]
FACTORS = ["Angle", "Handle Size", "End-effector Length", "Weight"]
SHORT = {"Angle": "Angle", "Handle Size": "Size", "End-effector Length": "Length", "Weight": "Weight"}
FACTOR_COLOURS = {"Angle": "#c8642a", "Size": "#3b6ea5", "Length": "#3e8e5e", "Weight": "#7d4e9e"}

# Participant IDs shown in the figure (names are never displayed). Names as
# they appear in the sheet. Assumed matches to the ranking sheet are marked.
IDS = {"Al-Azar": "P002",   # assumed = "Az" in the ranking sheet
       "Owen": "P003", "Rob": "P004",
       "Chris S": "P005",   # assumed = "Chris" in the ranking sheet
       "Adam": "P006", "Amanda": "P007", "Moggy": "P008", "Nathan": "P009",
       "Duncan": "P010", "Chris C": "P011"}


def load(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = ["name", "factor"] + [c.strip().capitalize() for c in df.columns[2:]]
    df["name"] = df["name"].replace(r"^\s*$", np.nan, regex=True).ffill().str.strip()
    df = df[df["factor"].notna() & (df["factor"].str.strip() != "")].copy()   # ignore blank rows
    df["factor"] = df["factor"].str.strip()
    for h in HEIGHTS:
        df[h] = pd.to_numeric(df[h], errors="coerce")
    unknown = sorted(set(df["name"]) - set(IDS))
    if unknown:
        sys.exit(f"Error: no participant ID for {unknown}; add them to IDS.")
    df["pid"] = df["name"].map(IDS)
    return df.drop(columns="name")


def valid_rankings(df):
    """{height: participants x factors DataFrame} of valid rankings only."""
    out = {}
    for h in HEIGHTS:
        rows = {}
        for pid, g in df.groupby("pid"):
            r = g.set_index("factor")[h].reindex(FACTORS)
            if r.isna().all():
                print(f"  [excluded] {pid} {h}: no rankings")
            elif r.isna().any() or sorted(r.astype(int)) != [1, 2, 3, 4]:
                print(f"  [excluded] {pid} {h}: ranks {r.tolist()} (each of 1-4 must be used once)")
            else:
                rows[pid] = r.astype(int)
        out[h] = pd.DataFrame(rows).T.sort_index()
    return out


def kendalls_w(R):
    """Kendall's W for an m raters x n items matrix of untied ranks."""
    m, n = R.shape
    S = np.sum((R.sum(axis=0) - m * (n + 1) / 2) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    return W, chi2, stats.chi2.sf(chi2, n - 1)


def borda_weights(R):
    """Share of Borda points per factor: rank 1 -> 3 points ... rank 4 -> 0."""
    points = (len(FACTORS) - R).sum(axis=0)
    return points / points.sum()


def main(path="factor_importance.tsv"):
    ranks = valid_rankings(load(path))
    rows, labels = [], []
    print(f"\n{'Height':<8} {'Factor':<22} {'Weight':>7} {'Mean rank':>10}")
    for h in HEIGHTS:
        R = ranks[h]
        W, chi2, p = kendalls_w(R.to_numpy(dtype=float))
        wts = borda_weights(R)
        for f in FACTORS:
            rows.append({"height": h, "factor": SHORT[f], "weight": float(wts[f]),
                         "mean_rank": float(R[f].mean()), "n": len(R), "kendall_w": W, "p": p})
            print(f"{h:<8} {f:<22} {wts[f]:>7.3f} {R[f].mean():>10.2f}")
        print(f"{h:<8} n = {len(R)}, Kendall's W = {W:.3f}, chi2({len(FACTORS)-1}) = {chi2:.2f}, p = {p:.4f}\n")
        labels.append(f"{h} workstation\n$n$ = {len(R)}, $W$ = {W:.2f} ($p$ = {p:.3f})")
    T = pd.DataFrame(rows)
    T.to_csv("factor_importance_weights.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x, w = np.arange(len(HEIGHTS)), 0.2
    order = [SHORT[f] for f in FACTORS]
    for i, f in enumerate(order):
        vals = [T[(T.height == h) & (T.factor == f)]["weight"].iloc[0] for h in HEIGHTS]
        xs = x + (i - (len(order) - 1) / 2) * w
        ax.bar(xs, vals, w, color=FACTOR_COLOURS[f], edgecolor="white", linewidth=0.8, label=f)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Factor weight")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Parameter", frameon=False, ncol=len(order), loc="upper right",
              fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig("factor_importance_rankings.pdf", bbox_inches="tight")
    fig.savefig("factor_importance_rankings.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:])