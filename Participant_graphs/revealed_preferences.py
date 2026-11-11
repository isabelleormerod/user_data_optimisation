#!/usr/bin/env python3
"""
revealed_preferences.py

Each participant's revealed preference for each prototype factor at each
workstation height, derived from their prototype rankings (rank 1 = most
preferred), shown against the level chosen by each analysis pipeline, with a
significance test of the group's preference.

Preference (Size, Length, Weight) = mean rank of the participant's prototypes
with the first level minus the mean rank of those with the second level (e.g.
Small - Large). Negative = the first level was ranked better.
Angle (three levels) is shown as each angle's mean rank relative to the
participant's average rank; it is indicative only, as participants ranked the
angles an unequal number of times (2, 3 and 3 prototypes).

SIGNIFICANCE (is there a group preference?):
  Size/Length/Weight: exact sign-flip permutation test. Under no preference,
    each participant's score is equally likely to be positive or negative, so
    every one of the 2^n sign combinations is equally likely; p = share of
    combinations whose |mean| is at least the observed |mean|. Exact and
    distribution-free, which suits n <= 10.
  Angle: Friedman test on each participant's mean rank of the three angles.
  p-values are unadjusted; Benjamini-Hochberg adjusted values (across all 12
  tests) are written to revealed_preferences_tests.csv.

PIPELINE CHOICES are read from the pipelines' own outputs:
  --mcda-consistency  mcda_consistency_by_height.csv   (05_rank_prototypes.py)
  --lda-consistency   recommended_config_by_height.csv (run_hierarchical_fpca.py)
  --mcda-heuristic    mcda_directional_by_height.csv   (05_rank_prototypes.py; optional,
                                                         for the appendix version)
Each pipeline's choice is its most consistent (or highest-scoring) level; a
hollow marker means the pipeline's own consistency test was not significant.
If no pipeline files are given, the values from the earlier chapter tables are
used, with a warning.

Input: rankings.tsv (the ranking sheet; participant names are dropped on load).
Invalid ranking sets (missing or duplicated ranks) are excluded automatically.
Outputs: revealed_preferences.pdf/.png, participant_pipeline_agreement.tex,
         revealed_preferences_tests.csv

Usage:
  python revealed_preferences.py \
      --mcda-consistency .../prototype_rankings/mcda_consistency_by_height.csv \
      --lda-consistency  .../fpca_results/recommended_config_by_height.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

HEIGHTS = ["High", "Medium", "Low"]
FACTORS = {"Size": ("Small", "Large"),
           "Length": ("Short", "Long"),
           "Weight": ("Not_weighted", "Front_weighted")}
ANGLES = ["A90", "A135", "A180"]
LEVEL_LABEL = {"Not_weighted": "Not weighted", "Front_weighted": "Front weighted"}
STYLE = {"MCDA consistency": ("s", "#2471a3"),
         "fPCA\u2013LDA consistency": ("D", "#7d3c98"),
         "MCDA heuristic": ("^", "#c0392b")}
SHORT = {"Small": "Small", "Large": "Large", "Short": "Short", "Long": "Long",
         "Not_weighted": "Not", "Front_weighted": "Front",
         "A90": "90\\textdegree{}", "A135": "135\\textdegree{}", "A180": "180\\textdegree{}"}
MAJ = {"Not_weighted": "Not weighted", "Front_weighted": "Front weighted"}

# Fallback only: values from the earlier chapter tables (pre-cleaning runs).
LEGACY = {
    "MCDA consistency": {
        "Size": {"High": "Small", "Medium": "Small", "Low": "Small"},
        "Length": {"High": "Short", "Medium": "Long", "Low": "Short"},
        "Weight": {"High": "Not_weighted", "Medium": "Not_weighted", "Low": "Not_weighted"},
        "Angle": {"High": "A90", "Medium": "A90", "Low": "A180"}},
    "fPCA\u2013LDA consistency": {
        "Size": {"High": "Large", "Medium": "Large", "Low": "Large"},
        "Length": {"High": "Long", "Medium": "Short", "Low": "Short"},
        "Weight": {"High": "Not_weighted", "Medium": "Front_weighted", "Low": "Front_weighted"},
        "Angle": {"High": "A90", "Medium": "A135", "Low": "A180"}},
}


# --------------------------------------------------------------------------- data
def load(path="rankings.tsv"):
    """Ranking sheet: PID, consent, name, stature, elbow, overhead, trial, Length,
    Size, Weight, Angle, DrillID, Recorded, Rank High, Rank Medium, Rank Low.
    Names are dropped on load; only participant IDs are used."""
    cols = ["pid", "consent", "name", "stature", "elbow", "overhead", "trial", "Length",
            "Size", "Weight", "Angle", "drill_id", "recorded", "High", "Medium", "Low"]
    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.shape[1] == len(cols) - 1:          # sheet already has the name column removed
        cols = [c for c in cols if c != "name"]
    df = raw.set_axis(cols, axis=1).drop(columns="name", errors="ignore")
    df["Weight"] = df["Weight"].str.strip().str.replace(" ", "_")
    for h in HEIGHTS:
        df[h] = pd.to_numeric(df[h], errors="coerce")
    return df


def valid_sets(df):
    """(pid, height) pairs whose rankings are a complete permutation of 1..8."""
    keep = set()
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if sorted(g[h].dropna().astype(int)) == list(range(1, len(g) + 1)):
                keep.add((pid, h))
            elif g[h].notna().any():
                print(f"  [excluded] {pid} {h}: ranks {sorted(g[h].dropna().astype(int))}")
            else:
                print(f"  [excluded] {pid} {h}: no rankings")
        for f in FACTORS:
            vc = g[f].value_counts()
            if len(vc) != 2 or vc.nunique() != 1:
                print(f"  [warning]  {pid} {f} unbalanced: {vc.to_dict()}")
    return keep


def preferences(df, keep):
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) not in keep:
                continue
            for f, (a, b) in FACTORS.items():
                rows.append({"pid": pid, "height": h, "factor": f,
                             "diff": g.loc[g[f] == a, h].mean() - g.loc[g[f] == b, h].mean()})
    return pd.DataFrame(rows)


def angle_deviations(df, keep):
    """Mean rank of each angle minus the participant's average rank (4.5)."""
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) not in keep:
                continue
            avg = g[h].mean()
            for ang, m in g.groupby("Angle")[h].mean().items():
                rows.append({"pid": pid, "height": h, "angle": ang, "dev": m - avg})
    return pd.DataFrame(rows)


def angle_preferences(df, keep):
    """Best-ranked angle per participant and height (None if tied)."""
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) not in keep:
                continue
            m = g.groupby("Angle")[h].mean()
            best = m[m == m.min()].index
            rows.append({"pid": pid, "height": h, "best": best[0] if len(best) == 1 else None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tests
def sign_flip_p(d):
    """Exact two-sided sign-flip permutation test of mean(d) = 0."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def friedman_p(angd, h):
    """Friedman test across the three angles (participants x angles mean-rank table)."""
    tab = angd[angd.height == h].pivot(index="pid", columns="angle", values="dev").reindex(columns=ANGLES).dropna()
    if len(tab) < 3:
        return np.nan, len(tab)
    return float(stats.friedmanchisquare(*[tab[a] for a in ANGLES]).pvalue), len(tab)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n:
        pv = p[ok]; o = np.argsort(pv)
        adj = np.minimum(np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1], 1.0)
        r = np.empty(n); r[o] = adj; out[ok] = r
    return out


def run_tests(pref, angd):
    rows = []
    for f in FACTORS:
        for h in HEIGHTS:
            d = pref[(pref.factor == f) & (pref.height == h)]["diff"]
            rows.append({"factor": f, "height": h, "test": "sign-flip (exact)", "n": len(d),
                         "mean_preference": float(d.mean()), "p": sign_flip_p(d)})
    for h in HEIGHTS:
        p, n = friedman_p(angd, h)
        rows.append({"factor": "Angle", "height": h, "test": "Friedman", "n": n,
                     "mean_preference": np.nan, "p": p})
    T = pd.DataFrame(rows)
    T["p_fdr"] = bh_adjust(T["p"])
    return T


# ------------------------------------------------------------------ pipeline choices
def _canon(factor, lev):
    s = str(lev).strip()
    if factor == "Angle":
        try:
            return f"A{int(float(s.lstrip('Aa')))}"
        except ValueError:
            return s
    return s


def load_pipelines(mcda_cons, lda_cons, mcda_heur):
    """{name: {"choice": {factor: {height: level}}, "sig": {factor: {height: bool|None}}}}"""
    pipes = {}
    if mcda_cons and Path(mcda_cons).is_file():
        t = pd.read_csv(mcda_cons)
        b = t[t["is_best"].astype(bool)]
        pipes["MCDA consistency"] = {
            "choice": {f: {r.height: _canon(f, r.level) for r in b[b.factor == f].itertuples()} for f in list(FACTORS) + ["Angle"]},
            "sig": {f: {r.height: bool(r.significant) for r in b[b.factor == f].itertuples()} for f in list(FACTORS) + ["Angle"]}}
    if lda_cons and Path(lda_cons).is_file():
        t = pd.read_csv(lda_cons).dropna(subset=["dispersion_full"])
        b = t.loc[t.groupby(["height", "factor"])["dispersion_full"].idxmin()]
        pipes["fPCA\u2013LDA consistency"] = {
            "choice": {f: {r.height: _canon(f, r.level) for r in b[b.factor == f].itertuples()} for f in list(FACTORS) + ["Angle"]},
            "sig": {f: {r.height: bool(r.significant) for r in b[b.factor == f].itertuples()} for f in list(FACTORS) + ["Angle"]}}
    if mcda_heur and Path(mcda_heur).is_file():
        t = pd.read_csv(mcda_heur)
        b = t[t["is_best"].astype(bool)]
        pipes["MCDA heuristic"] = {
            "choice": {f: {r.height: _canon(f, r.level) for r in b[b.factor == f].itertuples()} for f in list(FACTORS) + ["Angle"]},
            "sig": {f: {h: None for h in HEIGHTS} for f in list(FACTORS) + ["Angle"]}}
    if not pipes:
        print("  [WARN] no pipeline output files given -- using values from the earlier chapter "
              "tables, which predate the current analysis. Pass --mcda-consistency and "
              "--lda-consistency to use your current results.")
        pipes = {n: {"choice": c, "sig": {f: {h: None for h in HEIGHTS} for f in c}} for n, c in LEGACY.items()}
    return pipes


# ----------------------------------------------------------------------------- table
def _pfmt(p):
    return "--" if p != p else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write_table(pref, angp, tests, pipes, path="participant_pipeline_agreement.tex"):
    """Agreement table: participant majority and its p-value against each pipeline's
    choice. Bold = pipeline matches the participant majority."""
    names = list(pipes)
    lines, matches = [], {n: 0 for n in names}
    for f, (a, b) in list(FACTORS.items()) + [("Angle", (None, None))]:
        lines.append("\\midrule")
        for i, h in enumerate(HEIGHTS):
            p = tests[(tests.factor == f) & (tests.height == h)]["p"].iloc[0]
            if f == "Angle":
                sub = angp[angp.height == h]
                n = len(sub)
                counts = sub["best"].value_counts()
                top = counts[counts == counts.max()].index
                maj = top[0] if len(top) == 1 else None
                cell = (f"{SHORT[maj]} ({counts.max()}/{n})" if maj else f"Split (n={n})")
            else:
                d = pref[(pref.factor == f) & (pref.height == h)]["diff"]
                n, na, nb = len(d), int((d < 0).sum()), int((d > 0).sum())
                maj = a if na > nb else (b if nb > na else None)
                cnt = na if maj == a else nb
                cell = (f"{MAJ.get(maj, maj)} ({cnt}/{n})" if maj else f"Split ({na}/{n} each)")
            ptxt = _pfmt(p)
            if p == p and p < 0.05:
                ptxt = f"\\textbf{{{ptxt}}}"
            cells = []
            for nm in names:
                c = pipes[nm]["choice"].get(f, {}).get(h)
                hit = maj is not None and c == maj
                matches[nm] += hit
                txt = SHORT.get(c, "--") if c else "--"
                cells.append(f"\\textbf{{{txt}}}" if hit else txt)
            label = (f"\\multirow{{3}}{{*}}{{{f}" + ("$^{\\ddagger}$" if f == "Angle" else "") + "}"
                     if i == 0 else "")
            lines.append(f"{label} & {h} & {cell} & {ptxt} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines[1:])
    k = len(names)
    head2 = " & ".join(nm.replace("\u2013", "--") for nm in names)
    tex = (r"""% Generated by revealed_preferences.py -- rerun after editing rankings.tsv
% Requires: booktabs, multirow
\begin{table}[htbp]
\centering
\small
\caption{Participants' preferred level of each prototype factor, derived from their
prototype rankings, against the level selected by each analysis pipeline. Counts show
how many participants preferred that level out of those with valid rankings at that
height (ties excluded). $p$ tests whether the group showed a preference (exact
sign-flip test; Friedman test for angle), in bold where $p < 0.05$, unadjusted.
\textbf{Bold} pipeline choices match the participant majority.}
\label{tab:participant_pipeline_agreement}
\begin{tabular}{@{}lll c """ + "c" * k + r"""@{}}
\toprule
 & & & & \multicolumn{""" + str(k) + r"""}{c}{Pipeline choice} \\
\cmidrule(l){5-""" + str(4 + k) + r"""}
Factor & Height & Participants & $p$ & """ + head2 + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\par\vspace{1ex}
\footnotesize
$^{\ddagger}$Indicative only: participants ranked the three angles an unequal
number of times (2, 3 and 3 prototypes).
\end{table}
""")
    open(path, "w").write(tex)
    print("  Matches with participant majority (of 12):",
          ", ".join(f"{nm} {matches[nm]}" for nm in names))


# ---------------------------------------------------------------------------- figure
def _ptxt(p):
    if p != p:
        return ""
    s = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    return s + (" *" if p < 0.05 else "")


def draw_marker(ax, x, y, name, sig):
    mk, col = STYLE.get(name, ("o", "0.3"))
    if sig is False:
        ax.scatter(x, y, marker=mk, s=30, facecolors="white", edgecolors=col, linewidths=1.2, zorder=5)
    else:
        ax.scatter(x, y, marker=mk, s=30, color=col, zorder=5, linewidths=0)


def plot(df, keep, pref, tests, pipes, out_stem="revealed_preferences"):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_ang = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(0)
    xlim, band = 6.3, 4.6          # participant data lie within +/-4; pipelines sit in the outer band
    names = list(pipes)
    for ax, (f, (a, b)) in zip(axes, FACTORS.items()):
        for side in (-1, 1):
            ax.axvspan(side * band, side * xlim, color="0.94", zorder=0, linewidth=0)
        for yi, h in enumerate(HEIGHTS):
            d = pref[(pref.factor == f) & (pref.height == h)]["diff"].values
            ax.scatter(d, yi + rng.uniform(-0.1, 0.1, len(d)), s=22,
                       color="0.55", alpha=0.8, zorder=3, linewidths=0)
            ax.scatter(d.mean(), yi, marker="|", s=320, color="black", linewidths=2.2, zorder=4)
            p = tests[(tests.factor == f) & (tests.height == h)]["p"].iloc[0]
            ax.text(band - 0.15, yi - 0.3, _ptxt(p), ha="right", va="center", fontsize=7.5,
                    color="black" if (p == p and p < 0.05) else "0.45",
                    fontweight="bold" if (p == p and p < 0.05) else "normal")
            for k, nm in enumerate(names):
                c = pipes[nm]["choice"].get(f, {}).get(h)
                if c is None:
                    continue
                off = band + 0.45 + k * 0.5          # fixed slot per pipeline
                draw_marker(ax, -off if c == a else off, yi, nm, pipes[nm]["sig"].get(f, {}).get(h))
        ax.axvline(0, color="0.75", linewidth=0.9, zorder=1)
        ax.set_xlim(-xlim, xlim)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_ylim(len(HEIGHTS) - 0.5, -0.6)
        ax.set_yticks(range(len(HEIGHTS)))
        ax.set_yticklabels(HEIGHTS)
        ax.set_title(f)
        ax.set_xlabel(f"\u2190 prefers {LEVEL_LABEL.get(a, a)}      prefers {LEVEL_LABEL.get(b, b)} \u2192",
                      fontsize=8.5)
        ax.grid(axis="x", alpha=0.2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    # --- Angle panel: three rows per height, one per angle ---
    angd = angle_deviations(df, keep)
    ang_right = 8.8                # extra room on the right of the angle panel for the height labels
    ytick, ylab, y = [], [], 0.0
    for h in HEIGHTS:
        for ang in ANGLES:
            d = angd[(angd.height == h) & (angd.angle == ang)]["dev"].values
            ax_ang.scatter(d, y + rng.uniform(-0.12, 0.12, len(d)), s=18,
                           color="0.55", alpha=0.8, zorder=3, linewidths=0)
            if len(d):
                ax_ang.scatter(d.mean(), y, marker="|", s=170, color="black", linewidths=2, zorder=4)
            for k, nm in enumerate(names):
                if pipes[nm]["choice"].get("Angle", {}).get(h) == ang:
                    draw_marker(ax_ang, -(band + 0.45 + k * 0.5), y, nm,
                                pipes[nm]["sig"].get("Angle", {}).get(h))
            ytick.append(y); ylab.append(ang[1:] + "\u00b0"); y += 1
        p = tests[(tests.factor == "Angle") & (tests.height == h)]["p"].iloc[0]
        sig = p == p and p < 0.05
        ax_ang.text(ang_right - 0.1, y - 2, f"{h}\n{_ptxt(p)}", ha="right", va="center", fontsize=7.5,
                    style="italic", color="black" if sig else "0.4", fontweight="bold" if sig else "normal")
        if h != HEIGHTS[-1]:
            ax_ang.axhline(y - 0.5 + 0.25, color="0.8", linewidth=0.8)
        y += 0.5
    ax_ang.axvspan(-xlim, -band, color="0.94", zorder=0, linewidth=0)
    ax_ang.axvline(0, color="0.75", linewidth=0.9, zorder=1)
    ax_ang.set_xlim(-xlim, ang_right)
    ax_ang.set_xticks([-4, -2, 0, 2, 4])
    ax_ang.set_ylim(y - 0.5, -0.6)
    ax_ang.set_yticks(ytick)
    ax_ang.set_yticklabels(ylab, fontsize=8)
    ax_ang.tick_params(axis="y", length=0)
    ax_ang.set_title("Angle$^{\\ast}$")
    ax_ang.set_xlabel("\u2190 ranked better      ranked worse \u2192\n(relative to participant's average)",
                      fontsize=8.5)
    ax_ang.grid(axis="x", alpha=0.2)
    for s_ in ("top", "right"):
        ax_ang.spines[s_].set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", color="0.55", label="Participant"),
               Line2D([0], [0], marker="|", ls="", color="black", markersize=12,
                      markeredgewidth=2.2, label="Participant mean")]
    handles += [Line2D([0], [0], marker=STYLE[n][0], ls="", color=STYLE[n][1], label=f"{n} choice")
                for n in names]
    if any(v is False for nm in names for f in pipes[nm]["sig"].values() for v in f.values()):
        handles.append(Line2D([0], [0], marker="s", ls="", markerfacecolor="white",
                              markeredgecolor="0.3", label="Hollow: pipeline difference not significant"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.995, 0.005, "* p < 0.05 (exact sign-flip test; Friedman test for angle; unadjusted).   "
             "$^{\\ast}$Angle indicative: participants ranked the angles an unequal number of times.",
             ha="right", fontsize=7, color="0.35")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.24)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankings", default="rankings.tsv")
    ap.add_argument("--mcda-consistency", default=None, help="mcda_consistency_by_height.csv")
    ap.add_argument("--lda-consistency", default=None, help="fPCA-LDA recommended_config_by_height.csv")
    ap.add_argument("--mcda-heuristic", default=None,
                    help="mcda_directional_by_height.csv (optional; adds the appendix heuristic pipeline)")
    a = ap.parse_args()

    df = load(a.rankings)
    keep = valid_sets(df)
    pref = preferences(df, keep)
    angd = angle_deviations(df, keep)
    tests = run_tests(pref, angd)
    tests.to_csv("revealed_preferences_tests.csv", index=False)
    print("\nGroup preference tests (p unadjusted; p_fdr = Benjamini-Hochberg across all 12):")
    print(tests.round(3).to_string(index=False))
    pipes = load_pipelines(a.mcda_consistency, a.lda_consistency, a.mcda_heuristic)
    write_table(pref, angle_preferences(df, keep), tests, pipes)
    plot(df, keep, pref, tests, pipes)


if __name__ == "__main__":
    main()