#!/usr/bin/env python3
r"""
compare_all.py -- Stage 1 and Stage 2 results from all three sources together.
Self-contained: reads the two pipelines' outputs and the two participant sheets.

STAGE 1 (which parameters matter?)
  stage1_weights.pdf/.png   for each parameter, the MCDA weight, the fPCA-LDA weight
                            and participants' stated importance (Borda share), each
                            averaged over the three workstation heights; the thin
                            line on each bar spans the three per-height values
  stage1_weights_by_height.pdf/.png  the same, one panel per height (appendix)
  stage1_weights.csv / .tex per-height values plus the means (table includes Kendall's W)
  Participant weights are Borda shares (rank 1 -> 3 points ... rank 4 -> 0, as a
  share of all points): even unanimous agreement gives at most 0.50, so compare the
  ORDER of parameters across sources, not absolute heights.

STAGE 2 (which setting of each parameter?)
  stage2_preferences.pdf/.png  participants' revealed preferences (from their
                               prototype rankings) with both pipelines' consistency
                               choices alongside (hollow marker = the pipeline's own
                               consistency test not significant)
  participant_pipeline_agreement.tex, revealed_preferences_tests.csv
  stage2_rankings.csv / .tex   per height: each pipeline's top-ranked configuration
                               and the Spearman correlation between the two
                               pipelines' rankings of all configurations

  Revealed preference (size, length, weight) = mean rank of the participant's
  prototypes with the first setting minus that of the second (negative = first
  setting ranked better); angle = each angle's mean rank minus the participant's
  average rank. Group preference: exact sign-flip test (two settings), Friedman
  test (angle); p unadjusted, Benjamini-Hochberg values in the tests CSV.

INPUTS
  --landmarks-root   -> <root>/metrics/prototype_rankings (MCDA) and
                        <root>/metrics/fpca_results (fPCA-LDA); or give
  --mcda-dir / --fpca-dir directly.
  --importance-sheet factor_importance.tsv (Participant Name | Factor | Low | Medium | High,
                     name on the first of each participant's four rows)
  --rankings-sheet   rankings.tsv (the prototype ranking sheet)
  Sheets default to being searched for in this script's folder, the current folder,
  and anywhere below this script's folder.

Usage:
  python compare_all.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
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

HERE = Path(__file__).resolve().parent
HEIGHTS = ["High", "Medium", "Low"]
PARAMS = ["Angle", "Size", "Length", "Weight"]

# ---- participant importance sheet ----
SHEET_FACTORS = ["Angle", "Handle Size", "End-effector Length", "Weight"]
SHEET_NAME = {"Angle": "Angle", "Handle Size": "Size", "End-effector Length": "Length", "Weight": "Weight"}
# Participant IDs (names are never displayed). Assumed matches to the ranking sheet are marked.
IDS = {"Al-Azar": "P002",   # assumed = "Az" in the ranking sheet
       "Owen": "P003", "Rob": "P004",
       "Chris S": "P005",   # assumed = "Chris" in the ranking sheet
       "Adam": "P006", "Amanda": "P007", "Moggy": "P008", "Nathan": "P009",
       "Duncan": "P010", "Chris C": "P011"}

# ---- prototype rankings / Stage 2 ----
BINARY = {"Size": ("Small", "Large"), "Length": ("Short", "Long"), "Weight": ("Not_weighted", "Front_weighted")}
ANGLES = ["A90", "A135", "A180"]
LEVEL_LABEL = {"Not_weighted": "Not weighted", "Front_weighted": "Front weighted"}
TEX_LEVEL = {"Small": "Small", "Large": "Large", "Short": "Short", "Long": "Long",
             "Not_weighted": "Not", "Front_weighted": "Front",
             "A90": "90\\textdegree{}", "A135": "135\\textdegree{}", "A180": "180\\textdegree{}"}
MAJORITY = {"Not_weighted": "Not weighted", "Front_weighted": "Front weighted"}

# ---- shared styling ----
SOURCES = ["MCDA", "fPCA\u2013LDA", "Participants"]
SOURCE_COLOURS = {"MCDA": "#2471a3", "fPCA\u2013LDA": "#7d3c98", "Participants": "#9a9a9a"}
PIPE_STYLE = {"MCDA consistency": ("s", "#2471a3"), "fPCA\u2013LDA consistency": ("D", "#7d3c98")}


# =================================================================== file finding
def find_file(fname, explicit=None):
    if explicit is not None:
        if Path(explicit).is_file():
            return Path(explicit)
        raise FileNotFoundError(f"{explicit} not found.")
    for d in (HERE, Path.cwd()):
        if (d / fname).is_file():
            return d / fname
    hits = sorted(HERE.rglob(fname))
    if hits:
        if len(hits) > 1:
            print(f"  [note] found {len(hits)} copies of {fname}; using {hits[0]}")
        return hits[0]
    raise FileNotFoundError(f"Could not find {fname} in {HERE}, {Path.cwd()} or below {HERE}; "
                            f"pass its path with --importance-sheet / --rankings-sheet.")


# =================================================================== Stage 1: participants
def load_importance(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = ["name", "factor"] + [c.strip().capitalize() for c in df.columns[2:]]
    df["name"] = df["name"].replace(r"^\s*$", np.nan, regex=True).ffill().str.strip()
    df = df[df["factor"].notna() & (df["factor"].str.strip() != "")].copy()
    df["factor"] = df["factor"].str.strip()
    for h in HEIGHTS:
        df[h] = pd.to_numeric(df[h], errors="coerce")
    unknown = sorted(set(df["name"]) - set(IDS))
    if unknown:
        raise SystemExit(f"Error: no participant ID for {unknown}; add them to IDS.")
    df["pid"] = df["name"].map(IDS)
    return df.drop(columns="name")


def valid_importance(df):
    """{height: participants x factors table} of complete rankings (1-4 each used once)."""
    out = {}
    for h in HEIGHTS:
        rows = {}
        for pid, g in df.groupby("pid"):
            r = g.set_index("factor")[h].reindex(SHEET_FACTORS)
            if r.isna().all():
                print(f"  [importance excluded] {pid} {h}: no rankings")
            elif r.isna().any() or sorted(r.astype(int)) != [1, 2, 3, 4]:
                print(f"  [importance excluded] {pid} {h}: ranks {r.tolist()}")
            else:
                rows[pid] = r.astype(int)
        out[h] = pd.DataFrame(rows).T.sort_index()
    return out


def kendalls_w(R):
    m, n = R.shape
    S = np.sum((R.sum(axis=0) - m * (n + 1) / 2) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    return W, chi2, stats.chi2.sf(chi2, n - 1)


def borda_weights(R):
    points = (len(SHEET_FACTORS) - R).sum(axis=0)
    return points / points.sum()


# =================================================================== Stage 1: table + figure
def stage1_table(mcda_dir, fpca_dir, importance_sheet):
    rows = []
    for r in pd.read_csv(mcda_dir / "mcda_factor_weights.csv").itertuples():
        n_sig, n_met = int(getattr(r, "n_sig_fdr", 0)), int(getattr(r, "n_metrics", 0))
        rows.append({"height": r.height, "parameter": r.factor, "source": "MCDA", "weight": r.data_weight,
                     "significance": f"{n_sig}/{n_met} metrics significant after FDR",
                     "significant": n_sig > 0, "n_sig": n_sig, "n_metrics": n_met, "p": np.nan})
    for r in pd.read_csv(fpca_dir / "lda_factor_weights.csv").itertuples():
        p = getattr(r, "p_value_lda1", np.nan)
        rows.append({"height": r.height, "parameter": r.factor, "source": "fPCA\u2013LDA", "weight": r.data_weight,
                     "significance": f"LDA permutation p = {p:.3f}", "significant": bool(p == p and p < 0.05),
                     "n_sig": np.nan, "n_metrics": np.nan, "p": p})
    ranks = valid_importance(load_importance(importance_sheet))
    agreement = {}
    for h in HEIGHTS:
        R = ranks[h]
        W, _, p = kendalls_w(R.to_numpy(dtype=float))
        agreement[h] = (len(R), W, p)
        for f_sheet, w in borda_weights(R).items():
            rows.append({"height": h, "parameter": SHEET_NAME[f_sheet], "source": "Participants", "weight": float(w),
                         "significance": f"Kendall's W = {W:.3f}, p = {p:.3f}", "significant": bool(p < 0.05),
                         "n_sig": np.nan, "n_metrics": np.nan, "p": p})
    return pd.DataFrame(rows), agreement


def plot_stage1(T, agreement, out):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(1, len(HEIGHTS), figsize=(10.5, 3.5), sharey=True)
    x, w = np.arange(len(PARAMS)), 0.26
    for ax, h in zip(axes, HEIGHTS):
        for i, src in enumerate(SOURCES):
            vals = []
            for p in PARAMS:
                r = T[(T.height == h) & (T.parameter == p) & (T.source == src)]
                vals.append(float(r.weight.iloc[0]) if len(r) else np.nan)
            xs = x + (i - 1) * w
            ax.bar(xs, vals, w, color=SOURCE_COLOURS[src], edgecolor="white", linewidth=0.8, label=src)
            for xi, v in zip(xs, vals):
                if v == v:
                    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, rotation=90)
        _, W, p = agreement[h]
        ptxt = "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
        ax.set_title(f"{h} workstation\nparticipants: $W$ = {W:.2f} ({ptxt})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(PARAMS)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Parameter weight")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(out / f"stage1_weights_by_height.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stage1_average(T, out, show_range=True):
    """Main Stage 1 figure: each source's weight for each parameter, averaged over
    the three workstation heights (each source's averages still sum to 1). The thin
    line on each bar spans the lowest to highest of the three per-height values."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x, w = np.arange(len(PARAMS)), 0.26
    for i, src in enumerate(SOURCES):
        sub = T[(T.source == src) & (T.height.isin(HEIGHTS))].groupby("parameter")["weight"]
        mean, lo, hi = sub.mean().reindex(PARAMS), sub.min().reindex(PARAMS), sub.max().reindex(PARAMS)
        xs = x + (i - 1) * w
        ax.bar(xs, mean.values, w, color=SOURCE_COLOURS[src], edgecolor="white", linewidth=0.8, label=src)
        if show_range:
            ax.vlines(xs, lo.values, hi.values, color="0.25", linewidth=0.9, zorder=3)
        for xi, m, top in zip(xs, mean.values, (hi if show_range else mean).values):
            if m == m:
                ax.text(xi, top + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAMS)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Mean parameter weight")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"stage1_weights.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_stage1_tex(T, agreement, path):
    """Stage 1 table: each source's weight with its significance result alongside.
    MCDA: number of metrics the parameter significantly affected after FDR correction;
    fPCA-LDA: LDA permutation p-value (bold if < 0.05); participants: Kendall's W and
    its p-value for each height. A final block gives the means across heights."""
    def pfmt(p):
        if p != p:
            return "--"
        t = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        return rf"\textbf{{{t}}}" if p < 0.05 else t

    lines = [r"% Generated by compare_all.py -- rerun to update. Requires booktabs, multirow.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Parameter weights at each workstation height from the MCDA and fPCA--LDA "
             r"pipelines and from participants' stated importance (Borda share, maximum 0.50), with the "
             r"significance of each. MCDA: number of metrics significantly affected by the parameter after "
             r"false discovery rate correction. fPCA--LDA: permutation $p$-value for the separation of the "
             r"parameter's settings. Participants: Kendall's coefficient of concordance $W$ and its "
             r"$p$-value. Bold $p$-values are below 0.05.}",
             r"\label{tab:stage1_weights}",
             r"\begin{tabular}{@{}ll cc cc c@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{MCDA} & \multicolumn{2}{c}{fPCA--LDA} & Participants \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(l){7-7}",
             r"Height & Parameter & Weight & Sig.\ metrics & Weight & $p$ & Weight \\", r"\midrule"]
    for h in HEIGHTS:
        _, W, pw = agreement[h]
        for pi, prm in enumerate(PARAMS):
            g = lambda src: T[(T.height == h) & (T.parameter == prm) & (T.source == src)]
            m, l, q = g("MCDA"), g("fPCA\u2013LDA"), g("Participants")
            cells = [f"{m.weight.iloc[0]:.2f}" if len(m) else "--",
                     f"{int(m.n_sig.iloc[0])}/{int(m.n_metrics.iloc[0])}" if len(m) else "--",
                     f"{l.weight.iloc[0]:.2f}" if len(l) else "--",
                     pfmt(l.p.iloc[0]) if len(l) else "--",
                     f"{q.weight.iloc[0]:.2f}" if len(q) else "--"]
            label = rf"\multirow{{4}}{{*}}{{{h}}}" if pi == 0 else ""
            lines.append(f"{label} & {prm} & " + " & ".join(cells) + r" \\")
        lines.append(rf" & \multicolumn{{6}}{{r}}{{\footnotesize Participants: $W$ = {W:.2f}, $p$ = {pfmt(pw)}}} \\")
        lines.append(r"\midrule")
    for pi, prm in enumerate(PARAMS):
        cells = []
        for src in SOURCES:
            v = T[(T.parameter == prm) & (T.source == src) & (T.height.isin(HEIGHTS))]["weight"]
            cells.append(f"{v.mean():.2f}" if len(v) else "--")
        label = r"\multirow{4}{*}{Mean}" if pi == 0 else ""
        lines.append(f"{label} & {prm} & {cells[0]} & & {cells[1]} & & {cells[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =================================================================== Stage 2: participants
def load_rankings(path):
    """Ranking sheet: PID, consent, name, stature, elbow, overhead, trial, Length, Size,
    Weight, Angle, DrillID, Recorded, Rank High, Rank Medium, Rank Low (name optional)."""
    cols = ["pid", "consent", "name", "stature", "elbow", "overhead", "trial", "Length",
            "Size", "Weight", "Angle", "drill_id", "recorded", "High", "Medium", "Low"]
    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.shape[1] == len(cols) - 1:
        cols = [c for c in cols if c != "name"]
    df = raw.set_axis(cols, axis=1).drop(columns="name", errors="ignore")
    df["Weight"] = df["Weight"].str.strip().str.replace(" ", "_")
    for h in HEIGHTS:
        df[h] = pd.to_numeric(df[h], errors="coerce")
    return df


def valid_rank_sets(df):
    keep = set()
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if sorted(g[h].dropna().astype(int)) == list(range(1, len(g) + 1)):
                keep.add((pid, h))
            elif g[h].notna().any():
                print(f"  [rankings excluded] {pid} {h}: ranks {sorted(g[h].dropna().astype(int))}")
            else:
                print(f"  [rankings excluded] {pid} {h}: no rankings")
        for f in BINARY:
            vc = g[f].value_counts()
            if len(vc) != 2 or vc.nunique() != 1:
                print(f"  [rankings warning] {pid} {f} unbalanced: {vc.to_dict()}")
    return keep


def preferences(df, keep):
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) in keep:
                for f, (a, b) in BINARY.items():
                    rows.append({"pid": pid, "height": h, "factor": f,
                                 "diff": g.loc[g[f] == a, h].mean() - g.loc[g[f] == b, h].mean()})
    return pd.DataFrame(rows)


def angle_deviations(df, keep):
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) in keep:
                avg = g[h].mean()
                for ang, m in g.groupby("Angle")[h].mean().items():
                    rows.append({"pid": pid, "height": h, "angle": ang, "dev": m - avg})
    return pd.DataFrame(rows)


def angle_preferences(df, keep):
    rows = []
    for pid, g in df.groupby("pid"):
        for h in HEIGHTS:
            if (pid, h) in keep:
                m = g.groupby("Angle")[h].mean()
                best = m[m == m.min()].index
                rows.append({"pid": pid, "height": h, "best": best[0] if len(best) == 1 else None})
    return pd.DataFrame(rows)


def sign_flip_p(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


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
    for f in BINARY:
        for h in HEIGHTS:
            d = pref[(pref.factor == f) & (pref.height == h)]["diff"]
            rows.append({"factor": f, "height": h, "test": "sign-flip (exact)", "n": len(d),
                         "mean_preference": float(d.mean()), "p": sign_flip_p(d)})
    for h in HEIGHTS:
        tab = angd[angd.height == h].pivot(index="pid", columns="angle", values="dev").reindex(columns=ANGLES).dropna()
        p = float(stats.friedmanchisquare(*[tab[a] for a in ANGLES]).pvalue) if len(tab) >= 3 else np.nan
        rows.append({"factor": "Angle", "height": h, "test": "Friedman", "n": len(tab),
                     "mean_preference": np.nan, "p": p})
    T = pd.DataFrame(rows)
    T["p_fdr"] = bh_adjust(T["p"])
    return T


# =================================================================== Stage 2: pipelines
def _canon(factor, lev):
    s = str(lev).strip()
    if factor == "Angle":
        try:
            return f"A{int(float(s.lstrip('Aa')))}"
        except ValueError:
            return s
    return s


def load_pipelines(mcda_dir, fpca_dir):
    """{name: {"choice": {factor: {height: level}}, "sig": {factor: {height: bool}}}}"""
    facs = list(BINARY) + ["Angle"]
    pipes = {}
    t = pd.read_csv(mcda_dir / "mcda_consistency_by_height.csv")
    b = t[t["is_best"].astype(bool)]
    pipes["MCDA consistency"] = {
        "choice": {f: {r.height: _canon(f, r.level) for r in b[b.factor == f].itertuples()} for f in facs},
        "sig": {f: {r.height: bool(r.significant) for r in b[b.factor == f].itertuples()} for f in facs}}
    t = pd.read_csv(fpca_dir / "recommended_config_by_height.csv").dropna(subset=["dispersion_full"])
    b = t.loc[t.groupby(["height", "factor"])["dispersion_full"].idxmin()]
    pipes["fPCA\u2013LDA consistency"] = {
        "choice": {f: {r.height: _canon(f, r.level) for r in b[b.factor == f].itertuples()} for f in facs},
        "sig": {f: {r.height: bool(r.significant) for r in b[b.factor == f].itertuples()} for f in facs}}
    return pipes


# =================================================================== Stage 2: table + figure
def _pfmt(p):
    return "--" if p != p else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write_agreement_tex(pref, angp, tests, pipes, path):
    names = list(pipes)
    lines, matches = [], {n: 0 for n in names}
    for f, (a, b) in list(BINARY.items()) + [("Angle", (None, None))]:
        lines.append("\\midrule")
        for i, h in enumerate(HEIGHTS):
            p = tests[(tests.factor == f) & (tests.height == h)]["p"].iloc[0]
            if f == "Angle":
                sub = angp[angp.height == h]
                counts = sub["best"].value_counts()
                top = counts[counts == counts.max()].index
                maj = top[0] if len(top) == 1 else None
                cell = f"{TEX_LEVEL[maj]} ({counts.max()}/{len(sub)})" if maj else f"Split (n={len(sub)})"
            else:
                d = pref[(pref.factor == f) & (pref.height == h)]["diff"]
                n, na, nb = len(d), int((d < 0).sum()), int((d > 0).sum())
                maj = a if na > nb else (b if nb > na else None)
                cell = (f"{MAJORITY.get(maj, maj)} ({na if maj == a else nb}/{n})" if maj
                        else f"Split ({na}/{n} each)")
            ptxt = _pfmt(p)
            if p == p and p < 0.05:
                ptxt = f"\\textbf{{{ptxt}}}"
            cells = []
            for nm in names:
                c = pipes[nm]["choice"].get(f, {}).get(h)
                hit = maj is not None and c == maj
                matches[nm] += hit
                txt = TEX_LEVEL.get(c, "--") if c else "--"
                cells.append(f"\\textbf{{{txt}}}" if hit else txt)
            label = f"\\multirow{{3}}{{*}}{{{f}}}" if i == 0 else ""
            lines.append(f"{label} & {h} & {cell} & {ptxt} & " + " & ".join(cells) + " \\\\")
    k = len(names)
    head = " & ".join(nm.replace("\u2013", "--") for nm in names)
    tex = (r"""% Generated by compare_all.py -- rerun to update. Requires booktabs, multirow.
\begin{table}[htbp]
\centering
\small
\caption{Participants' preferred setting of each parameter, derived from their
prototype rankings, against the setting selected by each pipeline. Counts show how
many participants preferred that setting out of those with valid rankings at that
height (ties excluded). $p$ tests whether the group showed a preference (exact
sign-flip test; Friedman test for angle), in bold where $p < 0.05$, unadjusted.
\textbf{Bold} pipeline choices match the participant majority.}
\label{tab:participant_pipeline_agreement}
\begin{tabular}{@{}lll c """ + "c" * k + r"""@{}}
\toprule
 & & & & \multicolumn{""" + str(k) + r"""}{c}{Pipeline choice} \\
\cmidrule(l){5-""" + str(4 + k) + r"""}
Parameter & Height & Participants & $p$ & """ + head + r""" \\
\midrule
""" + "\n".join(lines[1:]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    path.write_text(tex, encoding="utf-8")
    print("  Matches with participant majority (of 12):",
          ", ".join(f"{nm} {matches[nm]}" for nm in names))


def _ptxt(p):
    if p != p:
        return ""
    return ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") + (" *" if p < 0.05 else "")


def _marker(ax, x, y, name, sig):
    mk, col = PIPE_STYLE[name]
    if sig is False:
        ax.scatter(x, y, marker=mk, s=30, facecolors="white", edgecolors=col, linewidths=1.2, zorder=5)
    else:
        ax.scatter(x, y, marker=mk, s=30, color=col, zorder=5, linewidths=0)


def plot_stage2(df, keep, pref, tests, pipes, out):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_ang = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(0)
    xlim, band = 6.3, 4.6
    names = list(pipes)
    for ax, (f, (a, b)) in zip(axes, BINARY.items()):
        for side in (-1, 1):
            ax.axvspan(side * band, side * xlim, color="0.94", zorder=0, linewidth=0)
        for yi, h in enumerate(HEIGHTS):
            d = pref[(pref.factor == f) & (pref.height == h)]["diff"].values
            ax.scatter(d, yi + rng.uniform(-0.1, 0.1, len(d)), s=22, color="0.55", alpha=0.8,
                       zorder=3, linewidths=0)
            ax.scatter(d.mean(), yi, marker="|", s=320, color="black", linewidths=2.2, zorder=4)
            p = tests[(tests.factor == f) & (tests.height == h)]["p"].iloc[0]
            sig = p == p and p < 0.05
            ax.text(band - 0.15, yi - 0.3, _ptxt(p), ha="right", va="center", fontsize=7.5,
                    color="black" if sig else "0.45", fontweight="bold" if sig else "normal")
            for k, nm in enumerate(names):
                c = pipes[nm]["choice"].get(f, {}).get(h)
                if c is not None:
                    off = band + 0.45 + k * 0.5
                    _marker(ax, -off if c == a else off, yi, nm, pipes[nm]["sig"].get(f, {}).get(h))
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
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    angd = angle_deviations(df, keep)
    ang_right = 8.8
    ytick, ylab, y = [], [], 0.0
    for h in HEIGHTS:
        for ang in ANGLES:
            d = angd[(angd.height == h) & (angd.angle == ang)]["dev"].values
            ax_ang.scatter(d, y + rng.uniform(-0.12, 0.12, len(d)), s=18, color="0.55", alpha=0.8,
                           zorder=3, linewidths=0)
            if len(d):
                ax_ang.scatter(d.mean(), y, marker="|", s=170, color="black", linewidths=2, zorder=4)
            for k, nm in enumerate(names):
                if pipes[nm]["choice"].get("Angle", {}).get(h) == ang:
                    _marker(ax_ang, -(band + 0.45 + k * 0.5), y, nm, pipes[nm]["sig"].get("Angle", {}).get(h))
            ytick.append(y); ylab.append(ang[1:] + "\u00b0"); y += 1
        p = tests[(tests.factor == "Angle") & (tests.height == h)]["p"].iloc[0]
        sig = p == p and p < 0.05
        ax_ang.text(ang_right - 0.1, y - 2, f"{h}\n{_ptxt(p)}", ha="right", va="center", fontsize=7.5,
                    style="italic", color="black" if sig else "0.4", fontweight="bold" if sig else "normal")
        if h != HEIGHTS[-1]:
            ax_ang.axhline(y - 0.25, color="0.8", linewidth=0.8)
        y += 0.5
    ax_ang.axvspan(-xlim, -band, color="0.94", zorder=0, linewidth=0)
    ax_ang.axvline(0, color="0.75", linewidth=0.9, zorder=1)
    ax_ang.set_xlim(-xlim, ang_right)
    ax_ang.set_xticks([-4, -2, 0, 2, 4])
    ax_ang.set_ylim(y - 0.5, -0.6)
    ax_ang.set_yticks(ytick)
    ax_ang.set_yticklabels(ylab, fontsize=8)
    ax_ang.tick_params(axis="y", length=0)
    ax_ang.set_title("Angle")
    ax_ang.set_xlabel("\u2190 ranked better      ranked worse \u2192\n(relative to participant's average)",
                      fontsize=8.5)
    ax_ang.grid(axis="x", alpha=0.2)
    for s in ("top", "right"):
        ax_ang.spines[s].set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", color="0.55", label="Participant"),
               Line2D([0], [0], marker="|", ls="", color="black", markersize=12, markeredgewidth=2.2,
                      label="Participant mean")]
    handles += [Line2D([0], [0], marker=PIPE_STYLE[n][0], ls="", color=PIPE_STYLE[n][1], label=f"{n} choice")
                for n in names]
    if any(v is False for nm in names for f in pipes[nm]["sig"].values() for v in f.values()):
        handles.append(Line2D([0], [0], marker="s", ls="", markerfacecolor="white", markeredgecolor="0.3",
                              label="Hollow: pipeline difference not significant"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.995, 0.005, "* p < 0.05 (exact sign-flip test; Friedman test for angle; unadjusted).",
             ha="right", fontsize=7, color="0.35")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.24)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"stage2_preferences.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =================================================================== Stage 2: pipeline rankings
def stage2_rankings(mcda_dir, fpca_dir):
    files = {"MCDA": mcda_dir / "mcda_weighted_config_ranking.csv",
             "fPCA-LDA": fpca_dir / "lda_weighted_config_ranking.csv"}
    missing = [f"{k}: {p}" for k, p in files.items() if not p.is_file()]
    if missing:
        print("\n  [skipped] weighted-ranking comparison -- file(s) not found:\n      " + "\n      ".join(missing) +
              "\n  These are written by the latest 05_rank_prototypes.py and run_hierarchical_fpca.py.")
        return pd.DataFrame()
    a, b = pd.read_csv(files["MCDA"]), pd.read_csv(files["fPCA-LDA"])
    rows = []
    for h in HEIGHTS:
        ah, bh = a[a.height == h], b[b.height == h]
        if ah.empty or bh.empty:
            continue
        m = ah[["config", "score"]].merge(bh[["config", "score"]], on="config", suffixes=("_mcda", "_lda"))
        rho, p = stats.spearmanr(m.score_mcda, m.score_lda)
        rows.append({"height": h, "top_mcda": ah.sort_values("rank").config.iloc[0],
                     "top_lda": bh.sort_values("rank").config.iloc[0],
                     "n_configs": len(m), "spearman_rho": rho, "p": p})
    return pd.DataFrame(rows)


def _cfg_words(cfg):
    parts = cfg.split("_")
    length, size, angle, weight = parts[0], parts[1], parts[-1], "_".join(parts[2:-1])
    weight_words = {"Not_weighted": "Non-weighted", "Front_weighted": "Weighted"}
    return (rf"\shortstack{{{length}, {size}\\{weight_words.get(weight, weight)}, "
            rf"{angle[1:]}\textdegree{{}}}}")


def write_rankings_tex(R, path):
    lines = [r"% Generated by compare_all.py -- rerun to update. Requires booktabs.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Top-ranked configuration at each workstation height from each pipeline's "
             r"weighted consistency ranking, and the agreement between the two pipelines' rankings of "
             r"all configurations (Spearman's $\rho$).}",
             r"\label{tab:stage2_rankings}",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Height & MCDA & fPCA--LDA & $\rho$ ($p$) \\", r"\midrule"]
    for r in R.itertuples():
        lines.append(f"{r.height} & {_cfg_words(r.top_mcda)} & {_cfg_words(r.top_lda)} & "
                     f"${r.spearman_rho:+.2f}$ ({_pfmt(r.p)}) \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--mcda-dir", type=Path, default=None,
                    help="05_rank_prototypes.py outputs (default <root>/metrics/prototype_rankings)")
    ap.add_argument("--fpca-dir", type=Path, default=None,
                    help="run_hierarchical_fpca.py outputs (default <root>/metrics/fpca_results)")
    ap.add_argument("--importance-sheet", type=Path, default=None)
    ap.add_argument("--rankings-sheet", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "comparison")
    a = ap.parse_args()
    if a.landmarks_root is None and (a.mcda_dir is None or a.fpca_dir is None):
        raise SystemExit("Error: give --landmarks-root, or both --mcda-dir and --fpca-dir.")
    mcda_dir = a.mcda_dir or (a.landmarks_root / "metrics" / "prototype_rankings")
    fpca_dir = a.fpca_dir or (a.landmarks_root / "metrics" / "fpca_results")
    importance_sheet = find_file("factor_importance.tsv", a.importance_sheet)
    rankings_sheet = find_file("rankings.tsv", a.rankings_sheet)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"Using: {mcda_dir}\n       {fpca_dir}\n       {importance_sheet}\n       {rankings_sheet}\n")

    # ---- Stage 1 ----
    T, agreement = stage1_table(mcda_dir, fpca_dir, importance_sheet)
    means = (T.groupby(["parameter", "source"], as_index=False)["weight"].mean().assign(height="Mean"))
    pd.concat([T, means], ignore_index=True).to_csv(a.out / "stage1_weights.csv", index=False)
    plot_stage1_average(T, a.out)
    plot_stage1(T, agreement, a.out)
    write_stage1_tex(T, agreement, a.out / "stage1_weights.tex")
    print("\nSTAGE 1 -- parameter weights")
    print(pd.concat([T, means]).pivot_table(index=["height", "parameter"], columns="source", values="weight")
          .reindex(columns=SOURCES).loc[HEIGHTS + ["Mean"]].round(2).to_string())
    for h in HEIGHTS:
        n, W, p = agreement[h]
        print(f"  {h:<7} participants n = {n}, Kendall's W = {W:.2f}, p = {p:.3f}")

    # ---- Stage 2 ----
    df = load_rankings(rankings_sheet)
    keep = valid_rank_sets(df)
    pref = preferences(df, keep)
    tests = run_tests(pref, angle_deviations(df, keep))
    tests.to_csv(a.out / "revealed_preferences_tests.csv", index=False)
    pipes = load_pipelines(mcda_dir, fpca_dir)
    print("\nSTAGE 2 -- participant preferences against pipeline choices")
    write_agreement_tex(pref, angle_preferences(df, keep), tests, pipes, a.out / "participant_pipeline_agreement.tex")
    plot_stage2(df, keep, pref, tests, pipes, a.out)
    R = stage2_rankings(mcda_dir, fpca_dir)
    if not R.empty:
        R.to_csv(a.out / "stage2_rankings.csv", index=False)
        write_rankings_tex(R, a.out / "stage2_rankings.tex")
        print("\nSTAGE 2 -- pipelines' weighted rankings")
        print(R.round(3).to_string(index=False))
    print(f"\nOutputs written to {a.out.resolve()}")


if __name__ == "__main__":
    main()