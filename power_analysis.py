"""
power_analysis.py  --  Monte-Carlo power for the ergonomic-pipeline study (all-Python).

WHY THIS METHOD (not G*Power / pwr closed-form)
-----------------------------------------------
The design is a repeated-measures, mixed-effects, partially-incomplete factorial:
each participant runs a full 2^3 on length x aperture x weight (once each); angle
is balanced-incomplete across participants; working height (High/Mid/Low) is a
within-subject, per-participant-normalised factor (rig set to each person's crown/
hip height); ~5 replicate placements per (participant x config x height) cell, and
placements within a cell are correlated (shared participant) -> an ICC. No closed-
form power formula covers this; treating the ~1,200 placements as independent is
pseudoreplication and overstates power. Monte-Carlo power (what simr does in R,
reproduced here) simulates from the ACTUAL model on the ACTUAL allocation, so the
within/between structure and incomplete blocks are honoured exactly.

WHAT DRIVES POWER FOR A WITHIN-SUBJECT EFFECT
---------------------------------------------
Under a random-INTERCEPT-only model the mean effect saturates near power 1.0 with
1,200 trials -- optimistic, and not the real question. The real question: given the
effect may DIFFER across participants (random SLOPE, SD = tau), can we detect the
AVERAGE effect? For a balanced within-subject design the random-slope mixed model
is exactly equivalent to a one-sample t-test on each participant's own effect
estimate (the summary-statistics method), so that is used here -- fast, exact, and
easy to defend. Consequence: power is governed mainly by the number of
PARTICIPANTS; extra placements per cell only shrink each participant's measurement
error and show diminishing returns.

Effects are in residual-SD units (d = mean_difference / sd_resid) because the
metrics (RULA, aperture, path deviation) are not yet computed. Once they are, set
sd_resid to the observed within-cell SD to read power in real units, and estimate
tau and icc from a variance-components fit of the pilot data.

Run:  python power_analysis.py        Deps: numpy pandas scipy matplotlib
Reads: allocation.csv (anonymised design)   Writes: power_curves.png, power_results.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260715)
ALLOC = pd.read_csv("allocation.csv")
TEMPLATES = [g.reset_index(drop=True) for _, g in ALLOC.groupby("subject")]
HEIGHTS = ["High", "Medium", "Low"]


def build_design(n_participants, k_trials):
    """One row per placement. n>len(templates) replicates allocation blocks with
    fresh ids, exactly as simr::extend would extend along participants."""
    rows = []
    for i in range(n_participants):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        for _, cfg in tpl.iterrows():
            for h in HEIGHTS:
                rows += [(f"S{i:03d}", cfg["weight"])] * k_trials
    d = pd.DataFrame(rows, columns=["participant", "weight"])
    d["target_c"] = np.where(d["weight"] == "Front_weighted", 0.5, -0.5)
    return d


def power_within(N, k, d_eff, icc, tau, nsim=4000, sd_resid=1.0, alpha=0.05):
    """Power for the within-subject binary main effect via the summary-statistics
    test (== random-slope mixed model for this balanced design). Vectorised.
      icc -> participant random-INTERCEPT SD (baseline differences between people)
      tau -> participant random-SLOPE SD    (the effect itself differs by person)
    """
    d = build_design(N, k)
    cats = pd.Categorical(d["participant"]); pc = cats.codes; P = len(cats.categories)
    tg = d["target_c"].values; fr = tg > 0; nobs = len(d)
    Af = np.zeros((P, nobs)); An = np.zeros((P, nobs))            # per-ppt averaging ops
    for p in range(P):
        fi = np.where((pc == p) & fr)[0]; Af[p, fi] = 1.0 / len(fi)
        ni = np.where((pc == p) & ~fr)[0]; An[p, ni] = 1.0 / len(ni)
    sd_b0 = np.sqrt(icc / (1 - icc)) * sd_resid; delta = d_eff * sd_resid
    B0 = RNG.normal(0, sd_b0, (P, nsim)); B1 = RNG.normal(0, tau, (P, nsim))
    Y = (delta * tg[:, None] + B0[pc, :] + B1[pc, :] * tg[:, None]
         + RNG.normal(0, sd_resid, (nobs, nsim)))
    eff = Af @ Y - An @ Y                                          # P x nsim per-ppt effect
    tstat = eff.mean(0) / (eff.std(0, ddof=1) / np.sqrt(P))
    return float(np.mean(np.abs(tstat) > stats.t.ppf(1 - alpha / 2, P - 1)))


def power_between(N, k, d_eff, icc, nsim=4000, sd_resid=1.0, alpha=0.05):
    """Counterfactual: the SAME effect if the factor had been assigned BETWEEN
    participants (each person sees one level). Shows the cost of not going within."""
    nper = 24 * k
    lv = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(N)])
    sd_b0 = np.sqrt(icc / (1 - icc)) * sd_resid; delta = d_eff * sd_resid
    ybar = (delta * lv[:, None] + RNG.normal(0, sd_b0, (N, nsim))
            + RNG.normal(0, sd_resid / np.sqrt(nper), (N, nsim)))
    A = lv > 0; nA = A.sum(); nB = (~A).sum()
    mA, mB = ybar[A].mean(0), ybar[~A].mean(0)
    sp = np.sqrt(((nA - 1) * ybar[A].var(0, ddof=1)
                  + (nB - 1) * ybar[~A].var(0, ddof=1)) / (nA + nB - 2))
    tstat = (mA - mB) / (sp * np.sqrt(1 / nA + 1 / nB))
    return float(np.mean(np.abs(tstat) > stats.t.ppf(1 - alpha / 2, nA + nB - 2)))


def mdes(N, k, icc, tau, target_power=0.8, nsim=4000):
    """Smallest standardized effect d reaching target power (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if power_within(N, k, mid, icc, tau, nsim) < target_power:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def main(nsim=4000):
    D = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    TAUS = [0.0, 0.25, 0.5]                       # homogeneous -> heterogeneous effect
    NG = [6, 8, 10, 12, 14, 16]
    KG = [1, 2, 3, 5, 10, 15]
    DF, ICC, TF = 0.3, 0.4, 0.25                  # realistic operating point

    A = {str(t): [power_within(10, 5, d, ICC, t, nsim) for d in D] for t in TAUS}
    B = [power_within(N, 5, DF, ICC, TF, nsim) for N in NG]
    C = [power_within(10, k, DF, ICC, TF, nsim) for k in KG]
    win = power_within(10, 5, DF, ICC, 0.0, nsim)
    bet = power_between(10, 5, DF, ICC, nsim)
    mdes_homog = mdes(10, 5, ICC, 0.0, nsim=nsim)
    mdes_hetero = mdes(10, 5, ICC, TF, nsim=nsim)

    print("A) power vs d (as-run N=10, k=5, icc=0.4)")
    for t in TAUS:
        print(f"   tau={t}: " + "  ".join(f"{d}:{p:.2f}" for d, p in zip(D, A[str(t)])))
    print("B) power vs #participants (d=0.3, k=5, tau=0.25): "
          + "  ".join(f"N{N}:{p:.2f}" for N, p in zip(NG, B)))
    print("C) power vs placements/cell (d=0.3, N=10, tau=0.25): "
          + "  ".join(f"k{k}:{p:.2f}" for k, p in zip(KG, C)))
    print(f"D) within={win:.2f}  between={bet:.2f}")
    print(f"MDES @80%: homogeneous d={mdes_homog}   heterogeneous(tau=0.25) d={mdes_hetero}")

    json.dump({"D": D, "A": A, "NG": NG, "B": B, "KG": KG, "C": C,
               "within": win, "between": bet,
               "mdes_homog": mdes_homog, "mdes_hetero": mdes_hetero},
              open("power_results.json", "w"), indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Monte-Carlo power  —  within-subject configuration main effect (mixed model)",
                 weight="bold", fontsize=13)
    for t in TAUS:
        ax[0, 0].plot(D, A[str(t)], marker="o", label=f"effect heterogeneity τ={t}")
    ax[0, 0].axhline(0.8, ls="--", c="grey", lw=1)
    ax[0, 0].legend(fontsize=8, title="SD of per-person effect")
    ax[0, 0].set(title="A   power vs effect size (as-run: N=10, 5/cell)",
                 xlabel="standardized effect d (residual-SD units)", ylabel="power", ylim=(0, 1.03))
    ax[0, 1].plot(NG, B, marker="o", c="C3"); ax[0, 1].axhline(0.8, ls="--", c="grey", lw=1)
    ax[0, 1].axvline(10, ls=":", c="k", lw=1)
    ax[0, 1].set(title="B   power vs #participants (d=0.3, τ=0.25)",
                 xlabel="participants", ylabel="power", ylim=(0, 1.03))
    ax[1, 0].plot(KG, C, marker="o", c="C2"); ax[1, 0].axhline(0.8, ls="--", c="grey", lw=1)
    ax[1, 0].axvline(5, ls=":", c="k", lw=1)
    ax[1, 0].set(title="C   power vs placements/cell (d=0.3, τ=0.25)",
                 xlabel="replicate placements per cell", ylabel="power", ylim=(0, 1.03))
    ax[1, 1].bar(["within-subject\n(as designed)", "between-subject\n(counterfactual)"],
                 [win, bet], color=["C0", "C3"])
    for i, v in enumerate([win, bet]):
        ax[1, 1].text(i, v + 0.03, f"{v:.2f}", ha="center")
    ax[1, 1].set(title="D   same effect, design comparison (d=0.3)", ylabel="power", ylim=(0, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("power_curves.png", dpi=150)
    print("saved -> power_curves.png, power_results.json")


if __name__ == "__main__":
    main()
