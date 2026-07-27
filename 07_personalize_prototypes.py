#!/usr/bin/env python3
r"""
personalize_prototypes.py - Per-Participant Effects & Personalised Prototype Ranking

Standalone companion to rank_prototypes.py / evaluate_difference.py. Where those
scripts estimate ONE population-average effect per factor (a mixed model with a
random INTERCEPT only -- every participant is assumed to respond to prototype
changes identically, differing only in their baseline level), this script asks
a different question: does a given factor actually affect DIFFERENT PEOPLE
DIFFERENTLY, and if so, what would that mean for a personalised recommendation?

WHY A SEPARATE SCRIPT, NOT A MIXED-MODEL EXTENSION:
Getting genuinely person-specific coefficients from the pooled mixed model would
require per-participant RANDOM SLOPES, which were explored earlier in this
project and found prone to convergence/boundary failures at n=10 (see
fit_mixed_models.py / evaluate_difference.py history). Because each participant
independently completed the FULL within-subject 2x2x2 on Length/Size/Weight
(~120 place events per person, ~60 vs ~60 at each level of each binary factor),
each person's own data is sufficient to fit a simple ordinary-least-squares
regression ENTIRELY WITHIN that one person -- no random effects needed, because
there is no other participant's data being pooled. Angle is weaker (each
participant saw only 2-3 of its 3 levels, unevenly paired with the binary
combinations), so per-participant Angle effects rest on fewer trials and should
be read with more caution; this is flagged in the output.

CRITICAL CAVEAT, stated up front and repeated in the output: a per-participant
estimate is a description of ONE PERSON'S data, with no participant-level
replication behind it. It cannot say whether a person's pattern is a stable
trait or noise from their one session (order effects, fatigue, a bad day).
Treat these as descriptive personalisation, not as generalisable evidence --
very different epistemic status to the pooled, 10-participant factor verdicts
in rank_prototypes.py.

WHAT THIS SCRIPT DOES:
  1. Fits ONE ordinary-least-squares regression PER PARTICIPANT PER METRIC:
         metric ~ C(Length) + C(Size) + C(Weight) + C(Angle)
     using only that participant's own place events -- their own "slopes".
  2. Summarises how much those slopes VARY across participants, per
     (factor, level, metric): mean, between-participant SD (a direct,
     descriptive analogue of the tau^2 discussed earlier in this project, but
     read straight off real per-person estimates rather than fit as a variance
     component), and what fraction of participants agree in sign with the
     majority direction -- a low agreement fraction is exactly where a
     population-average recommendation would serve individuals poorly.
  3. Produces a PERSONALISED prototype recommendation for every participant,
     using the same signed/standardised/domain-weighted verdict logic as
     rank_prototypes.py's compute_factor_level_verdict, but fed each person's
     own slopes -- and compares it against the population verdict (loaded from
     evaluate_difference.py's output via --comparison-dir if available, else
     approximated as the across-participant average of the slopes computed
     here) so agreement/disagreement per participant is directly visible.

Usage:
  python 07_personalize_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python 07_personalize_prototypes.py --pen-csv place_metrics_combined.csv --posture-csv posture_features_combined.csv --comparison-dir combined_comparison
"""

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
import statsmodels.formula.api as smf

# =========================================================================== #
# SECTION 1: METRIC REGISTRY & PARAM FACTORS (kept in sync with rank_prototypes.py)
# =========================================================================== #

METRIC_REGISTRY = {
    "duration_s":                 {"domain": "Performance",     "dir": "min", "label": "Task Duration (s)"},
    "perp_mean_deg":              {"domain": "Performance",     "dir": "min", "label": "Perpendicularity"},
    "leftright_mean_deg":         {"domain": "Performance",     "dir": "min", "label": "L/R Tilt"},
    "updown_mean_deg":            {"domain": "Performance",     "dir": "min", "label": "U/D Tilt"},
    "pos_jitter_mm":              {"domain": "Performance",     "dir": "min", "label": "Positional Jitter"},
    "ang_jitter_deg":             {"domain": "Performance",     "dir": "min", "label": "Angular Jitter"},
    "reba_score_a":               {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score A"},
    "reba_score_b_right":         {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (R)"},
    "reba_score_b_left":          {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (L)"},
    "reba_grand_right":           {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (R)"},
    "reba_grand_left":            {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (L)"},
    "reach_ratio_mean":           {"domain": "Postural_Risk",   "dir": "min", "label": "Reach Ratio"},
    "right_grip_comfort_score":   {"domain": "Grip_Ergonomics", "dir": "max", "label": "R Grip Comfort"},
    "right_sparc_linear":         {"domain": "Grip_Ergonomics", "dir": "max", "label": "R SPARC (Linear)"},
    "right_sparc_angular":        {"domain": "Grip_Ergonomics", "dir": "max", "label": "R SPARC (Angular)"},
}

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
FACTOR_REFERENCE = {"Length": "Long", "Size": "Large", "Weight": "Front_weighted", "Angle": "A90"}
FACTOR_LEVEL_LABELS = {
    "Length": {"Long": "Long", "Short": "Short"},
    "Size": {"Large": "Large", "Small": "Small"},
    "Weight": {"Front_weighted": "Front_weighted", "Not_weighted": "Not_weighted"},
    "Angle": {"90": "A90", "135": "A135", "180": "A180"},
}
MIN_TRIALS_PER_PARTICIPANT_METRIC = 15  # unused (kept only to avoid breaking external references);
                                        # see MIN_CELLS_PER_PARTICIPANT_METRIC, which gates on
                                        # aggregated config x height cells, not raw place events


# =========================================================================== #
# SECTION 2: PARSING & DESIRABILITY (mirrors rank_prototypes.py)
# =========================================================================== #

def parse_params(trial_val) -> dict:
    out = {k: "Other" for k in PARAM_FACTORS}
    if trial_val is None or pd.isna(trial_val):
        return out
    tokens = [t.strip() for t in str(trial_val).strip().split("_") if t.strip()]
    joined_low = "_".join(tokens).lower()
    if "not_weighted" in joined_low or "notweighted" in joined_low:
        out["Weight"] = "Not_weighted"
    elif "front_weighted" in joined_low or "frontweighted" in joined_low:
        out["Weight"] = "Front_weighted"
    for tok in tokens:
        t_low = tok.lower(); t_cap = tok.capitalize()
        if t_cap in ("Long", "Short"): out["Length"] = t_cap
        elif t_cap in ("Large", "Small"): out["Size"] = t_cap
        elif t_low.startswith("a") and t_low[1:].isdigit(): out["Angle"] = f"A{t_low[1:]}"
        elif tok.isdigit() and int(tok) in (0, 45, 90, 135, 180, 225, 270, 315): out["Angle"] = f"A{tok}"
    return out


CANONICAL_LEVELS = {
    "Length": {"Long", "Short"}, "Size": {"Large", "Small"},
    "Weight": {"Front_weighted", "Not_weighted"},
    "Angle": {"A90", "A135", "A180"},
}


def _normalize_angle(val) -> str:
    """Angle may arrive as a bare number (int/float/string, e.g. 90, 90.0, '90')
    from an existing 'Angle' column already in place_metrics_combined.csv (set
    by metrics.py), OR as an 'A90'-style string from this script's own
    parse_params fallback. Normalise both to the same 'A<int>' form before any
    comparison -- the earlier crash was exactly this mismatch: the validation
    below expected 'A90' but the real column already contained bare '90'."""
    s = str(val).strip()
    if s.upper().startswith("A") and s[1:].replace(".", "", 1).isdigit():
        s = s[1:]
    try:
        s = str(int(float(s)))
    except (ValueError, TypeError):
        return str(val).strip()  # leave unparseable values as-is; the canonical check below will flag them
    return f"A{s}"


def add_prototype_label(df: pd.DataFrame) -> pd.DataFrame:
    """LOUDLY reports and quarantines (drops) any row where a factor could not
    be cleanly parsed into its two/three canonical levels -- a naming
    inconsistency in the raw trial strings should be visible and fixable here,
    not discovered several analysis stages later as an unexplained spurious
    category (this is exactly what happened upstream in evaluate_difference.py:
    a silent 'weighted' fallback bucket was corrupting the population-level
    mixed models and leaking into every downstream verdict)."""
    df_clean = df.copy()
    parsed = df_clean["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        if c not in df_clean.columns or df_clean[c].isna().all():
            df_clean[c] = parsed[c]
        else:
            df_clean[c] = df_clean[c].fillna(parsed[c])
        df_clean[c] = df_clean[c].fillna("Other").astype(str).str.strip()
    df_clean["Angle"] = df_clean["Angle"].apply(_normalize_angle)

    print("\nObserved factor levels (sanity check -- each should show exactly the "
          "expected canonical set, with no unexpected extra category):")
    bad_mask = pd.Series(False, index=df_clean.index)
    for c in PARAM_FACTORS:
        observed = set(df_clean[c].dropna().unique().tolist())
        print(f"  {c}: {sorted(observed, key=str)}")
        unexpected = observed - CANONICAL_LEVELS[c]
        if unexpected:
            print(f"    [WARN] unexpected level(s) for {c}: {sorted(unexpected)}")
            bad_mask = bad_mask | df_clean[c].isin(unexpected)

    if bad_mask.any():
        n_remaining = len(df_clean) - bad_mask.sum()
        if n_remaining < 0.5 * len(df_clean):
            sys.exit(f"\nError: quarantining unexpected factor levels would remove "
                     f"{bad_mask.sum()}/{len(df_clean)} rows ({100*bad_mask.sum()/len(df_clean):.0f}%). "
                     f"This is almost certainly a genuine format mismatch, not scattered bad data -- "
                     f"stopping rather than analysing a near-empty or empty dataset. Check the "
                     f"'unexpected level(s)' warning(s) above.")
        bad_trials = df_clean.loc[bad_mask, "trial"].unique() if "trial" in df_clean.columns else []
        print(f"\n  [WARN] {bad_mask.sum()} row(s) across {len(bad_trials)} distinct trial name(s) "
              f"have a non-canonical factor level and are being QUARANTINED (dropped):")
        for t in list(bad_trials)[:10]:
            print(f"      '{t}'")
        if len(bad_trials) > 10:
            print(f"      ... and {len(bad_trials) - 10} more")
        df_clean = df_clean[~bad_mask].copy()

    df_clean["Prototype_Config"] = df_clean[["Length", "Size", "Weight", "Angle"]].agg("_".join, axis=1)
    return df_clean


# =========================================================================== #
# SECTION 3: PER-PARTICIPANT SLOPE ESTIMATION (OLS, one person's data at a time)
# =========================================================================== #

def _fit_ols(data: pd.DataFrame, response: str, factors: list):
    """OLS within one participant's own trials -- no random effects, since
    there is no other participant's data being pooled here. Uses CLUSTER-ROBUST
    standard errors, clustered by (trial, height) -- i.e. by the burst of ~5
    replicate place events run back-to-back for one config at one height.

    WHY: classical OLS standard errors assume every row is an independent draw.
    But the 5 replicate trials within one (config, height) burst were run
    consecutively, in a short window, and almost certainly share correlated
    noise (momentary fatigue, grip warm-up, environmental drift) that trials
    from a DIFFERENT burst would not share. Verified directly: on data built
    with genuinely zero true effect but realistic within-burst correlation,
    classical OLS understated the standard error by ~1.7x for a single
    participant -- since Cochran's Q (used downstream in
    summarise_heterogeneity) scales with 1/SE^2, that alone produces a
    spuriously large weight and inflated apparent heterogeneity across almost
    every metric, which is exactly the pattern this was introduced to fix.
    Clustering by (trial, height) treats each burst as one unit of information
    for variance-estimation purposes, without requiring any assumption about
    true chronological trial order (which isn't reliably available across
    different config files for one participant).

    Returns (result or None, factors present, failure reason)."""
    present = [f for f in factors if data[f].nunique() >= 2]
    if not present:
        return None, present, "no factor has >=2 levels for this participant"
    formula = f"{response} ~ " + " + ".join(f"C({f})" for f in present)
    try:
        cluster_groups = data["trial"].astype(str) + "||" + data["height"].astype(str)
        n_clusters = cluster_groups.nunique()
        if n_clusters < 8:
            # too few clusters for cluster-robust asymptotics to be trustworthy;
            # fall back to classical SE rather than report an unstable robust SE
            res = smf.ols(formula, data=data).fit()
        else:
            res = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
        return res, present, None
    except Exception as ex:
        return None, present, f"{type(ex).__name__}: {ex}"


def _term_level_effects(res, factor: str):
    """Per-level coefficients relative to the reference (same parsing approach
    as evaluate_difference.py's population-level version)."""
    names = [n for n in res.params.index if n.startswith(f"C({factor})")]
    out = []
    for n in names:
        m = re.search(r"\[T\.(.+?)\]", n)
        level_raw = m.group(1) if m else n
        try:
            level_raw = str(int(float(level_raw)))
        except ValueError:
            pass
        b = float(res.params[n])
        se = float(res.bse[n]) if n in res.bse.index else np.nan
        p = float(res.pvalues[n]) if n in res.pvalues.index else np.nan
        out.append({"level": level_raw, "effect": b, "se": se, "p_value": p})
    return out


def _fit_ols(data: pd.DataFrame, response: str, factors: list):
    """OLS on one participant's data, ALREADY AGGREGATED to one row per
    (trial, height) cell -- see compute_participant_slopes, which does the
    aggregation before calling this. No random effects needed (a single
    participant), and no cluster-robust correction needed either, because
    aggregation removes the within-cluster structure that made cluster-robust
    SE unreliable here in the first place (see compute_participant_slopes
    docstring). Returns (result or None, factors present, failure reason)."""
    present = [f for f in factors if data[f].nunique() >= 2]
    if not present:
        return None, present, "no factor has >=2 levels for this participant"
    formula = f"{response} ~ " + " + ".join(f"C({f})" for f in present)
    try:
        res = smf.ols(formula, data=data).fit()
        return res, present, None
    except Exception as ex:
        return None, present, f"{type(ex).__name__}: {ex}"


MIN_CELLS_PER_PARTICIPANT_METRIC = 8  # below this, a per-person OLS is too noisy to report
                                      # (see compute_participant_slopes: this counts AGGREGATED
                                      # config x height cells, not raw place events)


def compute_participant_slopes(df: pd.DataFrame, metrics: list, factors: list = PARAM_FACTORS,
                               verbose: bool = True) -> pd.DataFrame:
    """One row per (participant, factor, level, metric): that participant's OWN
    regression coefficient, fit entirely within their own place events.

    CRITICAL: fits on data AGGREGATED to one row per (trial, height) cell, not
    on raw place events. Length/Size/Weight/Angle are all properties of the
    CONFIG (trial), constant across every one of the ~5 replicate place events
    run for that config at that height -- so the true number of independent
    data points for estimating any of these coefficients is the number of
    (config, height) cells (~24 per participant), not the number of raw place
    events (~120). Fitting on raw events, even with cluster-robust standard
    errors, was verified to still produce badly miscalibrated (far too small)
    p-values: on synthetic data with a TRUE effect of exactly zero, trial-level
    OLS gave p<0.0001, and cluster-robust trial-level OLS *also* gave p<0.0001,
    because cluster-robust correction is known to perform poorly precisely when
    the predictor of interest is constant within each cluster and the number of
    clusters is modest (Cameron & Miller 2015). Aggregating to cell means FIRST
    removes this problem entirely and was verified to give a well-calibrated
    p=0.41 on the same null data. This is the same pseudoreplication issue
    solved earlier in this project at the population level (place events
    pseudoreplicating within participant), recurring one level down (place
    events pseudoreplicating within config, for a single participant's own
    regression)."""
    records = []
    diag = {"fit": 0, "skip": 0, "fail": 0}
    for participant, sub in df.groupby("participant"):
        for col in metrics:
            if col not in sub.columns:
                continue
            d = sub.dropna(subset=[col] + factors).copy().rename(columns={col: "_y"})
            if pd.to_numeric(d["_y"], errors="coerce").nunique() <= 1:
                continue
            # aggregate to one row per (trial, height) cell -- see docstring
            group_cols = ["trial", "height"] + factors
            group_cols = [c for c in dict.fromkeys(group_cols) if c in d.columns]  # dedupe, preserve order
            agg = d.groupby(group_cols, as_index=False)["_y"].mean()

            tag = f"[{participant}] {col}"
            if len(agg) < MIN_CELLS_PER_PARTICIPANT_METRIC:
                diag["skip"] += 1
                if verbose:
                    print(f"    {tag}: skipped ({len(agg)} config x height cells < {MIN_CELLS_PER_PARTICIPANT_METRIC})")
                continue
            res, present, note = _fit_ols(agg, "_y", factors)
            if res is None:
                diag["fail"] += 1
                if verbose:
                    print(f"    {tag}: fit failed: {note}")
                continue
            diag["fit"] += 1
            resid_df = float(res.df_resid)
            for f in factors:
                level_effects = _term_level_effects(res, f)
                for le in level_effects:
                    records.append({"participant": participant, "factor": f, "level": le["level"],
                                    "metric": col, "effect": le["effect"], "se": le["se"],
                                    "p_value": le["p_value"], "n": len(agg), "resid_df": resid_df})
    if verbose:
        total = sum(diag.values())
        print(f"\n  Per-participant OLS fit summary: {diag['fit']}/{total} fitted, "
              f"{diag['skip']}/{total} skipped (too few config x height cells), {diag['fail']}/{total} failed.")
    return pd.DataFrame(records)


# =========================================================================== #
# SECTION 4: HETEROGENEITY SUMMARY -- how much do slopes vary across people?
# =========================================================================== #

MIN_RESID_DF_FOR_HETEROGENEITY = 8   # below this, a participant's own SE is too unstable to trust
                                     # (recalibrated for the aggregated ~24-cell regression: a
                                     # typical fit has resid_df ~ 24 - 7 params = 17, so 8 is a
                                     # conservative floor, not the 20 used when fits ran on ~120
                                     # raw trials before the aggregation fix)
                                     # in the inverse-variance weighting (see summarise_heterogeneity)
MAX_WEIGHT_RATIO = 50  # safety net: cap any one participant's weight at this multiple of the row's
                       # median weight, so a single implausibly tiny SE cannot single-handedly
                       # dominate Q even if it technically cleared the df gate above


def summarise_heterogeneity(slopes: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """For each (factor, level, metric): a formal test of whether participants
    genuinely disagree, versus merely looking scattered because each person's
    own estimate is individually noisy (Cochran's Q / I-squared, the standard
    meta-analysis tool for exactly this question).

    WHY NOT JUST 'between-participant SD / |mean effect|' (an earlier version):
    that ratio blows up whenever the population-average effect happens to be
    near zero, regardless of whether participants genuinely disagree or are
    simply all estimating a true-zero effect noisily.

    SAFEGUARD -- why some participants are excluded from this specific test:
    inverse-variance weighting (weight = 1/se^2) means a participant whose own
    standard error is small gets disproportionate influence on Q. If that small
    SE is itself unreliable -- which happens easily with few residual degrees
    of freedom (thin data after tracking-dropout missingness, or a coarse/
    near-constant metric like REBA) -- that one person can single-handedly
    manufacture a 'significant' Q even when the other participants agree with
    each other perfectly. Participants whose fit had fewer than
    MIN_RESID_DF_FOR_HETEROGENEITY residual degrees of freedom are therefore
    excluded from the Q-test for that row (they remain in participant_slopes.csv
    and the personalised verdict -- only the heterogeneity TEST excludes them).
    As a second safety net, any remaining participant's weight is capped at
    MAX_WEIGHT_RATIO times the row's median weight."""
    records = []
    for (factor, level, metric), grp in slopes.groupby(["factor", "level", "metric"]):
        sd = pd.to_numeric(df[metric], errors="coerce").std(ddof=1) if metric in df.columns else np.nan
        if not sd or pd.isna(sd) or sd < 1e-9:
            continue
        n_before = len(grp)
        grp = grp[grp["resid_df"] >= MIN_RESID_DF_FOR_HETEROGENEITY]
        n_excluded = n_before - len(grp)
        if len(grp) < 2:
            continue
        effects = (grp["effect"] / sd).values
        ses = (grp["se"] / sd).values
        n_ppt = len(effects)
        if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
            continue

        weights = 1.0 / (ses ** 2)
        med_w = np.median(weights)
        weights = np.minimum(weights, med_w * MAX_WEIGHT_RATIO)   # cap runaway influence

        weighted_mean = float(np.sum(weights * effects) / np.sum(weights))
        Q = float(np.sum(weights * (effects - weighted_mean) ** 2))
        dfree = n_ppt - 1
        p_het = float(stats.chi2.sf(Q, dfree)) if dfree > 0 else np.nan
        i_sq = max(0.0, (Q - dfree) / Q) * 100.0 if Q > 0 else 0.0

        majority_sign = np.sign(weighted_mean) if weighted_mean != 0 else 1
        agree_frac = float(np.mean(np.sign(effects) == majority_sign))

        records.append({
            "factor": factor, "level": level, "metric": metric,
            "n_participants": n_ppt, "n_excluded_low_df": n_excluded,
            "weighted_mean_effect": round(weighted_mean, 4),
            "Q": round(Q, 2), "df": dfree, "p_heterogeneity": round(p_het, 4) if pd.notna(p_het) else np.nan,
            "I_squared_pct": round(i_sq, 1),
            "pct_agree_with_majority": round(agree_frac * 100, 1),
        })
    return pd.DataFrame(records).sort_values(["p_heterogeneity", "I_squared_pct"], ascending=[True, False])


def compute_shrunk_slopes(slopes: pd.DataFrame, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Partial pooling (DerSimonian & Laird, 1986): for each (factor, level,
    metric), estimate how much of the between-participant spread is genuine
    heterogeneity (tau^2) versus each person's own estimation noise, then pull
    every participant's raw estimate partway toward the heterogeneity-aware
    population consensus (theta_RE), by an amount (lambda_i) that depends on
    how reliable THAT participant's own estimate is.

    This is the same logic that produces a random-slopes mixed model's shrunk
    per-participant coefficients, computed here in closed form from the
    already-fitted per-participant slopes and the Q statistic, rather than by
    jointly re-fitting all participants at once (avoiding the convergence risk
    that motivated the two-stage design in the first place).

        tau^2_DL = max(0, (Q - df) / C),  C = sum(w_i) - sum(w_i^2)/sum(w_i)
        w_i*     = 1 / (se_i^2 + tau^2_DL)
        theta_RE = sum(w_i* . effect_i) / sum(w_i*)
        lambda_i = tau^2_DL / (tau^2_DL + se_i^2)
        shrunk_i = lambda_i . effect_i + (1 - lambda_i) . theta_RE

    A participant with a large SE (noisy/thin own data) gets lambda_i near 0
    and is pulled strongly toward the group; a participant with a small SE
    keeps close to their own raw estimate. If tau^2 is itself small (little
    real heterogeneity), everyone is pulled hard toward one shared answer,
    which is the correct behaviour when the apparent disagreement was mostly
    noise. Uses the same residual-df exclusion and weight cap as
    summarise_heterogeneity, for the same reasons."""
    records = []
    for (factor, level, metric), grp in slopes.groupby(["factor", "level", "metric"]):
        sd = pd.to_numeric(df[metric], errors="coerce").std(ddof=1) if metric in df.columns else np.nan
        if not sd or pd.isna(sd) or sd < 1e-9:
            continue
        grp = grp[grp["resid_df"] >= MIN_RESID_DF_FOR_HETEROGENEITY]
        if len(grp) < 2:
            continue
        effects = (grp["effect"] / sd).values
        ses = (grp["se"] / sd).values
        participants = grp["participant"].values
        if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
            continue

        w = 1.0 / (ses ** 2)
        med_w = np.median(w)
        w = np.minimum(w, med_w * MAX_WEIGHT_RATIO)
        theta_fe = float(np.sum(w * effects) / np.sum(w))
        Q = float(np.sum(w * (effects - theta_fe) ** 2))
        k = len(effects)
        dfree = k - 1
        C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0

        w_star = 1.0 / (ses ** 2 + tau2)
        theta_re = float(np.sum(w_star * effects) / np.sum(w_star))

        for i in range(k):
            lam = tau2 / (tau2 + ses[i] ** 2) if (tau2 + ses[i] ** 2) > 0 else 1.0
            shrunk_std = lam * effects[i] + (1 - lam) * theta_re
            records.append({
                "factor": factor, "level": level, "metric": metric,
                "participant": participants[i],
                "raw_std_effect": round(float(effects[i]), 4),
                "tau2": round(tau2, 4), "theta_RE": round(theta_re, 4),
                "lambda": round(float(lam), 3),
                "shrunk_std_effect": round(float(shrunk_std), 4),
                # stored back in RAW units (x sd) so this table can be fed into
                # compute_verdict exactly like any other effect table, which
                # itself divides by sd -- recovering shrunk_std_effect intact.
                "effect": shrunk_std * sd,
                "se": grp["se"].values[i],
            })
    out = pd.DataFrame(records)
    if verbose and len(out):
        avg_lambda = out["lambda"].mean()
        print(f"\n  Partial pooling applied: average shrinkage weight (lambda) across all "
              f"(factor, level, metric, participant) combinations = {avg_lambda:.2f} "
              f"(1.0 = fully trust own data, 0.0 = fully replaced by group consensus).")
    return out


def print_heterogeneity_summary(hetero: pd.DataFrame, top_n: int = 15):
    print(f"\n{'='*110}\nSLOPE HETEROGENEITY -- Cochran's Q / I-squared TEST (real disagreement vs. per-person noise)\n{'='*110}")
    print(" Q tests whether participants' own effects disagree MORE than each person's individual")
    print(" sampling noise (their own SE) would predict. p_heterogeneity < 0.05 = genuine disagreement,")
    print(" not just noisy individual estimates. I_squared = % of the apparent spread that is real")
    print(" heterogeneity rather than estimation noise. Excl = participants dropped from THIS test")
    print(" for having too few residual degrees of freedom for their own SE to be trustworthy.\n")
    print(f" {'Factor':<8} {'Level':<14} {'Metric':<24} {'Np':>3} {'Excl':>4} {'WtMean':>8} {'Q':>7} {'p_het':>8} {'I^2%':>7} {'%Agree':>8}")
    print(f" {'-'*8} {'-'*14} {'-'*24} {'-'*3} {'-'*4} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for _, r in hetero.head(top_n).iterrows():
        p_str = f"{r['p_heterogeneity']:>8.4f}" if pd.notna(r["p_heterogeneity"]) else f"{'n/a':>8}"
        print(f" {r['factor']:<8} {r['level']:<14} {r['metric']:<24} {int(r['n_participants']):>3} "
              f"{int(r['n_excluded_low_df']):>4} {r['weighted_mean_effect']:>8.3f} {r['Q']:>7.2f} {p_str} "
              f"{r['I_squared_pct']:>6.1f}% {r['pct_agree_with_majority']:>7.1f}%")

    genuine = hetero[(hetero["p_heterogeneity"] < 0.05) & (hetero["I_squared_pct"] >= 50)]
    if len(genuine):
        print(f"\n  {len(genuine)} (factor, level, metric) combination(s) show STATISTICALLY GENUINE heterogeneity")
        print("  (p_heterogeneity < 0.05 AND I^2 >= 50%) -- real disagreement between participants, not noise:")
        for _, r in genuine.head(10).iterrows():
            excl_note = f", {int(r['n_excluded_low_df'])} participant(s) excluded as low-confidence" if r["n_excluded_low_df"] > 0 else ""
            print(f"      {r['factor']} ({r['level']}) on {r['metric']}: p={r['p_heterogeneity']:.4f}, "
                  f"I^2={r['I_squared_pct']:.0f}%, {r['pct_agree_with_majority']:.0f}% agree{excl_note}")
    else:
        print("\n  No combinations show statistically genuine heterogeneity (p_heterogeneity < 0.05 and I^2 >= 50%).")
        print("  Apparent per-participant disagreement elsewhere in this table is consistent with ordinary")
        print("  individual estimation noise, not real between-person differences -- read those with caution.")

    high_excl = hetero[hetero["n_excluded_low_df"] >= 3]
    if len(high_excl):
        print(f"\n  Note: {len(high_excl)} row(s) had 3+ participants excluded for low residual degrees of "
              f"freedom (thin data, likely tracking dropout for that metric) -- treat those rows' results as")
        print("  based on a smaller, potentially unrepresentative subset of the sample.")


# =========================================================================== #
# SECTION 5: DESIRABILITY-WEIGHTED VERDICT (population AND per-participant)
#   Same logic as rank_prototypes.py's compute_factor_level_verdict, generalised
#   to accept either the pooled mixed-model table OR one participant's own
#   slope table.
# =========================================================================== #

def compute_metric_weights(df: pd.DataFrame, active_domains: dict, effect_table: pd.DataFrame) -> dict:
    """Same standardised-sum-of-squared-effects weighting as rank_prototypes.py's
    method (A), computed from whichever effect table is supplied (population or
    one participant)."""
    weights = {}
    for domain, cols in active_domains.items():
        for col in cols:
            raw = col.replace("_dscore", "")
            if raw not in df.columns:
                weights[col] = 0.01; continue
            sd = pd.to_numeric(df[raw], errors="coerce").std(ddof=1)
            rows = effect_table[(effect_table["metric"] == raw) & (effect_table["factor"].isin(PARAM_FACTORS))] \
                if effect_table is not None else pd.DataFrame()
            if not sd or pd.isna(sd) or sd < 1e-9 or rows.empty:
                weights[col] = 0.01; continue
            sq = [(r["effect"] / sd) ** 2 for _, r in rows.iterrows() if pd.notna(r["effect"])]
            weights[col] = max(0.01, float(sum(sq))) if sq else 0.01
    return weights


def compute_verdict(df_scored: pd.DataFrame, active_domains: dict, metric_weights: dict,
                    effect_table: pd.DataFrame) -> dict:
    """Identical logic to rank_prototypes.py's compute_factor_level_verdict."""
    if effect_table is None or effect_table.empty:
        return {f: {"winner": None, "reason": "no effect data"} for f in PARAM_FACTORS}
    domain_w = {d: np.mean([metric_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}
    tot_dom_w = sum(domain_w.values()) or 1.0
    verdicts = {}
    for factor in PARAM_FACTORS:
        ref_label = FACTOR_REFERENCE[factor]
        level_scores = defaultdict(float)
        level_scores[ref_label] = 0.0
        any_metric = False
        for domain, cols in active_domains.items():
            dom_w_sum = sum(metric_weights.get(c, 0.01) for c in cols) or 1.0
            inter_w = domain_w.get(domain, 0.0) / tot_dom_w
            for col in cols:
                raw = col.replace("_dscore", "")
                meta = METRIC_REGISTRY.get(raw)
                if meta is None or raw not in df_scored.columns:
                    continue
                sd = pd.to_numeric(df_scored[raw], errors="coerce").std(ddof=1)
                rows = effect_table[(effect_table["metric"] == raw) & (effect_table["factor"] == factor)]
                if rows.empty or not sd or pd.isna(sd) or sd < 1e-9:
                    continue
                direction = 1.0 if meta["dir"] == "max" else -1.0
                intra_w = metric_weights.get(col, 0.01) / dom_w_sum
                for _, r in rows.iterrows():
                    if pd.isna(r["effect"]) or pd.isna(r.get("level")):
                        continue
                    level_label = FACTOR_LEVEL_LABELS.get(factor, {}).get(str(r["level"]), str(r["level"]))
                    std_effect = r["effect"] / sd
                    level_scores[level_label] += std_effect * direction * intra_w * inter_w
                    any_metric = True
        if not any_metric:
            verdicts[factor] = {"winner": None, "reason": "no metric had an estimable effect"}
            continue
        winner = max(level_scores, key=level_scores.get)
        verdicts[factor] = {"winner": winner, "level_scores": dict(level_scores)}
    return verdicts


def calculate_desirability_scores(df: pd.DataFrame) -> tuple:
    df_scored = df.copy()
    active_domains = defaultdict(list)
    for col, meta in METRIC_REGISTRY.items():
        if col not in df_scored.columns:
            continue
        vals = pd.to_numeric(df_scored[col], errors="coerce")
        if vals.dropna().empty or vals.nunique() <= 1:
            continue
        if any(w in col for w in ("deg", "dev", "flex", "tilt")):
            vals = vals.abs()
        v_min, v_max = vals.min(), vals.max()
        if abs(v_max - v_min) < 1e-9:
            continue
        score_col = f"{col}_dscore"
        df_scored[score_col] = (100.0 * (v_max - vals) / (v_max - v_min) if meta["dir"] == "min"
                                else 100.0 * (vals - v_min) / (v_max - v_min))
        active_domains[meta["domain"]].append(score_col)
    return df_scored, active_domains


def load_population_effects(comparison_dir: Path):
    """Loads evaluate_difference.py's stat_tests.csv, then validates its 'level'
    column against the expected raw (pre-relabelling) non-reference levels --
    this catches contamination even in a STALE file generated before the
    upstream parse_params fix (see module history: a silent fallback bucket
    named literally 'weighted' was leaking into this exact file). Only
    non-reference levels ever appear as rows here (the reference level is
    absorbed into the model intercept and has no coefficient of its own)."""
    if comparison_dir is None:
        return None
    path = comparison_dir / "stat_tests.csv"
    if not path.is_file():
        print(f"  [WARN] {path} not found -- population verdict will be approximated as the "
              f"across-participant average of the per-participant slopes computed here, rather "
              f"than loaded from evaluate_difference.py's pooled mixed model.")
        return None
    tbl = pd.read_csv(path)

    expected_raw_levels = {"Length": {"Short"}, "Size": {"Small"},
                           "Weight": {"Not_weighted"}, "Angle": {"135", "180"}}
    if "level" in tbl.columns:
        bad_mask = pd.Series(False, index=tbl.index)
        for factor, allowed in expected_raw_levels.items():
            sub = tbl["factor"] == factor
            observed = set(tbl.loc[sub, "level"].dropna().astype(str).unique())
            unexpected = observed - allowed
            if unexpected:
                print(f"\n  [WARN] {path.name}: factor '{factor}' has unexpected level(s) "
                      f"{sorted(unexpected)} (expected only {sorted(allowed)}). This usually means "
                      f"the file was generated before a trial-name parsing fix -- RERUN "
                      f"evaluate_difference.py to regenerate it. Rows with the unexpected level are "
                      f"being dropped from the population verdict as a precaution.")
                bad_mask = bad_mask | (sub & tbl["level"].astype(str).isin(unexpected))
        if bad_mask.any():
            tbl = tbl[~bad_mask].copy()
    return tbl


# =========================================================================== #
# SECTION 6: MAIN
# =========================================================================== #

def config_string(verdict: dict) -> str:
    parts = []
    for f in PARAM_FACTORS:
        w = verdict.get(f, {}).get("winner")
        parts.append(w if w else "?")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None)
    ap.add_argument("--pen-csv", type=Path, default=None)
    ap.add_argument("--posture-csv", type=Path, default=None)
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="evaluate_difference.py output dir, for the population verdict "
                         "(default: <landmarks-root>/metrics/combined_comparison)")
    args = ap.parse_args()

    if args.landmarks_root:
        pen_path = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        posture_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
        out_dir = args.landmarks_root / "metrics" / "personalised_rankings"
        comparison_dir = args.comparison_dir or (args.landmarks_root / "metrics" / "combined_comparison")
    else:
        pen_path, posture_path = args.pen_csv, args.posture_csv
        out_dir = (pen_path or posture_path).parent / "personalised_rankings"
        comparison_dir = args.comparison_dir

    df_pen = pd.read_csv(pen_path) if (pen_path and pen_path.is_file()) else pd.DataFrame()
    df_posture = pd.read_csv(posture_path) if (posture_path and posture_path.is_file()) else pd.DataFrame()
    if df_pen.empty and df_posture.empty:
        sys.exit("Error: Both pen and posture CSVs are empty or missing.")

    if not df_pen.empty and not df_posture.empty:
        # Normalise a known naming drift between the two upstream extraction
        # scripts (metrics.py historically used 'trial_num' where
        # evaluate_difference.py uses 'place_index' for the same concept).
        for d in (df_pen, df_posture):
            if "place_index" not in d.columns and "trial_num" in d.columns:
                d.rename(columns={"trial_num": "place_index"}, inplace=True)

        intended_key = ["participant", "trial", "place_index", "height"]
        missing_pen = [c for c in intended_key if c not in df_pen.columns]
        missing_posture = [c for c in intended_key if c not in df_posture.columns]
        if missing_pen or missing_posture:
            sys.exit(f"\nError: the intended merge key {intended_key} is missing column(s) "
                     f"{missing_pen or '[]'} from the pen table and {missing_posture or '[]'} from "
                     f"the posture table. Merging on a SUBSET of this key (e.g. dropping "
                     f"'place_index') would silently produce a non-unique join and a many-to-many "
                     f"merge blowup -- refusing to proceed rather than repeat that bug. Check the "
                     f"column names in place_metrics_combined.csv and posture_features_combined.csv.")
        common = intended_key

        # --- diagnostic: check the merge KEY is actually unique in each source
        # table BEFORE merging. If it isn't, pandas silently produces the
        # cartesian product of every matching group, duplicating rows -- which
        # would make a per-participant regression's effective sample size look
        # artificially large, artificially SHRINKING its computed standard
        # error without adding any real information. This is a strong
        # candidate for implausibly small per-participant SEs and inflated
        # heterogeneity Q statistics seen elsewhere in this pipeline. ---
        for name, d in (("pen", df_pen), ("posture", df_posture)):
            dup_counts = d.groupby(common).size()
            dups = dup_counts[dup_counts > 1]
            if len(dups):
                print(f"\n  [WARN] {name} table: merge key {common} is NOT unique -- "
                      f"{len(dups)} key combination(s) appear more than once "
                      f"(up to {dups.max()}x). Example duplicated key(s):")
                for key, n in dups.head(5).items():
                    key_str = dict(zip(common, key if isinstance(key, tuple) else (key,)))
                    print(f"      {key_str}: appears {n} times")
                print(f"  This will cause a many-to-many merge blowup (see below) unless the "
                      f"upstream extraction script that produced {name}_path is fixed to emit one "
                      f"row per (participant, trial, place_index, height).")

        n_pen, n_posture = len(df_pen), len(df_posture)
        df = pd.merge(df_pen, df_posture, on=common, how="inner", suffixes=("", "_posture"))
        expected_max = max(n_pen, n_posture)
        if len(df) > 1.2 * expected_max:
            sys.exit(f"\nError: MERGE BLOWUP detected. Pen table has {n_pen} rows, posture table has "
                     f"{n_posture} rows, but the merge produced {len(df)} rows -- far more than either "
                     f"input. This means the merge key {common} is not unique in at least one source "
                     f"table (see [WARN] messages above), causing pandas to produce the cartesian "
                     f"product of matching groups rather than a clean 1-1 join. Every downstream "
                     f"result from this run would be corrupted by duplicated rows. Fix the upstream "
                     f"extraction so (participant, trial, place_index, height) is unique in both "
                     f"place_metrics_combined.csv and posture_features_combined.csv, then rerun.")
    else:
        df = df_pen if not df_pen.empty else df_posture
    print(f"Loaded {len(df)} Place events, {df['participant'].nunique()} participants.")

    if "height" in df.columns:
        mask = ~df["height"].isin({"High", "Medium", "Low"}) | df["height"].isna()
        if mask.any():
            print(f"[QUARANTINE] Dropping {mask.sum()} events with no valid height.")
            df = df[~mask].copy()

    df = add_prototype_label(df)
    df_scored, active_domains = calculate_desirability_scores(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_present = [c for c in METRIC_REGISTRY if c in df_scored.columns]

    print(f"\n{'='*80}\nFITTING PER-PARTICIPANT REGRESSIONS ({len(metrics_present)} metrics x "
          f"{df_scored['participant'].nunique()} participants)\n{'='*80}")
    slopes = compute_participant_slopes(df_scored, metrics_present, PARAM_FACTORS)
    slopes.to_csv(out_dir / "participant_slopes.csv", index=False)
    print(f"Wrote {out_dir / 'participant_slopes.csv'} ({len(slopes)} rows)")

    hetero = summarise_heterogeneity(slopes, df_scored)
    hetero.to_csv(out_dir / "slope_heterogeneity_summary.csv", index=False)
    print_heterogeneity_summary(hetero)

    # ------------------------------------------------------------------ #
    # Population verdict (from evaluate_difference.py if available, else
    # approximated from the average of the per-participant slopes here)
    # ------------------------------------------------------------------ #
    pop_effects = load_population_effects(comparison_dir)
    if pop_effects is None:
        pop_effects = (slopes.groupby(["factor", "level", "metric"], as_index=False)["effect"].mean())
    pop_weights = compute_metric_weights(df_scored, active_domains, pop_effects)
    pop_verdict = compute_verdict(df_scored, active_domains, pop_weights, pop_effects)

    print(f"\n{'='*80}\nPOPULATION VERDICT (for comparison)\n{'='*80}")
    for f in PARAM_FACTORS:
        v = pop_verdict.get(f, {})
        print(f"    {f:<8}: {v.get('winner', '?')}")
    print(f"    Population recommendation: {config_string(pop_verdict)}")

    # ------------------------------------------------------------------ #
    # Partial pooling: shrink each participant's raw slope toward the
    # heterogeneity-aware group consensus (see compute_shrunk_slopes)
    # ------------------------------------------------------------------ #
    shrunk_slopes = compute_shrunk_slopes(slopes, df_scored)
    shrunk_slopes.to_csv(out_dir / "participant_slopes_shrunk.csv", index=False)
    print(f"Wrote {out_dir / 'participant_slopes_shrunk.csv'} ({len(shrunk_slopes)} rows)")

    # ------------------------------------------------------------------ #
    # Per-participant verdicts: RAW (own data only) vs SHRUNK (partially
    # pooled with the other participants) -- reported side by side so the
    # effect of borrowing strength from the group is directly visible.
    # ------------------------------------------------------------------ #
    print(f"\n{'='*100}\nPERSONALISED PROTOTYPE RECOMMENDATIONS -- RAW (own data only)\n{'='*100}")
    print(f" {'Participant':<12} {'Length':<8} {'Size':<8} {'Weight':<16} {'Angle':<7} {'Matches Population?':<20}")
    print(f" {'-'*12} {'-'*8} {'-'*8} {'-'*16} {'-'*7} {'-'*20}")

    records = []
    n_match_raw = 0
    n_match_shrunk = 0
    n_changed_by_shrinkage = 0
    for participant, ppt_slopes in slopes.groupby("participant"):
        ppt_weights = compute_metric_weights(df_scored, active_domains, ppt_slopes)
        ppt_verdict = compute_verdict(df_scored, active_domains, ppt_weights, ppt_slopes)
        cfg_raw = config_string(ppt_verdict)
        matches_raw = cfg_raw == config_string(pop_verdict)
        n_match_raw += int(matches_raw)
        length = ppt_verdict.get("Length", {}).get("winner", "?")
        size = ppt_verdict.get("Size", {}).get("winner", "?")
        weight = ppt_verdict.get("Weight", {}).get("winner", "?")
        angle = ppt_verdict.get("Angle", {}).get("winner", "?")
        print(f" {participant:<12} {length:<8} {size:<8} {weight:<16} {angle:<7} {'YES' if matches_raw else 'no, differs':<20}")

        ppt_shrunk = shrunk_slopes[shrunk_slopes["participant"] == participant]
        cfg_shrunk, matches_shrunk = cfg_raw, matches_raw
        if len(ppt_shrunk):
            shrunk_weights = compute_metric_weights(df_scored, active_domains, ppt_shrunk)
            shrunk_verdict = compute_verdict(df_scored, active_domains, shrunk_weights, ppt_shrunk)
            cfg_shrunk = config_string(shrunk_verdict)
            matches_shrunk = cfg_shrunk == config_string(pop_verdict)
        n_match_shrunk += int(matches_shrunk)
        changed = cfg_shrunk != cfg_raw
        n_changed_by_shrinkage += int(changed)

        rec = {"participant": participant, "Length_raw": length, "Size_raw": size,
              "Weight_raw": weight, "Angle_raw": angle, "Recommended_Config_raw": cfg_raw,
              "Matches_Population_raw": matches_raw, "Recommended_Config_shrunk": cfg_shrunk,
              "Matches_Population_shrunk": matches_shrunk, "Changed_by_shrinkage": changed}
        for f in PARAM_FACTORS:
            rec[f"{f}_scores_raw"] = ppt_verdict.get(f, {}).get("level_scores")
        records.append(rec)

    n_ppt = slopes["participant"].nunique()
    print(f"\n{'='*100}\nSHRINKAGE EFFECT ON RECOMMENDATIONS\n{'='*100}")
    print(f"  RAW (own data only):        {n_match_raw}/{n_ppt} match the population pick ({100*n_match_raw/n_ppt:.0f}%)")
    print(f"  SHRUNK (partially pooled):  {n_match_shrunk}/{n_ppt} match the population pick ({100*n_match_shrunk/n_ppt:.0f}%)")
    print(f"  {n_changed_by_shrinkage}/{n_ppt} participants' recommended configuration CHANGED once partial "
          f"pooling was applied -- these are the participants whose own data was too thin/noisy on some "
          f"factor to stand on its own, and were pulled toward the group consensus.")

    pd.DataFrame(records).to_csv(out_dir / "personalised_verdicts.csv", index=False)
    print(f"\nWrote {out_dir / 'personalised_verdicts.csv'}")
    print(f"\nAll outputs saved to:\n  -> {out_dir}")
    print("\nReminder: per-participant estimates rest on ONE person's own session (no "
          "participant-level replication) and should be read as descriptive personalisation, "
          "not as generalisable findings -- see module docstring.")


if __name__ == "__main__":
    main()