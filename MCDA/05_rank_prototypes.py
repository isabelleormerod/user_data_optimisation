#!/usr/bin/env python3
r"""
rank_prototypes.py - Complete & Unified MCDA Prototype Selection Engine

Integrates 0-100 Desirability Scoring, Mixed-Model-Derived Sensitivity Weighting,
Equal Weighting Benchmarking, Height-Stratified Evaluations, a Factor-Level
"Synthesised Best Prototype" summary, and a 3-Pillar Decision Sensitivity
Analysis into a single executable pipeline.

DATA SOURCE: reads the single combined_place_metrics.csv produced by
evaluate_difference.py (pen + posture + hand metrics on one row per Place event).
Falls back to the legacy place_metrics_combined.csv + posture_features_combined.csv
merge only if the combined file is absent. Honours the clean_place_events.py
exclusion manifest, and never analyses a non-High/Medium/Low height stratum.

SCORING MODES (--score-mode):
  directional (default): metrics scored toward their known ergonomic optimum
      (lower REBA/duration/jitter better; higher comfort/smoothness better),
      via METRIC_REGISTRY's 'dir' field. This is the substantive best-prototype
      analysis.
  normative: metrics scored by CLOSENESS to their within-height mean, discarding
      directional knowledge. This mirrors the LDA/fPCA pipeline's distance-from-
      mean target so the two can be compared like-for-like (same optimisation
      target, differing only in feature set). See calculate_normative_scores.
  both: run each, print both leaderboards, and a directional-vs-normative shift
      matrix showing how far each configuration's rank moves between targets.

SENSITIVITY WEIGHTING -- WITHIN-SUBJECT MIXED-MODEL (the metric weighting):
  For each metric, the per-factor signed effect sizes estimated by
  evaluate_difference.py's mixed-effects models (stat_tests.csv /
  stratified_stat_tests.csv) are standardised by that metric's own pooled
  standard deviation, and the squared standardised effects are summed across the
  four prototype factors -- one "how prototype-sensitive is this metric" weight
  per metric. This uses ALL 10 participants via paired, within-subject
  comparisons (every participant tested every level of Length/Size/Weight and all
  three Angle levels), the well-powered comparison established in this project's
  power analysis. If the effect table is absent (evaluate_difference.py not run
  yet), weighting falls back to EQUAL, with a warning.

  An EQUAL-weighting leaderboard is always produced alongside as a benchmark, so
  the data-driven ranking can be read against a neutral baseline. (An earlier
  between-config Kruskal-Wallis weighting has been removed: with only 2-4
  participants per config it was a between-subject test at ~0.09 power vs ~1.0
  for the within-subject mixed model, so it added noise, not information.)

Also computes a FACTOR-LEVEL "synthesised best prototype" per height: rather
than reading off the top of the 24-cell leaderboard (each cell resting on only
2-4 participants), this asks, one factor at a time using the well-powered
within-subject comparison, which single level (e.g. Short vs Long) is
preferable, then combines the winning levels into a recommended configuration.
This is a genuinely different question to "which of the 24 tested combinations
scored highest" and the two are reported side by side so agreement/disagreement
between them is visible. The synthesis now covers ALL FOUR factors including
Angle: evaluate_difference.py's _term_level_effects stores one row per
non-reference LEVEL (so both Angle contrasts, 135-vs-90 and 180-vs-90, are
retained), letting all three angle levels be scored with A90 pinned at 0.

FACTOR-LEVEL TABLE OUTPUT (for the thesis write-up):
  For each scoring mode, in addition to the console verdict, a clean per-parameter
  net-desirability table is written:
    <mode>_factor_table.csv       long form: factor, parameter, S_p, is_winner (per stratum)
    <mode>_factor_table_wide.csv  pivoted:   one column per height (global/High/Medium/Low),
                                             one row per (factor, parameter), each cell = S_p
  S_p is the net WEIGHTED effect of a parameter relative to its reference baseline
  (the intra/inter-domain weights are already folded in); positive = expected
  ergonomic improvement over the reference. This is the table-ready artifact; the
  richer <mode>_metric_weights.csv and <mode>_factor_verdict_detail.csv are kept
  as supplementary per-metric evidence, not for the main table.

Usage:
  python MCDA/05_rank_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python rank_prototypes.py --score-mode both --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python rank_prototypes.py --combined-csv path/to/combined_place_metrics.csv --comparison-dir path/to/combined_comparison
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================================== #
# SECTION 1: METRIC REGISTRY & OPTIMALITY DIRECTION
# =========================================================================== #

METRIC_REGISTRY = {
    # --- DOMAIN 1: Task Performance (min = lower error/duration is better) ---
    "duration_s":                 {"domain": "Performance",     "dir": "min", "label": "Task Duration (s)"},
    "perp_mean_deg":              {"domain": "Performance",     "dir": "min", "label": "Perpendicularity"},
    "leftright_mean_deg":         {"domain": "Performance",     "dir": "min", "label": "L/R Tilt"},
    "updown_mean_deg":            {"domain": "Performance",     "dir": "min", "label": "U/D Tilt"},
    "pos_jitter_mm":              {"domain": "Performance",     "dir": "min", "label": "Positional Jitter"},
    "ang_jitter_deg":             {"domain": "Performance",     "dir": "min", "label": "Angular Jitter"},

    # --- DOMAIN 2: Postural Risk / REBA (min = lower strain/reach is better) ---
    "reba_score_a":               {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score A"},
    "reba_score_b_right":         {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (R)"},
    "reba_score_b_left":          {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Score B (L)"},
    "reba_grand_right":           {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (R)"},
    "reba_grand_left":            {"domain": "Postural_Risk",   "dir": "min", "label": "REBA Grand (L)"},
    "reach_ratio_mean":           {"domain": "Postural_Risk",   "dir": "min", "label": "Reach Ratio"},

    # --- DOMAIN 3: Grip Ergonomics (max = higher comfort / smoother movement is better) ---
    "right_grip_comfort_score":   {"domain": "Grip_Ergonomics", "dir": "max", "label": "R Grip Comfort"},
    "right_sparc_linear":         {"domain": "Grip_Ergonomics", "dir": "max", "label": "R SPARC (Linear)"},
    "right_sparc_angular":        {"domain": "Grip_Ergonomics", "dir": "max", "label": "R SPARC (Angular)"},
}

PARAM_FACTORS = ["Length", "Size", "Weight", "Angle"]
VALID_HEIGHTS = ["High", "Medium", "Low"]    # the only height strata ever analysed

# Reference (baseline) level for each factor, per the default alphabetical/
# numerical treatment-coding used throughout this project:
#   Length: Long < Short   -> reference = Long
#   Size:   Large < Small  -> reference = Large
#   Weight: Front_weighted < Not_weighted -> reference = Front_weighted
#   Angle:  90 < 135 < 180 -> reference = 90
# A positive stored coefficient means "this level scores higher on this metric
# than the reference level". Angle is now included: evaluate_difference.py
# stores one row per LEVEL (not just the single largest contrast), so all
# three angle levels -- including the reference, implicitly at effect=0 -- can
# be compared directly.
FACTOR_REFERENCE = {"Length": "Long", "Size": "Large", "Weight": "Front_weighted", "Angle": "A90"}
FACTOR_LEVEL_LABELS = {
    "Length": {"Long": "Long", "Short": "Short"},
    "Size": {"Large": "Large", "Small": "Small"},
    "Weight": {"Front_weighted": "Front_weighted", "Not_weighted": "Not_weighted"},
    "Angle": {"90": "A90", "135": "A135", "180": "A180"},
}


# =========================================================================== #
# SECTION 2: ROBUST PARSING & DESIRABILITY SCORING ENGINE
# =========================================================================== #

def parse_params(trial_val) -> dict:
    """Robustly parses prototype parameters from trial strings without formatting sensitivity."""
    out = {k: "Other" for k in PARAM_FACTORS}
    if trial_val is None or pd.isna(trial_val):
        return out

    clean_str = str(trial_val).strip()
    tokens = [t.strip() for t in clean_str.split("_") if t.strip()]
    joined_low = "_".join(tokens).lower()

    if "not_weighted" in joined_low or "notweighted" in joined_low:
        out["Weight"] = "Not_weighted"
    elif "front_weighted" in joined_low or "frontweighted" in joined_low:
        out["Weight"] = "Front_weighted"
    elif "weighted" in tokens:
        out["Weight"] = "weighted"

    for tok in tokens:
        t_low = tok.lower(); t_cap = tok.capitalize()
        if t_cap in ("Long", "Short"): out["Length"] = t_cap
        elif t_cap in ("Large", "Small"): out["Size"] = t_cap
        elif t_low.startswith("a") and t_low[1:].isdigit(): out["Angle"] = f"A{t_low[1:]}"
        elif tok.isdigit() and int(tok) in (0, 45, 90, 135, 180, 225, 270, 315): out["Angle"] = f"A{tok}"

    return out


def add_prototype_label(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a unified 'Prototype_Config' column, using existing columns or parsing from 'trial'."""
    df_clean = df.copy()
    parsed = df_clean["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_FACTORS:
        if c not in df_clean.columns or df_clean[c].isna().all() or (df_clean[c] == "Unknown").any():
            df_clean[c] = parsed[c]
        else:
            df_clean[c] = df_clean[c].fillna(parsed[c])
        df_clean[c] = df_clean[c].fillna("Other").astype(str).str.strip()

    df_clean["Prototype_Config"] = df_clean[["Length", "Size", "Weight", "Angle"]].agg("_".join, axis=1)
    return df_clean


def calculate_desirability_scores(df: pd.DataFrame) -> tuple:
    """Converts raw metrics into a standardized 0-100 Desirability Score.

    DIRECTIONAL mode: each metric is scored toward its known ergonomic optimum
    (METRIC_REGISTRY 'dir'). This is the substantive best-prototype analysis."""
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
        if meta["dir"] == "min":
            df_scored[score_col] = 100.0 * (v_max - vals) / (v_max - v_min)
        else:
            df_scored[score_col] = 100.0 * (vals - v_min) / (v_max - v_min)

        active_domains[meta["domain"]].append(score_col)

    return df_scored, active_domains


def calculate_normative_scores(df: pd.DataFrame, stratum_col: str = "height") -> tuple:
    """Alternative 0-100 desirability mirroring the LDA/fPCA pipeline's
    "closeness to normal" target instead of the directional optimum.

    For each metric, an event is desirable insofar as it keeps the participant
    CLOSE TO the metric's central value rather than pushing them to an extreme
    in either direction -- the direct analogue of the LDA distance-from-mean
    score |c_{i,f}|, applied to hand-crafted metrics. This makes the two
    pipelines optimise for the SAME thing (normative stability), so any
    remaining ranking disagreement isolates to the feature set (emergent fPCA
    components vs. named metrics) rather than to the optimisation target.

    "Normal" is the metric's mean computed WITHIN each height stratum (matching
    the LDA, which mean-centres per stratum), so height differences do not leak
    into "deviation". Desirability is the min-max-normalised inverse of the
    absolute deviation:

        d_{i,m} = 100 * (max|m - mean_h| - |m_i - mean_h|)
                        / (max|m - mean_h| - min|m - mean_h|)

    NOTE: this deliberately DISCARDS the directional knowledge in
    METRIC_REGISTRY's 'dir' field (that lower REBA is genuinely better, etc.).
    That is appropriate ONLY because the purpose is to match the LDA's
    optimisation target for a like-for-like comparison -- it is not a claim that
    closeness-to-mean is the right way to rank metrics whose good direction is
    known. Returns (df_scored, active_domains) using the same _dscore column
    names, so all downstream weighting/aggregation is reused unchanged."""
    df_scored = df.copy()
    active_domains = defaultdict(list)
    has_stratum = stratum_col in df_scored.columns

    for col, meta in METRIC_REGISTRY.items():
        if col not in df_scored.columns:
            continue
        vals = pd.to_numeric(df_scored[col], errors="coerce")
        if vals.dropna().empty or vals.nunique() <= 1:
            continue

        if has_stratum:
            stratum_mean = df_scored.groupby(stratum_col)[col].transform(
                lambda x: pd.to_numeric(x, errors="coerce").mean())
            abs_dev = (vals - stratum_mean).abs()
        else:
            abs_dev = (vals - vals.mean()).abs()

        v_min, v_max = abs_dev.min(), abs_dev.max()
        if pd.isna(v_max) or abs(v_max - v_min) < 1e-9:
            continue

        score_col = f"{col}_dscore"
        # 100 = smallest deviation (closest to normal); 0 = largest deviation.
        df_scored[score_col] = 100.0 * (v_max - abs_dev) / (v_max - v_min)
        active_domains[meta["domain"]].append(score_col)

    return df_scored, active_domains


def _equal_weights(active_domains: dict) -> dict:
    """Neutral fallback weighting: every active metric weighted 1.0 (so the
    data-driven ranking degrades gracefully to equal weighting when no
    mixed-model effect table is available)."""
    return {c: 1.0 for cols in active_domains.values() for c in cols}


def load_mixedmodel_effects(comparison_dir: Path, stratified: bool):
    """Load evaluate_difference.py's mixed-model effect tables. Returns None (with
    a console warning) if the expected file is missing, so callers can fall back
    gracefully rather than crash."""
    if comparison_dir is None:
        return None
    fname = "stratified_stat_tests.csv" if stratified else "stat_tests.csv"
    path = comparison_dir / fname
    if not path.is_file():
        print(f"  [WARN] {path} not found -- run evaluate_difference.py (--mode compare) first "
              f"to enable within-subject mixed-model weighting.")
        return None
    return pd.read_csv(path)


def compute_mixedmodel_weights(df_scored: pd.DataFrame, active_domains: dict,
                               effect_table: pd.DataFrame, stratum: str = None) -> dict:
    """
    (A) PREFERRED. For each metric, standardise its mixed-model coefficient for
    each prototype factor by that metric's own pooled standard deviation (an
    unstandardised regression coefficient in the metric's raw units is not
    comparable across metrics with different units), then sum the squared
    standardised effects across Length/Size/Weight/Angle. This gives one
    unitless "how much does prototype choice move this metric" weight per
    metric, computed from the well-powered within-subject test.
    """
    if effect_table is None:
        return None
    tbl = effect_table
    if stratum is not None and "stratum" in tbl.columns:
        tbl = tbl[tbl["stratum"] == stratum]
    elif stratum is not None:
        return None  # asked for a stratum but table has none -- caller should fall back

    weights = {}
    for domain, cols in active_domains.items():
        for col in cols:
            raw = col.replace("_dscore", "")
            if raw not in df_scored.columns:
                weights[col] = 0.01
                continue
            sd = pd.to_numeric(df_scored[raw], errors="coerce").std(ddof=1)
            rows = tbl[(tbl["metric"] == raw) & (tbl["factor"].isin(PARAM_FACTORS))]
            if not sd or pd.isna(sd) or sd < 1e-9 or rows.empty:
                weights[col] = 0.01
                continue
            std_effects_sq = [(r["effect"] / sd) ** 2 for _, r in rows.iterrows() if pd.notna(r["effect"])]
            weights[col] = max(0.01, float(sum(std_effects_sq))) if std_effects_sq else 0.01
    return weights


def _event_key(participant, trial, height, place_index):
    """Match key against the cleaning manifest; place_index is per-height (== the
    manifest's place_index_in_height)."""
    try:
        pi = int(place_index)
    except (ValueError, TypeError):
        return None
    return (str(participant), str(trial), str(height), pi)


def load_excluded_events(exclude_csv):
    """Load excluded_place_events.csv (clean_place_events.py) as a set of
    (participant, trial, height, place_index) keys. Empty set if absent."""
    if exclude_csv is None or not Path(exclude_csv).is_file():
        print(f"  [exclude] no cleaning manifest at {exclude_csv}; no events excluded.")
        return set()
    ex = pd.read_csv(exclude_csv)
    need = {"participant", "trial", "height", "place_index_in_height"}
    if not need.issubset(ex.columns):
        print(f"  [exclude] manifest missing {need - set(ex.columns)}; skipping exclusion.")
        return set()
    keys = {k for k in (_event_key(r["participant"], r["trial"], r["height"], r["place_index_in_height"])
                        for _, r in ex.iterrows()) if k is not None}
    print(f"  [exclude] loaded {len(keys)} rejected place event(s) from {Path(exclude_csv).name}")
    return keys


# =========================================================================== #
# SECTION 3: UNIFIED SCORING & RANKING ENGINE
# =========================================================================== #

def score_and_rank(df: pd.DataFrame, active_domains: dict, metric_weights: dict,
                   strategy: str = "data_driven", domain_weights_override: dict = None) -> pd.DataFrame:
    """
    Unified scoring engine. metric_weights supplies the intra/inter-domain
    weighting (from whichever method computed it -- (A), (B), or equal).

    Grand_Score is computed via TWO-STAGE aggregation -- first to one row per
    (Prototype_Config, participant), THEN to one row per Prototype_Config --
    rather than averaging raw place events directly, preventing a participant
    who contributed more trials to a config from pulling its score toward their
    own results. Score_SD is the standard deviation ACROSS PARTICIPANT-LEVEL
    MEANS (between-participant spread), not across raw trials.
    """
    df_calc = df.copy()
    domain_names = list(active_domains.keys())

    # 1. Intra-Domain Weighting (per place-event row; deterministic function of that row)
    for domain, cols in active_domains.items():
        if strategy == "equal":
            weights = {c: (1.0 / len(cols)) for c in cols}
        else:
            dom_w_sum = sum(metric_weights.get(c, 0.01) for c in cols)
            weights = {c: (metric_weights.get(c, 0.01) / dom_w_sum) for c in cols}

        df_calc[f"Domain_{domain}"] = 0.0
        for c in cols:
            df_calc[f"Domain_{domain}"] += df_calc[c].fillna(0.0) * weights[c]

    # 2. Inter-Domain Weighting
    if domain_weights_override:
        domain_w = domain_weights_override
    elif strategy == "equal":
        domain_w = {d: 1.0 for d in domain_names}
    else:
        domain_w = {d: np.mean([metric_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}

    tot_dom_w = sum(domain_w.values()) if sum(domain_w.values()) > 0 else 1.0

    df_calc["Grand_Score"] = 0.0
    for d in domain_names:
        w = domain_w.get(d, 0.0) / tot_dom_w
        df_calc["Grand_Score"] += df_calc[f"Domain_{d}"] * w

    # 3. TWO-STAGE aggregation: place events -> participant means -> config means
    ppt_agg = {f"Domain_{d}": (f"Domain_{d}", "mean") for d in domain_names}
    ppt_agg["Grand_Score"] = ("Grand_Score", "mean")
    ppt_agg["N_Events"] = ("Grand_Score", "size")
    has_participant = "participant" in df_calc.columns
    group_keys = ["Prototype_Config", "participant"] if has_participant else ["Prototype_Config"]
    stage1 = df_calc.groupby(group_keys, dropna=False).agg(**ppt_agg).reset_index()

    if has_participant:
        stage2 = stage1.groupby("Prototype_Config", dropna=False).agg(
            Grand_Score=("Grand_Score", "mean"),
            Score_SD=("Grand_Score", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
            N_Participants=("participant", "nunique"),
            N_Events=("N_Events", "sum"),
            **{d: (f"Domain_{d}", "mean") for d in domain_names},
        ).reset_index()
    else:
        stage2 = stage1.rename(columns={f"Domain_{d}": d for d in domain_names})
        stage2["Score_SD"] = 0.0
        stage2["N_Participants"] = np.nan

    rankings = stage2.sort_values(by="Grand_Score", ascending=False).reset_index(drop=True)
    rankings["Rank"] = rankings.index + 1

    cols = ["Rank", "Prototype_Config", "Grand_Score", "Score_SD", "N_Participants", "N_Events"] + domain_names
    return rankings[[c for c in cols if c in rankings.columns]]


def _aggregate_dispersion(by_pid, active_domains, min_participants, metric_weights=None,
                          return_breakdown=False):
    """Collapse the per-metric between-participant SDs (of participant-mean
    desirability) into a single dispersion for one level.

    metric_weights is None  -> equal weight within a domain, equal across domains
                               (the original behaviour).
    metric_weights supplied -> weighted mean within a domain by each metric's
                               sensitivity weight, then weighted mean across
                               domains by each domain's MEAN metric weight. The
                               weights are magnitudes only (squared standardised
                               effects), so NO directionality enters -- this is
                               importance-weighting of a spread, not optimisation.
    Weight keys are the '<metric>_dscore' column names (same keys metric_weights
    already uses). Returns the scalar dispersion, or (scalar, {domain: disp}) if
    return_breakdown."""
    domain_disp = {}
    for domain, cols in active_domains.items():
        sd_pairs = []          # (col, sd) for metrics with enough participants
        for c in cols:
            ppt_means = by_pid[c].mean().dropna()
            if len(ppt_means) >= max(2, min_participants):
                sd_pairs.append((c, float(ppt_means.std(ddof=1))))
        if not sd_pairs:
            domain_disp[domain] = np.nan
            continue
        if metric_weights:
            num = sum(metric_weights.get(c, 0.01) * sd for c, sd in sd_pairs)
            den = sum(metric_weights.get(c, 0.01) for c, sd in sd_pairs)
            domain_disp[domain] = (num / den) if den else np.nan
        else:
            domain_disp[domain] = float(np.mean([sd for _, sd in sd_pairs]))

    present = [d for d, v in domain_disp.items() if v == v]
    if not present:
        agg = np.nan
    elif metric_weights:
        # inter-domain: weight each domain by its mean metric sensitivity, so a
        # domain of more-prototype-sensitive metrics counts for more (mirrors the
        # directional synthesis's inter-domain weighting).
        dom_w = {d: float(np.mean([metric_weights.get(c, 0.01) for c in active_domains[d]]))
                 for d in present}
        num = sum(dom_w[d] * domain_disp[d] for d in present)
        den = sum(dom_w[d] for d in present)
        agg = (num / den) if den else np.nan
    else:
        agg = float(np.nanmean([domain_disp[d] for d in present]))

    return (agg, domain_disp) if return_breakdown else agg


def _mcda_level_dispersions(df_sub, labels, active_domains, domain_names, min_participants,
                            metric_weights=None):
    """Level -> between-participant dispersion of desirability scores, for a given
    (possibly permuted) level-label assignment aligned to df_sub rows. Aggregation
    (equal or sensitivity-weighted) is delegated to _aggregate_dispersion so the
    permutation null uses EXACTLY the same weighting as the observed value."""
    tmp = df_sub.assign(_lev=labels)
    disps = {}
    for lev, lev_df in tmp.groupby("_lev"):
        by_pid = lev_df.groupby("participant")
        disps[str(lev)] = _aggregate_dispersion(by_pid, active_domains, min_participants, metric_weights)
    return disps


def _disp_gap(disps):
    vals = [v for v in disps.values() if v == v]
    return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan


def _consistency_perm_pvalue(df_sub, factor, active_domains, domain_names, min_participants,
                             n_perm, seed=0, metric_weights=None):
    """Within-participant permutation test on the per-level dispersion GAP
    (max - min level dispersion). Shuffling level labels WITHIN each participant
    preserves each person's own movement and only breaks the level<->dispersion
    link -- the null 'this factor's levels don't differ in between-participant
    consistency'. p = fraction of shuffled gaps >= observed. metric_weights is
    passed through unchanged to BOTH the observed and permuted dispersions, so the
    null is on the same (weighted or equal) scale as the observed statistic."""
    labels0 = df_sub[factor].astype(str).values
    obs = _disp_gap(_mcda_level_dispersions(df_sub, labels0, active_domains, domain_names,
                                            min_participants, metric_weights))
    if not (obs == obs):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    parts = df_sub["participant"].values
    idx_by_p = [np.where(parts == p)[0] for p in pd.unique(parts)]
    ge, valid = 0, 0
    for _ in range(n_perm):
        perm = labels0.copy()
        for idx in idx_by_p:
            perm[idx] = rng.permutation(labels0[idx])
        g = _disp_gap(_mcda_level_dispersions(df_sub, perm, active_domains, domain_names,
                                              min_participants, metric_weights))
        if g == g:
            valid += 1
            if g >= obs - 1e-12:
                ge += 1
    return (obs, (ge + 1) / (valid + 1) if valid else np.nan)


def compute_factor_level_consistency(df_scored: pd.DataFrame, active_domains: dict,
                                     min_participants: int = 2, n_perm: int = 1000, alpha: float = 0.05,
                                     metric_weights: dict = None):
    """FACTOR-LEVEL consistency -- the WELL-POWERED analogue of the config-level
    dispersion leaderboard. Instead of asking how tightly participants converge on
    each of the 24 exact configs (2-4 people each), ask how tightly they converge
    on each LEVEL of each factor (Short vs Long, ...). Because the design is
    within-subject on every factor, ALL participants contribute to every level, so
    each estimate uses the full pool.

    For each (factor, level): average each participant's desirability scores first,
    take the between-participant SD across the whole pool, aggregate equal-weight
    across a domain's metrics then across domains. Lower dispersion = that level
    makes people converge more.

    A within-participant permutation test (n_perm) on the best-minus-worst
    dispersion gap gives a p-value per factor; a factor whose p >= alpha is marked
    NOT significant and given NO winner (so a 0.2%, height-flipping 'win' like
    Length isn't reported as real). Returns (rows_df, verdict), verdict[factor] =
    {'winner', 'level_disp', 'p_value', 'significant'}.

    CAVEAT: a participant's per-level mean pools over the other three factors, so
    some spread reflects those rather than pure between-person disagreement (this
    also slightly inflates the permutation significance). 'Most convergent' is NOT
    'ergonomically best' -- compare against the directional factor verdict."""
    if "participant" not in df_scored.columns:
        return pd.DataFrame(), {}
    domain_names = list(active_domains.keys())
    rows, verdict = [], {}
    for factor in PARAM_FACTORS:
        if factor not in df_scored.columns:
            continue
        level_disp = {}
        for lev, lev_df in df_scored.groupby(factor, dropna=True):
            by_pid = lev_df.groupby("participant")
            n_part = by_pid.ngroups
            # Same aggregator (and same weights) the permutation null uses, so the
            # observed dispersion and the null distribution are always comparable.
            disp, domain_d = _aggregate_dispersion(by_pid, active_domains, min_participants,
                                                   metric_weights, return_breakdown=True)
            level_disp[str(lev)] = disp
            row = {"factor": factor, "level": str(lev), "dispersion": disp,
                   "n_participants": n_part, "n_events": len(lev_df)}
            for d in domain_names:
                row[d] = domain_d.get(d, np.nan)
            rows.append(row)
        valid = {k: v for k, v in level_disp.items() if v == v}
        best = min(valid, key=valid.get) if valid else None
        sub = df_scored[df_scored[factor].notna()]
        _, pval = _consistency_perm_pvalue(sub, factor, active_domains, domain_names, min_participants,
                                           n_perm, metric_weights=metric_weights) \
            if best is not None else (np.nan, np.nan)
        sig = bool(pval == pval and pval < alpha)
        verdict[factor] = {"winner": best if sig else None, "best_level": best,
                           "level_disp": level_disp, "p_value": pval, "significant": sig}
        for r in rows:
            if r["factor"] == factor:
                r["p_value"] = pval
                r["significant"] = sig
    return pd.DataFrame(rows), verdict


def print_factor_level_consistency(verdict: dict, label: str):
    print(f"\n--- Factor-Level CONSISTENCY Verdict -- {label} (most-convergent level per factor) ---")
    parts = []
    for f in PARAM_FACTORS:
        v = verdict.get(f)
        if not v or v.get("best_level") is None:
            print(f"    {f:<8}: no verdict (insufficient participants per level)")
            parts.append("?")
            continue
        ordered = sorted(v["level_disp"].items(), key=lambda x: (x[1] if x[1] == x[1] else 9e99))
        scores = ", ".join(f"{lv}={d:.3f}" for lv, d in ordered)
        p = v.get("p_value", np.nan)
        ptxt = f"p={p:.3f}" if p == p else "p=n/a"
        if v.get("significant"):
            print(f"    {f:<8}: {v['winner']:<14} [dispersion: {scores}]  {ptxt}")
            parts.append(v["winner"])
        else:
            print(f"    {f:<8}: (n.s.) best={v['best_level']:<9} [dispersion: {scores}]  {ptxt}  -> no winner")
            parts.append("?")
    print(f"    Synthesised MOST-CONVERGENT config: {'_'.join(parts)}   (significant factors only; ? = n.s./undecided)")


# =========================================================================== #
# SECTION 3B: FACTOR-LEVEL "SYNTHESISED BEST PROTOTYPE"
#   A different question to the 24-cell leaderboard above: which single LEVEL
#   of each factor is preferable, using the well-powered within-subject
#   comparison, then combine winning levels into a recommended configuration.
#   See module docstring for the distinction.
# =========================================================================== #

def compute_factor_level_verdict(df_scored: pd.DataFrame, active_domains: dict,
                                 metric_weights: dict, effect_table: pd.DataFrame,
                                 stratum: str = None) -> dict:
    """Returns {factor: {"winner": level, "level_scores": {level: score}, "detail": [...]}}
    for EVERY prototype factor, including Angle. Uses the SAME intra/inter domain
    weights as Grand_Score, applied to signed, standardised mixed-model effects
    rather than raw desirability scores -- so a level's aggregate score is "how
    much would choosing this level over the reference be expected to move
    Grand_Score, on the evidence of the within-subject effects." The reference
    level always has an implicit score of 0 (it IS the baseline every other
    level is measured against); non-reference levels are compared to that
    baseline AND to each other, and the overall maximum wins.

    'detail' rows are (raw_metric, level_label, std_effect, eff_weight, contribution)
    where eff_weight = intra_w * inter_w (the actual weight applied to that metric
    in this verdict) and contribution = std_effect * direction * eff_weight (the
    signed amount that metric adds to the level's score). Summing 'contribution'
    over a (factor, level) reproduces that level's entry in 'level_scores'.

    NOTE: this synthesis is inherently DIRECTIONAL (it uses METRIC_REGISTRY's
    'dir' to decide which way is better), so it is only meaningful for the
    directional score mode. It is skipped for the normative comparison mode,
    where "better direction" is deliberately undefined."""
    if effect_table is None:
        return {f: {"winner": None, "reason": "no mixed-model effect table available"} for f in PARAM_FACTORS}

    tbl = effect_table
    if stratum is not None and "stratum" in tbl.columns:
        tbl = tbl[tbl["stratum"] == stratum]
    elif stratum is not None:
        return {f: {"winner": None, "reason": f"no per-stratum effects available for {stratum}"} for f in PARAM_FACTORS}

    domain_names = list(active_domains.keys())
    domain_w = {d: np.mean([metric_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}
    tot_dom_w = sum(domain_w.values()) or 1.0

    verdicts = {}
    for factor in PARAM_FACTORS:
        ref_label = FACTOR_REFERENCE[factor]
        level_scores = defaultdict(float)   # includes the reference, implicitly 0 unless touched
        level_scores[ref_label] = 0.0
        detail = []
        any_metric = False
        for domain, cols in active_domains.items():
            dom_w_sum = sum(metric_weights.get(c, 0.01) for c in cols) or 1.0
            inter_w = (domain_w.get(domain, 0.0) / tot_dom_w)
            for col in cols:
                raw = col.replace("_dscore", "")
                meta = METRIC_REGISTRY.get(raw)
                if meta is None or raw not in df_scored.columns:
                    continue
                sd = pd.to_numeric(df_scored[raw], errors="coerce").std(ddof=1)
                rows = tbl[(tbl["metric"] == raw) & (tbl["factor"] == factor)]
                if rows.empty or not sd or pd.isna(sd) or sd < 1e-9:
                    continue
                direction = 1.0 if meta["dir"] == "max" else -1.0
                intra_w = metric_weights.get(col, 0.01) / dom_w_sum
                for _, r in rows.iterrows():
                    if pd.isna(r["effect"]) or pd.isna(r.get("level")):
                        continue
                    level_label = FACTOR_LEVEL_LABELS.get(factor, {}).get(str(r["level"]), str(r["level"]))
                    std_effect = r["effect"] / sd
                    contribution = std_effect * direction * intra_w * inter_w
                    level_scores[level_label] += contribution
                    eff_weight = intra_w * inter_w          # the actual weight on this metric
                    detail.append((raw, level_label, round(std_effect, 3),
                                   round(eff_weight, 4), round(contribution, 4)))
                    any_metric = True
        if not any_metric:
            verdicts[factor] = {"winner": None, "reason": "no metric had an estimable effect for this factor here"}
            continue
        winner = max(level_scores, key=level_scores.get)
        verdicts[factor] = {"winner": winner, "level_scores": dict(level_scores), "detail": detail}
    return verdicts


def print_factor_level_verdicts(verdicts: dict, label: str, show_weights: bool = True):
    print(f"\n--- Factor-Level Synthesised Verdict -- {label} ---")
    config_parts = []
    for f in PARAM_FACTORS:
        v = verdicts.get(f, {})
        if v.get("winner") is None:
            print(f"    {f:<8}: no verdict ({v.get('reason', 'unknown')})")
            config_parts.append("?")
            continue
        scores_str = ", ".join(f"{lvl}={sc:+.4f}" for lvl, sc in sorted(v["level_scores"].items(), key=lambda x: -x[1]))
        print(f"    {f:<8}: {v['winner']:<14} [{scores_str}]")
        config_parts.append(v["winner"])

        if show_weights and v.get("detail"):
            # Effective weight is per-metric (same across that metric's levels),
            # so collapse to one row per metric; show the signed contribution at
            # the WINNING level so the sign lines up with why it won.
            eff_w = {}
            win_contrib = {}
            for raw, level_label, std_effect, eff_weight, contribution in v["detail"]:
                eff_w[raw] = eff_weight
                if level_label == v["winner"]:
                    win_contrib[raw] = contribution
            for raw in sorted(eff_w, key=lambda m: -eff_w[m]):
                wc = win_contrib.get(raw)
                wc_str = f"{wc:+.4f}" if wc is not None else "   ref"
                print(f"        {raw:<26} weight={eff_w[raw]:.4f}  contrib@{v['winner']}={wc_str}")

    length, size, weight, angle = config_parts
    print(f"    Synthesised recommendation: {length}_{size}_{weight}_{angle}")


def print_factor_level_table(verdicts: dict, label: str):
    """Table-ready net-desirability (S_p) per parameter level. One line per
    (factor, level); reference level shows +0.0000; winner marked with *.
    This is the clean per-parameter view for the thesis table -- no per-metric
    weights, just the aggregate S_p each parameter earns."""
    print(f"\n--- Net Desirability by Parameter (S_p) -- {label} ---")
    print(f"    {'Factor':<8} {'Parameter':<16} {'S_p':>9}  win")
    print(f"    {'-'*8} {'-'*16} {'-'*9}  ---")
    for f in PARAM_FACTORS:
        v = verdicts.get(f, {})
        if v.get("winner") is None:
            print(f"    {f:<8} {'(no verdict)':<16} {'n/a':>9}")
            continue
        for lvl, sc in sorted(v["level_scores"].items(), key=lambda x: -x[1]):
            star = "*" if lvl == v["winner"] else " "
            print(f"    {f:<8} {lvl:<16} {sc:>+9.4f}   {star}")


# =========================================================================== #
# SECTION 5: FORMATTED ASCII REPORTING
# =========================================================================== #

def print_ascii_leaderboard(rankings: pd.DataFrame, title: str, top_n: int = 10):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    domains = [d for d in rankings.columns if d not in ("Rank", "Prototype_Config", "Grand_Score", "Score_SD", "N_Participants", "N_Events")]
    domain_headers = "  ".join([f"{d[:10]:>10}" for d in domains])

    print(f" {'Rk':<3} {'Prototype Configuration':<32} {'Grand':>6} {'(SD)':>6} {'Np':>3} {'Ne':>4} | {domain_headers}")
    print(f" {'-'*3} {'-'*32} {'-'*6} {'-'*6} {'-'*3} {'-'*4}-+-{'-'*len(domain_headers)}")

    for _, r in rankings.head(top_n).iterrows():
        domain_vals = "  ".join([f"{r[d]:>10.1f}" for d in domains])
        np_val = r.get("N_Participants", np.nan)
        np_str = f"{int(np_val):>3}" if pd.notna(np_val) else "  ?"
        print(f" {int(r['Rank']):<3} {r['Prototype_Config']:<32} {r['Grand_Score']:>6.1f} ({r['Score_SD']:>4.1f}) {np_str} {int(r['N_Events']):>4} | {domain_vals}")

    if len(rankings) > top_n:
        print(f" ... and {len(rankings) - top_n} more configurations.")
    print(f" *(Scores 0-100; Np = distinct participants contributing to this config -- typically 2-4")
    print(f"   of 10, given the incomplete-block allocation. Individual config ranks therefore rest on")
    print(f"   substantially weaker evidence than the factor-level verdicts above.)*")


def print_target_shift_matrix(directional_ranks: pd.DataFrame, normative_ranks: pd.DataFrame, out_dir: Path):
    """Directional-vs-normative comparison: how far each configuration's rank
    moves when the optimisation target switches from 'toward the ergonomic
    optimum' to 'closest to the population norm'. A configuration that ranks
    well under BOTH is robust to the choice of target; a large shift flags a
    configuration whose standing depends entirely on which target is chosen.
    Also reports Spearman rank correlation between the two orderings as a single
    summary of how much the target choice matters."""
    print(f"\n{'='*85}\nOPTIMISATION-TARGET SHIFT MATRIX (Directional 'best' vs. Normative 'closest-to-mean')\n{'='*85}")

    merged = pd.merge(
        directional_ranks[["Prototype_Config", "Rank", "Grand_Score"]].rename(
            columns={"Rank": "Dir_Rank", "Grand_Score": "Dir_Score"}),
        normative_ranks[["Prototype_Config", "Rank", "Grand_Score"]].rename(
            columns={"Rank": "Norm_Rank", "Grand_Score": "Norm_Score"}),
        on="Prototype_Config", how="outer",
    )
    merged["Rank_Shift"] = (merged["Dir_Rank"] - merged["Norm_Rank"]).abs()
    merged = merged.sort_values("Dir_Rank").reset_index(drop=True)

    print(f" {'Prototype Configuration':<32} | {'Dir Rk':>7} {'(Score)':>8} | {'Norm Rk':>8} {'(Score)':>8} | {'|Shift|':>7}")
    print(f" {'-'*32}-+-{'-'*7}-{'-'*8}-+-{'-'*8}-{'-'*8}-+-{'-'*7}")
    for _, r in merged.iterrows():
        dr = f"{int(r['Dir_Rank'])}" if pd.notna(r["Dir_Rank"]) else "n/a"
        nr = f"{int(r['Norm_Rank'])}" if pd.notna(r["Norm_Rank"]) else "n/a"
        ds = f"{r['Dir_Score']:.1f}" if pd.notna(r["Dir_Score"]) else "  n/a"
        ns = f"{r['Norm_Score']:.1f}" if pd.notna(r["Norm_Score"]) else "  n/a"
        sh = f"{int(r['Rank_Shift'])}" if pd.notna(r["Rank_Shift"]) else "n/a"
        print(f" {r['Prototype_Config']:<32} | {dr:>7} {ds:>8} | {nr:>8} {ns:>8} | {sh:>7}")

    both = merged.dropna(subset=["Dir_Rank", "Norm_Rank"])
    if len(both) >= 3:
        rho, p = stats.spearmanr(both["Dir_Rank"], both["Norm_Rank"])
        print(f"\n Spearman rank correlation (directional vs normative ordering): "
              f"rho = {rho:+.3f} (p = {p:.3f}, n = {len(both)})")
        if rho >= 0.7:
            print("   -> Strong agreement: the two optimisation targets largely coincide.")
        elif rho >= 0.4:
            print("   -> Moderate agreement: target choice shifts some, not all, of the ranking.")
        else:
            print("   -> Weak agreement: the ranking depends substantially on which target is chosen.")
    print("\n *(Directional = metrics scored toward their known ergonomic optimum.")
    print("   Normative = metrics scored by closeness to the within-height mean, matching")
    print("   the LDA/fPCA pipeline's target. Comparing the two isolates the effect of the")
    print("   optimisation TARGET; comparing normative here against the fPCA leaderboard then")
    print("   isolates the effect of the FEATURE SET.)*")

    merged.to_csv(out_dir / "target_shift_matrix.csv", index=False)


# =========================================================================== #
# SECTION 6: COMMAND LINE INTERFACE & MAIN EXECUTION
# =========================================================================== #

def _verdict_rows(verdict, mode_tag, stratum):
    """Flatten a factor-level verdict dict into tidy rows for CSV: one row per
    (factor, level) with its synthesis score and a winner flag, plus a note row
    when a factor had no estimable verdict."""
    rows = []
    for factor, v in verdict.items():
        if v.get("winner") is None:
            rows.append({"mode": mode_tag, "stratum": stratum, "factor": factor,
                         "level": None, "score": np.nan, "is_winner": False,
                         "note": v.get("reason", "no verdict")})
            continue
        for lvl, sc in v.get("level_scores", {}).items():
            rows.append({"mode": mode_tag, "stratum": stratum, "factor": factor,
                         "level": lvl, "score": float(sc), "is_winner": bool(lvl == v["winner"]),
                         "note": ""})
    return rows


def _weights_to_rows(weights, active_domains, mode_tag, stratum):
    """metric_weights dict -> tidy rows. These are the RAW per-metric sensitivity
    weights (sum of squared standardised effects across the four factors), i.e.
    what compute_mixedmodel_weights returned BEFORE any intra/inter normalisation.
    See _verdict_detail_rows for the effective weight actually applied per verdict."""
    col_to_domain = {c: d for d, cols in active_domains.items() for c in cols}
    rows = []
    for score_col, w in (weights or {}).items():
        rows.append({"mode": mode_tag, "stratum": stratum,
                     "domain": col_to_domain.get(score_col, "?"),
                     "metric": score_col.replace("_dscore", ""),
                     "raw_weight": float(w)})
    return rows


def _verdict_detail_rows(verdict, mode_tag, stratum):
    """Per-metric, per-level contribution breakdown behind each factor verdict.
    'contribution' already folds in direction * intra_w * inter_w, so summing
    contribution over (factor, level) reproduces that level's synthesis score.
    This is the supplementary/appendix evidence, NOT the main table."""
    rows = []
    for factor, v in verdict.items():
        for raw, level_label, std_effect, eff_weight, contribution in v.get("detail", []):
            rows.append({"mode": mode_tag, "stratum": stratum, "factor": factor,
                         "metric": raw, "level": level_label,
                         "std_effect": std_effect, "eff_weight": eff_weight,
                         "contribution": contribution})
    return rows


def _verdict_table_rows(verdict, mode_tag, stratum):
    """Flat (factor, parameter, S_p, is_winner) rows for the clean thesis table.
    S_p == the parameter's entry in level_scores (net weighted effect vs the
    reference baseline; reference sits at 0). One row per (factor, parameter)."""
    rows = []
    for factor, v in verdict.items():
        if v.get("winner") is None:
            continue
        for lvl, sc in v.get("level_scores", {}).items():
            rows.append({"mode": mode_tag, "stratum": stratum, "factor": factor,
                         "parameter": lvl, "S_p": round(float(sc), 4),
                         "is_winner": bool(lvl == v["winner"])})
    return rows


def run_full_ranking(df_scored, active_domains, global_effects, strat_effects,
                     comparison_dir, out_dir, mode_tag, do_verdicts):
    """Run the full weighting -> ranking -> stratified -> sensitivity flow for a
    single scoring mode. Returns the global data-driven ranking (for the
    directional-vs-normative comparison). mode_tag prefixes output filenames so
    directional and normative outputs don't collide. do_verdicts controls the
    factor-level synthesis, which is only meaningful for the directional mode."""
    mm_weights = compute_mixedmodel_weights(df_scored, active_domains, global_effects, stratum=None)

    if mm_weights is None:
        print(f"\n[WARN] ({mode_tag}) No mixed-model effect table -- falling back to EQUAL metric "
              "weighting. Run evaluate_difference.py (--mode compare) to enable within-subject weighting.")
        global_weights = _equal_weights(active_domains)
    else:
        global_weights = mm_weights

    data_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="data_driven")
    equal_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="equal")
    data_ranks.to_csv(out_dir / f"{mode_tag}_rankings_data_driven_global.csv", index=False)
    equal_ranks.to_csv(out_dir / f"{mode_tag}_rankings_equal_weight_global.csv", index=False)

    print_ascii_leaderboard(data_ranks,
        f"[{mode_tag.upper()}] GLOBAL PROTOTYPE LEADERBOARD (24-Cell Configurations)", top_n=10)

    # Accumulate the weighting artifacts as we go, so every stratum's weights and
    # every verdict's contribution breakdown are captured, not just the global one.
    weight_rows = _weights_to_rows(global_weights, active_domains, mode_tag, "global")
    verdict_rows, verdict_detail_rows, table_rows = [], [], []

    if do_verdicts:
        global_verdict = compute_factor_level_verdict(df_scored, active_domains, global_weights, global_effects, stratum=None)
        # show_weights=False -> keep the console verdict clean (no per-metric lines);
        # the per-parameter S_p table is printed by print_factor_level_table below.
        print_factor_level_verdicts(global_verdict, f"{mode_tag.upper()} GLOBAL (all heights pooled)", show_weights=False)
        print_factor_level_table(global_verdict, f"{mode_tag.upper()} GLOBAL")
        verdict_rows        += _verdict_rows(global_verdict, mode_tag, "global")
        verdict_detail_rows += _verdict_detail_rows(global_verdict, mode_tag, "global")
        table_rows          += _verdict_table_rows(global_verdict, mode_tag, "global")

    # Stratified leaderboards -- only ever the real High/Medium/Low strata.
    strata = [h for h in VALID_HEIGHTS if "height" in df_scored.columns and h in set(df_scored["height"].dropna())]
    for s in strata:
        sub_df = df_scored[df_scored["height"] == s]
        if sub_df.empty:
            continue
        s_mm = compute_mixedmodel_weights(sub_df, active_domains, strat_effects, stratum=s)
        s_weights = s_mm if s_mm is not None else _equal_weights(active_domains)
        weight_rows += _weights_to_rows(s_weights, active_domains, mode_tag, s)
        s_ranks = score_and_rank(sub_df, active_domains, s_weights, strategy="data_driven")
        s_ranks.to_csv(out_dir / f"{mode_tag}_rankings_stratified_{s}.csv", index=False)
        print_ascii_leaderboard(s_ranks, f"[{mode_tag.upper()}] STRATIFIED LEADERBOARD -- {s.upper()} WORKSTATION", top_n=5)

        # Stratified factor-level synthesised verdict (per-height "best level" per
        # factor, from the stratified within-subject effects). Prototype effects
        # vary by working height, so this is reported per stratum, not just global.
        if do_verdicts:
            s_verdict = compute_factor_level_verdict(sub_df, active_domains, s_weights, strat_effects, stratum=s)
            print_factor_level_verdicts(s_verdict, f"{mode_tag.upper()} -- {s.upper()} WORKSTATION", show_weights=False)
            print_factor_level_table(s_verdict, f"{mode_tag.upper()} -- {s.upper()}")
            verdict_rows        += _verdict_rows(s_verdict, mode_tag, s)
            verdict_detail_rows += _verdict_detail_rows(s_verdict, mode_tag, s)
            table_rows          += _verdict_table_rows(s_verdict, mode_tag, s)

    # --- persist the weighting artifacts ---
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(out_dir / f"{mode_tag}_metric_weights.csv", index=False)
    if verdict_rows:
        pd.DataFrame(verdict_rows).to_csv(out_dir / f"{mode_tag}_factor_verdict.csv", index=False)
    if verdict_detail_rows:
        pd.DataFrame(verdict_detail_rows).to_csv(out_dir / f"{mode_tag}_factor_verdict_detail.csv", index=False)

    # --- clean per-parameter S_p table for the thesis (long + wide) ---
    if table_rows:
        tdf = pd.DataFrame(table_rows)
        tdf.to_csv(out_dir / f"{mode_tag}_factor_table.csv", index=False)
        # Wide form: one column per height (global/High/Medium/Low), row per
        # (factor, parameter), each cell the net weighted effect S_p. Pivots
        # straight into a LaTeX table. Winner per column = max S_p in that column.
        wide = tdf.pivot_table(index=["factor", "parameter"], columns="stratum",
                               values="S_p", aggfunc="first")
        order = [c for c in ["global"] + VALID_HEIGHTS if c in wide.columns]
        wide = wide[order].reset_index()
        wide.to_csv(out_dir / f"{mode_tag}_factor_table_wide.csv", index=False)

    return data_ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--combined-csv", type=Path, default=None,
                    help="Explicit path to combined_place_metrics.csv (single-source, from evaluate_difference.py)")
    ap.add_argument("--pen-csv", type=Path, default=None, help="Legacy: explicit path to place_metrics_combined.csv")
    ap.add_argument("--posture-csv", type=Path, default=None, help="Legacy: explicit path to posture_features_combined.csv")
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="Directory containing evaluate_difference.py's stat_tests.csv / "
                         "stratified_stat_tests.csv (default: <landmarks-root>/metrics/combined_comparison)")
    ap.add_argument("--score-mode", choices=["directional", "normative", "both"], default="directional",
                    help="'directional' (default): score toward known ergonomic optimum -- the substantive "
                         "analysis. 'normative': score by closeness to the within-height mean, matching the "
                         "LDA/fPCA target for like-for-like comparison. 'both': run each and print a "
                         "directional-vs-normative shift matrix.")
    ap.add_argument("--no-exclude", action="store_true", help="Keep events even if in the cleaning manifest")
    ap.add_argument("--no-consistency", action="store_true",
                    help="Skip the consistency-MCDA (between-participant convergence) leaderboard")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="excluded_place_events.csv (default <root>/metrics/cleaning/excluded_place_events.csv)")
    args = ap.parse_args()

    if args.landmarks_root:
        combined_path  = args.combined_csv or (args.landmarks_root / "metrics" / "combined_place_metrics.csv")
        pen_path       = args.pen_csv or (args.landmarks_root / "metrics" / "place_metrics_combined.csv")
        posture_path   = args.posture_csv or (args.landmarks_root / "metrics" / "posture_features_combined.csv")
        out_dir        = args.landmarks_root / "metrics" / "prototype_rankings"
        comparison_dir = args.comparison_dir or (args.landmarks_root / "metrics" / "combined_comparison")
        exclude_csv    = args.exclude_csv or (args.landmarks_root / "metrics" / "cleaning" / "excluded_place_events.csv")
    else:
        combined_path  = args.combined_csv
        pen_path       = args.pen_csv
        posture_path   = args.posture_csv
        base = combined_path or pen_path or posture_path
        if base is None:
            sys.exit("Error: Provide --landmarks-root OR --combined-csv (or --pen-csv/--posture-csv).")
        out_dir        = base.parent / "prototype_rankings"
        comparison_dir = args.comparison_dir
        exclude_csv    = args.exclude_csv

    # Prefer the single combined table (evaluate_difference.py output); fall back
    # to the legacy pen+posture merge only if the combined file is absent.
    if combined_path and Path(combined_path).is_file():
        df = pd.read_csv(combined_path)
        print(f"Loaded {len(df)} Place events from combined table {combined_path}.")
    else:
        df_pen = pd.read_csv(pen_path) if (pen_path and Path(pen_path).is_file()) else pd.DataFrame()
        df_posture = pd.read_csv(posture_path) if (posture_path and Path(posture_path).is_file()) else pd.DataFrame()
        if df_pen.empty and df_posture.empty:
            sys.exit("Error: no combined_place_metrics.csv found and both pen/posture CSVs are missing/empty.")
        if not df_pen.empty and not df_posture.empty:
            common = [c for c in ["participant", "trial", "place_index", "height"] if c in df_pen.columns and c in df_posture.columns]
            df = pd.merge(df_pen, df_posture, on=common, how="inner", suffixes=("", "_posture"))
            print(f"[legacy] Merged Pen ({len(df_pen)}) and Posture ({len(df_posture)}) into {len(df)} Place events.")
        else:
            df = df_pen if not df_pen.empty else df_posture
            print(f"[legacy] Loaded {len(df)} Place events from single available source.")

    # Exclude rejected place events (clean_place_events.py manifest)
    if not args.no_exclude:
        excl = load_excluded_events(exclude_csv)
        if excl and {"participant", "trial", "height", "place_index"}.issubset(df.columns):
            keys = df.apply(lambda r: _event_key(r["participant"], r["trial"], r["height"], r["place_index"]), axis=1)
            before = len(df); df = df[~keys.isin(excl)].copy()
            if before - len(df):
                print(f"  [exclude] removed {before - len(df)} rejected place event(s); {len(df)} remain.")

    if "height" in df.columns:
        df["height"] = df["height"].astype(str).str.strip()
        unknown_mask = ~df["height"].isin(VALID_HEIGHTS)
        if unknown_mask.any():
            print(f"\n[QUARANTINE] Filtering out {unknown_mask.sum()} Place events that failed to match a valid High/Medium/Low window.")
            df = df[~unknown_mask].copy()

    df = add_prototype_label(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_effects = load_mixedmodel_effects(comparison_dir, stratified=False)
    strat_effects = load_mixedmodel_effects(comparison_dir, stratified=True)

    directional_global = None
    normative_global = None

    if args.score_mode in ("directional", "both"):
        print(f"\n{'#'*85}\n# DIRECTIONAL SCORING (toward known ergonomic optimum -- substantive best-prototype)\n{'#'*85}")
        df_dir, dom_dir = calculate_desirability_scores(df)
        directional_global = run_full_ranking(df_dir, dom_dir, global_effects, strat_effects,
                                               comparison_dir, out_dir, "directional", do_verdicts=True)

    if args.score_mode in ("normative", "both"):
        print(f"\n{'#'*85}\n# NORMATIVE SCORING (closeness to within-height mean -- matches LDA/fPCA target)\n{'#'*85}")
        df_norm, dom_norm = calculate_normative_scores(df, stratum_col="height")
        normative_global = run_full_ranking(df_norm, dom_norm, global_effects, strat_effects,
                                            comparison_dir, out_dir, "normative", do_verdicts=False)

    if args.score_mode == "both" and directional_global is not None and normative_global is not None:
        print_target_shift_matrix(directional_global, normative_global, out_dir)

    # CONSISTENCY-MCDA: FACTOR-LEVEL between-participant convergence (well-powered
    # -- every participant sees every level, so each estimate uses the full pool).
    # Direction-agnostic, so computed once from the directional desirability
    # regardless of --score-mode. The config-level consistency leaderboard (which
    # rested on only 2-4 participants per 24-cell config) has been removed.
    if not args.no_consistency:
        print(f"\n{'#'*85}\n# CONSISTENCY-MCDA (factor-level between-participant convergence)\n{'#'*85}")
        df_d, dom_d = calculate_desirability_scores(df)
        # Sensitivity-weight the consistency aggregation by the SAME within-subject
        # mixed-model weights used elsewhere (magnitudes only -- squared standardised
        # effects -- so no directionality enters; this measures weighted spread, it
        # does not optimise). Falls back to equal weighting when no effect table is
        # available, exactly as before.
        cons_w_global = compute_mixedmodel_weights(df_d, dom_d, global_effects, stratum=None)
        if cons_w_global is None:
            print("  [consistency] no mixed-model effect table -- using EQUAL metric weighting.")
        fl_rows, fl_verdict = compute_factor_level_consistency(df_d, dom_d, metric_weights=cons_w_global)
        if fl_rows.empty:
            print("  [consistency] no factor-level result (needs a 'participant' column).")
        else:
            print_factor_level_consistency(fl_verdict, "GLOBAL (all heights pooled)")
            fl_rows.assign(stratum="global").to_csv(out_dir / "consistency_factor_level_global.csv", index=False)
            for s in [h for h in VALID_HEIGHTS if "height" in df_d.columns and h in set(df_d["height"].dropna())]:
                sub_s = df_d[df_d["height"] == s]
                cons_w_s = compute_mixedmodel_weights(sub_s, dom_d, strat_effects, stratum=s)
                r_s, v_s = compute_factor_level_consistency(sub_s, dom_d, metric_weights=cons_w_s)
                if not r_s.empty:
                    print_factor_level_consistency(v_s, f"{s.upper()} WORKSTATION")
                    r_s.assign(stratum=s).to_csv(out_dir / f"consistency_factor_level_{s}.csv", index=False)

    print(f"\nAll ranking evaluations saved to:\n  -> {out_dir}")
    if args.score_mode == "both":
        print("\nTo complete the two-way comparison: set the fPCA pipeline's leaderboard (which is")
        print("already normative) beside the NORMATIVE leaderboard here. Agreement there isolates")
        print("the effect of the feature set, since the optimisation target is now matched.")


if __name__ == "__main__":
    main()