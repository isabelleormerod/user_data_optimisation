#!/usr/bin/env python3
r"""
rank_prototypes.py - Complete & Unified MCDA Prototype Selection Engine

Integrates 0-100 Desirability Scoring, Mixed-Model-Derived Sensitivity Weighting,
Equal Weighting Benchmarking, Height-Stratified Evaluations, a Factor-Level
"Synthesised Best Prototype" summary, and a 3-Pillar Decision Sensitivity
Analysis into a single executable pipeline.

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

METHODOLOGICAL NOTE -- TWO SENSITIVITY-WEIGHTING METHODS, AND WHY ONE IS PREFERRED:

  (A) WITHIN-SUBJECT MIXED-MODEL WEIGHTING (default, "data_driven").
      Reads the per-factor, per-metric, per-height signed effect sizes already
      estimated by evaluate_difference.py's mixed-effects models (stat_tests.csv /
      stratified_stat_tests.csv), standardises each effect by that metric's own
      pooled standard deviation, and sums the squared standardised effects across
      the four prototype factors to get one "how prototype-sensitive is this
      metric" weight. This uses ALL 10 participants via paired, within-subject
      comparisons (every participant tested every level of Length/Size/Weight,
      and all three Angle levels) -- the well-powered comparison established
      earlier in this project's power analysis.

  (B) BETWEEN-CONFIG KRUSKAL-WALLIS WEIGHTING (legacy, kept only for comparison).
      Aggregates to one row per (participant, Prototype_Config), then runs a
      Kruskal-Wallis test ACROSS the 24 specific prototype configurations. Because
      each of those 24 configs was tested by only 2-4 participants (the
      incomplete-block allocation), this is a BETWEEN-subject comparison, which
      the project's earlier Monte-Carlo power analysis showed has roughly 0.09
      power where the equivalent within-subject comparison has roughly 1.0, for
      the same true effect size. It is retained here only so the two methods can
      be shown side by side (Table 1) -- method (A) is the recommended default.

Both weighting methods feed the SAME 0-100 desirability-scoring and domain-
aggregation machinery; only "how much should each metric count" differs between
them, and that difference is reported explicitly rather than silently chosen.

Also computes a FACTOR-LEVEL "synthesised best prototype" per height: rather
than reading off the top of the 24-cell leaderboard (each cell resting on only
2-4 participants), this asks, one factor at a time using the well-powered
within-subject comparison, which single level (e.g. Short vs Long) is
preferable, then combines the winning levels into a recommended configuration.
This is a genuinely different question to "which of the 24 tested combinations
scored highest" and the two are reported side by side so agreement/disagreement
between them is visible. NOTE: this synthesis currently covers the three binary
factors (Length, Size, Weight) only. Angle is a 3-level factor, and
evaluate_difference.py's stored 'effect' column retains only the SINGLE largest
of Angle's two contrasts (135 vs 90, 180 vs 90) -- not enough information to
safely reconstruct which specific angle "wins". Angle is flagged, not guessed.

Usage:
  python 06_rank_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python 06_rank_prototypes.py --score-mode both --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python 06_rank_prototypes.py --pen-csv path/to/place_metrics.csv --posture-csv path/to/posture.csv --comparison-dir path/to/combined_comparison
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


def compute_epsilon_weights_between_config(df: pd.DataFrame, active_domains: dict) -> dict:
    """
    (B) LEGACY / COMPARISON-ONLY. Kruskal-Wallis Epsilon-Squared (E_R^2 = H/(n-1))
    computed BETWEEN the 24 specific Prototype_Config groups, after aggregating to
    one row per (participant, config) to avoid raw-event pseudoreplication.

    This is still a between-subject test at the config level (each config is
    tested by only 2-4 participants), and is retained only so its weights can be
    shown alongside method (A)'s within-subject weights in Table 1 -- see module
    docstring for why (A) is preferred as the default.
    """
    if "participant" in df.columns:
        df_agg = df.groupby(["Prototype_Config", "participant"], dropna=False, as_index=False).mean(numeric_only=True)
    else:
        df_agg = df.copy()

    n = len(df_agg)
    weights = {}
    for domain, cols in active_domains.items():
        for col in cols:
            groups = [grp[col].dropna().values for _, grp in df_agg.groupby("Prototype_Config", dropna=False) if len(grp[col].dropna()) > 0]
            if len(groups) >= 2 and n > 1:
                try:
                    h_stat, _ = stats.kruskal(*groups)
                    e_sq = max(0.01, float(h_stat / (n - 1)))
                except ValueError:
                    e_sq = 0.01
            else:
                e_sq = 0.01
            weights[col] = e_sq
    return weights


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
    metric, analogous in role to the epsilon-squared weight it replaces, but
    computed from the well-powered within-subject test.
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


# =========================================================================== #
# SECTION 3B: FACTOR-LEVEL "SYNTHESISED BEST PROTOTYPE"
#   A different question to the 24-cell leaderboard above: which single LEVEL
#   of each binary factor is preferable, using the well-powered within-subject
#   comparison, then combine winning levels into a recommended configuration.
#   See module docstring for the distinction and the Angle caveat.
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
                    detail.append((raw, level_label, round(std_effect, 3), round(contribution, 4)))
                    any_metric = True
        if not any_metric:
            verdicts[factor] = {"winner": None, "reason": "no metric had an estimable effect for this factor here"}
            continue
        winner = max(level_scores, key=level_scores.get)
        verdicts[factor] = {"winner": winner, "level_scores": dict(level_scores), "detail": detail}
    return verdicts


def print_factor_level_verdicts(verdicts: dict, label: str):
    print(f"\n--- Factor-Level Synthesised Verdict -- {label} ---")
    config_parts = []
    for f in PARAM_FACTORS:
        v = verdicts.get(f, {})
        if v.get("winner") is None:
            print(f"    {f:<8}: no verdict ({v.get('reason', 'unknown')})")
            config_parts.append("?")
        else:
            scores_str = ", ".join(f"{lvl}={sc:+.4f}" for lvl, sc in sorted(v["level_scores"].items(), key=lambda x: -x[1]))
            print(f"    {f:<8}: {v['winner']:<14} [{scores_str}]")
            config_parts.append(v["winner"])
    length, size, weight, angle = config_parts
    print(f"    Synthesised recommendation: {length}_{size}_{weight}_{angle}")




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

def run_full_ranking(df_scored, active_domains, global_effects, strat_effects,
                     comparison_dir, out_dir, mode_tag, do_verdicts):
    """Run the full weighting -> ranking -> stratified -> sensitivity flow for a
    single scoring mode. Returns the global data-driven ranking (for the
    directional-vs-normative comparison). mode_tag prefixes output filenames so
    directional and normative outputs don't collide. do_verdicts controls the
    factor-level synthesis, which is only meaningful for the directional mode."""
    kw_weights = compute_epsilon_weights_between_config(df_scored, active_domains)
    mm_weights = compute_mixedmodel_weights(df_scored, active_domains, global_effects, stratum=None)

    if mm_weights is None:
        print(f"\n[WARN] ({mode_tag}) Falling back to between-config Kruskal-Wallis weighting -- "
              "weaker between-subject comparison; run evaluate_difference.py to enable the "
              "recommended within-subject weighting.")
        global_weights = kw_weights
        weighting_method_used = "Between-Config Kruskal-Wallis (fallback -- weaker)"
    else:
        global_weights = mm_weights
        weighting_method_used = "Within-Subject Mixed-Model (recommended)"

    data_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="data_driven")
    equal_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="equal")
    data_ranks.to_csv(out_dir / f"{mode_tag}_rankings_data_driven_global.csv", index=False)
    equal_ranks.to_csv(out_dir / f"{mode_tag}_rankings_equal_weight_global.csv", index=False)

    print_ascii_leaderboard(data_ranks,
        f"[{mode_tag.upper()}] GLOBAL PROTOTYPE LEADERBOARD (24-Cell Configurations)", top_n=10)

    if do_verdicts:
        global_verdict = compute_factor_level_verdict(df_scored, active_domains, global_weights, global_effects, stratum=None)
        print_factor_level_verdicts(global_verdict, f"{mode_tag.upper()} GLOBAL (all heights pooled)")

    # Stratified leaderboards
    strata = sorted(df_scored["height"].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9)) if "height" in df_scored.columns else []
    for s in strata:
        sub_df = df_scored[df_scored["height"] == s]
        if sub_df.empty:
            continue
        s_mm = compute_mixedmodel_weights(sub_df, active_domains, strat_effects, stratum=s)
        s_weights = s_mm if s_mm is not None else compute_epsilon_weights_between_config(sub_df, active_domains)
        s_ranks = score_and_rank(sub_df, active_domains, s_weights, strategy="data_driven")
        s_ranks.to_csv(out_dir / f"{mode_tag}_rankings_stratified_{s}.csv", index=False)
        print_ascii_leaderboard(s_ranks, f"[{mode_tag.upper()}] STRATIFIED LEADERBOARD -- {s.upper()} WORKSTATION", top_n=5)

    return data_ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--pen-csv", type=Path, default=None, help="Explicit path to place_metrics_combined.csv")
    ap.add_argument("--posture-csv", type=Path, default=None, help="Explicit path to posture_features_combined.csv")
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="Directory containing evaluate_difference.py's stat_tests.csv / "
                         "stratified_stat_tests.csv (default: <landmarks-root>/metrics/combined_comparison)")
    ap.add_argument("--score-mode", choices=["directional", "normative", "both"], default="directional",
                    help="'directional' (default): score toward known ergonomic optimum -- the substantive "
                         "analysis. 'normative': score by closeness to the within-height mean, matching the "
                         "LDA/fPCA target for like-for-like comparison. 'both': run each and print a "
                         "directional-vs-normative shift matrix.")
    args = ap.parse_args()

    if args.landmarks_root:
        pen_path     = args.landmarks_root / "metrics" / "place_metrics_combined.csv"
        posture_path = args.landmarks_root / "metrics" / "posture_features_combined.csv"
        out_dir      = args.landmarks_root / "metrics" / "prototype_rankings"
        comparison_dir = args.comparison_dir or (args.landmarks_root / "metrics" / "combined_comparison")
    else:
        pen_path     = args.pen_csv
        posture_path = args.posture_csv
        out_dir      = (pen_path or posture_path).parent / "prototype_rankings"
        comparison_dir = args.comparison_dir

    if not pen_path and not posture_path:
        sys.exit("Error: Provide --landmarks-root OR specify explicit paths via --pen-csv / --posture-csv.")

    df_pen = pd.read_csv(pen_path) if (pen_path and pen_path.is_file()) else pd.DataFrame()
    df_posture = pd.read_csv(posture_path) if (posture_path and posture_path.is_file()) else pd.DataFrame()

    if df_pen.empty and df_posture.empty:
        sys.exit("Error: Both pen and posture CSVs are empty or missing. Nothing to evaluate.")

    if not df_pen.empty and not df_posture.empty:
        common = [c for c in ["participant", "trial", "place_index", "height"] if c in df_pen.columns and c in df_posture.columns]
        df = pd.merge(df_pen, df_posture, on=common, how="inner", suffixes=("", "_posture"))
        print(f"Merged Pen ({len(df_pen)}) and Posture ({len(df_posture)}) into {len(df)} total Place events.")
    else:
        df = df_pen if not df_pen.empty else df_posture
        print(f"Loaded {len(df)} Place events from single available data source.")

    if "height" in df.columns:
        valid_strata = {"High", "Medium", "Low"}
        unknown_mask = ~df["height"].isin(valid_strata) | df["height"].isna()
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
        # full sensitivity suite only for the substantive directional analysis
        kw = compute_epsilon_weights_between_config(df_dir, dom_dir)
        mm = compute_mixedmodel_weights(df_dir, dom_dir, global_effects, stratum=None)

    if args.score_mode in ("normative", "both"):
        print(f"\n{'#'*85}\n# NORMATIVE SCORING (closeness to within-height mean -- matches LDA/fPCA target)\n{'#'*85}")
        df_norm, dom_norm = calculate_normative_scores(df, stratum_col="height")
        normative_global = run_full_ranking(df_norm, dom_norm, global_effects, strat_effects,
                                            comparison_dir, out_dir, "normative", do_verdicts=False)

    if args.score_mode == "both" and directional_global is not None and normative_global is not None:
        print_target_shift_matrix(directional_global, normative_global, out_dir)

    print(f"\nAll ranking evaluations saved to:\n  -> {out_dir}")
    if args.score_mode == "both":
        print("\nTo complete the two-way comparison: set the fPCA pipeline's leaderboard (which is")
        print("already normative) beside the NORMATIVE leaderboard here. Agreement there isolates")
        print("the effect of the feature set, since the optimisation target is now matched.")


if __name__ == "__main__":
    main()