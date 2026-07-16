#!/usr/bin/env python3
r"""
rank_prototypes.py - Complete & Unified MCDA Prototype Selection Engine

Integrates 0-100 Desirability Scoring, Mixed-Model-Derived Sensitivity Weighting,
Equal Weighting Benchmarking, Height-Stratified Evaluations, a Factor-Level
"Synthesised Best Prototype" summary, and a 3-Pillar Decision Sensitivity
Analysis into a single executable pipeline.

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
  python rank_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python rank_prototypes.py --pen-csv path/to/place_metrics.csv --posture-csv path/to/posture.csv --comparison-dir path/to/combined_comparison
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
    """Converts raw metrics into a standardized 0-100 Desirability Score."""
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

    FIX vs. earlier version: Grand_Score is now computed via TWO-STAGE
    aggregation -- first to one row per (Prototype_Config, participant), THEN to
    one row per Prototype_Config -- rather than averaging raw place events
    directly. This mirrors the aggregation already used (correctly) for the
    sensitivity weights themselves, and prevents a participant who happened to
    contribute more trials to a given config from pulling its score toward
    their own results. Score_SD is now the standard deviation ACROSS
    PARTICIPANT-LEVEL MEANS (between-participant spread), not across raw trials.
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

    This works for multi-level factors (Angle: 3 levels, 2 non-reference rows in
    effect_table) exactly as it does for binary factors (1 non-reference row),
    because evaluate_difference.py now stores one row per LEVEL rather than
    collapsing a factor to a single 'largest effect' value."""
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
# SECTION 4: 3-PILLAR DECISION SENSITIVITY ANALYSIS ENGINE
# =========================================================================== #

def run_sensitivity_analysis(df: pd.DataFrame, active_domains: dict, metric_weights: dict, out_dir: Path, top_n: int = 8):
    """Executes Scenario Stress-Testing, Leave-One-Domain-Out (LODO), and Monte Carlo Simulation."""
    print(f"\n{'='*85}\n3-PILLAR DECISION SENSITIVITY ANALYSIS (Robustness & Confidence Testing)\n{'='*85}")

    domain_names = list(active_domains.keys())
    base_ranks = score_and_rank(df, active_domains, metric_weights, strategy="data_driven")
    top_configs = base_ranks["Prototype_Config"].head(top_n).tolist()

    pure_dom_w = {d: np.mean([metric_weights.get(c, 0.01) for c in cols]) for d, cols in active_domains.items()}

    scenarios = {
        "Data-Driven (Effect)": None,
        "Equal Weighting":      "EQUAL_FLAG",
        "Speed Only":           {d: (1.0 if d == "Performance" else 0.0) for d in domain_names},
        "Ergo Only":            {d: (1.0 if d in ("Postural_Risk", "Grip_Ergonomics") else 0.0) for d in domain_names},
    }

    scenario_matrix = {cfg: [] for cfg in top_configs}
    for sc_name, sc_override in scenarios.items():
        if sc_override == "EQUAL_FLAG":
            sc_ranks = score_and_rank(df, active_domains, metric_weights, strategy="equal")
        else:
            sc_ranks = score_and_rank(df, active_domains, metric_weights, strategy="data_driven", domain_weights_override=sc_override)
        for cfg in top_configs:
            rk = sc_ranks[sc_ranks["Prototype_Config"] == cfg]["Rank"].values
            scenario_matrix[cfg].append(int(rk[0]) if len(rk) > 0 else np.nan)

    print(f"\n--- TABLE 6A: Operational Scenario Stress-Testing (Top {top_n} Ranks) ---")
    sc_headers = "  ".join([f"{k[:18]:>18}" for k in scenarios.keys()])
    print(f" {'Prototype Configuration':<32} | {sc_headers}")
    print(f" {'-'*32}-+-{'-'*len(sc_headers)}")

    sc_records = []
    for cfg in top_configs:
        ranks_str = "  ".join([f"{r:>18}" for r in scenario_matrix[cfg]])
        print(f" {cfg:<32} | {ranks_str}")
        rec = {"Prototype_Config": cfg}; rec.update(zip(scenarios.keys(), scenario_matrix[cfg]))
        sc_records.append(rec)
    pd.DataFrame(sc_records).to_csv(out_dir / "sensitivity_scenarios.csv", index=False)

    lodo_matrix = {cfg: [] for cfg in top_configs}
    for dropped_dom in domain_names:
        lodo_override = {d: (0.0 if d == dropped_dom else pure_dom_w[d]) for d in domain_names}
        lodo_ranks = score_and_rank(df, active_domains, metric_weights, strategy="data_driven", domain_weights_override=lodo_override)
        for cfg in top_configs:
            rk = lodo_ranks[lodo_ranks["Prototype_Config"] == cfg]["Rank"].values
            lodo_matrix[cfg].append(int(rk[0]) if len(rk) > 0 else np.nan)

    print(f"\n--- TABLE 6B: Leave-One-Domain-Out (LODO) Rank Stability ---")
    lodo_headers = "  ".join([f"-{d[:10]:>10}" for d in domain_names])
    print(f" {'Prototype Configuration':<32} | {'Data-Driven':>11} {lodo_headers}")
    print(f" {'-'*32}-+-{'-'*12}-{'-'*len(lodo_headers)}")

    lodo_records = []
    for cfg in top_configs:
        base_rk = int(base_ranks[base_ranks["Prototype_Config"] == cfg]["Rank"].values[0])
        ranks_str = "  ".join([f"{r:>10}" for r in lodo_matrix[cfg]])
        print(f" {cfg:<32} | {base_rk:>11} {ranks_str}")
        rec = {"Prototype_Config": cfg, "Data_Driven_Rank": base_rk}; rec.update(zip([f"Drop_{d}" for d in domain_names], lodo_matrix[cfg]))
        lodo_records.append(rec)
    pd.DataFrame(lodo_records).to_csv(out_dir / "sensitivity_lodo.csv", index=False)

    print(f"\n--- TABLE 6C: Monte Carlo Rank Confidence (1,000 Random Data-Weight Simulations) ---")
    print(" Simulating +/- 50% random perturbation around the within-subject-effect-derived domain weights...")

    n_sims = 1000
    win_counts = defaultdict(int); top3_counts = defaultdict(int)

    np.random.seed(42)
    for _ in range(n_sims):
        sim_override = {}
        for d in domain_names:
            base_w = pure_dom_w[d]
            sim_override[d] = max(0.001, base_w * np.random.uniform(0.5, 1.5))
        sim_ranks = score_and_rank(df, active_domains, metric_weights, strategy="data_driven", domain_weights_override=sim_override)
        winner = sim_ranks.iloc[0]["Prototype_Config"]
        win_counts[winner] += 1
        for _, r in sim_ranks.head(3).iterrows():
            top3_counts[r["Prototype_Config"]] += 1

    print(f" {'Prototype Configuration':<32} | {'Win Rate (%)':>14} {'Top-3 Finish (%)':>18} {'Robustness Status':>20}")
    print(f" {'-'*32}-+-{'-'*14}-{'-'*18}-{'-'*20}")

    mc_records = []
    all_cfgs_sorted = sorted(list(set(list(win_counts.keys()) + list(top3_counts.keys()) + top_configs)), key=lambda x: win_counts[x], reverse=True)

    for cfg in all_cfgs_sorted[:top_n]:
        w_pct = (win_counts[cfg] / n_sims) * 100.0; t3_pct = (top3_counts[cfg] / n_sims) * 100.0
        if w_pct >= 60.0: status = "** DOMINANT CHOICE"
        elif t3_pct >= 75.0: status = "* HIGHLY ROBUST"
        elif t3_pct >= 30.0: status = "  COMPETITIVE"
        else: status = "  SENSITIVE / NICHE"
        print(f" {cfg:<32} | {w_pct:>13.1f}% {t3_pct:>17.1f}% {status:>20}")
        mc_records.append({"Prototype_Config": cfg, "Win_Rate_Pct": w_pct, "Top3_Rate_Pct": t3_pct, "Status": status.strip()})
    pd.DataFrame(mc_records).to_csv(out_dir / "sensitivity_monte_carlo.csv", index=False)


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


# =========================================================================== #
# SECTION 6: COMMAND LINE INTERFACE & MAIN EXECUTION
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--pen-csv", type=Path, default=None, help="Explicit path to place_metrics_combined.csv")
    ap.add_argument("--posture-csv", type=Path, default=None, help="Explicit path to posture_features_combined.csv")
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="Directory containing evaluate_difference.py's stat_tests.csv / "
                         "stratified_stat_tests.csv (default: <landmarks-root>/metrics/combined_comparison)")
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
    df_scored, active_domains = calculate_desirability_scores(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load mixed-model effect tables (method A); fall back to KW (method B)
    # ------------------------------------------------------------------ #
    global_effects = load_mixedmodel_effects(comparison_dir, stratified=False)
    strat_effects = load_mixedmodel_effects(comparison_dir, stratified=True)

    kw_weights = compute_epsilon_weights_between_config(df_scored, active_domains)
    mm_weights = compute_mixedmodel_weights(df_scored, active_domains, global_effects, stratum=None)

    if mm_weights is None:
        print("\n[WARN] Falling back to between-config Kruskal-Wallis weighting (method B) -- "
              "this is the WEAKER, between-subject comparison; see module docstring. Run "
              "evaluate_difference.py first to enable the recommended within-subject weighting.")
        global_weights = kw_weights
        weighting_method_used = "Between-Config Kruskal-Wallis (fallback -- weaker)"
    else:
        global_weights = mm_weights
        weighting_method_used = "Within-Subject Mixed-Model (recommended)"

    data_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="data_driven")
    equal_ranks = score_and_rank(df_scored, active_domains, global_weights, strategy="equal")

    data_ranks.to_csv(out_dir / "rankings_data_driven_global.csv", index=False)
    equal_ranks.to_csv(out_dir / "rankings_equal_weight_global.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 1: Sensitivity-Weighting Method Comparison (A vs B, side by side)
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*90}\nTABLE 1: METRIC SENSITIVITY WEIGHTING -- METHOD COMPARISON\n{'='*90}")
    print(f" Active weighting method for all rankings below: {weighting_method_used}")
    print(f"\n {'Domain':<15} {'Metric':<26} {'(A) Within-Subj':>16} {'(B) Between-Cfg':>16} {'Agree?':>8}")
    print(f" {'-'*15} {'-'*26} {'-'*16} {'-'*16} {'-'*8}")
    weight_compare_records = []
    for domain, cols in active_domains.items():
        for col in cols:
            raw = col.replace("_dscore", "")
            a_val = mm_weights.get(col) if mm_weights else None
            b_val = kw_weights.get(col)
            a_str = f"{a_val:>16.3f}" if a_val is not None else f"{'n/a':>16}"
            b_str = f"{b_val:>16.3f}" if b_val is not None else f"{'n/a':>16}"
            agree = ""
            if a_val is not None and b_val is not None:
                # crude rank-direction agreement flag: both "high" (>median-ish) or both "low"
                agree = "~" if (a_val > 0.05) == (b_val > 0.05) else "X"
            print(f" {domain:<15} {raw:<26} {a_str} {b_str} {agree:>8}")
            weight_compare_records.append({"Domain": domain, "Metric": raw,
                                           "Weight_A_WithinSubject": a_val, "Weight_B_BetweenConfig": b_val})
    pd.DataFrame(weight_compare_records).to_csv(out_dir / "weighting_method_comparison.csv", index=False)
    print(" (A) = within-subject mixed-model, standardised effect^2 summed across factors (all N=10, recommended)")
    print(" (B) = between-config Kruskal-Wallis on 24-cell groups (~2-4 participants/config, weaker; comparison only)")

    # ----------------------------------------------------------------------- #
    # TABLE 2: Strategy Comparison Matrix
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*85}\nSTRATEGY COMPARISON: {weighting_method_used.upper()} VS. EQUAL WEIGHTING\n{'='*85}")
    print(f" {'Prototype Configuration':<32} | {'Weighted Score (Rk)':>22} | {'Equal Score (Rk)':>22}")
    print(f" {'-'*32}-+-{'-'*22}-+-{'-'*22}")
    for cfg in data_ranks["Prototype_Config"].head(15):
        dr_row = data_ranks[data_ranks["Prototype_Config"] == cfg].iloc[0]
        er_row = equal_ranks[equal_ranks["Prototype_Config"] == cfg].iloc[0]
        dr_str = f"{dr_row['Grand_Score']:>6.1f} (#{int(dr_row['Rank']):<2})"
        er_str = f"{er_row['Grand_Score']:>6.1f} (#{int(er_row['Rank']):<2})"
        print(f" {cfg:<32} | {dr_str:>22} | {er_str:>22}")

    # ----------------------------------------------------------------------- #
    # TABLE 3: Global Leaderboard + Factor-Level Verdict
    # ----------------------------------------------------------------------- #
    print_ascii_leaderboard(data_ranks, "GLOBAL PROTOTYPE LEADERBOARD (24-Cell Configurations)", top_n=10)
    global_verdict = compute_factor_level_verdict(df_scored, active_domains, global_weights, global_effects, stratum=None)
    print_factor_level_verdicts(global_verdict, "GLOBAL (all heights pooled)")

    # ----------------------------------------------------------------------- #
    # TABLE 4: Stratified Leaderboards + per-height Factor-Level Verdicts
    # ----------------------------------------------------------------------- #
    strata = sorted(df_scored["height"].dropna().unique(), key=lambda s: {"High":0,"Medium":1,"Low":2}.get(s,9)) if "height" in df_scored.columns else []
    stratified_results = {}
    verdict_records = []
    for s in strata:
        sub_df = df_scored[df_scored["height"] == s]
        if sub_df.empty: continue
        s_mm_weights = compute_mixedmodel_weights(sub_df, active_domains, strat_effects, stratum=s)
        s_weights = s_mm_weights if s_mm_weights is not None else compute_epsilon_weights_between_config(sub_df, active_domains)
        s_ranks = score_and_rank(sub_df, active_domains, s_weights, strategy="data_driven")
        stratified_results[s] = s_ranks
        s_ranks.to_csv(out_dir / f"rankings_stratified_{s}.csv", index=False)
        print_ascii_leaderboard(s_ranks, f"STRATIFIED LEADERBOARD -- {s.upper()} WORKSTATION", top_n=5)

        s_verdict = compute_factor_level_verdict(sub_df, active_domains, s_weights, strat_effects, stratum=s)
        print_factor_level_verdicts(s_verdict, s.upper())
        rec = {"height": s}
        for f in PARAM_FACTORS:
            rec[f"{f}_winner"] = s_verdict.get(f, {}).get("winner")
            v = s_verdict.get(f, {})
            rec[f"{f}_score"] = v.get("level_scores", {}).get(v.get("winner"))
        verdict_records.append(rec)
    pd.DataFrame(verdict_records).to_csv(out_dir / "factor_level_verdicts_by_height.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 5: Prototype Ranking Shift Matrix
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*85}\nPROTOTYPE RANKING SHIFT MATRIX (Global vs. Height-Stratified Models)\n{'='*85}")
    r_header = f" {'Prototype Configuration':<32} | {'Global Rk':>10} {'(Score)':>8} |" + "".join([f" {s[:3]} Rk (Scr) |" for s in strata])
    print(r_header); print(f" {'-'*32}-+-{'-'*10}-{'-'*8}-+" + "".join([f"{'-'*13}-+" for _ in strata]))

    shift_records = []
    for _, gr in data_ranks.iterrows():
        p_cfg = gr["Prototype_Config"]
        row_str = f" {p_cfg:<32} | {int(gr['Rank']):>10} {gr['Grand_Score']:>8.1f} |"
        rec = {"Prototype_Config": p_cfg, "Global_Rank": int(gr['Rank']), "Global_Score": gr['Grand_Score']}
        for s in strata:
            sr = stratified_results[s]
            match = sr[sr["Prototype_Config"] == p_cfg]
            if not match.empty:
                r_val, s_val = int(match.iloc[0]["Rank"]), match.iloc[0]["Grand_Score"]
                row_str += f" {r_val:>5} ({s_val:>4.1f}) |"
                rec[f"{s}_Rank"] = r_val; rec[f"{s}_Score"] = s_val
            else:
                row_str += f" {'n/a':>11} |"
                rec[f"{s}_Rank"] = np.nan; rec[f"{s}_Score"] = np.nan
        print(row_str)
        shift_records.append(rec)
    pd.DataFrame(shift_records).to_csv(out_dir / "rankings_shift_matrix.csv", index=False)

    # ----------------------------------------------------------------------- #
    # TABLE 6: 3-Pillar Decision Sensitivity Analysis
    # ----------------------------------------------------------------------- #
    run_sensitivity_analysis(df_scored, active_domains, global_weights, out_dir, top_n=8)

    print(f"\nAll ranking evaluations, strategy comparisons, and sensitivity tables saved to:\n  -> {out_dir}")


if __name__ == "__main__":
    main()