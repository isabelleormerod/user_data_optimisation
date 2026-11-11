#!/usr/bin/env python3
r"""
rank_prototypes.py - MCDA pipeline (Approach 1): factor importance and
consistency-based prototype ranking.

Two stages, each run independently for every working height (High / Medium /
Low), mirroring run_hierarchical_fpca.py so the two approaches can be compared
like-for-like.

STAGE 1 - factor IMPORTANCE (screening)
  Fits a main-effects linear mixed-effects model for every metric within each
  working height, on the same cleaned events used throughout this script, using
  the mixed-model engine in 04_evaluate_difference.py (loaded from this folder):
      metric ~ C(Length) + C(Size) + C(Weight) + C(Angle) + (1 | participant)
  A joint Wald test per factor gives one p-value per factor and metric; these are
  reported with Benjamini-Hochberg FDR correction across each height's tests.
  --use-existing-effects reads a previously written stratified_stat_tests.csv
  instead of refitting.
  For each metric m and factor f, the standardised range of f's level effects
      R_fm = (max(0, max beta) - min(0, min beta)) / SD_m
  is the largest difference between f's levels in SD units (reference level = 0).
  Metrics are combined using the pipeline's sensitivity weights
      w_m = sum over factors and levels of (beta / SD_m)^2,
  normalised within each ergonomic domain, with each domain weighted by its mean
  w_m, so a domain's influence does not depend on how many metrics it contains:
      importance_f = sum_m W_m * R_fm,     weight_f = importance_f / sum_f importance_f
  weight_f lies in [0, 1] and sums to 1 across the four factors, on the same
  footing as the fPCA-LDA factor weight (normalised summed LDA eigenvalue). The
  number of metrics on which the factor's joint Wald test is significant is
  reported alongside as a screening count.
  --importance-weighting equal weights every domain equally and every metric
  equally within its domain (the pipeline's equal-weight benchmark). Because W_m
  is built from the same effects it then weights, running both is a useful check
  that the factor ordering does not depend on the weighting.

STAGE 2 - PROTOTYPE RANKING BY CONSISTENCY
  Metrics are scored 0-100 toward their ergonomic optimum (desirability). For each
  level of each factor, each participant's desirability scores are averaged and
  the between-participant SD of those means is taken per metric; the SDs are
  combined with the same sensitivity weights (within, then across domains). The
  level with the lowest dispersion is the most consistent. A within-participant
  permutation test (level labels shuffled within each participant) on the
  best-minus-worst dispersion gap gives a p-value per factor. Only significant
  factors contribute a level to the recommended configuration ('?' otherwise);
  the best level of every factor, significant or not, is also reported.
  CAVEAT: a participant's per-level mean pools over the other three factors, so
  some of the spread reflects those factors rather than pure between-person
  disagreement.
  WEIGHTED CONFIGURATION RANKING: every configuration (all 24) is then scored
      S_c = sum_f  W_f * sigma_{f, level_f(c)} / mean_l(sigma_{f, l})
  using the Stage 1 factor weights W_f, i.e. a weighted mean of each level's
  dispersion relative to its own factor's average (1 = average, lower = more
  consistent). The top-ranked configuration always combines each factor's most
  consistent level; the weights order the remaining configurations, so an
  inconsistent level of a heavily weighted factor costs more. Same calculation as
  run_hierarchical_fpca.py, so the two pipelines' rankings are comparable.

APPENDIX - HEURISTIC (DIRECTIONAL) RANKING
  For the appendix, the factor-level synthesis score of each level is also
  computed from the same fitted effects and the same sensitivity weights:
      S_p = sum_m (beta_pm / SD_m) * D_m * W_m
  where D_m = +1 if higher values of metric m are ergonomically better and -1 if
  lower values are, and each factor's reference level is fixed at S_p = 0. The
  highest-scoring level of each factor is selected. The synthesis has no
  significance test of its own.

OUTPUTS (<landmarks-root>/metrics/prototype_rankings/):
  mcda_stratified_stat_tests.csv          Stage 1 mixed-model fits (effects, SEs, Wald p)
  mcda_factor_weights.csv                 Stage 1; shares its columns with the fPCA
                                          script's lda_factor_weights.csv where they
                                          overlap (height, factor, n_levels,
                                          data_weight, equal_weight, rank)
  mcda_metric_weights.csv                 per-height raw and effective metric weights
  mcda_significant_metrics.csv / .tex     Stage 1 screening detail: every factor x metric
                                          test, raw and Benjamini-Hochberg FDR-adjusted p
  mcda_factor_importance.pdf / .png       Stage 1 figure (weights 0-1, grouped by height, one bar per factor)
  mcda_consistency_by_height.csv          Stage 2, one row per factor level
  mcda_recommended_config_by_height.csv   Stage 2 configurations
  mcda_weighted_config_ranking.csv        Stage 2: all configurations per height, ranked by
                                          the weighted consistency score
  mcda_consistency_table.tex              Stage 2 LaTeX table (workstations side by side)
  mcda_stage1_effects_<height>.pdf/.png   Stage 1 figure: each metric's contribution to each
                                          factor's factor weight
  mcda_stage2_consistency_<height>.pdf/.png  Stage 2 figure: each metric's between-participant
                                          SD per setting and their weighted mean sigma
  mcda_directional_by_height.csv          Appendix: S_p per factor level
  mcda_directional_table.tex              Appendix LaTeX table, same layout as Stage 2

The normative scoring and 24-cell leaderboards of the previous version are no
longer produced.

Usage:
  python MCDA/05_rank_prototypes.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
  python MCDA/05_rank_prototypes.py --combined-csv path/to/combined_place_metrics.csv \
      --comparison-dir path/to/combined_comparison
  Optional: --n-perm 1000  --alpha 0.05  --importance-weighting pipeline|equal
            --min-n 8  --use-existing-effects  --no-exclude  --exclude-csv PATH
"""

import argparse
import importlib.util
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================== #
# SECTION 1: METRIC REGISTRY & factorS
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

PARAM_factorS = ["Length", "Size", "Weight", "Angle"]    # configuration-name order
VALID_HEIGHTS = ["High", "Medium", "Low"]                # the only height strata ever analysed
FIG_factorS = ["Angle", "Size", "Length", "Weight"]      # display order (matches participant figures)
TEX_factorS = ["Weight", "Size", "Length", "Angle"]      # row order of the thesis tables
LEVEL_ORDER = {"Length": ["Short", "Long"], "Size": ["Small", "Large"],
               "Weight": ["Not_weighted", "Front_weighted"], "Angle": ["A90", "A135", "A180"]}
TEX_LEVEL = {"Short": "Short", "Long": "Long", "Small": "Small", "Large": "Large",
             "Not_weighted": "Non-weighted", "Front_weighted": "Weighted",
             "A90": r"90\textdegree", "A135": r"135\textdegree", "A180": r"180\textdegree"}
factor_REFERENCE = {"Length": "Long", "Size": "Large", "Weight": "Front_weighted", "Angle": "A90"}
EFFECT_LEVEL = {"Angle": {"90": "A90", "135": "A135", "180": "A180"}}   # model level label -> config label
factor_COLOURS = {"Angle": "#c8642a", "Size": "#3b6ea5",          # as stated_vs_revealed.py and
                  "Length": "#3e8e5e", "Weight": "#7d4e9e"}       # importance_comparison.py


# =========================================================================== #
# SECTION 2: LOADING, PARSING & DESIRABILITY SCORING
# =========================================================================== #

def parse_params(trial_val) -> dict:
    """Robustly parses prototype factors from trial strings without formatting sensitivity."""
    out = {k: "Other" for k in PARAM_factorS}
    if trial_val is None or pd.isna(trial_val):
        return out

    clean_str = str(trial_val).strip()
    tokens = [t.strip() for t in clean_str.split("_") if t.strip()]
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
        elif tok.isdigit() and int(tok) in (90, 135, 180): out["Angle"] = f"A{tok}"

    return out


def add_prototype_label(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Length/Size/Weight/Angle from 'trial' and quarantine rows that fail to
    parse (matching evaluate_difference.py), so no spurious 'Other' level is analysed."""
    df_clean = df.copy()
    parsed = df_clean["trial"].apply(parse_params).apply(pd.Series)
    for c in PARAM_factorS:
        df_clean[c] = parsed[c].astype(str).str.strip()
    bad = (df_clean[PARAM_factorS] == "Other").any(axis=1)
    if bad.any():
        print(f"  [QUARANTINE] {int(bad.sum())} event(s) in {df_clean.loc[bad, 'trial'].nunique()} trial(s) "
              f"have an unparseable configuration and are excluded.")
        df_clean = df_clean[~bad].copy()
    df_clean["Prototype_Config"] = df_clean[PARAM_factorS].agg("_".join, axis=1)
    return df_clean


def calculate_desirability_scores(df: pd.DataFrame) -> tuple:
    """Convert raw metrics into 0-100 desirability scores, scored toward each
    metric's known ergonomic optimum (METRIC_REGISTRY 'dir')."""
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


def _equal_weights(active_domains: dict) -> dict:
    """Every active metric weighted 1.0: within the domain-then-domain aggregation
    this gives every domain equal weight and every metric equal weight within it."""
    return {c: 1.0 for cols in active_domains.values() for c in cols}


def _load_evaluate_difference():
    """Load MCDA/04_evaluate_difference.py (numeric filename prefix, so loaded by
    path from this script's folder) to reuse its mixed-model engine."""
    path = Path(__file__).resolve().parent / "04_evaluate_difference.py"
    if not path.is_file():
        sys.exit(f"Error: Stage 1 needs {path} (the mixed-model engine) beside this script, "
                 f"or run with --use-existing-effects.")
    spec = importlib.util.spec_from_file_location("evaluate_difference_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_stratified_effects(df: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    """Fit the stratified main-effects mixed models for every MCDA metric on the
    cleaned events, using 04_evaluate_difference.py's mixed_tests. Angle is passed
    as an integer (90/135/180) so its level labels match that script's output."""
    ed = _load_evaluate_difference()
    fit = df.copy()
    fit["Angle"] = fit["Angle"].str.upper().str.lstrip("A").astype(int)
    fit["participant"] = fit["participant"].astype(str)
    metrics = [(c, m["label"], "") for c, m in METRIC_REGISTRY.items()
               if c in fit.columns and pd.to_numeric(fit[c], errors="coerce").dropna().nunique() > 1]
    print(f"\n  Fitting mixed models: {len(metrics)} metrics x {len(VALID_HEIGHTS)} heights ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tests = ed.mixed_tests(fit, metrics, PARAM_factorS, stratum_col="height",
                               min_n=min_n, verbose=False)
    status = tests.drop_duplicates(["stratum", "metric"])["fit_status"]
    print(f"  Model fits: {int((status == 'ok').sum())}/{len(status)} ok"
          + (f"; {int((status != 'ok').sum())} skipped or failed" if (status != "ok").any() else ""))
    return tests


def load_stratified_effects(comparison_dir: Path):
    """Load evaluate_difference.py's stratified mixed-model effects, or None."""
    if comparison_dir is None:
        return None
    path = comparison_dir / "stratified_stat_tests.csv"
    if not path.is_file():
        print(f"  [WARN] {path} not found -- run evaluate_difference.py (--mode compare) first.")
        return None
    return pd.read_csv(path)


def compute_mixedmodel_weights(df_scored: pd.DataFrame, active_domains: dict,
                               effect_table: pd.DataFrame, stratum: str = None) -> dict:
    """Raw per-metric sensitivity weight w_m: the metric's mixed-model coefficients,
    standardised by its SD, squared and summed across the four factors (all levels).
    Keys are '<metric>_dscore'. Metrics without estimable effects get 0.01."""
    if effect_table is None:
        return None
    tbl = effect_table
    if stratum is not None and "stratum" in tbl.columns:
        tbl = tbl[tbl["stratum"] == stratum]
    elif stratum is not None:
        return None

    weights = {}
    for domain, cols in active_domains.items():
        for col in cols:
            raw = col.replace("_dscore", "")
            if raw not in df_scored.columns:
                weights[col] = 0.01
                continue
            sd = pd.to_numeric(df_scored[raw], errors="coerce").std(ddof=1)
            rows = tbl[(tbl["metric"] == raw) & (tbl["factor"].isin(PARAM_factorS))]
            if not sd or pd.isna(sd) or sd < 1e-9 or rows.empty:
                weights[col] = 0.01
                continue
            std_effects_sq = [(r["effect"] / sd) ** 2 for _, r in rows.iterrows() if pd.notna(r["effect"])]
            weights[col] = max(0.01, float(sum(std_effects_sq))) if std_effects_sq else 0.01
    return weights


def effective_weights(metric_weights: dict, active_domains: dict) -> dict:
    """Effective weight per metric = intra-domain share x inter-domain share, where
    each domain is weighted by its mean raw weight (as the pipeline aggregates).
    The effective weights sum to 1."""
    dom = {d: cols for d, cols in active_domains.items() if cols}
    dom_w = {d: float(np.mean([metric_weights.get(c, 0.01) for c in cols])) for d, cols in dom.items()}
    tot = sum(dom_w.values()) or 1.0
    eff = {}
    for d, cols in dom.items():
        s = sum(metric_weights.get(c, 0.01) for c in cols) or 1.0
        for c in cols:
            eff[c] = metric_weights.get(c, 0.01) / s * dom_w[d] / tot
    return eff


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (NaNs left as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    adj = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    res = np.empty(n)
    res[order] = adj
    out[ok] = res
    return out


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
# SECTION 3: STAGE 1 -- factor IMPORTANCE
# =========================================================================== #

def factor_importance(df_h: pd.DataFrame, active_domains: dict, effects: pd.DataFrame,
                      height: str, weighting: str = "pipeline", alpha: float = 0.05,
                      p_adjust: str = "fdr"):
    """Stage 1 for one height. p_adjust = "fdr" (Benjamini-Hochberg across the height's
    tests, the default) or "none" (raw p-values; the significance columns then hold the
    unadjusted results, labelled as such in every caption). Returns (factor_rows_df, metric_rows_df, metric_weights)
    where metric_weights is the raw weight dict used (also passed to Stage 2)."""
    tbl = effects[effects["stratum"] == height]
    raw_w = compute_mixedmodel_weights(df_h, active_domains, effects, stratum=height)
    used_w = raw_w if (weighting == "pipeline" and raw_w is not None) else _equal_weights(active_domains)
    eff_w = effective_weights(used_w, active_domains)

    col_domain = {c: d for d, cols in active_domains.items() for c in cols}
    metric_rows = [{"height": height, "domain": col_domain[c], "metric": c.replace("_dscore", ""),
                    "raw_weight": (raw_w or {}).get(c, np.nan), "effective_weight": w}
                   for c, w in eff_w.items()]

    imp, n_sig, n_met, tests = {}, {}, {}, []
    for f in PARAM_factorS:
        s, ns, nm = 0.0, 0, 0
        for c, w in eff_w.items():
            raw = c.replace("_dscore", "")
            sd = pd.to_numeric(df_h[raw], errors="coerce").std(ddof=1)
            rows = tbl[(tbl["metric"] == raw) & (tbl["factor"] == f)]
            e = rows["effect"].dropna()
            if e.empty or not np.isfinite(sd) or sd < 1e-9:
                continue
            nm += 1
            s += w * (max(0.0, e.max()) - min(0.0, e.min())) / sd
            p = rows["p_value"].dropna() if "p_value" in rows.columns else pd.Series(dtype=float)
            if len(p):
                tests.append({"height": height, "factor": f, "metric": raw,
                              "label": METRIC_REGISTRY[raw]["label"], "p_value": float(p.iloc[0]),
                              "weighted_range": w * (max(0.0, e.max()) - min(0.0, e.min())) / sd})
                if p.iloc[0] < alpha:
                    ns += 1
        imp[f], n_sig[f], n_met[f] = s, ns, nm

    total = sum(imp.values())
    rows = [{"height": height, "factor": f, "n_levels": int(df_h[f].nunique()),
             "importance": imp[f], "data_weight": (imp[f] / total) if total > 0 else np.nan,
             "equal_weight": 1.0 / len(PARAM_factorS), "n_sig_metrics": n_sig[f],
             "n_metrics": n_met[f], "weighting": weighting} for f in PARAM_factorS]
    fdf = pd.DataFrame(rows).sort_values("data_weight", ascending=False).reset_index(drop=True)
    fdf["rank"] = range(1, len(fdf) + 1)

    # screening detail: every factor x metric test at this height, with Benjamini-Hochberg
    # FDR correction across this height's tests (one height = one screening family)
    tdf = pd.DataFrame(tests, columns=["height", "factor", "metric", "label", "p_value", "weighted_range"])
    tdf["contribution"] = tdf["weighted_range"] / total if total > 0 else np.nan   # sums to the factor weight
    tdf["p_fdr"] = bh_adjust(tdf["p_value"]) if len(tdf) else []
    tdf["significant"] = tdf["p_value"] < alpha
    tdf["significant_fdr"] = tdf["p_fdr"] < alpha
    if p_adjust == "none":                       # report unadjusted significance instead
        tdf["p_fdr"] = tdf["p_value"]
        tdf["significant_fdr"] = tdf["significant"]
    tdf["p_adjust"] = p_adjust
    fdf["p_adjust"] = p_adjust
    fdf["n_sig_fdr"] = fdf["factor"].map(tdf[tdf["significant_fdr"]].groupby("factor").size()).fillna(0).astype(int)
    return fdf, pd.DataFrame(metric_rows), used_w, tdf


def print_factor_leaderboard(fdf, height, tdf=None):
    print(f"\n{'='*74}\nfactor-IMPORTANCE LEADERBOARD -- {height.upper()} "
          f"(weight = normalised sensitivity-weighted effect range)\n{'='*74}")
    print(f"  {'Rk':<3} {'factor':<8} {'Weight':>7} {'Importance':>11} {'EqualW':>7} {'Sig. metrics':>12} {'FDR':>5} {'Levels':>7}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*11} {'-'*7} {'-'*12} {'-'*5} {'-'*7}")
    for _, r in fdf.iterrows():
        w = f"{r['data_weight']:.3f}" if pd.notna(r["data_weight"]) else "n/a"
        sig = f"{int(r['n_sig_metrics'])}/{int(r['n_metrics'])}"
        print(f"  {int(r['rank']):<3} {r['factor']:<8} {w:>7} {r['importance']:>11.4f} "
              f"{r['equal_weight']:>7.3f} {sig:>12} {int(r.get('n_sig_fdr', 0)):>5} {int(r['n_levels']):>7}")
    print("  *(Weight = the factor's share of total importance (0-1; the four factors sum to 1).")
    print("    Importance = the unnormalised value the weight is calculated from.")
    print("    Sig. metrics = how many of the metrics the factor significantly affected at this")
    print("    height in the mixed models (p < 0.05), e.g. 9/15 = 9 of the 15 metrics.")
    print("    FDR = how many remain significant after Benjamini-Hochberg correction across"
          if fdf.get("p_adjust", pd.Series(["fdr"])).iloc[0] == "fdr" else
          "    FDR column = unadjusted count (--p-adjust none: no correction applied)"
          "\n   ")
    print("    this height's tests.)*")
    if tdf is not None and len(tdf):
        print("\n  Significant metrics by factor (p < 0.05; * = also significant after FDR correction):")
        for f in fdf["factor"]:
            sig = tdf[(tdf["factor"] == f) & tdf["significant"]].sort_values("p_value")
            names = ", ".join(f"{r.label}{'*' if r.significant_fdr else ''} (p={r.p_value:.3f})"
                              for r in sig.itertuples()) or "none"
            print(f"      {f:<8}: {names}")


def plot_factor_importance(fw: pd.DataFrame, out_dir: Path):
    """Stage 1 figure: factor weights (0-1) on a single axis, grouped by working
    height, one bar per factor, so the change in each weight across heights is
    visible. factor colours match the other thesis figures."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    heights = [h for h in VALID_HEIGHTS if h in set(fw["height"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x, w = np.arange(len(heights)), 0.2
    for i, f in enumerate(FIG_factorS):
        vals = []
        for h in heights:
            r = fw[(fw["height"] == h) & (fw["factor"] == f)]
            vals.append(float(r["data_weight"].iloc[0]) if len(r) else np.nan)
        xs = x + (i - (len(FIG_factorS) - 1) / 2) * w
        ax.bar(xs, vals, w, color=factor_COLOURS[f], edgecolor="white", linewidth=0.8, label=f)
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} workstation" for h in heights])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Factor weight")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="factor", frameon=False, ncol=len(FIG_factorS),
              loc="upper right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"mcda_factor_importance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_significance_table(tests: pd.DataFrame, path: Path, alpha: float):
    """Stage 1 screening table: for each factor and height, the metrics it
    significantly affected (p < alpha); bold = survives FDR correction."""
    lines = [r"% Generated by 05_rank_prototypes.py -- rerun to update",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Metrics significantly affected by each prototype factor at each working "
             r"height (mixed-effects models, joint Wald test, $p < " + f"{alpha:g}" + r"$"
             + (r", unadjusted). " if tests["p_adjust"].iloc[0] == "none" else
                r"). Metrics in \textbf{bold} remain significant after Benjamini--Hochberg false "
                r"discovery rate correction across that height's tests. ")
             + r"Counts give the number of the " + str(tests["metric"].nunique()) + r" metrics tested.}",
             r"\label{tab:mcda_significant_metrics}",
             r"\begin{tabular}{@{}l p{4cm} p{4cm} p{4cm}@{}}", r"\toprule",
             r"factor & High & Medium & Low \\", r"\midrule"]
    for f in FIG_factorS:
        cells = []
        for h in VALID_HEIGHTS:
            sig = tests[(tests["height"] == h) & (tests["factor"] == f) & tests["significant"]].sort_values("p_value")
            bold = tests["p_adjust"].iloc[0] != "none"
            names = [rf"\textbf{{{r.label}}}" if (bold and r.significant_fdr) else r.label for r in sig.itertuples()]
            cells.append(f"({len(names)}) " + (", ".join(names) if names else "none"))
        lines.append(f"{f} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================================== #
# SECTION 4: STAGE 2 -- PROTOTYPE RANKING BY CONSISTENCY
# =========================================================================== #

def _aggregate_dispersion(by_pid, active_domains, min_participants, metric_weights=None,
                          return_breakdown=False):
    """Collapse the per-metric between-participant SDs (of participant-mean
    desirability) into a single dispersion for one level: weighted mean within a
    domain by each metric's sensitivity weight, then weighted mean across domains
    by each domain's mean metric weight (equal weights if metric_weights is None).
    Returns the scalar dispersion, or (scalar, {domain: disp}) if return_breakdown."""
    domain_disp = {}
    for domain, cols in active_domains.items():
        sd_pairs = []
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
        dom_w = {d: float(np.mean([metric_weights.get(c, 0.01) for c in active_domains[d]]))
                 for d in present}
        num = sum(dom_w[d] * domain_disp[d] for d in present)
        den = sum(dom_w[d] for d in present)
        agg = (num / den) if den else np.nan
    else:
        agg = float(np.nanmean([domain_disp[d] for d in present]))

    return (agg, domain_disp) if return_breakdown else agg


def _level_dispersions(df_sub, labels, active_domains, min_participants, metric_weights=None):
    """Level -> dispersion for a (possibly permuted) labelling aligned to df_sub rows,
    so the permutation null uses exactly the same aggregation as the observed value."""
    tmp = df_sub.assign(_lev=labels)
    return {str(lev): _aggregate_dispersion(lev_df.groupby("participant"), active_domains,
                                            min_participants, metric_weights)
            for lev, lev_df in tmp.groupby("_lev")}


def _disp_gap(disps):
    vals = [v for v in disps.values() if v == v]
    return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan


def _consistency_perm_pvalue(df_sub, factor, active_domains, min_participants,
                             n_perm, seed=0, metric_weights=None):
    """Within-participant permutation test on the best-minus-worst dispersion gap.
    Shuffling level labels within each participant keeps each person's own data
    and only breaks the level-dispersion link. p = share of shuffled gaps >= observed."""
    labels0 = df_sub[factor].astype(str).values
    obs = _disp_gap(_level_dispersions(df_sub, labels0, active_domains, min_participants, metric_weights))
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
        g = _disp_gap(_level_dispersions(df_sub, perm, active_domains, min_participants, metric_weights))
        if g == g:
            valid += 1
            if g >= obs - 1e-12:
                ge += 1
    return obs, ((ge + 1) / (valid + 1) if valid else np.nan)


def consistency_ranking(df_h: pd.DataFrame, active_domains: dict, metric_weights: dict,
                        height: str, n_perm: int = 1000, alpha: float = 0.05, min_participants: int = 2):
    """Stage 2 for one height. Returns (level_rows_df, verdict)."""
    rows, verdict = [], {}
    for factor in PARAM_factorS:
        level_disp, level_n = {}, {}
        for lev, lev_df in df_h.groupby(factor, dropna=True):
            level_disp[str(lev)] = _aggregate_dispersion(lev_df.groupby("participant"), active_domains,
                                                         min_participants, metric_weights)
            level_n[str(lev)] = lev_df["participant"].nunique()
        valid = {k: v for k, v in level_disp.items() if v == v}
        best = min(valid, key=valid.get) if valid else None
        pval = (_consistency_perm_pvalue(df_h, factor, active_domains, min_participants, n_perm,
                                         metric_weights=metric_weights)[1] if best else np.nan)
        sig = bool(pval == pval and pval < alpha)
        verdict[factor] = {"winner": best if sig else None, "best_level": best,
                           "level_disp": level_disp, "p_value": pval, "significant": sig}
        for lev, d in level_disp.items():
            rows.append({"height": height, "factor": factor, "level": lev, "dispersion": d,
                         "n_participants": level_n[lev], "is_best": lev == best,
                         "p_value": pval, "significant": sig})
    return pd.DataFrame(rows), verdict


def config_strings(verdict):
    """(recommended config using significant factors only, best level of every factor)."""
    rec = "_".join(verdict[f]["winner"] or "?" for f in PARAM_factorS)
    best = "_".join(verdict[f]["best_level"] or "?" for f in PARAM_factorS)
    return rec, best


def weighted_config_ranking(verdict: dict, weights: dict, height: str) -> pd.DataFrame:
    """Rank all configurations at one height by the weighted consistency score
        S_c = sum_f W_f * sigma_{f, level_f(c)} / mean_l(sigma_{f, l})
    (see module docstring). weights = {factor: Stage 1 weight}. Lower = more
    consistent; the top-ranked configuration always combines each factor's most
    consistent level."""
    import itertools
    rel = {}
    for f in PARAM_factorS:
        disp = {lev: d for lev, d in (verdict.get(f, {}).get("level_disp") or {}).items() if d == d}
        if not disp:
            print(f"  [WARN] weighted ranking skipped at {height}: no consistency result for {f} "
                  f"(too few participants per level?)")
            return pd.DataFrame()
        if weights.get(f, np.nan) != weights.get(f, np.nan):
            print(f"  [WARN] weighted ranking skipped at {height}: no Stage 1 weight for {f}")
            return pd.DataFrame()
        mean_d = float(np.mean(list(disp.values())))
        rel[f] = {lev: d / mean_d for lev, d in disp.items()} if mean_d > 0 else {lev: 1.0 for lev in disp}
    rows = []
    ordered = [sorted(rel[f], key=lambda l: LEVEL_ORDER[f].index(l) if l in LEVEL_ORDER[f] else 99)
               for f in PARAM_factorS]
    for combo in itertools.product(*ordered):
        levels = dict(zip(PARAM_factorS, combo))
        row = {"height": height, "config": "_".join(combo),
               "score": sum(weights[f] * rel[f][levels[f]] for f in PARAM_factorS)}
        for f in PARAM_factorS:
            row[f] = levels[f]
            row[f"{f}_relative_dispersion"] = rel[f][levels[f]]
            row[f"{f}_weight"] = weights[f]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    out.insert(1, "rank", range(1, len(out) + 1))
    best = out["score"].iloc[0]
    out["pct_worse_than_best"] = 100.0 * (out["score"] - best) / best
    return out


def print_weighted_ranking(rank_df: pd.DataFrame, height: str, top_n: int = 5):
    print(f"\n  WEIGHTED CONFIGURATION RANKING (consistency x Stage 1 weights) -- {height}")
    print(f"      {'Rk':<3} {'Configuration':<34} {'Score':>6} {'vs best':>8}")
    for _, r in rank_df.head(top_n).iterrows():
        print(f"      {int(r['rank']):<3} {r['config']:<34} {r['score']:>6.3f} {r['pct_worse_than_best']:>+7.1f}%")
    print(f"      ... {len(rank_df)} configurations in total; score = weighted mean relative "
          f"dispersion (lower = more consistent)")


def print_consistency_recommendation(verdict, weights, height):
    rec, best = config_strings(verdict)
    print(f"\n  RECOMMENDATION (consistency) -- {height}: {rec}")
    for f in PARAM_factorS:
        v = verdict[f]
        if v["best_level"] is None:
            print(f"      {f:<8}: no verdict (insufficient participants per level)")
            continue
        ordered = sorted(v["level_disp"].items(), key=lambda x: (x[1] if x[1] == x[1] else 9e99))
        ds = ", ".join(f"{lv}={d:.3f}" for lv, d in ordered)
        wt = weights.get(f, np.nan)
        wts = f"{wt:.2f}" if wt == wt else "n/a"
        ptxt = f"p={v['p_value']:.3f}" if v["p_value"] == v["p_value"] else "p=n/a"
        if v["significant"]:
            print(f"      {f:<8}: {v['winner']:<14} weight={wts}  {ptxt}  [disp {ds}]")
        else:
            print(f"      {f:<8}: (n.s.) best={v['best_level']:<9} weight={wts}  {ptxt}  [disp {ds}]  -> no winner")
    print(f"  Best level per factor (significant or not): {best}")


def _tex(s):
    return str(s).replace("_", r"\_")


def write_consistency_table(results, path: Path, alpha: float):
    """Stage 2 LaTeX table: one column per workstation, one row per factor level.
    Bold = most consistent level of each factor at that height; the p row is the
    within-participant permutation test (bold if significant). The two bottom rows
    give the most consistent configuration (every factor's most consistent level)
    and the recommended configuration (significant factors only). Needs booktabs."""
    heights = list(results)
    n = len(heights)
    cfg_label = {"Length": None, "Size": None, "Weight": None, "Angle": None}

    def level_text(lev):
        return TEX_LEVEL[lev]

    lines = [r"% Generated by 05_rank_prototypes.py -- rerun to update. Requires booktabs.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{factor-level between-participant consistency (MCDA). Values are the "
             r"between-participant SD of desirability scores (0--100 points) for each level of each "
             r"factor; lower values indicate closer agreement between participants. Bold marks the most "
             r"consistent level of each factor at each workstation (unbolded where levels tie at the "
             r"precision shown). $p$ is from a within-participant "
             r"permutation test of the gap between the most and least consistent levels, in bold where "
             rf"$p < {alpha:g}$. The most consistent configuration combines the most consistent level of "
             r"every factor; the recommended configuration includes only factors with a significant "
             r"difference.}",
             r"\label{tab:mcda_consistency}",
             r"\begin{tabular}{@{}ll" + "c" * n + r"@{}}", r"\toprule",
             rf" & & \multicolumn{{{n}}}{{c}}{{Workstation}} \\",
             rf"\cmidrule(l){{3-{n + 2}}}",
             "factor & Level & " + " & ".join(heights) + r" \\", r"\midrule"]
    for fi, f in enumerate(FIG_factorS):
        # bold the most consistent level only if it is visibly lowest at 2 dp
        shown = {h: {lev: (f"{d:.2f}" if d == d else None)
                     for lev in LEVEL_ORDER[f] for d in [results[h][0][f]["level_disp"].get(lev, np.nan)]}
                 for h in heights}
        for li, lev in enumerate(LEVEL_ORDER[f]):
            cells = []
            for h in heights:
                best = results[h][0][f]["best_level"]
                txt = shown[h][lev]
                tied = best is not None and sum(x == shown[h].get(best) for x in shown[h].values()) > 1
                cells.append("--" if txt is None else (rf"\textbf{{{txt}}}" if (lev == best and not tied) else txt))
            lines.append(f"{f if li == 0 else ''} & {level_text(lev)} & " + " & ".join(cells) + r" \\")
        pcells = []
        for h in heights:
            v = results[h][0][f]
            p = v["p_value"]
            pcells.append("--" if p != p else (rf"\textbf{{{p:.3f}}}" if v["significant"] else f"{p:.3f}"))
        lines.append(r" & \textit{p} & " + " & ".join(pcells) + r" \\")
        if fi < len(FIG_factorS) - 1:
            lines.append(r"\addlinespace")
    lines.append(r"\midrule")

    best_cells, rec_cells = [], []
    for h in heights:
        v = results[h][0]
        lv = {f: v[f]["best_level"] for f in PARAM_factorS}
        if all(lv.values()):
            best_cells.append(rf"\shortstack{{{level_text(lv['Length'])}, {level_text(lv['Size'])}\\"
                              rf"{level_text(lv['Weight'])}, {level_text(lv['Angle'])}}}")
        else:
            best_cells.append("--")
        sig = [level_text(v[f]["winner"]) for f in PARAM_factorS if v[f]["winner"]]
        rec_cells.append(", ".join(sig) if sig else "none")
    lines.append(r"\multicolumn{2}{@{}l}{Most consistent configuration} & " + " & ".join(best_cells) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{2}{@{}l}{Recommended (significant factors)} & " + " & ".join(rec_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================================== #
# SECTION 4A: THESIS FIGURES -- ONE PER STAGE PER HEIGHT
# =========================================================================== #

STAGE1_DOMAIN_COLOURS = {"Performance": "#c8642a",      # soft orange palette, as the other
                         "Postural_Risk": "#e39a62",    # thesis figures (dark -> light)
                         "Grip_Ergonomics": "#f2c8a2"}
DOMAIN_STYLE = {"Performance": ("Task performance", "#3d3d3d"),
                "Postural_Risk": ("Postural risk", "#8a8a8a"),
                "Grip_Ergonomics": ("Grip ergonomics", "#c9c9c9")}
DISPLAY_LEVEL = {"Short": "Short", "Long": "Long", "Small": "Small", "Large": "Large",
                 "Not_weighted": "Non-weighted", "Front_weighted": "Weighted",
                 "A90": "90\u00b0", "A135": "135\u00b0", "A180": "180\u00b0"}


def plot_mcda_stages(df_h, active_domains, metric_weights, tests, fdf, verdict, height, out_dir):
    """Two figures per height, each a 2x2 grid with one panel per factor.
      mcda_stage1_effects_<height>   each metric's contribution to the factor's
            factor weight (bars sum to the weight in the title), shaded by ergonomic
            domain; filled = significant after FDR correction, outline = not.
      mcda_stage2_consistency_<height>  for each setting, each metric's
            between-participant SD of desirability (dot size = metric weight) and their
            weighted mean sigma (black line; the value in the consistency table); most
            consistent setting in bold."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    eff_w = effective_weights(metric_weights, active_domains)
    col_domain = {c: d for d, cols in active_domains.items() for c in cols}
    metrics = [c.replace("_dscore", "") for d in DOMAIN_STYLE for c in active_domains.get(d, [])]
    weights = dict(zip(fdf["factor"], fdf["data_weight"]))
    fmt = lambda p: ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") if p == p else "p = n/a"

    # ---------------- Stage 1 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), sharex=True, sharey=True)
    xmax = max(0.01, tests["contribution"].max() * 1.15) if len(tests) else 1
    y = np.arange(len(metrics))
    for ax, f in zip(axes.ravel(), FIG_factorS):
        sub = tests[tests["factor"] == f].set_index("metric")
        for yi, m in enumerate(metrics):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            colour = STAGE1_DOMAIN_COLOURS[col_domain[m + "_dscore"]]
            if bool(r["significant_fdr"]):
                ax.barh(yi, r["contribution"], height=0.7, color=colour, edgecolor=colour)
            else:
                ax.barh(yi, r["contribution"], height=0.7, color="white", edgecolor=colour, linewidth=1)
        n_sig = int(sub["significant_fdr"].sum()) if len(sub) else 0
        ax.set_title(f"{f}: Factor weight {weights.get(f, np.nan):.2f}"
                     , fontsize=9, loc="center")
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([METRIC_REGISTRY[m]["label"] for m in metrics], fontsize=7.5)
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
    for ax in axes[1, :]:
        ax.set_xlabel("Contribution to Factor Weight ")
    unadjusted = len(tests) and tests["p_adjust"].iloc[0] == "none"
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE1_DOMAIN_COLOURS[d], label=DOMAIN_STYLE[d][0])
               for d in DOMAIN_STYLE]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#c8642a",
                                 label=("Outline: not significant (p \u2265 0.05, unadjusted)" if unadjusted
                                        else "Outline: not significant after FDR correction")))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, -0.01))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.12, hspace=0.35, wspace=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"mcda_stage1_effects_{height}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Stage 2 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    wmax = max(eff_w.values()) if eff_w else 1
    xmax, xmin = 0.0, np.inf
    panel_data = []
    for f in FIG_factorS:
        v = verdict[f]
        levels = [l for l in LEVEL_ORDER[f] if l in v["level_disp"]]
        rows = []
        for lev in levels:
            lev_df = df_h[df_h[f] == lev]
            sds = []
            for c, w in eff_w.items():
                pm = lev_df.groupby("participant")[c].mean().dropna()
                if len(pm) >= 2:
                    sds.append((c, float(pm.std(ddof=1)), w))
            rows.append((lev, sds, v["level_disp"].get(lev, np.nan)))
            xmax = max(xmax, max((sd for _, sd, _ in sds), default=0))
            xmin = min(xmin, min((sd for _, sd, _ in sds), default=np.inf))
        panel_data.append((f, v, rows))
    for ax, (f, v, rows) in zip(axes.ravel(), panel_data):
        n = len(rows)
        for i, (lev, sds, sigma) in enumerate(rows):
            yi = n - 1 - i                                   # first setting on top
            for c, sd, w in sds:
                ax.scatter(sd, yi, s=15 + 110 * w / wmax, color=DOMAIN_STYLE[col_domain[c]][1],
                           edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
            if sigma == sigma:
                ax.plot([sigma, sigma], [yi - 0.25, yi + 0.25], color="black", linewidth=2, zorder=4)
                ax.text(sigma, yi + 0.3, f"\u03c3 = {sigma:.2f}", ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold" if lev == v["best_level"] else "normal")
        ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_yticklabels([DISPLAY_LEVEL[r[0]] for r in rows])
        for t, r in zip(ax.get_yticklabels(), rows):
            if r[0] == v["best_level"]:
                t.set_fontweight("bold")
        ax.set_ylim(-0.6, n - 0.25)
        span = (xmax - xmin) if np.isfinite(xmin) and xmax > xmin else 1.0
        ax.set_xlim(xmin - 0.08 * span, xmax + 0.08 * span)
        ax.set_title(f"{f}: Factor weight {weights.get(f, np.nan):.2f}\nconsistency {fmt(v['p_value'])}", fontsize=9, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Between-participant SD of desirability (points)")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, markersize=6, label=n)
               for n, c in DOMAIN_STYLE.values()]
    handles.append(plt.Line2D([0], [0], color="black", linewidth=2, label="\u03c3: weighted mean (dot size = metric weight)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.14, hspace=0.5, wspace=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"mcda_stage2_consistency_{height}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# SECTION 4B: APPENDIX -- HEURISTIC (DIRECTIONAL) SYNTHESIS
# =========================================================================== #

def directional_synthesis(df_h: pd.DataFrame, active_domains: dict, metric_weights: dict,
                          effects: pd.DataFrame, height: str) -> dict:
    """factor-level synthesis S_p for one height: each level's standardised
    mixed-model effects, signed toward the ergonomic optimum and combined with the
    effective sensitivity weights. Reference levels stay at 0.
    Returns {factor: {"best_level", "level_scores"}}."""
    tbl = effects[effects["stratum"] == height]
    eff_w = effective_weights(metric_weights, active_domains)
    out = {}
    for f in PARAM_factorS:
        scores = {lev: 0.0 for lev in LEVEL_ORDER[f]}
        any_metric = False
        for c, w in eff_w.items():
            raw = c.replace("_dscore", "")
            sd = pd.to_numeric(df_h[raw], errors="coerce").std(ddof=1)
            rows = tbl[(tbl["metric"] == raw) & (tbl["factor"] == f)]
            if rows.empty or not np.isfinite(sd) or sd < 1e-9:
                continue
            direction = 1.0 if METRIC_REGISTRY[raw]["dir"] == "max" else -1.0
            for _, r in rows.iterrows():
                if pd.isna(r["effect"]) or pd.isna(r.get("level")):
                    continue
                lev = EFFECT_LEVEL.get(f, {}).get(str(r["level"]), str(r["level"]))
                if lev in scores:
                    scores[lev] += r["effect"] / sd * direction * w
                    any_metric = True
        out[f] = {"best_level": max(scores, key=scores.get) if any_metric else None,
                  "level_scores": scores}
    return out


def print_directional(verdict, height):
    best = "_".join(verdict[f]["best_level"] or "?" for f in PARAM_factorS)
    print(f"\n  HEURISTIC SYNTHESIS (appendix) -- {height}: {best}")
    for f in PARAM_factorS:
        sc = verdict[f]["level_scores"]
        txt = ", ".join(f"{lev}={sc[lev]:+.4f}{' (ref)' if lev == factor_REFERENCE[f] else ''}"
                        for lev in sorted(sc, key=lambda k: -sc[k]))
        print(f"      {f:<8}: {txt}")


def _config_cell(levels: dict) -> str:
    """Configuration written out in words over two lines, for the table summary rows."""
    if not all(levels.get(f) for f in PARAM_factorS):
        return "--"
    return (rf"\shortstack{{{TEX_LEVEL[levels['Length']]}, {TEX_LEVEL[levels['Size']]}\\"
            rf"{TEX_LEVEL[levels['Weight']]}, {TEX_LEVEL[levels['Angle']]}}}")


def write_directional_table(dir_results: dict, path: Path):
    """Appendix LaTeX table of S_p, same layout as the Stage 2 consistency table."""
    heights = list(dir_results)
    n = len(heights)
    lines = [r"% Generated by 05_rank_prototypes.py -- rerun to update. Requires booktabs.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{factor-level synthesis scores ($S_p$) from the heuristic MCDA ranking. "
             r"$S_p$ is the sensitivity-weighted sum of each level's standardised mixed-model "
             r"effects, signed so that positive values indicate a more favourable expected "
             r"ergonomic outcome than the factor's reference level, which is fixed at zero. Bold "
             r"marks the highest-scoring level of each factor at each workstation. The synthesis "
             r"has no associated significance test.}",
             r"\label{tab:mcda_directional}",
             r"\begin{tabular}{@{}ll" + "c" * n + r"@{}}", r"\toprule",
             rf" & & \multicolumn{{{n}}}{{c}}{{Workstation}} \\",
             rf"\cmidrule(l){{3-{n + 2}}}",
             "factor & Level & " + " & ".join(heights) + r" \\", r"\midrule"]
    for fi, f in enumerate(FIG_factorS):
        for li, lev in enumerate(LEVEL_ORDER[f]):
            cells = []
            for h in heights:
                v = dir_results[h][f]
                val = v["level_scores"].get(lev, np.nan)
                best = lev == v["best_level"]
                if lev == factor_REFERENCE[f]:
                    txt = r"\textbf{0 (ref.)}" if best else "0 (ref.)"
                elif val != val:
                    txt = "--"
                else:                      # maths mode for a true minus sign; \boldmath to embolden it
                    txt = (rf"{{\boldmath${val:+.3f}$}}" if best else f"${val:+.3f}$")
                cells.append(txt)
            lines.append(f"{f if li == 0 else ''} & {TEX_LEVEL[lev]} & " + " & ".join(cells) + r" \\")
        if fi < len(FIG_factorS) - 1:
            lines.append(r"\addlinespace")
    lines.append(r"\midrule")
    cells = [_config_cell({f: dir_results[h][f]["best_level"] for f in PARAM_factorS}) for h in heights]
    lines.append(r"\multicolumn{2}{@{}l}{Highest-scoring configuration} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================================== #
# SECTION 5: CLI
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, default=None, help="Root directory containing metrics/ CSVs")
    ap.add_argument("--combined-csv", type=Path, default=None,
                    help="Explicit path to combined_place_metrics.csv (from evaluate_difference.py)")
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="Directory containing evaluate_difference.py's stratified_stat_tests.csv "
                         "(default: <landmarks-root>/metrics/combined_comparison)")
    ap.add_argument("--p-adjust", choices=["fdr", "none"], default="fdr",
                    help="Stage 1 significance: 'fdr' (Benjamini-Hochberg, default) or 'none' (unadjusted)")
    ap.add_argument("--importance-weighting", choices=["pipeline", "equal"], default="pipeline",
                    help="Stage 1 metric weighting (default: the pipeline's sensitivity weights)")
    ap.add_argument("--use-existing-effects", action="store_true",
                    help="Read stratified_stat_tests.csv from --comparison-dir instead of refitting the models")
    ap.add_argument("--min-n", type=int, default=8,
                    help="Minimum events per height and metric for a model to be fitted (default 8)")
    ap.add_argument("--n-perm", type=int, default=1000, help="Permutations for the Stage 2 test (default 1000)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    ap.add_argument("--no-exclude", action="store_true", help="Keep events even if in the cleaning manifest")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="excluded_place_events.csv (default <root>/metrics/cleaning/excluded_place_events.csv)")
    args = ap.parse_args()

    if args.landmarks_root:
        combined_path  = args.combined_csv or (args.landmarks_root / "metrics" / "combined_place_metrics.csv")
        out_dir        = args.landmarks_root / "metrics" / "prototype_rankings"
        comparison_dir = args.comparison_dir or (args.landmarks_root / "metrics" / "combined_comparison")
        exclude_csv    = args.exclude_csv or (args.landmarks_root / "metrics" / "cleaning" / "excluded_place_events.csv")
    elif args.combined_csv:
        combined_path  = args.combined_csv
        out_dir        = combined_path.parent / "prototype_rankings"
        comparison_dir = args.comparison_dir
        exclude_csv    = args.exclude_csv
    else:
        sys.exit("Error: provide --landmarks-root OR --combined-csv.")

    if not Path(combined_path).is_file():
        sys.exit(f"Error: {combined_path} not found. Run evaluate_difference.py --mode extract first.")
    df = pd.read_csv(combined_path)
    print(f"Loaded {len(df)} Place events from {combined_path}.")

    if not args.no_exclude:
        excl = load_excluded_events(exclude_csv)
        if excl and {"participant", "trial", "height", "place_index"}.issubset(df.columns):
            keys = df.apply(lambda r: _event_key(r["participant"], r["trial"], r["height"], r["place_index"]), axis=1)
            before = len(df); df = df[~keys.isin(excl)].copy()
            print(f"  [exclude] removed {before - len(df)} rejected place event(s); {len(df)} remain.")

    df["height"] = df["height"].astype(str).str.strip()
    unknown = ~df["height"].isin(VALID_HEIGHTS)
    if unknown.any():
        print(f"  [QUARANTINE] {int(unknown.sum())} event(s) matched no High/Medium/Low window and are excluded.")
        df = df[~unknown].copy()
    df = add_prototype_label(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_existing_effects:
        effects = load_stratified_effects(comparison_dir)
        if effects is None:
            sys.exit("Error: --use-existing-effects needs stratified_stat_tests.csv in --comparison-dir.")
        print("  Using existing mixed-model results from stratified_stat_tests.csv.")
    else:
        effects = fit_stratified_effects(df, args.min_n)
        effects.to_csv(out_dir / "mcda_stratified_stat_tests.csv", index=False)
    df_d, dom_d = calculate_desirability_scores(df)

    factor_rows, metric_rows, level_rows, rec_rows, test_rows, results = [], [], [], [], [], {}
    dir_rows, dir_results = [], {}
    ranking_rows = []
    for h in VALID_HEIGHTS:
        df_h = df_d[df_d["height"] == h]
        if df_h.empty:
            print(f"\n[{h}] no events; skipped.")
            continue
        print(f"\n{'='*74}\nHEIGHT: {h}  ({len(df_h)} events, {df_h['participant'].nunique()} participants)\n{'='*74}")

        # ---- Stage 1: factor importance ----
        fdf, mdf, used_w, tdf = factor_importance(df_h, dom_d, effects, h, args.importance_weighting,
                                                  args.alpha, args.p_adjust)
        print_factor_leaderboard(fdf, h, tdf)
        test_rows.append(tdf)
        factor_rows.append(fdf)
        metric_rows.append(mdf)
        weights = dict(zip(fdf["factor"], fdf["data_weight"]))

        # ---- Stage 2: consistency ranking (same sensitivity weights) ----
        ldf, verdict = consistency_ranking(df_h, dom_d, used_w, h, args.n_perm, args.alpha)
        print_consistency_recommendation(verdict, weights, h)
        level_rows.append(ldf.assign(data_weight=ldf["factor"].map(weights)))
        rec, best = config_strings(verdict)
        rec_rows.append({"height": h, "recommended_config": rec, "best_level_config": best})
        rank_df = weighted_config_ranking(verdict, weights, h)
        if not rank_df.empty:
            print_weighted_ranking(rank_df, h)
            ranking_rows.append(rank_df)
        results[h] = (verdict, weights)
        plot_mcda_stages(df_h, dom_d, used_w, tdf, fdf, verdict, h, out_dir)

        # ---- Appendix: heuristic (directional) synthesis, same effects and weights ----
        dverdict = directional_synthesis(df_h, dom_d, used_w, effects, h)
        print_directional(dverdict, h)
        dir_results[h] = dverdict
        for f in PARAM_factorS:
            for lev, sc in dverdict[f]["level_scores"].items():
                dir_rows.append({"height": h, "factor": f, "level": lev, "S_p": sc,
                                 "is_reference": lev == factor_REFERENCE[f],
                                 "is_best": lev == dverdict[f]["best_level"]})

    if not results:
        sys.exit("No height stratum had events.")

    fw = pd.concat(factor_rows, ignore_index=True)
    fw.to_csv(out_dir / "mcda_factor_weights.csv", index=False)
    pd.concat(metric_rows, ignore_index=True).to_csv(out_dir / "mcda_metric_weights.csv", index=False)
    pd.concat(level_rows, ignore_index=True).to_csv(out_dir / "mcda_consistency_by_height.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(out_dir / "mcda_recommended_config_by_height.csv", index=False)
    if ranking_rows:
        pd.concat(ranking_rows, ignore_index=True).to_csv(out_dir / "mcda_weighted_config_ranking.csv", index=False)
    else:
        print("  [WARN] mcda_weighted_config_ranking.csv NOT written: the ranking was skipped at every "
              "height (see warnings above).")
    plot_factor_importance(fw, out_dir)
    tests = pd.concat(test_rows, ignore_index=True)
    tests.to_csv(out_dir / "mcda_significant_metrics.csv", index=False)
    write_significance_table(tests, out_dir / "mcda_significant_metrics.tex", args.alpha)
    write_consistency_table(results, out_dir / "mcda_consistency_table.tex", args.alpha)
    pd.DataFrame(dir_rows).to_csv(out_dir / "mcda_directional_by_height.csv", index=False)
    write_directional_table(dir_results, out_dir / "mcda_directional_table.tex")

    print(f"\n{'='*64}\nRECOMMENDED CONFIG PER HEIGHT (consistency; significant factors only)\n{'='*64}")
    for r in rec_rows:
        print(f"  {r['height']:<7}: {r['recommended_config']:<34} (best levels: {r['best_level_config']})")
    print(f"\nOutputs written to:\n  -> {out_dir}")
    print(f"  Stage 1 figure: {(out_dir / 'mcda_factor_importance.pdf').resolve()}")
    names = ([] if args.use_existing_effects else ["mcda_stratified_stat_tests.csv"]) + [
        "mcda_factor_weights.csv", "mcda_metric_weights.csv", "mcda_factor_importance.pdf",
        "mcda_significant_metrics.csv", "mcda_significant_metrics.tex",
        "mcda_consistency_by_height.csv", "mcda_recommended_config_by_height.csv",
        "mcda_weighted_config_ranking.csv", "mcda_stage1_effects_<height>.pdf",
        "mcda_stage2_consistency_<height>.pdf",
        "mcda_consistency_table.tex", "mcda_directional_by_height.csv", "mcda_directional_table.tex"]
    for name in names:
        print(f"     {name}")


if __name__ == "__main__":
    main()