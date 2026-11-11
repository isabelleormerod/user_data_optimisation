# User Data Optimisation

Analysis code for the end-effector prototype study in the PhD thesis. Participants did a Place task with 24 prototype configurations at three workstation heights (High / Medium / Low) while pen, body and hand motion was tracked. The prototypes varied in:

| Factor | Levels |
|---|---|
| Length | Short, Long |
| Size | Small, Large |
| Weight | Not_weighted, Front_weighted |
| Angle | A90, A135, A180 |

The pipeline turns the raw tracking recordings and BORIS behaviour labels into cleaned Place events. Two independent methods then rank the factors and prototypes:

- **MCDA**: hand-crafted ergonomic metrics, linear mixed-effects models, and weighted desirability or consistency.
- **fPCA–LDA**: functional PCA on the joint-angle curves, then a Fisher LDA for each factor.

The results of both methods are then compared with the participants' own rankings.

---

## 1. Setup

Python 3.10+ is required. The code was last run on Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas scipy matplotlib statsmodels scikit-learn fdasrsf plotly
```

**Run every command from the repo root, with the repo root on `PYTHONPATH`.** The scripts import the shared `utils/` and `preprocessing/` packages. Without this setting they fail with `ModuleNotFoundError: No module named 'utils'`.

```powershell
cd C:\Python_directory\Repos\user_data_optimisation
$env:PYTHONPATH = (Get-Location).Path
```

For convenience, set the data root once per session. Every later command uses it:

```powershell
$ROOT  = "A:\Automated_chain_BETA\Participant_Landmarks"
$BORIS = "A:\Automated_chain_BETA\BORIS_csvs"
```

The data is not in the repo: `*.json`, `*.csv` and `*.tsv` are git-ignored. The defaults point to the `A:\Automated_chain_BETA` drive, which also appears in `user_data_optimisation.code-workspace`.

---

## 2. Data layout

Raw input:

```
Participant_Landmarks/
  <stem>.csv.json            one master JSON per recording (pen + body + hand, from Unity)
BORIS_csvs/
  *.tsv | *.csv              BORIS behaviour exports (Place, High/Medium/Low, calibration "Point"...)
```

Here `<stem>` is the trial ID, for example `P003_Long_Large_Front_weighted_A180`. Its first token is the participant ID (`P003`).

The preprocessing builds one folder per trial, and every later step reads from these folders:

```
Participant_Landmarks/
  plane_quality_log.csv                      (step 2)
  <PID>/<stem>/
    <stem>_pen.csv / _body.csv / _hand.csv   (step 1: synced streams, shared t_s column)
    <stem>_sync.json                         (step 1: sync diagnostics)
    <stem>_boris_synced.csv                  (step 1: BORIS events on the t_s timeline)
    <stem>_pen_flattened.csv                 (step 2)
    <stem>_*_labelled.csv                    (step 3: one 0/1 column per behaviour)
    <stem>_body_zfilt.csv                    (step 4: median-filtered body z)
  metrics/                                   (all analysis outputs, see below)
```

---

## 3. Running the pipeline

Run the steps in order. Every script accepts `--help`, and most accept `--participants P003,P004` so you can run a subset.

### Stage A: Preprocessing (`preprocessing/`)

**1. Sync streams and convert BORIS times.** This step runs `sync_to_pen_time.py` and then `convert_boris_to_synced_time.py` on every recording. It puts pen, body and hand on a single timeline (`t_s` = seconds since the first body sample) and maps each BORIS event onto that timeline.

```powershell
python preprocessing/01_batch_sync.py --json-folder $ROOT --boris-folder $BORIS
```

- `--skip-sync` re-runs only the BORIS conversion.
- `--dry-run` shows what would be processed without doing it.
- Participants can be given as `3,4` or as `P003,P004`.

**2. Flatten the pen data onto the calibration plane.** This step fits a plane to the BORIS `Point` calibration events and writes `_pen_flattened.csv` and `plane_quality_log.csv`.

```powershell
python preprocessing/02_flatten_pen_to_plane.py --batch --landmarks-root $ROOT
```

**3. Label the tracking data with BORIS behaviours.** This step needs the `_pen_flattened.csv` from step 2.

```powershell
python preprocessing/03_label_tracking_with_boris.py $ROOT --batch
```

**4. Clean the Place events.** This step applies a coverage filter to every Place event and a median filter to body z. It sweeps four coverage thresholds (0.60, 0.70, 0.80 and 0.90) and writes one exclusion manifest per threshold.

```powershell
python preprocessing/clean_place_events.py --landmarks-root $ROOT
```

Outputs go to `metrics/cleaning_sensitivity/threshold_X.XX/excluded_place_events.csv`, plus `sensitivity_summary.csv`.

> **Important: choose the manifest.** The later scripts look for the manifest at `metrics/cleaning/excluded_place_events.csv` by default, but step 4 does not write a file there. Copy your chosen threshold's folder into place:
>
> ```powershell
> Copy-Item -Recurse "$ROOT\metrics\cleaning_sensitivity\threshold_0.80" "$ROOT\metrics\cleaning"
> ```
>
> Alternatively, pass `--exclude-csv <path>` to every analysis script. **If neither is done, no events are excluded and you get no warning.**

### Stage B: Optional quality check (`quality_check/`)

This step builds a lazy-loading HTML dashboard of the raw tracking. It shows dropouts, jumps and the Place and height windows.

```powershell
python quality_check/quality_viz.py --landmarks-root $ROOT --out .\quality_check
```

Open `quality_check/quality_dashboard.html` in a browser. The default `--out` is `$ROOT\metrics\quality_viz`.

To inspect a single trial frame by frame, open `Data_viewer_with_overlay/hand_pose_viewer.html` in a browser. Then load a trial's hand CSV, pen CSV and, optionally, its events CSV. Everything runs locally and nothing is uploaded.

### Stage C: MCDA (`MCDA/`)

**5. Extract metrics and fit the mixed models.** For every Place event, this step computes the pen task-performance, posture/REBA, grip and SPARC smoothness metrics. It then fits the following model for each metric, stratified by height:

`metric ~ C(Length) + C(Size) + C(Weight) + C(Angle) + (1 | participant)`

```powershell
python MCDA/04_evaluate_difference.py --mode all --landmarks-root $ROOT
```

- `--mode extract` writes `metrics/combined_place_metrics.csv`. This is the slow step.
- `--mode compare` re-runs the statistics on the existing CSV and writes to `metrics/combined_comparison/`.
- `--mode sensitivity` repeats the comparison at every cleaning threshold from step 4 and writes to `metrics/sensitivity_comparison/`. Use it to check that the results are robust to the threshold.

**6. Rank factors and prototypes (thesis version).** This step has two stages:

- Stage 1 gives the factor-importance weights.
- Stage 2 ranks the factor levels and all 24 configurations by consistency between participants, with a permutation test.

It also writes the directional ranking used in the appendix.

```powershell
python MCDA/05_analysis.py --landmarks-root $ROOT
```

Outputs go to `metrics/prototype_rankings/`. They include `mcda_factor_weights.csv`, `mcda_consistency_by_height.csv`, `mcda_weighted_config_ranking.csv`, `mcda_directional_by_height.csv`, the LaTeX tables and the figures.

- `--use-existing-effects` reuses the step 5 model fits instead of refitting them.
- `--importance-weighting equal` runs the equal-weight benchmark.

`MCDA/05_rank_prototypes.py` is an earlier, alternative ranking engine. It uses desirability leaderboards with `--score-mode directional|normative|both`, writes its own files to the same `prototype_rankings/` folder, and is not needed for the thesis figures.

### Stage D: fPCA–LDA (`fPCA_LDA/`)

**7. Run the hierarchical fPCA and LDA.** This step works in four parts:

1. It resamples the joint-angle curves of each Place event. Elastic registration is off by default.
2. It fits an fPCA for each domain (pen, body and right hand) and each height.
3. It fits an LDA for each factor.
4. It ranks the factors and configurations by consistency, using the same approach as step 6.

```powershell
python fPCA_LDA/run_hierarchical_fpca.py --mode all --landmarks-root $ROOT
```

- `--mode extract` caches the curves to `metrics/fpca_cache/`. After that, `--mode compare` re-runs only the modelling, which is fast.
- Results go to `metrics/fpca_results/`. They include `lda_factor_weights.csv`, `recommended_config_by_height.csv`, `lda_weighted_config_ranking.csv` and the figures.
- Robustness variants:
  - `--registration` turns elastic registration on and writes to `fpca_results_reg/`.
  - `--timing-block` adds a timing block and writes to `fpca_results_timing/`.
  - `--n-components-{pen,body,hand} N` sets the number of fPCA components kept for each domain.

### Stage E: Compare the pipelines with the participants (`comparison/`)

**8. Compare the MCDA, fPCA–LDA and participant results.** This step reads the outputs of steps 6 and 7 together with two participant sheets. Both sheets are TSV files and are not in the repo:

- `factor_importance.tsv`: each participant's stated importance of the four factors at each height. The columns are `Participant Name | Factor | Low | Medium | High`, with the name on the first of each participant's four rows.
- `rankings.tsv`: each participant's rankings of the prototypes.

```powershell
python comparison/compare_all.py --landmarks-root $ROOT `
    --importance-sheet path\to\factor_importance.tsv `
    --rankings-sheet   path\to\rankings.tsv `
    --out comparison
```

- **Always pass `--out comparison`.** Without it, the outputs go to `comparison/comparison/`.
- If you leave out the sheet paths, the script searches `comparison/`, the current folder and the folders below `comparison/` for files with those names.
- `--mcda-dir` and `--fpca-dir` replace the default input folders (`metrics/prototype_rankings` and `metrics/fpca_results`), for example to compare the `fpca_results_reg` variant.

Outputs:

| File | Content |
|---|---|
| `stage1_weights.pdf/.png` | Each factor's weight from MCDA, fPCA–LDA and the participants, averaged over the three heights. The thin line on each bar spans the per-height values. |
| `stage1_weights_by_height.pdf/.png` | The same, with one panel per height (appendix). |
| `stage1_weights.csv/.tex` | Per-height weights, their means and Kendall's W. |
| `stage2_preferences.pdf/.png` | Participants' revealed preferences, with each pipeline's most consistent setting alongside. A hollow marker means that pipeline's own consistency test was not significant. |
| `participant_pipeline_agreement.tex`, `revealed_preferences_tests.csv` | The participants' group preference for each factor against the two pipelines' choices, with the significance tests. |
| `stage2_rankings.csv/.tex` | Each pipeline's top-ranked configuration per height, and the Spearman correlation between the two pipelines' rankings of all configurations. |

Participant weights are Borda shares: a participant's 1st choice gets 3 points and the 4th gets 0. Even unanimous agreement therefore gives at most 0.50, so compare the **order** of the factors across sources rather than the size of the bars.

The participant names in `factor_importance.tsv` are mapped to participant IDs through the `IDS` dictionary near the top of `compare_all.py`, and names never appear in the outputs. Add any new participants to that dictionary.

> The script's help text and console messages refer to `05_rank_prototypes.py`, but the files it reads are written by `MCDA/05_analysis.py`.

---

## 4. Quick reference

| Step | Script | Main input | Main output |
|---|---|---|---|
| 1 | `preprocessing/01_batch_sync.py` | master JSONs, BORIS exports | `<PID>/<stem>/` synced CSVs |
| 2 | `preprocessing/02_flatten_pen_to_plane.py` | `_pen.csv`, `_boris_synced.csv` | `_pen_flattened.csv`, `plane_quality_log.csv` |
| 3 | `preprocessing/03_label_tracking_with_boris.py` | step 1–2 CSVs | `_*_labelled.csv` |
| 4 | `preprocessing/clean_place_events.py` | labelled CSVs | `metrics/cleaning_sensitivity/` and `_body_zfilt.csv` |
| – | `quality_check/quality_viz.py` | labelled CSVs | HTML dashboard |
| 5 | `MCDA/04_evaluate_difference.py` | labelled CSVs and the manifest | `metrics/combined_place_metrics.csv`, `metrics/combined_comparison/` |
| 6 | `MCDA/05_analysis.py` | step 5 | `metrics/prototype_rankings/mcda_*` |
| 7 | `fPCA_LDA/run_hierarchical_fpca.py` | labelled CSVs and the manifest | `metrics/fpca_results/` |
| 8 | `comparison/compare_all.py` | step 6 and 7 outputs, participant TSVs | `comparison/` figures and LaTeX tables |

The figures and tables committed in `comparison/` are the versions used in the thesis.

## 5. Repository structure

```
preprocessing/       steps 1-4 (sync, flatten, label, clean)
quality_check/       raw-data quality dashboard
Data_viewer_with_overlay/  standalone browser viewer for one trial
MCDA/                metric extraction, mixed models, MCDA ranking
fPCA_LDA/            functional PCA + LDA pipeline
comparison/          compare_all.py + final comparison figures/tables for the thesis
utils/               shared I/O, discovery, trial-name parsing, statistics
```

## License

See [LICENSE](LICENSE).
