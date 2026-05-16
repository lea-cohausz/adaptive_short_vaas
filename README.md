# Adaptive Questionnaire Experiment 

Adaptive questionnaire simulation for political party matching (Saarbrücken/Halle dataset).
The system selects items intelligently, predicts missing responses, and evaluates
whether a voter's best-matching party can be identified with fewer than the full
set of items.

The below documentation shows the project structure for Saarbrücken. You can use the same project structure for Halle, 
you only need to replace the files and the paths when starting experiments.

---

## Project structure

```
project/
├── src/                          # Shared utilities (used across all scripts)
│   ├── load_data.py              # Data loading & validation
│   ├── party_scoring.py          # Match-score arithmetic
│   ├── predictors.py             # Ridge, NaiveBayes, RandomForest, kNN
│   ├── item_selector.py          # Item selection strategies
│   └── stopping_rules.py        # Stopping criteria (fixed 50%, entropy, error)
│
├── adaptive_v2/src/              # Experiment layer
│   ├── simulation.py             # Core per-person simulation loop
│   ├── evaluation.py             # Metrics & bootstrap CIs
│   ├── plots.py                  # Publication-quality plots
│   ├── run_experiment.py         # ← MAIN entry point
│   ├── run_learning_curve_experiment.py
│   ├── run_pred_stopping_experiment.py
│   └── make_plots.py             # Regenerate plots from saved CSVs
│
└── data/
    ├── saar_item_ready_for_experiments.csv
    ├── saar_party_ready_for_experiments.csv
    └── Saarbru_prepared_Party.csv
```

---

## Requirements

**Python ≥ 3.9**

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Data files needed

Three CSV files must be present (paths can be overridden via CLI flags):

| File | Default path | Description |
|---|---|---|
| Item responses | `data/saar_item_ready_for_experiments.csv` | 5 224 persons × 38 cols (`id` + 37 items, values 0–4) |
| Party scores | `data/saar_party_ready_for_experiments.csv` | 5 224 persons × 13 cols (`id` + 6 parties × {res, acc}) |
| Party positions | `data/Saarbru_prepared_Party.csv` | 6 parties × 42 cols (`party` + item positions 0–4) |

These files are **not included** in the repository and must be obtained separately.

The data used in this project was obtained from the Harvard Dataverse:

Dataset: doi:10.7910/DVN/PAH9O1
Harvard Dataverse. https://doi.org/10.7910/DVN/PAH9O1

Replication Instructions
To replicate the dataset used in this analysis, download the full dataset from the link above and filter by the gmd_name (Saarbrücken/Halle) column to extract the relevant subset. For example, using Python:

---

## Running the main experiment

```bash
python adaptive_v2/src/run_experiment.py \
    --responses        data/saar_item_ready_for_experiments.csv \
    --party-scores     data/saar_party_ready_for_experiments.csv \
    --party-positions  data/Saarbru_prepared_Party.csv \
    --out-dir          adaptive_v2/results/ \
    --predictors       ridge naive_bayes \
    --strategies       entropy uncertainty random variance_reduction top1_change top3_change \
    --test-fraction    0.2 \
    --seed             42
```

This runs all predictor × strategy combinations (2 × 6 = 12 by default), saves
results, and generates plots automatically.

### Quick sanity-check run (ridge + entropy only)

```bash
python adaptive_v2/src/run_experiment.py \
    --responses        data/saar_item_ready_for_experiments.csv \
    --party-scores     data/saar_party_ready_for_experiments.csv \
    --party-positions  data/Saarbru_prepared_Party.csv \
    --quick
```

### Skip plot generation

Add `--no-plots` to any command to skip the matplotlib step.

---

## Outputs

All outputs are written to `--out-dir` (default `adaptive_v2/results/`):

```
results/
├── summary.csv                        # One row per predictor×strategy combination
├── per_person/
│   └── {predictor}__{strategy}.csv   # One row per test person
├── party_breakdown/
│   └── {predictor}__{strategy}__{condition}.csv
├── reports/
│   └── {predictor}__{strategy}.json  # Bootstrap CI summary
└── plots/
    └── *.png / *.pdf
```

### Key metrics reported

| Metric | Description |
|---|---|
| `obs_top1_50` | Top-1 accuracy at 50% of items (observed responses only) |
| `pred_top1_50` | Top-1 accuracy at 50% of items (observed + predicted responses) |
| `obs_top3_50` | Top-3 accuracy at 50% (observed only) |
| `pred_top3_50` | Top-3 accuracy at 50% (observed + predicted) |
| `mean_items_top1` | Mean items needed for adaptive top-1 stopping |
| `mean_items_top3` | Mean items needed for adaptive top-3 stopping |
| `reached_top1/3` | Fraction of persons where adaptive stopping was reached before cap |

All metrics include 95% bootstrap confidence intervals (1 000 resamples).

---

## Additional experiments

### Learning curve (effect of training set size)

```bash
python adaptive_v2/src/run_learning_curve_experiment.py \
    --responses        data/saar_item_ready_for_experiments.csv \
    --party-scores     data/saar_party_ready_for_experiments.csv \
    --party-positions  data/Saarbru_prepared_Party.csv \
    --out-dir          adaptive_v2/results_learning_curve/
```

Evaluates Ridge + entropy across training sizes from 50 to the full training set,
in steps of 50 (up to 299), 100 (300–999), and 250 (1 000+).

### Predictive stopping (obs-only vs obs+pred stopping criterion)

```bash
python adaptive_v2/src/run_pred_stopping_experiment.py \
    --responses        data/saar_item_ready_for_experiments.csv \
    --party-scores     data/saar_party_ready_for_experiments.csv \
    --party-positions  data/Saarbru_prepared_Party.csv \
    --out-dir          adaptive_v2/results_pred_stopping/
```

Compares stopping when observed-only scores match ground truth vs when
observed+predicted scores match, for Ridge with entropy and random selection.

### Regenerate plots from saved results

```bash
python adaptive_v2/src/make_plots.py \
    --results-dir adaptive_v2/results/ \
    --out-dir     adaptive_v2/results/plots/ \
    --dpi 300
```

Reads the per-person CSVs produced by `run_experiment.py` and regenerates all
plots without re-running the simulation.

---

## Available predictors and strategies

### Predictors (`--predictors`)

| Name | Description |
|---|---|
| `ridge` | Ridge regression (continuous → rounded to 1–5); uncertainty from prediction distance to nearest integer |
| `naive_bayes` | Multinomial Naïve Bayes over 5 Likert classes |
| `random_forest` | Random forest classifier |
| `knn` | k-Nearest Neighbours classifier |

### Item selection strategies (`--strategies`)

| Name | Description |
|---|---|
| `entropy` | Select item with highest predicted Shannon entropy |
| `uncertainty` | Select item with highest predicted uncertainty |
| `random` | Uniformly random item selection (baseline) |
| `variance_reduction` | Select item maximising expected reduction in party-score variance |
| `top1_change` | Select item most likely to change the current top-1 party |
| `top3_change` | Select item most likely to change the current top-3 set |

---

## Design notes

- **Value ranges:** Raw data is 0–4. The pipeline shifts to 1–5 on load; `party_scoring.py` converts back internally.
- **Train/test split:** Stratified by each person's true best-matching party; default 80/20.
- **Model caching:** Within a simulation run all test persons share the same training set, so fitted models are cached by `(observed_columns, target_item)` to avoid redundant fitting.
- **Ground truth:** Party ranking computed from all items before any adaptive selection begins; used only for evaluation, never exposed to the model during selection.
- **Adaptive cap:** If a stopping condition is never satisfied, the person's result is recorded at `n_items_total` items with `top1_reached = False` / `top3_reached = False`.

---

## What is missing to run the experiments

1. **Data files** — the three CSVs listed under *Data files needed* above are not
   included. Without them nothing will run.

2. **Project root on `PYTHONPATH`** — the scripts assume they are run from the
   project root (the directory that contains both `src/` and `adaptive_v2/`).
   Running from a different working directory will cause import errors. Either
   `cd` to the project root before running, or set:

   ```bash
   export PYTHONPATH=/path/to/project/root:$PYTHONPATH
   ```
