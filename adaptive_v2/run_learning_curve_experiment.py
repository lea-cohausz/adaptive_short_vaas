"""
run_learning_curve_experiment.py
=================================
Evaluates ridge regression + entropy selection at the fixed 50% stopping
criterion across increasing training set sizes.

For each training size n, the model is trained on a random subsample of n
persons from the full training set (same train/test split as all previous
experiments), then evaluated on the fixed test set.

Metrics recorded per training size:
  - obs-only  : top-1 accuracy, top-3 accuracy, mean MAE
  - obs+pred  : top-1 accuracy, top-3 accuracy, mean MAE

Training sizes
--------------
  Steps of  50 from    50 to  299  (inclusive)
  Steps of 100 from   300 to  999
  Steps of 250 from 1,000 to full training set size

Outputs (saved to --out-dir):
  per_person/n{n_train}.csv   — one row per test person for each training size
  summary.csv                 — one row per training size with bootstrap CIs

Usage
-----
    python run_learning_curve_experiment.py \
        --responses   data/saar_item_ready_for_experiments.csv \
        --party-scores data/saar_party_ready_for_experiments.csv \
        --party-positions data/Saarbru_prepared_Party.csv \
        --out-dir adaptive_v2/results_learning_curve/ \
        [--test-fraction 0.2] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent.parent / "src"
for _p in [str(_SRC), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import load_all
from party_scoring import compute_match_scores
from predictors import BasePredictor, get_predictor


# ---------------------------------------------------------------------------
# Training size schedule
# ---------------------------------------------------------------------------

def _training_sizes(n_train_total: int) -> list[int]:
    """
    Build the list of training sizes to evaluate:
      steps of  50 from   50 to  299
      steps of 100 from  300 to  999
      steps of 250 from 1000 to n_train_total
    Always includes n_train_total as the final entry.
    """
    sizes = []
    n = 50
    while n < n_train_total:
        sizes.append(n)
        if n < 300:
            n += 50
        elif n < 1000:
            n += 100
        else:
            n += 250
    sizes.append(n_train_total)
    return sizes


# ---------------------------------------------------------------------------
# Model cache (keyed on training size too, since models change with n_train)
# ---------------------------------------------------------------------------

class _FoldCache:
    def __init__(self):
        self._store: dict[tuple, BasePredictor] = {}
        self.hits = self.misses = 0

    def get(self, n_train: int, obs_cols: list[str],
            target: str) -> Optional[BasePredictor]:
        k = (n_train, frozenset(obs_cols), target)
        m = self._store.get(k)
        if m is not None:
            self.hits += 1
        else:
            self.misses += 1
        return m

    def set(self, n_train: int, obs_cols: list[str],
            target: str, model: BasePredictor):
        self._store[(n_train, frozenset(obs_cols), target)] = model

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t > 0 else 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _best(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)

def _top3(scores: dict[str, float]) -> frozenset:
    return frozenset(sorted(scores, key=scores.__getitem__, reverse=True)[:3])

def _party_mae(
    true_scores: dict[str, float],
    pred_scores: dict[str, float],
) -> dict[str, float]:
    return {
        p: abs(true_scores[p] - pred_scores[p])
        for p in true_scores
        if p in pred_scores
    }

def _score_obs_only(
    observed: dict[str, int | float],
    party_positions: pd.DataFrame,
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_match_scores(
            observed, party_positions, input_range="1-5"
        ).to_dict()


# ---------------------------------------------------------------------------
# Entropy-based next-item selection
# ---------------------------------------------------------------------------

def _select_next_entropy(
    observed: dict[str, int | float],
    unasked: list[str],
    train_responses: pd.DataFrame,
    n_train: int,
    fold_cache: _FoldCache,
) -> str:
    obs_cols = list(observed.keys())
    X_train  = train_responses[obs_cols].values.astype(float)
    X_person = np.array([[observed[c] for c in obs_cols]])

    entropies: dict[str, float] = {}
    for item in unasked:
        model = fold_cache.get(n_train, obs_cols, item)
        if model is None:
            y = train_responses[item].values.astype(int)
            model = get_predictor("ridge")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y)
            fold_cache.set(n_train, obs_cols, item, model)
        entropies[item] = float(model.predict_entropy(X_person)[0])

    max_e     = max(entropies.values())
    top_items = [k for k, v in entropies.items() if v == max_e]
    return top_items[0]   # deterministic tie-break


# ---------------------------------------------------------------------------
# 50% checkpoint evaluation for one person x one training size
# ---------------------------------------------------------------------------

def _evaluate_50pct(
    person_responses: pd.Series,
    train_responses: pd.DataFrame,
    true_scores_dict: dict[str, float],
    true_best: str,
    true_top3: frozenset,
    party_positions: pd.DataFrame,
    n_train: int,
    fold_cache: _FoldCache,
    rng: np.random.Generator,
) -> dict:
    """
    Run the entropy adaptive loop to the 50% checkpoint for one person.
    Returns a dict of per-person metric values.
    """
    all_items = list(person_responses.index)
    n_total   = len(all_items)
    n_50      = int(np.ceil(0.5 * n_total))

    # Start with one random item
    first_item = str(rng.choice(all_items))
    observed: dict[str, int] = {first_item: int(person_responses[first_item])}

    # Ask items via entropy until we reach the 50% checkpoint
    while len(observed) < n_50:
        unasked   = [i for i in all_items if i not in observed]
        next_item = _select_next_entropy(
            observed, unasked, train_responses, n_train, fold_cache
        )
        observed[next_item] = int(person_responses[next_item])

    # ── Obs-only scores at 50% ────────────────────────────────────────
    obs_scores   = _score_obs_only(observed, party_positions)
    obs_top1     = float(_best(obs_scores) == true_best)   if obs_scores else np.nan
    obs_top3     = float(_top3(obs_scores) == true_top3)   if obs_scores else np.nan
    obs_mae      = _party_mae(true_scores_dict, obs_scores) if obs_scores else {}
    obs_mae_mean = float(np.mean(list(obs_mae.values())))  if obs_mae    else np.nan

    # ── Obs+pred scores at 50% ────────────────────────────────────────
    obs_cols  = list(observed.keys())
    unasked   = [i for i in all_items if i not in observed]
    predicted: dict[str, float] = {}

    if unasked:
        X_train  = train_responses[obs_cols].values.astype(float)
        X_person = np.array([[observed[c] for c in obs_cols]])
        for item in unasked:
            model = fold_cache.get(n_train, obs_cols, item)
            if model is None:
                y = train_responses[item].values.astype(int)
                model = get_predictor("ridge")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y)
                fold_cache.set(n_train, obs_cols, item, model)
            predicted[item] = float(model.predict(X_person)[0])

    full_responses = {**{k: float(v) for k, v in observed.items()}, **predicted}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_scores = compute_match_scores(
            full_responses, party_positions, input_range="1-5"
        ).to_dict()

    pred_top1     = float(_best(pred_scores) == true_best)
    pred_top3     = float(_top3(pred_scores) == true_top3)
    pred_mae      = _party_mae(true_scores_dict, pred_scores)
    pred_mae_mean = float(np.mean(list(pred_mae.values()))) if pred_mae else np.nan

    return {
        "n_items_asked_50": len(observed),
        "obs_top1":         obs_top1,
        "obs_top3":         obs_top3,
        "obs_mae_mean":     obs_mae_mean,
        "pred_top1":        pred_top1,
        "pred_top3":        pred_top3,
        "pred_mae_mean":    pred_mae_mean,
        **{f"obs_mae_{p}":  v for p, v in obs_mae.items()},
        **{f"pred_mae_{p}": v for p, v in pred_mae.items()},
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v   = np.asarray(values, dtype=float)
    v   = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    alpha = (1 - ci) / 2
    return (
        float(np.mean(v)),
        float(np.quantile(boots, alpha)),
        float(np.quantile(boots, 1 - alpha)),
    )


def _fmt_ci(m, lo, hi, pct=True) -> str:
    if np.isnan(m):
        return "—"
    if pct:
        return f"{m*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_all(
    responses: pd.DataFrame,
    party_scores_gt: pd.DataFrame,
    party_positions: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
    out_dir: Path = Path("adaptive_v2/results_learning_curve/"),
    verbose: bool = True,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_person").mkdir(exist_ok=True)

    # ── Same train/test split as all previous experiments ─────────────
    best_party = party_scores_gt.idxmax(axis=1)
    train_idx, test_idx = train_test_split(
        np.arange(len(responses)),
        test_size=test_fraction,
        stratify=best_party.values,
        random_state=seed,
    )
    full_train_responses = responses.iloc[train_idx].copy()
    test_responses       = responses.iloc[test_idx].copy()
    test_gt              = party_scores_gt.iloc[test_idx].copy()
    n_train_total        = len(full_train_responses)

    sizes = _training_sizes(n_train_total)

    if verbose:
        print(f"Train (full): {n_train_total}  |  Test: {len(test_responses)}")
        print(f"Training sizes: {sizes}")
        print(f"Total checkpoints: {len(sizes)}\n")

    # Shared cache — models are keyed on (n_train, obs_cols, target)
    fold_cache = _FoldCache()
    summary_rows = []
    t_total = time.time()

    for n_train in sizes:
        t0 = time.time()

        # Deterministic subsample — nested: smaller sets are subsets of larger.
        # seed + n_train gives a distinct but reproducible subsample per size.
        rng_sub = np.random.default_rng(seed + n_train)
        sub_idx = rng_sub.choice(n_train_total, size=n_train, replace=False)
        train_responses = full_train_responses.iloc[sub_idx].copy()

        # Reset person-level rng per training size so item sequences are
        # identical across sizes (only model quality varies).
        rng_persons = np.random.default_rng(seed)

        person_rows = []
        for person_id in test_responses.index:
            true_scores_dict = test_gt.loc[person_id].to_dict()
            true_best        = _best(true_scores_dict)
            true_top3        = _top3(true_scores_dict)

            metrics = _evaluate_50pct(
                person_responses=test_responses.loc[person_id],
                train_responses=train_responses,
                true_scores_dict=true_scores_dict,
                true_best=true_best,
                true_top3=true_top3,
                party_positions=party_positions,
                n_train=n_train,
                fold_cache=fold_cache,
                rng=rng_persons,
            )
            person_rows.append({
                "person_id":       str(person_id),
                "true_best_party": true_best,
                "n_train":         n_train,
                **metrics,
            })

        elapsed    = time.time() - t0
        df_persons = pd.DataFrame(person_rows)
        df_persons.to_csv(
            out_dir / "per_person" / f"n{n_train:05d}.csv", index=False
        )

        # ── Summary row ───────────────────────────────────────────────
        obs_top1_ci  = _bootstrap_ci(df_persons["obs_top1"].values)
        obs_top3_ci  = _bootstrap_ci(df_persons["obs_top3"].values)
        obs_mae_ci   = _bootstrap_ci(df_persons["obs_mae_mean"].values)
        pred_top1_ci = _bootstrap_ci(df_persons["pred_top1"].values)
        pred_top3_ci = _bootstrap_ci(df_persons["pred_top3"].values)
        pred_mae_ci  = _bootstrap_ci(df_persons["pred_mae_mean"].values)

        summary_rows.append({
            "n_train":         n_train,
            "n_test":          len(df_persons),
            "obs_top1":        obs_top1_ci[0],
            "obs_top1_ci_lo":  obs_top1_ci[1],
            "obs_top1_ci_hi":  obs_top1_ci[2],
            "obs_top3":        obs_top3_ci[0],
            "obs_top3_ci_lo":  obs_top3_ci[1],
            "obs_top3_ci_hi":  obs_top3_ci[2],
            "obs_mae":         obs_mae_ci[0],
            "obs_mae_ci_lo":   obs_mae_ci[1],
            "obs_mae_ci_hi":   obs_mae_ci[2],
            "pred_top1":       pred_top1_ci[0],
            "pred_top1_ci_lo": pred_top1_ci[1],
            "pred_top1_ci_hi": pred_top1_ci[2],
            "pred_top3":       pred_top3_ci[0],
            "pred_top3_ci_lo": pred_top3_ci[1],
            "pred_top3_ci_hi": pred_top3_ci[2],
            "pred_mae":        pred_mae_ci[0],
            "pred_mae_ci_lo":  pred_mae_ci[1],
            "pred_mae_ci_hi":  pred_mae_ci[2],
            "elapsed_s":       elapsed,
        })

        if verbose:
            print(
                f"  n_train={n_train:>5}  "
                f"obs_top1={_fmt_ci(*obs_top1_ci)}  "
                f"pred_top1={_fmt_ci(*pred_top1_ci)}  "
                f"({elapsed:.0f}s)"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    if verbose:
        total_elapsed = time.time() - t_total
        print(f"\n✓ Done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
        print(f"✓ Summary → {out_dir / 'summary.csv'}")
        print(f"  Cache: {len(fold_cache._store):,} models stored  |  "
              f"hit rate: {fold_cache.hit_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Learning curve: ridge + entropy at fixed 50% stopping.")
    p.add_argument("--responses",       required=True)
    p.add_argument("--party-scores",    required=True)
    p.add_argument("--party-positions", required=True)
    p.add_argument("--out-dir",
                   default="adaptive_v2/results_learning_curve/")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    responses, party_scores_gt, party_positions = load_all(
        item_path=args.responses,
        party_score_path=args.party_scores,
        party_pos_path=args.party_positions,
        validate=True,
        verbose=True,
    )
    run_all(
        responses=responses,
        party_scores_gt=party_scores_gt,
        party_positions=party_positions,
        test_fraction=args.test_fraction,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        verbose=True,
    )


if __name__ == "__main__":
    main()