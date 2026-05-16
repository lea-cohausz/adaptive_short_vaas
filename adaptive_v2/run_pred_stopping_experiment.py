"""
run_pred_stopping_experiment.py
================================
Compares adaptive stopping under two criteria — obs-only and obs+predicted —
for two selection strategies (entropy and random), using ridge regression.

The four runs are:
  1. ridge / entropy  — stops when obs-only    top-1/top-3 matches ground truth
  2. ridge / entropy  — stops when obs+pred     top-1/top-3 matches ground truth
  3. ridge / random   — stops when obs-only     top-1/top-3 matches ground truth
  4. ridge / random   — stops when obs+pred     top-1/top-3 matches ground truth

Uses the same train/test split (seed + test_fraction) as the original experiment
so test persons are identical to existing results.

Outputs (saved to --out-dir):
  per_person/{label}.csv   — one row per test person
  summary.csv              — bootstrap CI summary across all four runs

Usage
-----
    python run_pred_stopping_experiment.py \
        --responses   data/saar_item_ready_for_experiments.csv \
        --party-scores data/saar_party_ready_for_experiments.csv \
        --party-positions data/Saarbru_prepared_Party.csv \
        --out-dir adaptive_v2/results_pred_stopping/ \
        [--test-fraction 0.2] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
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
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PredStoppingResult:
    """Per-person results for the pred-stopping experiment."""

    person_id:       str
    true_best_party: str
    true_top3_parties: list[str]
    n_items_total:   int

    # Adaptive top-1 — obs only
    n_items_top1_obs:   int  = 0
    top1_reached_obs:   bool = False

    # Adaptive top-1 — obs + pred
    n_items_top1_pred:  int  = 0
    top1_reached_pred:  bool = False

    # Adaptive top-3 — obs only
    n_items_top3_obs:   int  = 0
    top3_reached_obs:   bool = False

    # Adaptive top-3 — obs + pred
    n_items_top3_pred:  int  = 0
    top3_reached_pred:  bool = False


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

class _FoldCache:
    def __init__(self):
        self._store: dict[tuple, BasePredictor] = {}
        self.hits = self.misses = 0

    def get(self, obs_cols: list[str], target: str) -> Optional[BasePredictor]:
        k = (frozenset(obs_cols), target)
        m = self._store.get(k)
        if m is not None:
            self.hits += 1
        else:
            self.misses += 1
        return m

    def set(self, obs_cols: list[str], target: str, model: BasePredictor):
        self._store[(frozenset(obs_cols), target)] = model

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

def _score_obs_only(
    observed: dict[str, int | float],
    party_positions: pd.DataFrame,
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_match_scores(
            observed, party_positions, input_range="1-5"
        ).to_dict()

def _score_obs_pred(
    observed: dict[str, int | float],
    all_items: list[str],
    train_responses: pd.DataFrame,
    party_positions: pd.DataFrame,
    fold_cache: _FoldCache,
    predictor_name: str,
    predictor_kwargs: dict,
) -> dict[str, float]:
    """Impute unasked items and return full obs+pred party scores."""
    obs_cols = list(observed.keys())
    unasked  = [i for i in all_items if i not in observed]

    predicted: dict[str, float] = {}
    if unasked:
        X_train  = train_responses[obs_cols].values.astype(float)
        X_person = np.array([[observed[c] for c in obs_cols]])
        for item in unasked:
            model = fold_cache.get(obs_cols, item)
            if model is None:
                y = train_responses[item].values.astype(int)
                model = get_predictor(predictor_name, **predictor_kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y)
                fold_cache.set(obs_cols, item, model)
            predicted[item] = float(model.predict(X_person)[0])

    full_responses = {**{k: float(v) for k, v in observed.items()}, **predicted}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_match_scores(
            full_responses, party_positions, input_range="1-5"
        ).to_dict()


# ---------------------------------------------------------------------------
# Next-item selection (entropy and random only)
# ---------------------------------------------------------------------------

def _select_next(
    observed: dict[str, int | float],
    unasked: list[str],
    train_responses: pd.DataFrame,
    party_positions: pd.DataFrame,
    strategy: str,
    rng: np.random.Generator,
    fold_cache: _FoldCache,
    predictor_name: str,
    predictor_kwargs: dict,
) -> str:
    if strategy == "random" or not unasked:
        return str(rng.choice(unasked))

    # entropy: ask the item whose predicted response distribution has the
    # highest Shannon entropy (most uncertain → most informative).
    obs_cols = list(observed.keys())
    X_train  = train_responses[obs_cols].values.astype(float)
    X_person = np.array([[observed[c] for c in obs_cols]])

    item_scores: dict[str, float] = {}

    for item in unasked:
        model = fold_cache.get(obs_cols, item)
        if model is None:
            y = train_responses[item].values.astype(int)
            model = get_predictor(predictor_name, **predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y)
            fold_cache.set(obs_cols, item, model)

        proba = model.predict_proba(X_person)[0]   # shape (n_classes,)
        total = proba.sum()
        if total > 0:
            proba = proba / total
        else:
            proba = np.ones_like(proba) / len(proba)

        # Shannon entropy: H = -sum(p * log(p)), ignoring zero-prob classes
        entropy = -float(np.sum(proba * np.log(proba + 1e-12)))
        item_scores[item] = entropy

    max_score = max(item_scores.values())
    top_items = [k for k, v in item_scores.items() if v == max_score]
    return str(rng.choice(top_items))


# ---------------------------------------------------------------------------
# Per-person simulation
# ---------------------------------------------------------------------------

def _simulate_person(
    person_id: str,
    person_responses: pd.Series,
    train_responses: pd.DataFrame,
    true_party_scores: pd.Series,
    party_positions: pd.DataFrame,
    predictor_name: str,
    predictor_kwargs: dict,
    strategy: str,
    rng: np.random.Generator,
    fold_cache: _FoldCache,
) -> PredStoppingResult:

    all_items = list(person_responses.index)
    n_total   = len(all_items)

    true_scores_dict = true_party_scores.to_dict()
    true_best        = _best(true_scores_dict)
    true_top3        = _top3(true_scores_dict)

    result = PredStoppingResult(
        person_id=str(person_id),
        true_best_party=true_best,
        true_top3_parties=list(true_top3),
        n_items_total=n_total,
    )

    # Start with one random item
    first_item = str(rng.choice(all_items))
    observed: dict[str, int] = {first_item: int(person_responses[first_item])}

    top1_obs_recorded  = False
    top3_obs_recorded  = False
    top1_pred_recorded = False
    top3_pred_recorded = False

    while True:
        n_obs = len(observed)

        # ── Obs-only stopping checks ──────────────────────────────────
        if not top1_obs_recorded or not top3_obs_recorded:
            obs_scores = _score_obs_only(observed, party_positions)
            if obs_scores:
                if not top1_obs_recorded and _best(obs_scores) == true_best:
                    result.n_items_top1_obs  = n_obs
                    result.top1_reached_obs  = True
                    top1_obs_recorded        = True
                if not top3_obs_recorded and _top3(obs_scores) == true_top3:
                    result.n_items_top3_obs  = n_obs
                    result.top3_reached_obs  = True
                    top3_obs_recorded        = True

        # ── Obs+pred stopping checks ──────────────────────────────────
        if not top1_pred_recorded or not top3_pred_recorded:
            pred_scores = _score_obs_pred(
                observed, all_items, train_responses, party_positions,
                fold_cache, predictor_name, predictor_kwargs,
            )
            if not top1_pred_recorded and _best(pred_scores) == true_best:
                result.n_items_top1_pred = n_obs
                result.top1_reached_pred = True
                top1_pred_recorded       = True
            if not top3_pred_recorded and _top3(pred_scores) == true_top3:
                result.n_items_top3_pred = n_obs
                result.top3_reached_pred = True
                top3_pred_recorded       = True

        # ── Break if all items asked or all criteria recorded ─────────
        if n_obs == n_total:
            break

        # ── Select next item ──────────────────────────────────────────
        unasked   = [i for i in all_items if i not in observed]
        next_item = _select_next(
            observed, unasked, train_responses, party_positions,
            strategy, rng, fold_cache, predictor_name, predictor_kwargs,
        )
        observed[next_item] = int(person_responses[next_item])

    # Cap any never-reached criteria at n_total
    if not top1_obs_recorded:
        result.n_items_top1_obs = n_total
    if not top3_obs_recorded:
        result.n_items_top3_obs = n_total
    if not top1_pred_recorded:
        result.n_items_top1_pred = n_total
    if not top3_pred_recorded:
        result.n_items_top3_pred = n_total

    return result


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


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

RUNS = [
    ("ridge", "entropy", "ridge__entropy__obs"),
    ("ridge", "entropy", "ridge__entropy__pred"),
    ("ridge", "random",  "ridge__random__obs"),
    ("ridge", "random",  "ridge__random__pred"),
]
# The obs/pred distinction is tracked within a single simulation per
# (predictor, strategy) pair — both criteria are recorded simultaneously.
# We therefore only need TWO simulation runs, not four.
UNIQUE_RUNS = [
    ("ridge", "entropy"),
    ("ridge", "random"),
]


def run_all(
    responses: pd.DataFrame,
    party_scores_gt: pd.DataFrame,
    party_positions: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
    out_dir: Path = Path("adaptive_v2/results_pred_stopping/"),
    verbose: bool = True,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_person").mkdir(exist_ok=True)

    # ── Same train/test split as original experiment ───────────────────
    best_party = party_scores_gt.idxmax(axis=1)
    train_idx, test_idx = train_test_split(
        np.arange(len(responses)),
        test_size=test_fraction,
        stratify=best_party.values,
        random_state=seed,
    )
    train_responses = responses.iloc[train_idx].copy()
    test_responses  = responses.iloc[test_idx].copy()
    test_gt         = party_scores_gt.iloc[test_idx].copy()

    if verbose:
        print(f"Train: {len(train_responses)}, Test: {len(test_responses)}")

    summary_rows = []

    for predictor_name, strategy in UNIQUE_RUNS:
        label = f"{predictor_name}__{strategy}"
        if verbose:
            print(f"\n── {label} ──────────────────────────────────────")

        t0         = time.time()
        rng        = np.random.default_rng(seed)
        fold_cache = _FoldCache()
        results: list[PredStoppingResult] = []

        for i, person_id in enumerate(test_responses.index, 1):
            r = _simulate_person(
                person_id=str(person_id),
                person_responses=test_responses.loc[person_id],
                train_responses=train_responses,
                true_party_scores=test_gt.loc[person_id],
                party_positions=party_positions,
                predictor_name=predictor_name,
                predictor_kwargs={},
                strategy=strategy,
                rng=rng,
                fold_cache=fold_cache,
            )
            results.append(r)
            if verbose and i % 50 == 0:
                print(f"  [{i:>5}/{len(test_responses)}]  "
                      f"cache hit rate: {fold_cache.hit_rate:.1%}")

        elapsed = time.time() - t0
        if verbose:
            print(f"  Done in {elapsed:.0f}s. "
                  f"Cache hit rate: {fold_cache.hit_rate:.1%}")

        # ── Save per-person CSV ────────────────────────────────────────
        rows = []
        for r in results:
            rows.append({
                "person_id":          r.person_id,
                "true_best_party":    r.true_best_party,
                "n_items_total":      r.n_items_total,
                "n_items_top1_obs":   r.n_items_top1_obs,
                "top1_reached_obs":   r.top1_reached_obs,
                "n_items_top1_pred":  r.n_items_top1_pred,
                "top1_reached_pred":  r.top1_reached_pred,
                "n_items_top3_obs":   r.n_items_top3_obs,
                "top3_reached_obs":   r.top3_reached_obs,
                "n_items_top3_pred":  r.n_items_top3_pred,
                "top3_reached_pred":  r.top3_reached_pred,
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "per_person" / f"{label}.csv", index=False)

        # ── Compute summary metrics ────────────────────────────────────
        for criterion, n_col, reached_col in [
            ("top1_obs",  "n_items_top1_obs",  "top1_reached_obs"),
            ("top1_pred", "n_items_top1_pred", "top1_reached_pred"),
            ("top3_obs",  "n_items_top3_obs",  "top3_reached_obs"),
            ("top3_pred", "n_items_top3_pred", "top3_reached_pred"),
        ]:
            n_vals       = df[n_col].values.astype(float)
            reached_vals = df[reached_col].values.astype(float)
            n_m, n_lo, n_hi         = _bootstrap_ci(n_vals)
            r_m, r_lo, r_hi         = _bootstrap_ci(reached_vals)
            # Items asked only among those who reached the criterion
            reached_mask = df[reached_col].values
            n_reached_vals = df.loc[reached_mask, n_col].values.astype(float)
            nr_m, nr_lo, nr_hi = _bootstrap_ci(n_reached_vals) \
                if len(n_reached_vals) > 0 else (np.nan, np.nan, np.nan)

            summary_rows.append({
                "run":                label,
                "predictor":          predictor_name,
                "strategy":           strategy,
                "criterion":          criterion,
                "n_persons":          len(results),
                "mean_items_asked":   n_m,
                "items_ci_lo":        n_lo,
                "items_ci_hi":        n_hi,
                "mean_items_if_reached": nr_m,
                "items_reached_ci_lo":   nr_lo,
                "items_reached_ci_hi":   nr_hi,
                "reached_fraction":   r_m,
                "reached_ci_lo":      r_lo,
                "reached_ci_hi":      r_hi,
                "elapsed_s":          elapsed,
            })

        if verbose:
            for row in summary_rows[-4:]:
                print(f"  {row['criterion']:<12}  "
                      f"items={row['mean_items_asked']:.1f} "
                      f"[{row['items_ci_lo']:.1f}, {row['items_ci_hi']:.1f}]  "
                      f"reached={row['reached_fraction']:.1%}")

    # ── Save summary CSV ───────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    if verbose:
        print(f"\n✓ Summary saved → {out_dir / 'summary.csv'}")
        print(summary_df[["run", "criterion", "mean_items_asked",
                           "reached_fraction"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run pred vs obs-only adaptive stopping comparison.")
    p.add_argument("--responses",       required=True)
    p.add_argument("--party-scores",    required=True)
    p.add_argument("--party-positions", required=True)
    p.add_argument("--out-dir",  default="adaptive_v2/results_pred_stopping/")
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