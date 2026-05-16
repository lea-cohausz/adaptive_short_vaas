"""
adaptive_v2/src/simulation.py
==============================
Redesigned adaptive questionnaire simulation.

Key differences from v1
-----------------------
- Initial item: ONE random item per person (unique per person, no shared seed)
- Two stopping conditions tracked INDEPENDENTLY per person:
    1. fixed_50pct : record state at exactly ceil(0.5 * n_items) items asked
    2. adaptive    : continue until obs-only top-1 OR top-3 matches ground
                     truth; cap at n_items (all 37); record n_items_asked
- At fixed_50pct two scoring paths are evaluated:
    obs_only  : party scores from observed items only
    obs_pred  : party scores from observed + predicted remaining items
- At adaptive stop: obs_only only (correct answer already reached)
- Ground truth: party ranking from ALL items (pre-computed, passed in)
- Metrics: top-1, top-3 (exact set), party MAE per party

Public API
----------
    PersonResult      — dataclass storing all per-person outputs
    run_simulation()  — main entry point (train/test split → results)
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import shared modules from project/src/
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from party_scoring import compute_match_scores
from predictors import BasePredictor, get_predictor


# ---------------------------------------------------------------------------
# Fold-level model cache (reuse fitted models across persons within a run)
# ---------------------------------------------------------------------------

class _FoldCache:
    """
    Cache fitted predictors keyed by (frozenset(observed_cols), target_item).
    All test persons share the same training set, so models can be reused.
    """
    def __init__(self):
        self._store: dict[tuple, BasePredictor] = {}
        self.hits = 0
        self.misses = 0

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
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PersonResult:
    """All outputs for one test person."""

    person_id: str

    # ── Fixed 50% condition ──────────────────────────────────────────────
    # Items asked at the 50% checkpoint
    items_asked_50: list[str] = field(default_factory=list)
    n_items_asked_50: int = 0

    # Obs-only party scores at 50%
    obs_scores_50: dict[str, float] = field(default_factory=dict)
    # Obs+pred party scores at 50%
    pred_scores_50: dict[str, float] = field(default_factory=dict)

    # Evaluation at 50% — obs only
    obs_top1_correct_50: Optional[bool] = None
    obs_top3_correct_50: Optional[bool] = None
    obs_party_mae_50: Optional[dict[str, float]] = None

    # Evaluation at 50% — obs + pred
    pred_top1_correct_50: Optional[bool] = None
    pred_top3_correct_50: Optional[bool] = None
    pred_party_mae_50: Optional[dict[str, float]] = None

    # ── Adaptive top-1 condition ─────────────────────────────────────────
    n_items_asked_top1: int = 0          # items asked when top-1 first correct
    top1_reached: bool = False            # False if capped at 37
    items_asked_top1: list[str] = field(default_factory=list)
    obs_party_mae_top1: Optional[dict[str, float]] = None

    # ── Adaptive top-3 condition ─────────────────────────────────────────
    n_items_asked_top3: int = 0          # items asked when top-3 first correct
    top3_reached: bool = False            # False if capped at 37
    items_asked_top3: list[str] = field(default_factory=list)
    obs_party_mae_top3: Optional[dict[str, float]] = None

    # ── Ground truth ────────────────────────────────────────────────────
    true_party_scores: dict[str, float] = field(default_factory=dict)
    true_best_party: str = ""
    true_top3_parties: list[str] = field(default_factory=list)
    n_items_total: int = 0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _best(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)

def _top3(scores: dict[str, float]) -> frozenset:
    return frozenset(sorted(scores, key=scores.__getitem__, reverse=True)[:3])

def _party_mae(true: dict[str, float], pred: dict[str, float]) -> dict[str, float]:
    return {p: abs(true[p] - pred.get(p, float("nan"))) for p in true}


# ---------------------------------------------------------------------------
# Core per-person simulation
# ---------------------------------------------------------------------------

def _simulate_person(
    person_id: str,
    person_responses: pd.Series,
    train_responses: pd.DataFrame,
    true_party_scores: pd.Series,
    party_positions: pd.DataFrame,
    predictor_name: str,
    predictor_kwargs: dict,
    strategy: Literal["entropy", "uncertainty", "random",
                      "variance_reduction", "top1_change", "top3_change"],
    rng: np.random.Generator,
    fold_cache: _FoldCache,
) -> PersonResult:

    all_items = list(person_responses.index)
    n_total   = len(all_items)
    n_50      = int(np.ceil(0.5 * n_total))

    true_scores_dict = true_party_scores.to_dict()
    true_best        = _best(true_scores_dict)
    true_top3        = _top3(true_scores_dict)

    result = PersonResult(
        person_id=str(person_id),
        n_items_total=n_total,
        true_party_scores=true_scores_dict,
        true_best_party=true_best,
        true_top3_parties=list(true_top3),
    )

    # ── Start: one random item unique to this person ──────────────────
    first_item = str(rng.choice(all_items))
    observed: dict[str, int] = {first_item: int(person_responses[first_item])}

    # Tracking for adaptive stopping
    top1_recorded = False
    top3_recorded = False

    # ── Main loop ─────────────────────────────────────────────────────
    while len(observed) < n_total:

        # ── Checkpoint: 50% ──────────────────────────────────────────
        if len(observed) == n_50 and result.n_items_asked_50 == 0:
            _record_50pct(
                result, observed, person_responses, train_responses,
                party_positions, predictor_name, predictor_kwargs,
                true_scores_dict, fold_cache,
            )

        # ── Adaptive stopping checks (obs-only) ──────────────────────
        if not top1_recorded or not top3_recorded:
            obs_scores = _score_obs_only(observed, party_positions)
            if obs_scores:
                if not top1_recorded and _best(obs_scores) == true_best:
                    result.n_items_asked_top1 = len(observed)
                    result.top1_reached       = True
                    result.items_asked_top1   = list(observed.keys())
                    result.obs_party_mae_top1 = _party_mae(true_scores_dict, obs_scores)
                    top1_recorded = True
                if not top3_recorded and _top3(obs_scores) == true_top3:
                    result.n_items_asked_top3 = len(observed)
                    result.top3_reached       = True
                    result.items_asked_top3   = list(observed.keys())
                    result.obs_party_mae_top3 = _party_mae(true_scores_dict, obs_scores)
                    top3_recorded = True

        # ── All items asked — record missing adaptive stops at cap ────
        if len(observed) == n_total:
            break

        # ── Select next item ─────────────────────────────────────────
        unasked = [i for i in all_items if i not in observed]
        next_item = _select_next(
            observed, unasked, train_responses,
            predictor_name, predictor_kwargs,
            strategy, rng, fold_cache,
            party_positions=party_positions,
        )
        observed[next_item] = int(person_responses[next_item])

    # ── Final 50% checkpoint if n_total < n_50 (edge case) ───────────
    if result.n_items_asked_50 == 0:
        _record_50pct(
            result, observed, person_responses, train_responses,
            party_positions, predictor_name, predictor_kwargs,
            true_scores_dict, fold_cache,
        )

    # ── Cap adaptive stops at n_total if never reached ────────────────
    final_obs_scores = _score_obs_only(observed, party_positions)
    if not top1_recorded:
        result.n_items_asked_top1 = n_total
        result.top1_reached       = False
        result.items_asked_top1   = list(observed.keys())
        result.obs_party_mae_top1 = _party_mae(true_scores_dict, final_obs_scores) \
                                    if final_obs_scores else None
    if not top3_recorded:
        result.n_items_asked_top3 = n_total
        result.top3_reached       = False
        result.items_asked_top3   = list(observed.keys())
        result.obs_party_mae_top3 = _party_mae(true_scores_dict, final_obs_scores) \
                                    if final_obs_scores else None

    return result


def _record_50pct(
    result: PersonResult,
    observed: dict[str, int],
    person_responses: pd.Series,
    train_responses: pd.DataFrame,
    party_positions: pd.DataFrame,
    predictor_name: str,
    predictor_kwargs: dict,
    true_scores_dict: dict[str, float],
    fold_cache: _FoldCache,
) -> None:
    """Record all 50% checkpoint metrics into result (in-place)."""
    obs_cols = list(observed.keys())
    result.n_items_asked_50 = len(obs_cols)
    result.items_asked_50   = obs_cols[:]

    # Obs-only scores
    obs_scores = _score_obs_only(observed, party_positions)
    result.obs_scores_50 = obs_scores
    if obs_scores:
        result.obs_top1_correct_50 = (_best(obs_scores) == result.true_best_party)
        result.obs_top3_correct_50 = (_top3(obs_scores) == frozenset(result.true_top3_parties))
        result.obs_party_mae_50    = _party_mae(true_scores_dict, obs_scores)

    # Obs+pred scores
    all_items = list(person_responses.index)
    unasked   = [i for i in all_items if i not in observed]
    predicted: dict[str, float] = {}

    if unasked:
        X_train = train_responses[obs_cols].values.astype(float)
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
    pred_scores = compute_match_scores(
        full_responses, party_positions, input_range="1-5"
    ).to_dict()

    result.pred_scores_50      = pred_scores
    result.pred_top1_correct_50 = (_best(pred_scores) == result.true_best_party)
    result.pred_top3_correct_50 = (_top3(pred_scores) == frozenset(result.true_top3_parties))
    result.pred_party_mae_50    = _party_mae(true_scores_dict, pred_scores)


def _score_obs_only(
    observed: dict[str, int],
    party_positions: pd.DataFrame,
) -> dict[str, float]:
    """Compute party scores from observed items only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_match_scores(
            observed, party_positions, input_range="1-5"
        ).to_dict()


# ---------------------------------------------------------------------------
# Next-item selection
# ---------------------------------------------------------------------------

def _select_next(
    observed: dict[str, int],
    unasked: list[str],
    train_responses: pd.DataFrame,
    predictor_name: str,
    predictor_kwargs: dict,
    strategy: str,
    rng: np.random.Generator,
    fold_cache: _FoldCache,
    party_positions: pd.DataFrame | None = None,
) -> str:

    if strategy == "random" or not unasked:
        return str(rng.choice(unasked))

    obs_cols = list(observed.keys())
    X_train  = train_responses[obs_cols].values.astype(float)
    X_person = np.array([[observed[c] for c in obs_cols]])

    # ── Fit / retrieve models for all unasked items ───────────────────────
    models: dict[str, BasePredictor] = {}
    for item in unasked:
        model = fold_cache.get(obs_cols, item)
        if model is None:
            y = train_responses[item].values.astype(int)
            model = get_predictor(predictor_name, **predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y)
            fold_cache.set(obs_cols, item, model)
        models[item] = model

    # ── Model-uncertainty strategies (entropy / uncertainty) ─────────────
    if strategy in ("entropy", "uncertainty"):
        scores = {}
        for item, model in models.items():
            if strategy == "uncertainty":
                scores[item] = float(model.predict_uncertainty(X_person)[0])
            else:
                scores[item] = float(model.predict_entropy(X_person)[0])
        max_score = max(scores.values())
        top_items = [k for k, v in scores.items() if v == max_score]
        return str(rng.choice(top_items))

    # ── Party-score–based strategies ─────────────────────────────────────
    # variance_reduction, top1_change, top3_change all require party_positions.
    # Fall back to random if it is missing.
    if party_positions is None:
        return str(rng.choice(unasked))

    RESPONSE_VALUES = [1, 2, 3, 4, 5]   # pipeline 1–5 range

    # Current party scores (before asking anything new)
    current_scores = _score_obs_only(observed, party_positions)
    current_top1   = _best(current_scores) if current_scores else None
    current_top3   = _top3(current_scores) if current_scores else frozenset()

    scores = {}
    for item, model in models.items():
        # predict_proba always returns (n_samples, 5) with columns = classes 1–5
        # in order (guaranteed by BasePredictor and all subclass implementations).
        # Direct positional indexing avoids any dependency on a classes_ attribute,
        # which is absent on RidgePredictor (regression wrapper, no classes_ set).
        proba = model.predict_proba(X_person)[0]  # shape (5,)

        # Normalise to guard against numerical drift
        total = proba.sum()
        prob_vec: dict[int, float] = {
            v: float(proba[v - 1] / total) if total > 0 else 0.2
            for v in RESPONSE_VALUES
        }

        if strategy == "variance_reduction":
            # Expected reduction in variance of party scores
            # = current_variance − E[variance after observing item]
            cur_var = float(np.var(list(current_scores.values()))) \
                      if current_scores else 0.0
            expected_post_var = 0.0
            for v, p in prob_vec.items():
                if p == 0.0:
                    continue
                hyp_observed = {**observed, item: v}
                hyp_scores   = _score_obs_only(hyp_observed, party_positions)
                if hyp_scores:
                    expected_post_var += p * float(
                        np.var(list(hyp_scores.values()))
                    )
            scores[item] = cur_var - expected_post_var   # higher = more reduction

        elif strategy == "top1_change":
            # Probability that observing this item changes the top-1 party
            change_prob = 0.0
            for v, p in prob_vec.items():
                if p == 0.0:
                    continue
                hyp_observed = {**observed, item: v}
                hyp_scores   = _score_obs_only(hyp_observed, party_positions)
                if hyp_scores and _best(hyp_scores) != current_top1:
                    change_prob += p
            scores[item] = change_prob

        elif strategy == "top3_change":
            # Probability that observing this item changes the top-3 set
            change_prob = 0.0
            for v, p in prob_vec.items():
                if p == 0.0:
                    continue
                hyp_observed = {**observed, item: v}
                hyp_scores   = _score_obs_only(hyp_observed, party_positions)
                if hyp_scores and _top3(hyp_scores) != current_top3:
                    change_prob += p
            scores[item] = change_prob

    if not scores:
        return str(rng.choice(unasked))

    max_score = max(scores.values())
    top_items = [k for k, v in scores.items() if v == max_score]
    return str(rng.choice(top_items))


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    responses: pd.DataFrame,
    party_scores_gt: pd.DataFrame,
    party_positions: pd.DataFrame,
    predictor_name: str = "ridge",
    strategy: Literal["entropy", "uncertainty", "random",
                      "variance_reduction", "top1_change", "top3_change"] = "entropy",
    predictor_kwargs: Optional[dict] = None,
    max_train_persons: Optional[int] = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[PersonResult], pd.DataFrame, pd.DataFrame]:
    """
    Run the full adaptive simulation on a stratified train/test split.

    Parameters
    ----------
    responses : pd.DataFrame        (n_persons, n_items) Likert 1–5
    party_scores_gt : pd.DataFrame  (n_persons, n_parties) ground-truth scores
    party_positions : pd.DataFrame  (n_parties, n_items) 0–4
    predictor_name : str
    strategy : str                  'entropy', 'uncertainty', 'random'
    predictor_kwargs : dict
    max_train_persons : int or None Cap training rows (speed optimisation)
    test_fraction : float
    seed : int
    verbose : bool

    Returns
    -------
    results : list[PersonResult]
    train_responses : pd.DataFrame
    test_responses : pd.DataFrame
    """
    predictor_kwargs = predictor_kwargs or {}

    # ── Stratified split ──────────────────────────────────────────────
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

    # ── Subsample training rows ────────────────────────────────────────
    if max_train_persons and len(train_responses) > max_train_persons:
        rng_sub = np.random.default_rng(seed)
        sub_idx = rng_sub.choice(
            len(train_responses), size=max_train_persons, replace=False
        )
        train_responses = train_responses.iloc[sub_idx]

    if verbose:
        print(f"Train: {len(train_responses)}, Test: {len(test_responses)}")
        print(f"Predictor: {predictor_name}, Strategy: {strategy}")

    rng        = np.random.default_rng(seed)
    fold_cache = _FoldCache()
    results    = []

    for i, person_id in enumerate(test_responses.index, 1):
        result = _simulate_person(
            person_id=str(person_id),
            person_responses=test_responses.loc[person_id],
            train_responses=train_responses,
            true_party_scores=test_gt.loc[person_id],
            party_positions=party_positions,
            predictor_name=predictor_name,
            predictor_kwargs=predictor_kwargs,
            strategy=strategy,
            rng=rng,
            fold_cache=fold_cache,
        )
        results.append(result)

        if verbose and i % 100 == 0:
            print(f"  [{i:>5}/{len(test_responses)}] "
                  f"cache hit rate: {fold_cache.hit_rate:.1%}")

    if verbose:
        print(f"  Done. Cache: {len(fold_cache._store)} models, "
              f"hit rate: {fold_cache.hit_rate:.1%}")

    return results, train_responses, test_responses


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SRC = Path(__file__).resolve().parent.parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from party_scoring import compute_match_scores_matrix
    from load_data import load_all

    rng = np.random.default_rng(0)
    n_persons, n_items, n_parties = 100, 20, 4
    item_cols   = [f"item_{i:02d}" for i in range(n_items)]
    party_names = ["A", "B", "C", "D"]
    person_ids  = list(range(1, n_persons + 1))

    latent = rng.standard_normal((n_persons, 3))
    raw    = latent @ rng.standard_normal((n_items, 3)).T \
             + 0.5 * rng.standard_normal((n_persons, n_items))
    responses = pd.DataFrame(
        np.clip(np.round(3 + raw), 1, 5).astype(int),
        columns=item_cols, index=person_ids,
    )
    party_pos = pd.DataFrame(
        rng.integers(0, 5, size=(n_parties, n_items)),
        index=party_names, columns=item_cols,
    )
    party_scores = compute_match_scores_matrix(
        responses, party_pos, input_range="1-5"
    )

    for strategy in ["entropy", "uncertainty", "random",
                      "variance_reduction", "top1_change", "top3_change"]:
        results, _, _ = run_simulation(
            responses, party_scores, party_pos,
            predictor_name="ridge",
            strategy=strategy,
            seed=42, verbose=False,
        )
        top1 = np.mean([r.obs_top1_correct_50 for r in results
                        if r.obs_top1_correct_50 is not None])
        n_top1 = np.mean([r.n_items_asked_top1 for r in results])
        n_top3 = np.mean([r.n_items_asked_top3 for r in results])
        print(f"  {strategy:<12s}  top1@50%={top1:.1%}  "
              f"avg_items_top1={n_top1:.1f}  avg_items_top3={n_top3:.1f}")

    print("simulation.py OK.")
