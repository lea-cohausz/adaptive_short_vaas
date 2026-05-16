"""
stopping_rules.py
=================
Stopping criteria for the adaptive questionnaire loop.

Three rules are implemented:

1. **fixed_50pct**
   Stop when the number of asked items reaches 50% of the total item count.
   This is the practically-used threshold in your domain and serves as the
   primary benchmark baseline. Simple, interpretable, domain-grounded.

2. **entropy**
   Stop when the mean predicted Shannon entropy across ALL remaining unasked
   items drops below a threshold τ, OR when the marginal entropy reduction
   from the last item asked falls below a minimum improvement ε.

   Operationally:
   - After each new item is revealed, recompute the mean entropy over all
     unasked items (using the selector's cached models).
   - Stop if:
       mean_entropy < tau                  (absolute threshold), OR
       delta_entropy < epsilon             (marginal gain too small)
   - τ and ε can be set explicitly or estimated from the training data
     (see `estimate_entropy_thresholds()`).

3. **error**
   Stop when the estimated prediction error (MAE) on unasked items is
   sufficiently low, based on an internal cross-validation estimate on the
   training set. Specifically:
   - After each step, estimate the current MAE via leave-one-out or
     k-fold CV on the training set using the current observed feature set.
   - Stop if:
       estimated_mae < tau                 (absolute threshold), OR
       delta_mae < epsilon                 (marginal improvement too small)
   - τ and ε can be set explicitly or estimated from training data
     (see `estimate_error_thresholds()`).

Design notes
------------
- All rules share a common `StoppingRule` base class with a single method:
    should_stop(state) -> bool
  where `state` is a `StoppingState` dataclass capturing everything needed.

- Threshold estimation helpers are provided so thresholds can be derived
  from training data rather than set by hand. This is important for
  comparability across datasets and experiments.

- All rules track their internal history (entropy / error curves) so that
  post-hoc analysis of "when did each rule fire?" is possible.

Public API
----------
    StoppingState          — dataclass passed to should_stop()
    StoppingRule           — abstract base class
    Fixed50PctRule         — fixed_50pct implementation
    EntropyRule            — entropy-based implementation
    ErrorRule              — error-based implementation
    get_stopping_rule()    — factory function
    estimate_entropy_thresholds()  — data-driven τ and ε for entropy rule
    estimate_error_thresholds()    — data-driven τ and ε for error rule
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from predictors import BasePredictor, get_predictor


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class StoppingState:
    """
    Everything the stopping rule needs to make its decision.

    Passed to `should_stop()` at every step of the adaptive loop.

    Attributes
    ----------
    person_observed : dict[str, int]
        Items already asked and their responses.
    all_items : list[str]
        All item names in the questionnaire.
    train_responses : pd.DataFrame
        Training persons × all items (Likert 1–5).
    predictor_name : str
        Which predictor is being used (needed to refit for error estimation).
    predictor_kwargs : dict
        Kwargs for the predictor.
    item_scores : pd.Series or None
        Pre-computed per-item scores (entropy or uncertainty) from the
        ItemSelector. If provided, the entropy rule uses these directly
        instead of refitting.
    previous_mean_entropy : float or None
        Mean entropy from the previous step (for delta computation).
    previous_mean_error : float or None
        Mean estimated MAE from the previous step (for delta computation).
    n_initial_items : int
        Number of fixed seed items (lower bound on items asked).
    step : int
        Current step index (0-based), i.e. number of items asked beyond
        the initial set.
    """
    person_observed: dict[str, int]
    all_items: list[str]
    train_responses: pd.DataFrame
    predictor_name: str = "lgbm"
    predictor_kwargs: dict = field(default_factory=dict)
    item_scores: Optional[pd.Series] = None   # entropy/uncertainty per item
    previous_mean_entropy: Optional[float] = None
    previous_mean_error: Optional[float] = None
    n_initial_items: int = 5
    step: int = 0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class StoppingRule(ABC):
    """Abstract base for all stopping rules."""

    name: str = "base"

    @abstractmethod
    def should_stop(self, state: StoppingState) -> bool:
        """Return True if the questionnaire should stop."""
        ...

    def reset(self) -> None:
        """Reset internal history. Call between persons."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Rule 1: Fixed 50%
# ---------------------------------------------------------------------------

class Fixed50PctRule(StoppingRule):
    """
    Stop when the number of observed items reaches `fraction` of all items.

    Parameters
    ----------
    fraction : float
        Fraction of items to ask (default 0.5 = 50%).
        The effective threshold is max(n_initial, round(fraction * n_total)).
    """

    name = "fixed_50pct"

    def __init__(self, fraction: float = 0.5):
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        self.fraction = fraction

    def should_stop(self, state: StoppingState) -> bool:
        n_total = len(state.all_items)
        threshold = max(
            state.n_initial_items,
            int(round(self.fraction * n_total)),
        )
        return len(state.person_observed) >= threshold

    def __repr__(self) -> str:
        return f"Fixed50PctRule(fraction={self.fraction})"


# ---------------------------------------------------------------------------
# Rule 2: Entropy-based
# ---------------------------------------------------------------------------

class EntropyRule(StoppingRule):
    """
    Stop when remaining item entropy is sufficiently low.

    Two sub-criteria (either triggers stopping):
    (a) Absolute: mean entropy over unasked items < tau
    (b) Marginal: reduction in mean entropy since last step < epsilon

    Parameters
    ----------
    tau : float or None
        Absolute entropy threshold (nats). If None, only the marginal
        criterion is used.
    epsilon : float or None
        Minimum entropy reduction per step. If None, only the absolute
        criterion is used. At least one of tau/epsilon must be set.
    min_items_asked : int
        Never stop before this many items have been asked (safeguard).
        Defaults to n_initial_items + 1.
    """

    name = "entropy"

    def __init__(
        self,
        tau: Optional[float] = None,
        epsilon: Optional[float] = None,
        min_items_asked: Optional[int] = None,
    ):
        if tau is None and epsilon is None:
            raise ValueError(
                "At least one of `tau` or `epsilon` must be set for EntropyRule."
            )
        self.tau = tau
        self.epsilon = epsilon
        self.min_items_asked = min_items_asked
        self._history: list[float] = []   # mean entropy per step

    def reset(self) -> None:
        self._history.clear()

    def _compute_mean_entropy(self, state: StoppingState) -> float:
        """
        Compute mean Shannon entropy across all unasked items for this person.

        Uses pre-computed item_scores if available (faster).
        Otherwise refits predictors — expensive; prefer passing item_scores.
        """
        unasked = [i for i in state.all_items if i not in state.person_observed]
        if not unasked:
            return 0.0

        if state.item_scores is not None:
            available = state.item_scores.reindex(unasked).dropna()
            if len(available) > 0:
                return float(available.mean())

        # Fallback: refit and compute entropy
        obs_cols = list(state.person_observed.keys())
        X_train = state.train_responses[obs_cols].values.astype(float)
        X_person = np.array([[state.person_observed[c] for c in obs_cols]])

        entropies = []
        for item in unasked:
            y_train = state.train_responses[item].values.astype(int)
            model = get_predictor(state.predictor_name, **state.predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            entropies.append(float(model.predict_entropy(X_person)[0]))

        return float(np.mean(entropies))

    def should_stop(self, state: StoppingState) -> bool:
        n_asked = len(state.person_observed)
        min_items = (
            self.min_items_asked
            if self.min_items_asked is not None
            else state.n_initial_items + 1
        )

        # Never stop too early
        if n_asked < min_items:
            return False

        # All items asked
        unasked = [i for i in state.all_items if i not in state.person_observed]
        if not unasked:
            return True

        mean_ent = self._compute_mean_entropy(state)
        self._history.append(mean_ent)

        # (a) Absolute threshold
        if self.tau is not None and mean_ent < self.tau:
            return True

        # (b) Marginal improvement threshold
        if self.epsilon is not None and len(self._history) >= 2:
            delta = self._history[-2] - self._history[-1]   # reduction
            if delta < self.epsilon:
                return True

        return False

    @property
    def entropy_history(self) -> list[float]:
        """Mean entropy per step for the current person."""
        return list(self._history)

    def __repr__(self) -> str:
        return f"EntropyRule(tau={self.tau}, epsilon={self.epsilon})"


# ---------------------------------------------------------------------------
# Rule 3: Error-based
# ---------------------------------------------------------------------------

class ErrorRule(StoppingRule):
    """
    Stop when the estimated prediction error on unasked items is low enough.

    Error is estimated via k-fold cross-validation on the training set,
    using only the currently observed feature columns.

    Two sub-criteria (either triggers stopping):
    (a) Absolute: mean CV-MAE across unasked items < tau
    (b) Marginal: reduction in mean CV-MAE since last step < epsilon

    Parameters
    ----------
    tau : float or None
        Absolute MAE threshold (in Likert units). A value of ~0.5 means
        on average predictions are within half a scale point.
    epsilon : float or None
        Minimum MAE reduction per step. If reduction is smaller, adding
        more items does not meaningfully improve predictions.
    cv_folds : int
        Number of CV folds for error estimation (default 3 for speed).
    subsample_items : int or None
        If set, estimate error on a random subsample of unasked items
        (much faster for large item sets). None = use all unasked items.
    min_items_asked : int or None
        Never stop before this many items have been asked.
    random_state : int
        Seed for subsample selection.
    """

    name = "error"

    def __init__(
        self,
        tau: Optional[float] = None,
        epsilon: Optional[float] = None,
        cv_folds: int = 3,
        subsample_items: Optional[int] = None,
        min_items_asked: Optional[int] = None,
        random_state: int = 42,
    ):
        if tau is None and epsilon is None:
            raise ValueError(
                "At least one of `tau` or `epsilon` must be set for ErrorRule."
            )
        self.tau = tau
        self.epsilon = epsilon
        self.cv_folds = cv_folds
        self.subsample_items = subsample_items
        self.min_items_asked = min_items_asked
        self.random_state = random_state
        self._history: list[float] = []   # mean CV-MAE per step
        self._rng = np.random.default_rng(random_state)

    def reset(self) -> None:
        self._history.clear()
        self._rng = np.random.default_rng(self.random_state)

    def _estimate_mean_mae(self, state: StoppingState) -> float:
        """
        Estimate the mean CV-MAE across unasked items given the current
        observed feature set on the training data.
        """
        unasked = [i for i in state.all_items if i not in state.person_observed]
        if not unasked:
            return 0.0

        obs_cols = list(state.person_observed.keys())
        X_train = state.train_responses[obs_cols].values.astype(float)

        # Subsample items for speed
        target_items = unasked
        if self.subsample_items is not None and len(unasked) > self.subsample_items:
            target_items = list(
                self._rng.choice(unasked, size=self.subsample_items, replace=False)
            )

        # Ensure we have enough training samples for CV
        n_train = len(state.train_responses)
        n_folds = min(self.cv_folds, n_train)
        if n_folds < 2:
            warnings.warn(
                f"Training set too small ({n_train}) for {self.cv_folds}-fold CV. "
                "Returning NaN error estimate.",
                UserWarning, stacklevel=3,
            )
            return np.nan

        maes = []
        for item in target_items:
            y_train = state.train_responses[item].values.astype(int)

            # Skip zero-variance targets (CV is meaningless)
            if len(np.unique(y_train)) < 2:
                continue

            model = get_predictor(state.predictor_name, **state.predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=n_folds,
                    scoring="neg_mean_absolute_error",
                    error_score="raise",
                )
            maes.append(-scores.mean())

        return float(np.mean(maes)) if maes else np.nan

    def should_stop(self, state: StoppingState) -> bool:
        n_asked = len(state.person_observed)
        min_items = (
            self.min_items_asked
            if self.min_items_asked is not None
            else state.n_initial_items + 1
        )

        if n_asked < min_items:
            return False

        unasked = [i for i in state.all_items if i not in state.person_observed]
        if not unasked:
            return True

        mean_mae = self._estimate_mean_mae(state)

        if np.isnan(mean_mae):
            return False  # cannot estimate; keep asking

        self._history.append(mean_mae)

        # (a) Absolute threshold
        if self.tau is not None and mean_mae < self.tau:
            return True

        # (b) Marginal improvement threshold
        if self.epsilon is not None and len(self._history) >= 2:
            delta = self._history[-2] - self._history[-1]   # reduction
            if delta < self.epsilon:
                return True

        return False

    @property
    def error_history(self) -> list[float]:
        """Mean CV-MAE per step for the current person."""
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"ErrorRule(tau={self.tau}, epsilon={self.epsilon}, "
            f"cv_folds={self.cv_folds})"
        )


# ---------------------------------------------------------------------------
# Threshold estimation helpers
# ---------------------------------------------------------------------------

def estimate_entropy_thresholds(
    train_responses: pd.DataFrame,
    predictor_name: str = "lgbm",
    predictor_kwargs: Optional[dict] = None,
    observed_fraction: float = 0.5,
    quantile_tau: float = 0.25,
    quantile_epsilon: float = 0.10,
    n_sample_persons: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """
    Estimate data-driven thresholds for EntropyRule from training data.

    Simulates the entropy trajectory at `observed_fraction` of items asked
    for a sample of training persons (leave-one-out style) and derives:
    - tau    : `quantile_tau` quantile of observed mean-entropy values
               (i.e. "low entropy" typical of a well-predicted response set)
    - epsilon: `quantile_epsilon` quantile of step-wise entropy reductions
               (i.e. "small marginal gain" typical of plateau)

    Parameters
    ----------
    train_responses : pd.DataFrame
    predictor_name : str
    predictor_kwargs : dict
    observed_fraction : float
        Fraction of items treated as "observed" for threshold estimation.
    quantile_tau : float
        Lower quantile for mean-entropy → tau (default 0.25).
    quantile_epsilon : float
        Lower quantile for delta-entropy → epsilon (default 0.10).
    n_sample_persons : int
        Number of training persons to sample for efficiency.
    seed : int

    Returns
    -------
    dict with keys 'tau' and 'epsilon'.
    """
    predictor_kwargs = predictor_kwargs or {}
    rng = np.random.default_rng(seed)
    items = list(train_responses.columns)
    n_items = len(items)
    n_obs = int(round(observed_fraction * n_items))

    persons = list(train_responses.index)
    if len(persons) > n_sample_persons:
        persons = list(rng.choice(persons, size=n_sample_persons, replace=False))

    mean_entropies = []
    delta_entropies = []

    for pid in persons:
        # Use all other persons as training
        others = train_responses.drop(index=pid)
        person = train_responses.loc[pid]

        # Randomly pick observed items
        obs_items = list(rng.choice(items, size=n_obs, replace=False))
        unasked = [i for i in items if i not in obs_items]

        X_train = others[obs_items].values.astype(float)
        X_person = np.array([[person[c] for c in obs_items]])

        ents = []
        for item in unasked:
            y_train = others[item].values.astype(int)
            if len(np.unique(y_train)) < 2:
                continue
            model = get_predictor(predictor_name, **predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            ents.append(float(model.predict_entropy(X_person)[0]))

        if ents:
            mean_entropies.append(np.mean(ents))

        # Also simulate one step earlier (n_obs - 1) for delta
        if n_obs > 1:
            obs_prev = obs_items[:-1]
            unasked_prev = [i for i in items if i not in obs_prev]
            X_train_prev = others[obs_prev].values.astype(float)
            X_person_prev = np.array([[person[c] for c in obs_prev]])
            ents_prev = []
            for item in unasked_prev:
                y_train = others[item].values.astype(int)
                if len(np.unique(y_train)) < 2:
                    continue
                model = get_predictor(predictor_name, **predictor_kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_prev, y_train)
                ents_prev.append(float(model.predict_entropy(X_person_prev)[0]))
            if ents and ents_prev:
                delta_entropies.append(np.mean(ents_prev) - np.mean(ents))

    tau = float(np.quantile(mean_entropies, quantile_tau)) if mean_entropies else 1.0
    epsilon = float(np.quantile(delta_entropies, quantile_epsilon)) if delta_entropies else 0.01

    return {"tau": tau, "epsilon": epsilon}


def estimate_error_thresholds(
    train_responses: pd.DataFrame,
    predictor_name: str = "lgbm",
    predictor_kwargs: Optional[dict] = None,
    observed_fraction: float = 0.5,
    quantile_tau: float = 0.25,
    quantile_epsilon: float = 0.10,
    cv_folds: int = 3,
    n_sample_persons: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    """
    Estimate data-driven thresholds for ErrorRule from training data.

    Analogous to `estimate_entropy_thresholds` but using CV-MAE.

    Returns
    -------
    dict with keys 'tau' and 'epsilon'.
    """
    predictor_kwargs = predictor_kwargs or {}
    rng = np.random.default_rng(seed)
    items = list(train_responses.columns)
    n_items = len(items)
    n_obs = int(round(observed_fraction * n_items))

    persons = list(train_responses.index)
    if len(persons) > n_sample_persons:
        persons = list(rng.choice(persons, size=n_sample_persons, replace=False))

    mean_maes = []
    delta_maes = []

    for pid in persons:
        others = train_responses.drop(index=pid)
        obs_items = list(rng.choice(items, size=n_obs, replace=False))
        unasked = [i for i in items if i not in obs_items]

        X_train = others[obs_items].values.astype(float)
        n_folds_eff = min(cv_folds, len(others))

        maes = []
        for item in unasked:
            y = others[item].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            model = get_predictor(predictor_name, **predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    model, X_train, y,
                    cv=n_folds_eff,
                    scoring="neg_mean_absolute_error",
                    error_score="raise",
                )
            maes.append(-scores.mean())

        if maes:
            mean_maes.append(np.mean(maes))

        # One step earlier for delta
        if n_obs > 1:
            obs_prev = obs_items[:-1]
            unasked_prev = [i for i in items if i not in obs_prev]
            X_train_prev = others[obs_prev].values.astype(float)
            maes_prev = []
            for item in unasked_prev:
                y = others[item].values.astype(int)
                if len(np.unique(y)) < 2:
                    continue
                model = get_predictor(predictor_name, **predictor_kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        model, X_train_prev, y,
                        cv=n_folds_eff,
                        scoring="neg_mean_absolute_error",
                        error_score="raise",
                    )
                maes_prev.append(-scores.mean())
            if maes and maes_prev:
                delta_maes.append(np.mean(maes_prev) - np.mean(maes))

    tau = float(np.quantile(mean_maes, quantile_tau)) if mean_maes else 0.5
    epsilon = float(np.quantile(delta_maes, quantile_epsilon)) if delta_maes else 0.01

    return {"tau": tau, "epsilon": epsilon}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STOPPING_RULE_REGISTRY: dict[str, type[StoppingRule]] = {
    "fixed_50pct": Fixed50PctRule,
    "entropy": EntropyRule,
    "error": ErrorRule,
}


def get_stopping_rule(
    name: Literal["fixed_50pct", "entropy", "error"],
    **kwargs,
) -> StoppingRule:
    """
    Instantiate a stopping rule by name.

    Parameters
    ----------
    name : str
        One of 'fixed_50pct', 'entropy', 'error'.
    **kwargs
        Passed to the rule constructor.

    Returns
    -------
    StoppingRule instance.
    """
    if name not in STOPPING_RULE_REGISTRY:
        raise ValueError(
            f"Unknown stopping rule '{name}'. "
            f"Available: {list(STOPPING_RULE_REGISTRY)}"
        )
    return STOPPING_RULE_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(7)
    n_persons, n_items = 120, 24
    item_names = [f"item_{i:02d}" for i in range(n_items)]

    latent = rng.standard_normal((n_persons, 3))
    loadings = rng.standard_normal((n_items, 3))
    raw = latent @ loadings.T + 0.5 * rng.standard_normal((n_persons, n_items))
    responses = pd.DataFrame(
        np.clip(np.round(3 + raw), 1, 5).astype(int),
        columns=item_names,
        index=[f"P{i:03d}" for i in range(n_persons)],
    )

    train = responses.iloc[:100]
    test_person = responses.iloc[100]

    # --- Fixed rule ---
    rule_fixed = get_stopping_rule("fixed_50pct")
    state = StoppingState(
        person_observed={f"item_{i:02d}": int(test_person[f"item_{i:02d}"]) for i in range(12)},
        all_items=item_names,
        train_responses=train,
        n_initial_items=5,
    )
    print(f"Fixed50Pct (12/24 asked): {rule_fixed.should_stop(state)}")  # expect True

    # --- Estimate thresholds ---
    print("\nEstimating entropy thresholds (ridge for speed)...")
    ent_thresh = estimate_entropy_thresholds(
        train, predictor_name="ridge", n_sample_persons=20, seed=42
    )
    print(f"  Entropy thresholds: {ent_thresh}")

    print("Estimating error thresholds (ridge for speed)...")
    err_thresh = estimate_error_thresholds(
        train, predictor_name="ridge", n_sample_persons=15, seed=42
    )
    print(f"  Error thresholds: {err_thresh}")

    # --- Entropy rule ---
    rule_ent = get_stopping_rule("entropy", tau=ent_thresh["tau"], epsilon=ent_thresh["epsilon"])
    rule_ent.reset()
    for n_asked in [6, 9, 12]:
        obs = {f"item_{i:02d}": int(test_person[f"item_{i:02d}"]) for i in range(n_asked)}
        state_ent = StoppingState(
            person_observed=obs,
            all_items=item_names,
            train_responses=train,
            predictor_name="ridge",
            n_initial_items=5,
        )
        result = rule_ent.should_stop(state_ent)
        hist = rule_ent.entropy_history
        print(f"  EntropyRule ({n_asked} asked): stop={result} | "
              f"mean_ent={hist[-1]:.4f}" if hist else f"  EntropyRule ({n_asked}): stop={result}")

    # --- Error rule ---
    rule_err = get_stopping_rule(
        "error", tau=err_thresh["tau"], epsilon=err_thresh["epsilon"],
        cv_folds=3, subsample_items=8
    )
    rule_err.reset()
    for n_asked in [6, 9, 12]:
        obs = {f"item_{i:02d}": int(test_person[f"item_{i:02d}"]) for i in range(n_asked)}
        state_err = StoppingState(
            person_observed=obs,
            all_items=item_names,
            train_responses=train,
            predictor_name="ridge",
            n_initial_items=5,
        )
        result = rule_err.should_stop(state_err)
        hist = rule_err.error_history
        print(f"  ErrorRule   ({n_asked} asked): stop={result} | "
              f"mean_mae={hist[-1]:.4f}" if hist else f"  ErrorRule ({n_asked}): stop={result}")

    print("\nstopping_rules.py OK.")
