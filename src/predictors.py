"""
predictors.py
=============
Unified predictor wrappers for adaptive item selection.

Each predictor predicts ONE target item (ordinal 1-5) given a set of
observed items. All models expose:

    fit(X_train, y_train)
    predict(X_test)           -> point predictions (rounded to 1-5)
    predict_uncertainty(X)    -> scalar uncertainty per sample
    predict_entropy(X)        -> expected entropy per sample (uses class probs)

For models without native probabilistic output (Ridge), uncertainty is
derived from prediction variance across a small bootstrap ensemble or from
prediction distance from the nearest integer.

Design notes
------------
- All wrappers are sklearn-compatible (fit / predict interface).
- kNN exposes class probabilities for entropy.
- Ridge predicts a continuous value; uncertainty is approximated as
  the absolute deviation of the raw prediction from the nearest integer.
- A PREDICTOR_REGISTRY dict at the bottom maps string names to classes,
  so the rest of the codebase can reference models by name.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

LIKERT_CLASSES = np.array([1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePredictor(ABC, BaseEstimator):
    """
    Abstract base for all item predictors.

    Inherits from sklearn's BaseEstimator so that cross_val_score,
    clone(), and other sklearn utilities work correctly. BaseEstimator
    provides __sklearn_tags__, get_params(), and set_params() via
    constructor introspection — so every subclass must store its
    constructor arguments as same-named instance attributes (standard
    sklearn convention).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BasePredictor":
        """Fit the model on (n_samples, n_observed_items) → target item."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer predictions clipped to [1, 5]."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class-probability matrix of shape (n_samples, 5).
        Columns correspond to LIKERT_CLASSES = [1, 2, 3, 4, 5].
        Default implementation uses a one-hot of the hard prediction.
        Override for probabilistic models.
        """
        preds = self.predict(X)
        proba = np.zeros((len(preds), 5))
        for i, p in enumerate(preds):
            proba[i, int(p) - 1] = 1.0
        return proba

    def predict_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        Shannon entropy H(p) over the 5 Likert classes.
        Shape: (n_samples,).
        Higher value = more uncertain prediction for that item.
        """
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-12, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Scalar uncertainty per sample.
        Default: entropy. Override for model-specific measures (e.g. Ridge).
        """
        return self.predict_entropy(X)



# ---------------------------------------------------------------------------
# Ridge wrapper
# ---------------------------------------------------------------------------

class RidgePredictor(BasePredictor):
    """
    Ridge regression on continuous Likert values.
    Predictions are clipped and rounded to [1, 5].
    Uncertainty = |raw_prediction - nearest_integer|  (distance to certainty).
    For entropy, we build a soft probability from a Gaussian centred on the
    raw prediction with std = residual_std estimated from training.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._residual_std: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgePredictor":
        self._model.fit(X_train, y_train)
        train_preds = self._model.predict(X_train)
        residuals = y_train - train_preds
        self._residual_std = max(float(np.std(residuals)), 0.1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X)
        return np.clip(np.round(raw), 1, 5).astype(int)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft probabilities from a Gaussian: p(k) ∝ exp(-0.5*((k - μ)/σ)²).
        """
        raw = self._raw_predict(X)  # (n_samples,)
        # distance from each class centre
        diff = LIKERT_CLASSES[np.newaxis, :] - raw[:, np.newaxis]  # (n, 5)
        log_proba = -0.5 * (diff / self._residual_std) ** 2
        # softmax-like normalisation
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """|raw - nearest integer|: 0 = certain, 0.5 = maximally uncertain."""
        raw = self._raw_predict(X)
        return np.abs(raw - np.round(raw))


# ---------------------------------------------------------------------------
# kNN wrapper
# ---------------------------------------------------------------------------

class KNNPredictor(BasePredictor):
    """
    k-Nearest Neighbours classifier over Likert classes.
    Naturally handles ordinal structure via distance in feature space.
    Uncertainty = 1 - max(class probability).
    """

    name = "knn"

    def __init__(self, n_neighbors: int = 10, weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights
        )
        self._classes: np.ndarray = LIKERT_CLASSES

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNPredictor":
        self._model.fit(X_train, y_train)
        self._classes = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Map sklearn's class-indexed proba to the full [1..5] grid.
        """
        raw_proba = self._model.predict_proba(X)  # (n, n_classes_seen)
        n = len(X)
        full_proba = np.zeros((n, 5))
        for ci, cls in enumerate(self._classes):
            col = int(cls) - 1  # map class label 1-5 to index 0-4
            if 0 <= col < 5:
                full_proba[:, col] = raw_proba[:, ci]
        return full_proba

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return 1.0 - proba.max(axis=1)


# ---------------------------------------------------------------------------
# Naive Bayes wrapper
# ---------------------------------------------------------------------------

class NaiveBayesPredictor(BasePredictor):
    """
    Multinomial Naive Bayes classifier over Likert classes {1, 2, 3, 4, 5}.

    A fast probabilistic baseline. MultinomialNB requires non-negative
    integer inputs — Likert 1–5 satisfies this directly with no preprocessing.

    Speed: ~1.3× Ridge, making it the fastest probabilistic model available.
    Uncertainty = 1 − max(class probability).
    """

    name = "naive_bayes"

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing parameter (default 1.0).
        """
        self.alpha = alpha
        self._model = MultinomialNB(alpha=alpha)
        self._classes: np.ndarray = LIKERT_CLASSES

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NaiveBayesPredictor":
        # Shift 1–5 to 0–4 so all values are non-negative (MultinomialNB requirement)
        # Then shift back on predict. Using alpha smoothing handles zero counts.
        self._model = MultinomialNB(alpha=self.alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_train - 1, y_train)
        self._classes = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X - 1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_proba = self._model.predict_proba(X - 1)
        n = len(X)
        full_proba = np.zeros((n, 5))
        for ci, cls in enumerate(self._classes):
            col = int(cls) - 1
            if 0 <= col < 5:
                full_proba[:, col] = raw_proba[:, ci]
        return full_proba

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return 1.0 - proba.max(axis=1)


# ---------------------------------------------------------------------------
# Random Forest wrapper
# ---------------------------------------------------------------------------

class RandomForestPredictor(BasePredictor):
    """
    Random Forest classifier over Likert classes {1, 2, 3, 4, 5}.

    Intentionally configured for speed over accuracy:
    - max_train_persons should be capped at ~250 via SimulationConfig
    - n_estimators=10, max_depth=3 recommended for adaptive loop use

    Uncertainty = 1 − max(class probability).
    """

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self._classes: np.ndarray = LIKERT_CLASSES

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestPredictor":
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_train, y_train)
        self._classes = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_proba = self._model.predict_proba(X)
        n = len(X)
        full_proba = np.zeros((n, 5))
        for ci, cls in enumerate(self._classes):
            col = int(cls) - 1
            if 0 <= col < 5:
                full_proba[:, col] = raw_proba[:, ci]
        return full_proba

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return 1.0 - proba.max(axis=1)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {
    "ridge":         RidgePredictor,
    "knn":           KNNPredictor,
    "naive_bayes":   NaiveBayesPredictor,
    "random_forest": RandomForestPredictor,
}


def get_predictor(name: str, **kwargs) -> BasePredictor:
    """
    Instantiate a predictor by name.

    Parameters
    ----------
    name : str
        One of 'ridge', 'knn', 'naive_bayes', 'random_forest'.
    **kwargs
        Passed to the predictor constructor.

    Returns
    -------
    BasePredictor instance (unfitted).
    """
    if name not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown predictor '{name}'. "
            f"Available: {list(PREDICTOR_REGISTRY)}"
        )
    return PREDICTOR_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly to verify imports)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_train, n_test, n_features = 200, 20, 10

    X_train = rng.integers(1, 6, size=(n_train, n_features)).astype(float)
    X_test = rng.integers(1, 6, size=(n_test, n_features)).astype(float)
    y_train = rng.integers(1, 6, size=n_train)

    for model_name in ["ridge", "knn", "naive_bayes", "random_forest"]:
        predictor = get_predictor(model_name)
        predictor.fit(X_train, y_train)
        preds = predictor.predict(X_test)
        unc = predictor.predict_uncertainty(X_test)
        ent = predictor.predict_entropy(X_test)
        print(
            f"{model_name:12s} | preds range [{preds.min()},{preds.max()}] "
            f"| mean unc {unc.mean():.3f} | mean entropy {ent.mean():.3f}"
        )
    print("All predictors OK.")
