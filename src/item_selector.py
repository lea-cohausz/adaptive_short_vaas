"""
item_selector.py
================
Adaptive next-item selection: given a person's observed responses so far,
pick the single best unasked item to ask next.

Two strategies are implemented:

1. **uncertainty**
   For each unasked item, fit a predictor (trained on the training set) and
   compute the model's uncertainty about its prediction for this person.
   Select the item with the HIGHEST uncertainty — the item we know least
   about for this specific person given what we already know.
   Uncertainty measure: 1 − max(class_probability)  (works for all models).

2. **entropy**
   For each unasked item, compute the Shannon entropy of the model's
   predicted class distribution for this person.
   Select the item with the HIGHEST predicted entropy — the item whose
   answer is most "spread out" in probability, i.e. hardest to pin down.
   For deterministic models (Ridge) this reduces to entropy of the soft
   Gaussian approximation. For classifiers it uses the native class probs.

Both strategies follow the same signature and differ only in the scoring
function applied to the predictor output.

Key design: Model caching
-------------------------
Fitting one predictor per unasked item at every adaptive step would be
O(n_items²) fits per person — expensive for LGBM and TabPFN.

We address this with a ModelCache that stores fitted predictors keyed by
(frozenset(observed_items), target_item). Within a single person's loop,
the observed feature set grows by exactly one item per step. We only need
to refit models whose training features changed — which is all of them, but
we only refit when explicitly asked (i.e., after each new item is revealed).

For TabPFN specifically, since it is in-context and stateless, refitting is
unavoidable; the cache simply avoids redundant fits within the same step.

For a further speed-up in production runs, the cache can be populated in
parallel across items (see `fit_all_parallel` flag).

Public API
----------
    ItemSelector(strategy, predictor_name, predictor_kwargs, ...)
    selector.fit_item_models(train_responses, observed_cols)
    selector.select_next_item(person_observed, all_items) -> str
    selector.score_all_items(person_observed, all_items)  -> pd.Series
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal, Optional

import numpy as np
import pandas as pd

from predictors import BasePredictor, get_predictor


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

class _ModelCache:
    """
    Stores fitted predictors keyed by target item name.
    A cache instance is valid for one (person, step) — i.e. one fixed set
    of observed features. Call reset() whenever observed_cols changes.
    """

    def __init__(self):
        self._cache: dict[str, BasePredictor] = {}

    def get(self, item: str) -> Optional[BasePredictor]:
        return self._cache.get(item)

    def set(self, item: str, model: BasePredictor) -> None:
        self._cache[item] = model

    def reset(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Core selector class
# ---------------------------------------------------------------------------

class ItemSelector:
    """
    Adaptive next-item selector.

    Parameters
    ----------
    strategy : str
        'uncertainty' — select item with highest 1 − max(prob).
        'entropy'     — select item with highest Shannon entropy H(p).
    predictor_name : str
        Which predictor to use for scoring: 'ridge', 'knn', 'lgbm',
        'tabpfn_v2'.
    predictor_kwargs : dict
        Forwarded to get_predictor().
    parallel : bool
        If True, fit per-item models in parallel using ProcessPoolExecutor.
        Recommended for 'ridge' and 'knn'; use with caution for 'tabpfn_v2'
        (GPU memory contention).
    n_jobs : int
        Number of parallel workers (default 4). Ignored if parallel=False.
    random_state : int
        Used to break ties in item scoring.
    """

    def __init__(
        self,
        strategy: Literal["uncertainty", "entropy", "random", "fixed"] = "entropy",
        predictor_name: str = "lgbm",
        predictor_kwargs: Optional[dict] = None,
        parallel: bool = False,
        n_jobs: int = 4,
        random_state: int = 42,
        fold_cache=None,   # FoldModelCache instance (opt. 2); avoids circular import
    ):
        self.strategy = strategy
        self.predictor_name = predictor_name
        self.predictor_kwargs = predictor_kwargs or {}
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.fold_cache = fold_cache   # shared across persons within a fold

        # Set after fit_item_models()
        self._train_responses: Optional[pd.DataFrame] = None
        self._observed_cols: list[str] = []
        self._cache = _ModelCache()
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_item_models(
        self,
        train_responses: pd.DataFrame,
        observed_cols: list[str],
        target_items: Optional[list[str]] = None,
    ) -> "ItemSelector":
        """
        Fit one predictor per target item using `observed_cols` as features.

        Parameters
        ----------
        train_responses : pd.DataFrame
            Training persons × all items. Likert 1-5.
        observed_cols : list[str]
            Items already observed for the current person (used as features).
        target_items : list[str] or None
            Items to fit models for. If None, fits for all columns not in
            observed_cols.

        Returns
        -------
        self
        """
        self._train_responses = train_responses
        self._observed_cols = list(observed_cols)
        self._cache.reset()

        # Random and fixed strategies need no fitted models
        if self.strategy in ("random", "fixed"):
            return self

        all_items = list(train_responses.columns)
        if target_items is None:
            target_items = [c for c in all_items if c not in observed_cols]

        X_train = train_responses[observed_cols].values.astype(float)

        # Separate items into those already in the fold cache vs those to fit
        items_to_fit = []
        for item in target_items:
            if self.fold_cache is not None:
                cached = self.fold_cache.get(observed_cols, item)
                if cached is not None:
                    self._cache.set(item, cached)
                    continue
            items_to_fit.append(item)

        if items_to_fit:
            if self.parallel and len(items_to_fit) > 4:
                self._fit_parallel(X_train, train_responses, items_to_fit)
            else:
                self._fit_sequential(X_train, train_responses, items_to_fit)

            # Store newly fitted models back into the fold cache
            if self.fold_cache is not None:
                for item in items_to_fit:
                    model = self._cache.get(item)
                    if model is not None:
                        self.fold_cache.set(observed_cols, item, model)

        return self

    def _fit_sequential(
        self,
        X_train: np.ndarray,
        train_responses: pd.DataFrame,
        target_items: list[str],
    ) -> None:
        for item in target_items:
            y_train = train_responses[item].values.astype(int)
            model = get_predictor(self.predictor_name, **self.predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            self._cache.set(item, model)

    def _fit_parallel(
        self,
        X_train: np.ndarray,
        train_responses: pd.DataFrame,
        target_items: list[str],
    ) -> None:
        """
        Fit models in parallel. Uses ProcessPoolExecutor.
        Note: TabPFN is not safe for multiprocessing with GPU; set
        parallel=False when using tabpfn_v2.
        """
        def _fit_one(item: str):
            y = train_responses[item].values.astype(int)
            m = get_predictor(self.predictor_name, **self.predictor_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_train, y)
            return item, m

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_fit_one, item): item for item in target_items}
            for future in as_completed(futures):
                item, model = future.result()
                self._cache.set(item, model)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_all_items(
        self,
        person_observed: dict[str, int],
        all_items: list[str],
    ) -> pd.Series:
        """
        Compute selection scores for all unasked items.

        Higher score = more informative to ask next.

        Parameters
        ----------
        person_observed : dict[str, int]
            {item_name: response_value} for already-asked items.
        all_items : list[str]
            All item names in the questionnaire.

        Returns
        -------
        pd.Series, index = unasked item names, values = scores.
        """
        obs_cols = list(person_observed.keys())
        X_person = np.array([[person_observed[c] for c in obs_cols]])

        unasked = [i for i in all_items if i not in person_observed]
        if not unasked:
            return pd.Series(dtype=float)

        scores = {}
        for item in unasked:
            if self.strategy == "random":
                scores[item] = float(self._rng.random())
                continue

            model = self._cache.get(item)
            if model is None:
                warnings.warn(
                    f"No fitted model found for item '{item}'. "
                    "Call fit_item_models() before score_all_items(). "
                    "Assigning score=0.",
                    UserWarning, stacklevel=2,
                )
                scores[item] = 0.0
                continue

            if self.strategy == "uncertainty":
                score = float(model.predict_uncertainty(X_person)[0])
            elif self.strategy == "entropy":
                score = float(model.predict_entropy(X_person)[0])
            else:
                raise ValueError(
                    f"Unknown strategy '{self.strategy}'. "
                    "Choose 'uncertainty', 'entropy', or 'random'."
                )
            scores[item] = score

        return pd.Series(scores)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_next_item(
        self,
        person_observed: dict[str, int],
        all_items: list[str],
    ) -> str:
        """
        Select the single best unasked item to ask next.

        Ties are broken uniformly at random (seeded for reproducibility).

        Parameters
        ----------
        person_observed : dict[str, int]
            {item_name: Likert response} for already-asked items.
        all_items : list[str]
            Full list of item names.

        Returns
        -------
        str : name of the selected item.

        Raises
        ------
        ValueError if all items have already been observed.
        """
        unasked = [i for i in all_items if i not in person_observed]
        if not unasked:
            raise ValueError("All items have already been observed.")

        scores = self.score_all_items(person_observed, all_items)

        if scores.empty or scores.isna().all():
            # Fallback: random selection
            warnings.warn(
                "All item scores are NaN or missing. Falling back to random selection.",
                UserWarning, stacklevel=2,
            )
            return str(self._rng.choice(unasked))

        # Break ties randomly
        max_score = scores.max()
        top_items = scores[scores == max_score].index.tolist()
        return str(self._rng.choice(top_items))

    # ------------------------------------------------------------------
    # Convenience: full step (fit + select)
    # ------------------------------------------------------------------

    def fit_and_select(
        self,
        train_responses: pd.DataFrame,
        person_observed: dict[str, int],
        all_items: list[str],
    ) -> str:
        """
        Convenience wrapper: fits item models for the current observed set
        and immediately returns the next item to ask.

        Use this in the simulation loop for clarity.
        """
        observed_cols = list(person_observed.keys())
        unasked = [i for i in all_items if i not in person_observed]
        self.fit_item_models(
            train_responses,
            observed_cols=observed_cols,
            target_items=unasked,
        )
        return self.select_next_item(person_observed, all_items)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_report(
        self,
        person_observed: dict[str, int],
        all_items: list[str],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return a diagnostic DataFrame showing scores for the top-N unasked
        items. Useful for inspection and debugging.

        Returns
        -------
        pd.DataFrame with columns: item, score, rank.
        """
        scores = self.score_all_items(person_observed, all_items)
        if scores.empty:
            return pd.DataFrame(columns=["item", "score", "rank"])
        df = (
            scores
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        df.columns = ["item", "score"]
        df["rank"] = np.arange(1, len(df) + 1)
        df["strategy"] = self.strategy
        df["predictor"] = self.predictor_name
        return df


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_selector(
    strategy: Literal["uncertainty", "entropy", "random", "fixed"] = "entropy",
    predictor_name: str = "lgbm",
    fold_cache=None,
    **kwargs,
) -> ItemSelector:
    """
    Instantiate an ItemSelector by strategy and predictor name.

    Parameters
    ----------
    strategy : str
        'uncertainty' or 'entropy'.
    predictor_name : str
        One of 'ridge', 'knn', 'lgbm', 'tabpfn_v2'.
    fold_cache : FoldModelCache or None
        Optional shared cache across persons within a fold (opt. 2).
    **kwargs
        Forwarded to ItemSelector constructor.
    """
    return ItemSelector(
        strategy=strategy,
        predictor_name=predictor_name,
        fold_cache=fold_cache,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n_train, n_items = 150, 20
    item_names = [f"item_{i:02d}" for i in range(n_items)]

    # Simulate correlated Likert data
    latent = rng.standard_normal((n_train, 3))
    loadings = rng.standard_normal((n_items, 3))
    raw = latent @ loadings.T + 0.5 * rng.standard_normal((n_train, n_items))
    train_df = pd.DataFrame(
        np.clip(np.round(3 + raw), 1, 5).astype(int),
        columns=item_names,
    )

    # Simulate one test person with 4 observed items
    person_obs = {f"item_{i:02d}": int(rng.integers(1, 6)) for i in range(4)}
    all_items = item_names

    for strategy in ["uncertainty", "entropy"]:
        for predictor_name in ["ridge", "knn", "lgbm"]:
            selector = get_selector(
                strategy=strategy,
                predictor_name=predictor_name,
                random_state=42,
            )
            next_item = selector.fit_and_select(train_df, person_obs, all_items)
            scores = selector.score_all_items(person_obs, all_items)
            print(
                f"strategy={strategy:12s} | predictor={predictor_name:6s} | "
                f"next_item={next_item} | "
                f"top score={scores.max():.4f} | "
                f"score std={scores.std():.4f}"
            )

    # Diagnostic report for one configuration
    print("\n--- Score report (entropy + lgbm, top 5) ---")
    selector = get_selector("entropy", "lgbm")
    selector.fit_item_models(train_df, list(person_obs.keys()))
    report = selector.score_report(person_obs, all_items, top_n=5)
    print(report.to_string(index=False))

    print("\nitem_selector.py OK.")
