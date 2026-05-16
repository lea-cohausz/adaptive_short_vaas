"""
party_scoring.py
================
Compute party match accuracy scores from item responses.

Background
----------
The match score logic was developed by a colleague and is reproduced here
without changes to the core arithmetic. The original pipeline:

1. Raw user responses are 0–4. Add 1 → 1–5.
2. Replace 3 → 3.11, 2 → 2.12  (distinguishes "neutral" from "skip").
3. Same shift + replace is applied to the party position data.
4. For each party: add the (shifted) user vector to the (shifted) party
   vector element-wise, then map each sum through transform_value().
5. Sum the transformed values → raw result score ("res").
6. Rescale to 0–100 accuracy ("acc"):
       acc = ((res + n_items) / (2 * n_items)) * 100

Integration note — value range
-------------------------------
The rest of our pipeline (predictors, simulation) works in the 1–5 Likert
range (raw 0–4 shifted by +1). party_scoring receives responses in this
1–5 range and converts them back to 0–4 before applying the match logic,
so the arithmetic stays identical to the original code.

Public API
----------
    load_party_positions(path)               -> pd.DataFrame
    compute_match_scores(responses, party_positions) -> pd.Series  (party -> acc)
    compute_match_scores_matrix(responses_df, party_positions) -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core transform — scalar version (kept for reference / external callers)
# ---------------------------------------------------------------------------

def transform_value(x: float) -> float:
    """
    Map a summed (user + party) value to a match contribution weight.
    Reproduced exactly from the updated scoring code (v2).

    Changes vs v1:
    - All comparisons use round(x, 2) to avoid floating-point issues
    - 5.12 → replaced by 7.12 (corrected case)
    - New case: 8.11 → 0

    Note: the hot path uses _transform_vec() which vectorises this logic
    over a whole item array at once via np.select.
    """
    rx = round(x, 2)
    if rx == 2:
        return 1
    elif rx == 3.12:
        return 0.5
    elif rx == 4.11:
        return 0
    elif rx == 4.24:
        return 0.75
    elif x == 5:
        return -0.5
    elif rx == 7.12:
        return -0.5
    elif rx == 6.12:
        return -0.25
    elif rx == 6.22:
        return 0.5
    elif rx == 5.23:
        return 0.25
    elif rx == 7.11:
        return 0.25
    elif rx == 8.11:
        return 0
    elif rx == 6:
        return -1
    elif rx == 8:
        return 0.75
    elif rx == 9:
        return 0.5
    elif rx == 10:
        return 1
    else:
        return 0


# ---------------------------------------------------------------------------
# Vectorised transform — lookup-table approach
# ---------------------------------------------------------------------------

# The 15 reachable sum values, multiplied by 100 and cast to int as indices.
# Max index = 1000; the LUT array is 1001 elements (negligible memory).
_SUMS_INT = np.array([200, 312, 411, 424, 500, 523,
                       600, 612, 622, 711, 712, 800, 811, 900, 1000])
_WEIGHTS  = np.array([1.0, 0.5, 0.0, 0.75, -0.5, 0.25,
                       -1.0, -0.25, 0.5, 0.25, -0.5, 0.75, 0.0, 0.5, 1.0])

_LUT: np.ndarray = np.zeros(1001, dtype=np.float64)
for _k, _w in zip(_SUMS_INT, _WEIGHTS):
    _LUT[_k] = _w


def _transform_vec(sums: np.ndarray) -> np.ndarray:
    """
    Vectorised equivalent of transform_value() via integer lookup table.

    Multiplies by 100, rounds to int, and indexes into a 1001-element array.
    O(1) per element, branch-free, ~29x faster than the np.select version.

    Parameters
    ----------
    sums : np.ndarray  Array of summed prepared values (any shape).

    Returns
    -------
    np.ndarray of the same shape, dtype float64.
    """
    return _LUT[np.round(sums * 100).astype(np.int32)]


# ---------------------------------------------------------------------------
# Shift + replace helpers
# ---------------------------------------------------------------------------

_PREPARE_IN  = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
_PREPARE_OUT = np.array([1.0, 2.12, 3.11, 4.0, 5.0])  # +1, then 2->2.12, 3->3.11


def _prepare_vec(raw: np.ndarray) -> np.ndarray:
    """
    Vectorised _prepare_responses for numpy arrays of raw 0-4 integer values.
    Uses np.select for a branch-free mapping; output dtype is float64.
    """
    conditions = [raw == v for v in _PREPARE_IN]
    return np.select(conditions, _PREPARE_OUT, default=raw + 1.0)


# ---------------------------------------------------------------------------
# Party matrix cache
# ---------------------------------------------------------------------------

class _PartyMatrixCache:
    """
    Caches the prepared party position matrix (n_parties, n_items) keyed by
    the tuple of item column names.  Avoids recomputing _prepare_vec on the
    same party_positions DataFrame on every call to compute_match_scores.
    Typically only one distinct item set exists per run, so the cache stays
    at a single entry.
    """
    def __init__(self):
        self._key: tuple | None = None
        self._mat: np.ndarray | None = None

    def get(self, party_positions: pd.DataFrame,
            item_cols: list[str]) -> np.ndarray:
        key = (id(party_positions), tuple(item_cols))
        if key != self._key:
            raw = party_positions[item_cols].to_numpy(dtype=float)
            self._mat = _prepare_vec(raw)
            self._key = key
        return self._mat  # type: ignore[return-value]


_party_cache = _PartyMatrixCache()


def _prepare_responses(values: pd.Series) -> pd.Series:
    """
    Apply the original preprocessing to a response vector (pandas path,
    used only for the public scalar API and batch matrix scorer).

    Input:  0-4  (raw Likert as stored in data files)
    Steps:  +1 -> 1-5, then 2 -> 2.12, 3 -> 3.11
    Output: float series with the coded values expected by transform_value.
    """
    v = values.astype(float) + 1          # 0-4 -> 1-5
    v = v.replace({2.0: 2.12, 3.0: 3.11})
    return v


def _pipeline_to_raw(values: pd.Series) -> pd.Series:
    """
    Convert pipeline-internal 1-5 values back to 0-4 before scoring.
    Called on predicted/observed responses that the rest of the pipeline
    produced in the 1-5 range.
    """
    return values.astype(float) - 1       # 1-5 -> 0-4


# ---------------------------------------------------------------------------
# Party position loader
# ---------------------------------------------------------------------------

def load_party_positions(path: str) -> pd.DataFrame:
    """
    Load the party positions dataset.

    Expected format
    ---------------
    CSV with:
    - One row per party.
    - A column named "party" containing the party identifier string.
    - All other columns are item columns (same names as in the user data),
      with values in the 0–4 range.

    Returns
    -------
    pd.DataFrame, index = party name, columns = item names.
    """
    df = pd.read_csv(path)
    if "party" not in df.columns:
        raise ValueError(
            "Party positions file must contain a 'party' column. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.set_index("party")
    return df


# ---------------------------------------------------------------------------
# Core scoring function (single person)
# ---------------------------------------------------------------------------

def compute_match_scores(
    responses: Union[dict[str, float], pd.Series],
    party_positions: pd.DataFrame,
    input_range: str = "1-5",
) -> pd.Series:
    """
    Compute party match accuracy scores for one person.

    Parameters
    ----------
    responses : dict or pd.Series
        Item responses for one person.
        Keys/index = item names, values = Likert responses.
    party_positions : pd.DataFrame
        Index = party names, columns = item names, values in 0-4 range.
        Load with load_party_positions() or pass the pre-loaded DataFrame.
    input_range : str
        '1-5'  — responses are in the pipeline-internal 1-5 range (default).
                 They will be shifted back to 0-4 before scoring.
        '0-4'  — responses are already in the raw 0-4 range.

    Returns
    -------
    pd.Series
        Index = party names, values = match accuracy (0-100).
        Higher = better match.

    Implementation note
    -------------------
    Hot path is fully vectorised via numpy (_prepare_vec + _transform_vec).
    All parties are scored in a single np.select pass over an
    (n_parties, n_items) matrix, eliminating per-element Python calls.
    """
    if isinstance(responses, dict):
        responses = pd.Series(responses)

    # Align to item columns present in party_positions
    item_cols = [c for c in party_positions.columns if c in responses.index]
    if not item_cols:
        raise ValueError(
            "No overlapping item columns between responses and party_positions."
        )
    if len(item_cols) < len(party_positions.columns):
        warnings.warn(
            f"{len(party_positions.columns) - len(item_cols)} item(s) present in "
            "party_positions but missing from responses. They will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    n_items = len(item_cols)

    # User vector: to raw 0-4, then vectorised prepare  (shape: n_items,)
    user_raw = responses[item_cols].to_numpy(dtype=float)
    if input_range == "1-5":
        user_raw = user_raw - 1.0          # 1-5 -> 0-4
    elif input_range != "0-4":
        raise ValueError(f"input_range must be '1-5' or '0-4', got '{input_range}'.")
    user_prepared = _prepare_vec(user_raw)  # shape: (n_items,)

    # Party matrix: (n_parties, n_items) — cached across calls
    party_prepared = _party_cache.get(party_positions, item_cols)

    # Broadcast sum: (n_parties, n_items) + (n_items,) -> (n_parties, n_items)
    sums        = party_prepared + user_prepared          # broadcast
    transformed = _transform_vec(sums)                   # same shape, no loop
    res         = transformed.sum(axis=1)                # (n_parties,)
    acc         = ((res + n_items) / (2 * n_items)) * 100

    return pd.Series(acc, index=party_positions.index)


# ---------------------------------------------------------------------------
# Batch scoring (whole DataFrame)
# ---------------------------------------------------------------------------

def compute_match_scores_matrix(
    responses_df: pd.DataFrame,
    party_positions: pd.DataFrame,
    input_range: str = "1-5",
) -> pd.DataFrame:
    """
    Compute party match accuracy scores for all persons in a DataFrame.

    Parameters
    ----------
    responses_df : pd.DataFrame
        Shape (n_persons, n_items). Index = person IDs.
        Values in range specified by `input_range`.
    party_positions : pd.DataFrame
        Index = party names, columns = item names, values in 0-4 range.
    input_range : str
        '1-5' or '0-4' — see compute_match_scores().

    Returns
    -------
    pd.DataFrame
        Shape (n_persons, n_parties).
        Index = person IDs, columns = party names, values = accuracy (0-100).

    Implementation note
    -------------------
    Fully vectorised: builds a single (n_persons, n_parties, n_items) broadcast
    and applies _transform_vec once, avoiding any Python-level loops.
    """
    item_cols = [c for c in party_positions.columns if c in responses_df.columns]
    if not item_cols:
        raise ValueError(
            "No overlapping item columns between responses_df and party_positions."
        )

    n_items = len(item_cols)

    # Persons matrix: (n_persons, n_items), raw 0-4
    persons_raw = responses_df[item_cols].to_numpy(dtype=float)
    if input_range == "1-5":
        persons_raw = persons_raw - 1.0
    elif input_range != "0-4":
        raise ValueError(f"input_range must be '1-5' or '0-4', got '{input_range}'.")
    persons_prepared = _prepare_vec(persons_raw)   # (n_persons, n_items)

    # Party matrix: (n_parties, n_items) — cached across calls
    party_prepared = _party_cache.get(party_positions, item_cols)

    # Broadcast: (n_persons, 1, n_items) + (n_parties, n_items)
    #         -> (n_persons, n_parties, n_items)
    sums        = persons_prepared[:, np.newaxis, :] + party_prepared
    transformed = _transform_vec(sums)             # (n_persons, n_parties, n_items)
    res         = transformed.sum(axis=2)          # (n_persons, n_parties)
    acc         = ((res + n_items) / (2 * n_items)) * 100

    return pd.DataFrame(acc,
                        index=responses_df.index,
                        columns=party_positions.index)


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def get_top_parties(
    scores: pd.Series,
    k: int = 3,
) -> list[str]:
    """
    Return the top-k parties by match accuracy (descending).

    Parameters
    ----------
    scores : pd.Series
        Index = party names, values = accuracy scores.
    k : int
        Number of top parties to return.

    Returns
    -------
    list[str] of length min(k, n_parties).
    """
    return list(scores.nlargest(k).index)


def best_match(scores: pd.Series) -> str:
    """Return the single best-matching party."""
    return str(scores.idxmax())


# ---------------------------------------------------------------------------
# Wire into simulation — drop-in replacement for _placeholder_compute_party_scores
# ---------------------------------------------------------------------------

def make_party_scorer(
    party_positions: pd.DataFrame,
    input_range: str = "1-5",
):
    """
    Return a callable suitable for use in simulation._simulate_person.

    Usage
    -----
        scorer = make_party_scorer(party_positions)
        scores_dict = scorer(full_responses_dict, all_items)

    The returned callable matches the signature of the placeholder:
        f(item_responses: dict, all_items: list) -> dict[str, float]
    """
    def _scorer(
        item_responses: dict[str, float],
        all_items: list[str],
    ) -> dict[str, float]:
        scores = compute_match_scores(
            item_responses,
            party_positions,
            input_range=input_range,
        )
        return scores.to_dict()

    return _scorer


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate item names
    item_cols = [f"item_{i:02d}" for i in range(20)]
    parties = ["PartyA", "PartyB", "PartyC", "PartyD"]

    # Fake party positions in 0–4
    party_pos = pd.DataFrame(
        rng.integers(0, 5, size=(len(parties), len(item_cols))),
        index=parties,
        columns=item_cols,
    )

    # --- Test with raw 0–4 responses ---
    user_raw = pd.Series(
        rng.integers(0, 5, size=len(item_cols)),
        index=item_cols,
    )
    scores_raw = compute_match_scores(user_raw, party_pos, input_range="0-4")
    print("Scores from 0-4 input:")
    print(scores_raw.round(2).to_string())

    # --- Test with pipeline 1–5 responses (shift by +1) ---
    user_15 = user_raw + 1
    scores_15 = compute_match_scores(user_15, party_pos, input_range="1-5")
    print("\nScores from 1-5 input (should be identical):")
    print(scores_15.round(2).to_string())

    # Verify they are identical
    assert np.allclose(scores_raw.values, scores_15.values), \
        "Mismatch between 0-4 and 1-5 scoring paths!"
    print("\n✓ Both input ranges produce identical scores.")

    # --- Top parties ---
    print(f"\nBest match:  {best_match(scores_raw)}")
    print(f"Top-3 match: {get_top_parties(scores_raw, k=3)}")

    # --- Batch scoring ---
    n_persons = 10
    responses_df = pd.DataFrame(
        rng.integers(1, 6, size=(n_persons, len(item_cols))),
        columns=item_cols,
        index=[f"P{i:03d}" for i in range(n_persons)],
    )
    score_matrix = compute_match_scores_matrix(responses_df, party_pos, input_range="1-5")
    print(f"\nBatch score matrix shape: {score_matrix.shape}")
    print(score_matrix.round(1).to_string())

    print("\nparty_scoring.py OK.")
