"""
adaptive_v2/src/evaluation.py
==============================
Evaluation and confidence interval computation for adaptive_v2 results.

Metrics
-------
For fixed 50% condition (obs-only and obs+pred):
    top-1 accuracy, top-3 accuracy, party MAE per party

For adaptive stopping (top-1 and top-3):
    n_items_asked distribution, reached fraction, party MAE per party

Confidence intervals
---------------------
Bootstrap CIs (n=1000 resamples) over the test persons.
Normal approximation also available for quick checks.

Public API
----------
    summarise_50pct(results)           → dict of metrics + CIs
    summarise_adaptive(results)        → dict of metrics + CIs
    party_breakdown(results, condition)→ pd.DataFrame per party
    bootstrap_ci(values, stat, n)      → (lower, upper)
    results_to_dataframe(results)      → long-form pd.DataFrame
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simulation import PersonResult


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat: Callable = np.mean,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic over `values`.

    Parameters
    ----------
    values : array-like   Per-person metric values (1-D).
    stat   : callable     Statistic to bootstrap (default np.mean).
    n      : int          Number of bootstrap resamples (default 1000).
    ci     : float        Confidence level (default 0.95).
    seed   : int

    Returns
    -------
    (lower, upper) : float, float
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots  = [stat(rng.choice(values, size=len(values), replace=True))
               for _ in range(n)]
    alpha  = (1 - ci) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def _ci(values, stat=np.mean, n=1000, seed=42):
    """Shorthand returning (mean, lower, upper)."""
    v   = np.asarray(values, dtype=float)
    m   = float(stat(v))
    lo, hi = bootstrap_ci(v, stat=stat, n=n, seed=seed)
    return m, lo, hi


# ---------------------------------------------------------------------------
# 50% condition summary
# ---------------------------------------------------------------------------

def summarise_50pct(
    results: list[PersonResult],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Aggregate metrics for the fixed 50% stopping condition.

    Returns a dict with keys:
        obs_top1, obs_top1_ci, obs_top3, obs_top3_ci,
        pred_top1, pred_top1_ci, pred_top3, pred_top3_ci,
        obs_party_mae  : {party: (mean, lo, hi)},
        pred_party_mae : {party: (mean, lo, hi)},
        n_items_asked  : (mean, lo, hi),
        n_persons
    """
    valid = [r for r in results if r.obs_top1_correct_50 is not None]

    obs_top1  = np.array([r.obs_top1_correct_50  for r in valid], dtype=float)
    obs_top3  = np.array([r.obs_top3_correct_50  for r in valid], dtype=float)
    pred_top1 = np.array([r.pred_top1_correct_50 for r in valid], dtype=float)
    pred_top3 = np.array([r.pred_top3_correct_50 for r in valid], dtype=float)
    n_asked   = np.array([r.n_items_asked_50      for r in valid], dtype=float)

    out = {
        "obs_top1":    _ci(obs_top1,  n=n_boot, seed=seed),
        "obs_top3":    _ci(obs_top3,  n=n_boot, seed=seed),
        "pred_top1":   _ci(pred_top1, n=n_boot, seed=seed),
        "pred_top3":   _ci(pred_top3, n=n_boot, seed=seed),
        "n_items_asked": _ci(n_asked, n=n_boot, seed=seed),
        "n_persons":   len(valid),
    }

    # Per-party MAE
    parties = list(valid[0].obs_party_mae_50.keys()) if valid else []
    obs_mae_by_party  = {}
    pred_mae_by_party = {}
    for party in parties:
        obs_vals  = [r.obs_party_mae_50.get(party, np.nan)  for r in valid]
        pred_vals = [r.pred_party_mae_50.get(party, np.nan) for r in valid]
        obs_mae_by_party[party]  = _ci(
            [v for v in obs_vals  if not np.isnan(v)], n=n_boot, seed=seed)
        pred_mae_by_party[party] = _ci(
            [v for v in pred_vals if not np.isnan(v)], n=n_boot, seed=seed)

    out["obs_party_mae"]  = obs_mae_by_party
    out["pred_party_mae"] = pred_mae_by_party
    return out


# ---------------------------------------------------------------------------
# Adaptive condition summary
# ---------------------------------------------------------------------------

def summarise_adaptive(
    results: list[PersonResult],
    condition: Literal["top1", "top3"] = "top1",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Aggregate metrics for the adaptive stopping condition.

    Parameters
    ----------
    condition : 'top1' or 'top3'

    Returns a dict with keys:
        mean_items_asked, items_asked_ci,
        reached_fraction, reached_ci,
        party_mae : {party: (mean, lo, hi)},
        items_distribution : np.ndarray  (per-person n_items_asked)
        n_persons
    """
    if condition == "top1":
        n_asked  = np.array([r.n_items_asked_top1 for r in results], dtype=float)
        reached  = np.array([r.top1_reached        for r in results], dtype=float)
        mae_dicts = [r.obs_party_mae_top1 for r in results]
    else:
        n_asked  = np.array([r.n_items_asked_top3 for r in results], dtype=float)
        reached  = np.array([r.top3_reached        for r in results], dtype=float)
        mae_dicts = [r.obs_party_mae_top3 for r in results]

    out = {
        "mean_items_asked":   _ci(n_asked,  n=n_boot, seed=seed),
        "reached_fraction":   _ci(reached,  n=n_boot, seed=seed),
        "items_distribution": n_asked,
        "n_persons": len(results),
    }

    # Per-party MAE
    valid_mae = [d for d in mae_dicts if d is not None]
    parties   = list(valid_mae[0].keys()) if valid_mae else []
    mae_by_party = {}
    for party in parties:
        vals = [d.get(party, np.nan) for d in valid_mae]
        mae_by_party[party] = _ci(
            [v for v in vals if not np.isnan(v)], n=n_boot, seed=seed)
    out["party_mae"] = mae_by_party
    return out


# ---------------------------------------------------------------------------
# Party breakdown
# ---------------------------------------------------------------------------

def party_breakdown(
    results: list[PersonResult],
    condition: Literal["50pct_obs", "50pct_pred", "top1", "top3"] = "50pct_obs",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Per-party metrics, stratified by each person's true best-matching party.

    One row per party with columns:
        party, n_persons, pct_persons,
        top1_accuracy, top1_ci_lo, top1_ci_hi,
        top3_accuracy, top3_ci_lo, top3_ci_hi,
        mean_mae, mae_ci_lo, mae_ci_hi,
        mean_items_asked (adaptive only), items_ci_lo, items_ci_hi
    """
    rows = []
    # Group by true best party
    by_party: dict[str, list[PersonResult]] = {}
    for r in results:
        p = r.true_best_party
        by_party.setdefault(p, []).append(r)

    n_total = len(results)

    for party, group in sorted(by_party.items()):
        row: dict = {"party": party, "n_persons": len(group),
                     "pct_persons": len(group) / n_total * 100}

        if condition == "50pct_obs":
            top1_vals = [r.obs_top1_correct_50  for r in group
                         if r.obs_top1_correct_50 is not None]
            top3_vals = [r.obs_top3_correct_50  for r in group
                         if r.obs_top3_correct_50 is not None]
            mae_vals  = [r.obs_party_mae_50.get(party, np.nan)
                         for r in group if r.obs_party_mae_50]

        elif condition == "50pct_pred":
            top1_vals = [r.pred_top1_correct_50 for r in group
                         if r.pred_top1_correct_50 is not None]
            top3_vals = [r.pred_top3_correct_50 for r in group
                         if r.pred_top3_correct_50 is not None]
            mae_vals  = [r.pred_party_mae_50.get(party, np.nan)
                         for r in group if r.pred_party_mae_50]

        elif condition == "top1":
            top1_vals = [float(r.top1_reached) for r in group]
            top3_vals = []
            mae_vals  = [r.obs_party_mae_top1.get(party, np.nan)
                         for r in group if r.obs_party_mae_top1]
            items = [r.n_items_asked_top1 for r in group]
            m, lo, hi = _ci(items, n=n_boot, seed=seed)
            row.update({"mean_items_asked": m,
                        "items_ci_lo": lo, "items_ci_hi": hi})

        elif condition == "top3":
            top1_vals = []
            top3_vals = [float(r.top3_reached) for r in group]
            mae_vals  = [r.obs_party_mae_top3.get(party, np.nan)
                         for r in group if r.obs_party_mae_top3]
            items = [r.n_items_asked_top3 for r in group]
            m, lo, hi = _ci(items, n=n_boot, seed=seed)
            row.update({"mean_items_asked": m,
                        "items_ci_lo": lo, "items_ci_hi": hi})

        # Top-1
        if top1_vals:
            m, lo, hi = _ci(top1_vals, n=n_boot, seed=seed)
            row.update({"top1_accuracy": m,
                        "top1_ci_lo": lo, "top1_ci_hi": hi})

        # Top-3
        if top3_vals:
            m, lo, hi = _ci(top3_vals, n=n_boot, seed=seed)
            row.update({"top3_accuracy": m,
                        "top3_ci_lo": lo, "top3_ci_hi": hi})

        # MAE
        clean_mae = [v for v in mae_vals if not np.isnan(v)]
        if clean_mae:
            m, lo, hi = _ci(clean_mae, n=n_boot, seed=seed)
            row.update({"mean_mae": m, "mae_ci_lo": lo, "mae_ci_hi": hi})

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "n_persons", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flat DataFrame export
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list[PersonResult]) -> pd.DataFrame:
    """
    Convert results list to a long-form DataFrame.
    One row per person with all metrics as columns.
    """
    rows = []
    for r in results:
        row = {
            "person_id":           r.person_id,
            "true_best_party":     r.true_best_party,
            "n_items_total":       r.n_items_total,
            # 50% condition
            "n_items_asked_50":    r.n_items_asked_50,
            "obs_top1_50":         r.obs_top1_correct_50,
            "obs_top3_50":         r.obs_top3_correct_50,
            "pred_top1_50":        r.pred_top1_correct_50,
            "pred_top3_50":        r.pred_top3_correct_50,
            # Adaptive top-1
            "n_items_top1":        r.n_items_asked_top1,
            "top1_reached":        r.top1_reached,
            # Adaptive top-3
            "n_items_top3":        r.n_items_asked_top3,
            "top3_reached":        r.top3_reached,
        }
        # Per-party MAE at 50% (obs)
        if r.obs_party_mae_50:
            for party, mae in r.obs_party_mae_50.items():
                row[f"obs_mae_50_{party}"] = mae
        # Per-party MAE at 50% (pred)
        if r.pred_party_mae_50:
            for party, mae in r.pred_party_mae_50.items():
                row[f"pred_mae_50_{party}"] = mae
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sim_v2', Path(__file__).parent / 'simulation.py')
    sim_v2 = importlib.util.module_from_spec(spec)
    sys.modules['sim_v2'] = sim_v2
    spec.loader.exec_module(sim_v2)

    import numpy as np, pandas as pd
    from party_scoring import compute_match_scores_matrix

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
        columns=item_cols, index=person_ids)
    party_pos = pd.DataFrame(
        rng.integers(0, 5, size=(n_parties, n_items)),
        index=party_names, columns=item_cols)
    party_scores = compute_match_scores_matrix(
        responses, party_pos, input_range="1-5")

    results, _, _ = sim_v2.run_simulation(
        responses, party_scores, party_pos,
        predictor_name="ridge", strategy="entropy",
        seed=42, verbose=False)

    s50  = summarise_50pct(results)
    sad1 = summarise_adaptive(results, condition="top1")
    sad3 = summarise_adaptive(results, condition="top3")

    print("=== 50% summary ===")
    print(f"  obs  top1: {s50['obs_top1'][0]:.1%}  CI [{s50['obs_top1'][1]:.1%}, {s50['obs_top1'][2]:.1%}]")
    print(f"  pred top1: {s50['pred_top1'][0]:.1%}  CI [{s50['pred_top1'][1]:.1%}, {s50['pred_top1'][2]:.1%}]")
    print(f"  obs  MAE (party A): {s50['obs_party_mae']['A'][0]:.3f}")

    print("=== Adaptive top-1 ===")
    print(f"  mean items: {sad1['mean_items_asked'][0]:.1f}  CI [{sad1['mean_items_asked'][1]:.1f}, {sad1['mean_items_asked'][2]:.1f}]")
    print(f"  reached:    {sad1['reached_fraction'][0]:.1%}")

    print("=== Party breakdown (50pct_obs) ===")
    pb = party_breakdown(results, condition="50pct_obs")
    print(pb[["party","n_persons","top1_accuracy","top1_ci_lo","top1_ci_hi"]].to_string(index=False))

    df = results_to_dataframe(results)
    print(f"\nFlat DataFrame: {df.shape}, cols: {list(df.columns[:8])}")
    print("evaluation.py OK.")
