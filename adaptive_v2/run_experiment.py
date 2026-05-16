"""
adaptive_v2/src/run_experiment.py
==================================
Entry point for adaptive_v2 experiments.

Runs all combinations of:
  predictors : ridge, naive_bayes, random_forest
  strategies : entropy, uncertainty, random

Saves results and generates all plots.

Usage
-----
    python adaptive_v2/src/run_experiment.py \
        --responses   data/saar_item_ready_for_experiments.csv \
        --party-scores data/saar_party_ready_for_experiments.csv \
        --party-positions data/Saarbru_prepared_Party.csv \
        --out-dir adaptive_v2/results/ \
        [--predictors ridge naive_bayes] \
        [--strategies entropy uncertainty random] \
        [--test-fraction 0.2] \
        [--seed 42] \
        [--no-plots]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS  = Path(__file__).resolve().parent
_SRC   = _THIS.parent.parent / "src"
for _p in [str(_SRC), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import load_all

# Load simulation from adaptive_v2/src explicitly to avoid collision with v1
import importlib.util as _ilu
_sim_spec = _ilu.spec_from_file_location("sim_v2", _THIS / "simulation.py")
_sim_mod  = _ilu.module_from_spec(_sim_spec)
import sys as _sys
_sys.modules["sim_v2"] = _sim_mod
_sim_spec.loader.exec_module(_sim_mod)
run_simulation = _sim_mod.run_simulation
PersonResult   = _sim_mod.PersonResult

_eval_spec = _ilu.spec_from_file_location("eval_v2", _THIS / "evaluation.py")
_eval_mod  = _ilu.module_from_spec(_eval_spec)
_sys.modules["eval_v2"] = _eval_mod
_eval_spec.loader.exec_module(_eval_mod)
summarise_50pct    = _eval_mod.summarise_50pct
summarise_adaptive = _eval_mod.summarise_adaptive
party_breakdown    = _eval_mod.party_breakdown
results_to_dataframe = _eval_mod.results_to_dataframe

_plots_spec = _ilu.spec_from_file_location("plots_v2", _THIS / "plots.py")
_plots_mod  = _ilu.module_from_spec(_plots_spec)
_sys.modules["plots_v2"] = _plots_mod
_plots_spec.loader.exec_module(_plots_mod)
PlotConfig    = _plots_mod.PlotConfig
make_all_plots = _plots_mod.make_all_plots


# ---------------------------------------------------------------------------
# Predictor kwargs and train caps
# ---------------------------------------------------------------------------

PREDICTOR_KWARGS = {
    "ridge":       {},
    "naive_bayes": {},
}

MAX_TRAIN_PERSONS = {
    "ridge":       1500,
    "naive_bayes": 1500,
}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _ci_to_dict(triple):
    m, lo, hi = triple
    return {"mean": m, "ci_lo": lo, "ci_hi": hi}


def _summary_to_dict(s50, sad1, sad3, label):
    d = {
        "label": label,
        "n_persons": s50["n_persons"],
        # 50% obs-only
        "obs_top1_50":  _ci_to_dict(s50["obs_top1"]),
        "obs_top3_50":  _ci_to_dict(s50["obs_top3"]),
        # 50% obs+pred
        "pred_top1_50": _ci_to_dict(s50["pred_top1"]),
        "pred_top3_50": _ci_to_dict(s50["pred_top3"]),
        # adaptive top-1
        "items_top1":       _ci_to_dict(sad1["mean_items_asked"]),
        "reached_top1":     _ci_to_dict(sad1["reached_fraction"]),
        # adaptive top-3
        "items_top3":       _ci_to_dict(sad3["mean_items_asked"]),
        "reached_top3":     _ci_to_dict(sad3["reached_fraction"]),
    }
    # Per-party MAE
    for party, triple in s50["obs_party_mae"].items():
        d[f"obs_mae_50_{party}"]  = _ci_to_dict(triple)
    for party, triple in s50["pred_party_mae"].items():
        d[f"pred_mae_50_{party}"] = _ci_to_dict(triple)
    for party, triple in sad1["party_mae"].items():
        d[f"mae_top1_{party}"]    = _ci_to_dict(triple)
    for party, triple in sad3["party_mae"].items():
        d[f"mae_top3_{party}"]    = _ci_to_dict(triple)
    return d


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    responses: pd.DataFrame,
    party_scores_gt: pd.DataFrame,
    party_positions: pd.DataFrame,
    predictors: list[str],
    strategies: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
    out_dir: Path = Path("adaptive_v2/results/"),
    run_plots: bool = True,
    verbose: bool = True,
) -> dict[tuple, list[PersonResult]]:

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[tuple, list[PersonResult]] = {}
    summary_rows = []

    combos = [(p, s) for p in predictors for s in strategies]
    n = len(combos)

    if verbose:
        print(f"\nRunning {n} combinations "
              f"({len(predictors)} predictors × {len(strategies)} strategies)\n")

    for i, (pred, strat) in enumerate(combos, 1):
        label = f"{pred}__{strat}"
        if verbose:
            print(f"[{i}/{n}] {label}")

        t0 = time.time()
        try:
            results, _, _ = run_simulation(
                responses, party_scores_gt, party_positions,
                predictor_name=pred,
                strategy=strat,
                predictor_kwargs=PREDICTOR_KWARGS.get(pred, {}),
                max_train_persons=MAX_TRAIN_PERSONS.get(pred, 1500),
                test_fraction=test_fraction,
                seed=seed,
                verbose=verbose,
            )
            elapsed = time.time() - t0

            # Evaluate
            s50  = summarise_50pct(results)
            sad1 = summarise_adaptive(results, "top1")
            sad3 = summarise_adaptive(results, "top3")

            # Store
            all_results[(pred, strat)] = results

            # Save per-person CSV
            _save_csv(
                results_to_dataframe(results),
                out_dir / "per_person" / f"{label}.csv",
            )

            # Save party breakdowns
            for cond in ["50pct_obs", "50pct_pred", "top1", "top3"]:
                pb = party_breakdown(results, cond)
                _save_csv(pb, out_dir / "party_breakdown" / f"{label}__{cond}.csv")

            # Save summary JSON
            sd = _summary_to_dict(s50, sad1, sad3, label)
            sd["elapsed_seconds"] = elapsed
            _save_json(sd, out_dir / "reports" / f"{label}.json")
            summary_rows.append({
                "label":       label,
                "predictor":   pred,
                "strategy":    strat,
                "obs_top1_50": s50["obs_top1"][0],
                "pred_top1_50": s50["pred_top1"][0],
                "obs_top3_50": s50["obs_top3"][0],
                "pred_top3_50": s50["pred_top3"][0],
                "mean_items_top1": sad1["mean_items_asked"][0],
                "mean_items_top3": sad3["mean_items_asked"][0],
                "reached_top1": sad1["reached_fraction"][0],
                "reached_top3": sad3["reached_fraction"][0],
                "elapsed_s":   elapsed,
                "status":      "ok",
            })

            if verbose:
                print(f"  → obs_top1@50%={s50['obs_top1'][0]:.1%}  "
                      f"pred_top1@50%={s50['pred_top1'][0]:.1%}  "
                      f"items_top1={sad1['mean_items_asked'][0]:.1f}  "
                      f"({elapsed:.0f}s)\n")

        except Exception as e:
            warnings.warn(f"  {label} FAILED: {e}", RuntimeWarning)
            summary_rows.append({"label": label, "predictor": pred,
                                  "strategy": strat, "status": f"error: {e}"})

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    _save_csv(summary_df, out_dir / "summary.csv")
    if verbose:
        print(f"✓ Summary saved → {out_dir / 'summary.csv'}")
        cols = ["label", "obs_top1_50", "pred_top1_50",
                "mean_items_top1", "reached_top1", "status"]
        available = [c for c in cols if c in summary_df.columns]
        print(summary_df[available].to_string(index=False))

    # Generate plots
    if run_plots and all_results:
        cfg = PlotConfig(
            out_dir=str(out_dir / "plots"),
            dpi=300,
            fmt=["png", "pdf"],
        )
        make_all_plots(all_results, cfg)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Run adaptive_v2 experiments.")
    p.add_argument("--responses",       required=True)
    p.add_argument("--party-scores",    required=True)
    p.add_argument("--party-positions", required=True)
    p.add_argument("--out-dir",         default="adaptive_v2/results/")
    p.add_argument("--predictors", nargs="+",
                   default=["ridge", "naive_bayes"])
    p.add_argument("--strategies", nargs="+",
                   default=["entropy", "uncertainty", "random",
                            "variance_reduction", "top1_change", "top3_change"])
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no-plots",      action="store_true")
    p.add_argument("--quick",         action="store_true",
                   help="Ridge only, entropy only, for testing.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.quick:
        args.predictors = ["ridge"]
        args.strategies = ["entropy"]
        print("Quick mode: ridge + entropy only.")

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
        predictors=args.predictors,
        strategies=args.strategies,
        test_fraction=args.test_fraction,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        run_plots=not args.no_plots,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    import numpy as np, pandas as pd, warnings
    sys.path.insert(0, str(_SRC))
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

    out_dir = Path("/tmp/adaptive_v2_smoke")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_results = run_all(
            responses=responses,
            party_scores_gt=party_scores,
            party_positions=party_pos,
            predictors=["ridge"],
            strategies=["entropy", "random", "variance_reduction", "top1_change", "top3_change"],
            test_fraction=0.2,
            seed=42,
            out_dir=out_dir,
            run_plots=True,
            verbose=True,
        )

    print("\n=== Output files ===")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"  ✓  {f.relative_to(out_dir)}  ({f.stat().st_size/1024:.1f} KB)")

    print("\nrun_experiment.py OK.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _smoke_test()
