"""
adaptive_v2/src/make_plots.py
==============================
Load saved per-person CSVs and regenerate all plots without re-running experiments.

Usage
-----
    python adaptive_v2/src/make_plots.py \
        --results-dir adaptive_v2/results/ \
        --out-dir     adaptive_v2/results/plots/ \
        [--dpi 300] [--no-pdf]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent.parent / "src"
for _p in [str(_SRC), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sim_mod   = _load_mod("sim_v2",   _THIS / "simulation.py")
_eval_mod  = _load_mod("eval_v2",  _THIS / "evaluation.py")
_plots_mod = _load_mod("plots_v2", _THIS / "plots.py")

PersonResult  = _sim_mod.PersonResult
PlotConfig    = _plots_mod.PlotConfig
make_all_plots = _plots_mod.make_all_plots


# ---------------------------------------------------------------------------
# Reconstruct PersonResult from a per-person CSV row
# ---------------------------------------------------------------------------

def _row_to_result(row: pd.Series, parties: list[str]) -> PersonResult:
    """Reconstruct a PersonResult from a saved CSV row."""
    r = PersonResult(
        person_id        = str(row["person_id"]),
        true_best_party  = str(row["true_best_party"]),
        n_items_total    = int(row["n_items_total"]),
        # 50% condition
        n_items_asked_50      = int(row["n_items_asked_50"]),
        obs_top1_correct_50   = bool(row["obs_top1_50"])  if pd.notna(row.get("obs_top1_50"))  else None,
        obs_top3_correct_50   = bool(row["obs_top3_50"])  if pd.notna(row.get("obs_top3_50"))  else None,
        pred_top1_correct_50  = bool(row["pred_top1_50"]) if pd.notna(row.get("pred_top1_50")) else None,
        pred_top3_correct_50  = bool(row["pred_top3_50"]) if pd.notna(row.get("pred_top3_50")) else None,
        # Adaptive top-1
        n_items_asked_top1 = int(row["n_items_top1"]),
        top1_reached       = bool(row["top1_reached"]),
        # Adaptive top-3
        n_items_asked_top3 = int(row["n_items_top3"]),
        top3_reached       = bool(row["top3_reached"]),
        # True party scores — not stored in CSV, reconstructed as empty
        # (only used for ground-truth ranking which is captured in true_best_party)
        true_party_scores    = {},
        true_top3_parties    = [],
    )

    # Per-party MAE at 50% (obs and pred)
    obs_mae_50  = {}
    pred_mae_50 = {}
    obs_mae_top1 = {}
    obs_mae_top3 = {}
    for party in parties:
        k_obs  = f"obs_mae_50_{party}"
        k_pred = f"pred_mae_50_{party}"
        k_top1 = f"obs_mae_top1_{party}"
        k_top3 = f"obs_mae_top3_{party}"
        if k_obs  in row and pd.notna(row[k_obs]):
            obs_mae_50[party]  = float(row[k_obs])
        if k_pred in row and pd.notna(row[k_pred]):
            pred_mae_50[party] = float(row[k_pred])
        if k_top1 in row and pd.notna(row[k_top1]):
            obs_mae_top1[party] = float(row[k_top1])
        if k_top3 in row and pd.notna(row[k_top3]):
            obs_mae_top3[party] = float(row[k_top3])

    r.obs_party_mae_50   = obs_mae_50  or None
    r.pred_party_mae_50  = pred_mae_50 or None
    r.obs_party_mae_top1 = obs_mae_top1 or None
    r.obs_party_mae_top3 = obs_mae_top3 or None

    return r


def load_results(results_dir: Path) -> dict[tuple, list[PersonResult]]:
    """
    Load all per-person CSVs from results_dir/per_person/ and return
    a dict keyed by (predictor, strategy).
    """
    per_person_dir = results_dir / "per_person"
    if not per_person_dir.exists():
        raise FileNotFoundError(f"No per_person/ directory found in {results_dir}")

    csv_files = sorted(per_person_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {per_person_dir}")

    all_results: dict[tuple, list[PersonResult]] = {}

    for csv_path in csv_files:
        # Filename format: {predictor}__{strategy}.csv
        parts = csv_path.stem.split("__")
        if len(parts) != 2:
            print(f"  Skipping unexpected filename: {csv_path.name}")
            continue
        predictor, strategy = parts[0], parts[1]

        df = pd.read_csv(csv_path)

        # Infer party names from MAE columns
        parties = sorted(set(
            col.replace("obs_mae_50_", "")
            for col in df.columns
            if col.startswith("obs_mae_50_")
        ))

        results = [_row_to_result(row, parties) for _, row in df.iterrows()]
        all_results[(predictor, strategy)] = results
        print(f"  Loaded {len(results):>5} persons  ←  {csv_path.name}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Regenerate plots from saved results.")
    p.add_argument("--results-dir", default="adaptive_v2/results/",
                   help="Directory containing per_person/ subfolder.")
    p.add_argument("--out-dir",     default=None,
                   help="Output directory for plots. Defaults to results-dir/plots/.")
    p.add_argument("--dpi",  type=int, default=300)
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir \
                  else results_dir / "plots"

    print(f"Loading results from {results_dir / 'per_person'} ...")
    all_results = load_results(results_dir)
    print(f"Loaded {len(all_results)} combinations.\n")

    cfg = PlotConfig(
        out_dir=str(out_dir),
        dpi=args.dpi,
        fmt=["png"] if args.no_pdf else ["png", "pdf"],
        show=False,
    )
    make_all_plots(all_results, cfg)


if __name__ == "__main__":
    main()
