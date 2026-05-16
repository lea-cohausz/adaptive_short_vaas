"""
load_data.py
============
Data loading, validation, and preparation for the Saarbrücken adaptive
questionnaire experiment.

Dataset summary (as inspected)
-------------------------------
saar_item_ready_for_experiments.csv
    5224 persons × 38 columns (id + 37 items)
    Items: bridge_03–bridge_2b, local_01–local_20  (float, 0–4, no NaNs)

saar_party_ready_for_experiments.csv
    5224 persons × 13 columns (id + 6×{party_res, party_acc})
    Parties: CDU, DIE LINKE, Die Grünen, FDP, SPD, bunt.saar
    Acc range: ~17–97  (already computed ground-truth match scores)

Saarbru_prepared_Party.csv
    6 parties × 42 columns (party + 41 item positions, values 0–4)
    Has 4 extra item columns not present in responses:
        bridge_14, bridge_22, local_13, local_14  → dropped automatically

Pipeline value range convention
--------------------------------
Raw data is 0–4.
The pipeline (predictors, item selector, stopping rules) works in 1–5.
Conversion: responses += 1  on load.
party_scoring.py handles the reverse shift internally when computing scores.

Public API
----------
    load_all(item_path, party_score_path, party_pos_path)
        -> (responses_df, party_scores_df, party_positions_df)

    validate_datasets(responses, party_scores, party_positions)
        -> ValidationReport  (prints issues and raises on critical ones)

    quick_summary(responses, party_scores, party_positions)
        -> prints a human-readable summary
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults — file paths relative to project root
# ---------------------------------------------------------------------------

DEFAULT_ITEM_PATH       = "data/saar_item_ready_for_experiments.csv"
DEFAULT_PARTY_SCORE_PATH = "data/saar_party_ready_for_experiments.csv"
DEFAULT_PARTY_POS_PATH  = "data/Saarbru_prepared_Party.csv"

ID_COL = "id"
PARTY_COL = "party"
PARTIES = ["CDU", "DIE LINKE", "Die Grünen", "FDP", "SPD", "bunt.saar"]


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Collects all findings from dataset validation."""
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info:     list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return len(self.errors) == 0

    def print(self) -> None:
        print("\n=== Dataset Validation Report ===")
        for msg in self.info:
            print(f"  ℹ  {msg}")
        for msg in self.warnings:
            print(f"  ⚠  {msg}")
        for msg in self.errors:
            print(f"  ✗  {msg}")
        if self.ok():
            print("  ✓  All checks passed.\n")
        else:
            print(f"\n  {len(self.errors)} error(s) found.\n")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_responses(path: str) -> pd.DataFrame:
    """
    Load item response CSV.
    - Sets 'id' as index.
    - Converts 0–4 values to 1–5 (pipeline convention).
    - Casts to float.
    """
    df = pd.read_csv(path)
    df = df.set_index(ID_COL)
    df = df.astype(float)
    df = df + 1          # 0–4  →  1–5
    return df


def _load_party_scores(path: str) -> pd.DataFrame:
    """
    Load party scores CSV.
    - Sets 'id' as index.
    - Keeps only the _acc columns (drops _res).
    - Renames PartyName_acc  →  PartyName for clean column names.
    """
    df = pd.read_csv(path)
    df = df.set_index(ID_COL)
    acc_cols = [c for c in df.columns if c.endswith("_acc")]
    df = df[acc_cols].copy()
    df.columns = [c.replace("_acc", "") for c in acc_cols]
    df = df.astype(float)
    return df


def _load_party_positions(path: str) -> pd.DataFrame:
    """
    Load party positions CSV.
    - Sets 'party' as index.
    - Keeps only item columns (drops any non-item extras).
    - Values remain 0–4 as required by party_scoring.py.
    """
    df = pd.read_csv(path)
    df = df.set_index(PARTY_COL)
    df = df.astype(float)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_datasets(
    responses: pd.DataFrame,
    party_scores: pd.DataFrame,
    party_positions: pd.DataFrame,
    raise_on_error: bool = True,
) -> ValidationReport:
    """
    Validate alignment and integrity of all three datasets.

    Checks
    ------
    - ID alignment between responses and party_scores
    - Item column alignment between responses and party_positions
    - Value ranges (responses 1–5, party_scores 0–100, positions 0–4)
    - Missing values
    - Party name consistency between party_scores and party_positions
    - Minimum dataset size for CV

    Parameters
    ----------
    responses : pd.DataFrame       Shape (n_persons, n_items), values 1–5.
    party_scores : pd.DataFrame    Shape (n_persons, n_parties), values 0–100.
    party_positions : pd.DataFrame Shape (n_parties, n_items), values 0–4.
    raise_on_error : bool          Raise ValueError if any errors found.

    Returns
    -------
    ValidationReport
    """
    r = ValidationReport()

    # ── ID alignment ──────────────────────────────────────────────────────
    common_ids = responses.index.intersection(party_scores.index)
    only_resp  = len(responses.index) - len(common_ids)
    only_party = len(party_scores.index) - len(common_ids)

    if len(common_ids) == 0:
        r.errors.append("No common IDs between responses and party_scores.")
    else:
        r.info.append(
            f"Person IDs: {len(common_ids)} in common, "
            f"{only_resp} only in responses, {only_party} only in party_scores."
        )
        if only_resp > 0:
            r.warnings.append(
                f"{only_resp} persons in responses have no party scores — "
                "they will be dropped."
            )
        if only_party > 0:
            r.warnings.append(
                f"{only_party} persons in party_scores have no item responses — "
                "they will be dropped."
            )

    # ── Item alignment ────────────────────────────────────────────────────
    resp_items = set(responses.columns)
    pos_items  = set(party_positions.columns)
    common_items = resp_items & pos_items
    only_in_resp = resp_items - pos_items
    only_in_pos  = pos_items - resp_items

    if not common_items:
        r.errors.append(
            "No overlapping item columns between responses and party_positions."
        )
    else:
        r.info.append(
            f"Item columns: {len(common_items)} shared, "
            f"{len(only_in_resp)} only in responses, "
            f"{len(only_in_pos)} only in party_positions."
        )
    if only_in_pos:
        r.warnings.append(
            f"Party positions has {len(only_in_pos)} extra item(s) not in "
            f"responses (will be ignored): {sorted(only_in_pos)}"
        )
    if only_in_resp:
        r.warnings.append(
            f"Responses has {len(only_in_resp)} item(s) not in party_positions "
            f"(will be ignored for scoring): {sorted(only_in_resp)}"
        )

    # ── Party name consistency ─────────────────────────────────────────────
    score_parties = set(party_scores.columns)
    pos_parties   = set(party_positions.index)
    if score_parties != pos_parties:
        only_scores = score_parties - pos_parties
        only_pos    = pos_parties - score_parties
        if only_scores:
            r.warnings.append(
                f"Parties in scores but not in positions: {sorted(only_scores)}"
            )
        if only_pos:
            r.warnings.append(
                f"Parties in positions but not in scores: {sorted(only_pos)}"
            )
    else:
        r.info.append(f"Party names consistent: {sorted(score_parties)}")

    # ── Value ranges ──────────────────────────────────────────────────────
    resp_min = responses.min().min()
    resp_max = responses.max().max()
    if resp_min < 1 or resp_max > 5:
        r.errors.append(
            f"Response values out of expected 1–5 range: "
            f"found [{resp_min}, {resp_max}]."
        )
    else:
        r.info.append(f"Response values: range [{resp_min:.0f}, {resp_max:.0f}] ✓")

    acc_min = party_scores.min().min()
    acc_max = party_scores.max().max()
    if acc_min < 0 or acc_max > 100:
        r.warnings.append(
            f"Party acc scores out of expected 0–100 range: "
            f"found [{acc_min:.2f}, {acc_max:.2f}]."
        )
    else:
        r.info.append(
            f"Party acc scores: range [{acc_min:.2f}, {acc_max:.2f}] ✓"
        )

    pos_min = party_positions.min().min()
    pos_max = party_positions.max().max()
    if pos_min < 0 or pos_max > 4:
        r.errors.append(
            f"Party position values out of expected 0–4 range: "
            f"found [{pos_min}, {pos_max}]."
        )
    else:
        r.info.append(
            f"Party position values: range [{pos_min:.0f}, {pos_max:.0f}] ✓"
        )

    # ── Missing values ────────────────────────────────────────────────────
    resp_na = int(responses.isnull().sum().sum())
    scores_na = int(party_scores.isnull().sum().sum())
    pos_na = int(party_positions.isnull().sum().sum())

    if resp_na > 0:
        r.warnings.append(
            f"Responses has {resp_na} missing value(s). "
            "Consider imputation or row removal."
        )
    else:
        r.info.append("No missing values in responses ✓")

    if scores_na > 0:
        r.warnings.append(f"Party scores has {scores_na} missing value(s).")
    if pos_na > 0:
        r.errors.append(f"Party positions has {pos_na} missing value(s).")

    # ── Minimum size for CV ───────────────────────────────────────────────
    if len(common_ids) < 10:
        r.errors.append(
            f"Only {len(common_ids)} persons — too few for cross-validation."
        )

    if raise_on_error and not r.ok():
        raise ValueError(
            f"Dataset validation failed with {len(r.errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in r.errors)
        )

    return r


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all(
    item_path: str = DEFAULT_ITEM_PATH,
    party_score_path: str = DEFAULT_PARTY_SCORE_PATH,
    party_pos_path: str = DEFAULT_PARTY_POS_PATH,
    validate: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, align, and validate all three datasets.

    Parameters
    ----------
    item_path : str
        Path to item responses CSV.
    party_score_path : str
        Path to ground-truth party scores CSV.
    party_pos_path : str
        Path to party positions CSV.
    validate : bool
        Run validation checks (recommended).
    verbose : bool
        Print summary after loading.

    Returns
    -------
    tuple of:
        responses       pd.DataFrame  (n_persons, n_items)   values 1–5
        party_scores    pd.DataFrame  (n_persons, n_parties) values 0–100
        party_positions pd.DataFrame  (n_parties, n_items)   values 0–4
    """
    if verbose:
        print("Loading datasets...")

    responses       = _load_responses(item_path)
    party_scores    = _load_party_scores(party_score_path)
    party_positions = _load_party_positions(party_pos_path)

    # ── Align IDs ─────────────────────────────────────────────────────────
    common_ids = responses.index.intersection(party_scores.index)
    responses    = responses.loc[common_ids]
    party_scores = party_scores.loc[common_ids]

    # ── Align items: keep only columns present in BOTH responses and positions
    common_items    = responses.columns.intersection(party_positions.columns)
    responses       = responses[common_items]
    party_positions = party_positions[common_items]

    # ── Validate ──────────────────────────────────────────────────────────
    if validate:
        report = validate_datasets(
            responses, party_scores, party_positions,
            raise_on_error=True,
        )
        report.print()

    if verbose:
        quick_summary(responses, party_scores, party_positions)

    return responses, party_scores, party_positions


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def quick_summary(
    responses: pd.DataFrame,
    party_scores: pd.DataFrame,
    party_positions: pd.DataFrame,
) -> None:
    """Print a human-readable summary of the three loaded datasets."""
    print("\n" + "═" * 52)
    print("  DATASET SUMMARY")
    print("═" * 52)
    print(f"  Persons            : {len(responses)}")
    print(f"  Items              : {len(responses.columns)}")
    print(f"  Parties            : {len(party_positions.index)}")
    print(f"  Party names        : {list(party_positions.index)}")
    print(f"  Response range     : {responses.min().min():.0f}–{responses.max().max():.0f} (pipeline: 1–5)")
    print(f"  Missing values     : {int(responses.isnull().sum().sum())}")
    print()
    print("  Party match score distribution (acc, 0–100):")
    for party in party_scores.columns:
        col = party_scores[party]
        print(f"    {party:<15s}  "
              f"mean={col.mean():.1f}  "
              f"std={col.std():.1f}  "
              f"min={col.min():.1f}  "
              f"max={col.max():.1f}")
    print()
    print("  Best-matching party distribution:")
    best = party_scores.idxmax(axis=1).value_counts()
    for party, count in best.items():
        pct = count / len(party_scores) * 100
        print(f"    {party:<15s}  n={count:>5d}  ({pct:.1f}%)")
    print("═" * 52 + "\n")


# ---------------------------------------------------------------------------
# Smoke test / real data check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Accept paths as positional CLI args, or fall back to data/ next to src/
    # Usage:
    #   python load_data.py                              # uses data/ defaults
    #   python load_data.py item.csv scores.csv pos.csv  # explicit paths
    if len(sys.argv) == 4:
        item_path        = sys.argv[1]
        party_score_path = sys.argv[2]
        party_pos_path   = sys.argv[3]
    else:
        # Default: data/ folder relative to the project root (one level up from src/)
        src_dir  = Path(__file__).parent
        data_dir = src_dir.parent / "data"
        item_path        = str(data_dir / "saar_item_ready_for_experiments.csv")
        party_score_path = str(data_dir / "saar_party_ready_for_experiments.csv")
        party_pos_path   = str(data_dir / "Saarbru_prepared_Party.csv")
        print(f"Using default data paths from: {data_dir}")

    responses, party_scores, party_positions = load_all(
        item_path=item_path,
        party_score_path=party_score_path,
        party_pos_path=party_pos_path,
        validate=True,
        verbose=True,
    )

    print("Shapes after loading and alignment:")
    print(f"  responses       : {responses.shape}")
    print(f"  party_scores    : {party_scores.shape}")
    print(f"  party_positions : {party_positions.shape}")
    print(f"\nItem columns : {list(responses.columns)}")
    print(f"\nFirst 3 response rows (1–5 range):")
    print(responses.head(3).to_string())
    print(f"\nFirst 3 party score rows:")
    print(party_scores.head(3).round(2).to_string())

    print("\nload_data.py OK.")
