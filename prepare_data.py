"""
prepare_data.py
===============
Full data preparation pipeline for the adaptive questionnaire experiments.

Runs three steps in order:
  1. Reshape raw voter responses and party positions (data_preparation.py logic)
  2. Compute party match scores for each voter (Matching.ipynb logic)
  3. Split the matched data into the three files expected by the experiment
     pipeline (Prepare_for_Experiments.ipynb logic)

Output files written to data/ (matching experiment pipeline defaults in load_data.py):
  data/saar_item_ready_for_experiments.csv   — voter item responses (id + 37 items)
  data/saar_party_ready_for_experiments.csv  — voter party match scores (id + 6×{res,acc})
  data/Saarbru_prepared_Party.csv            — party positions (party + item columns)

  data/halle_item_ready_for_experiments.csv  — Halle voter item responses
  data/halle_party_ready_for_experiments.csv — Halle voter party match scores

Usage
-----
    python prepare_data.py \
        --saar-responses   Data/Saarbrücken0503.csv \
        --halle-responses  Data/Halle0503.csv \
        --party-positions  Data/all_partyposthesen.csv \
        --out-dir          data/

All arguments are optional; the defaults above are used if not provided.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Reshape raw data
# ---------------------------------------------------------------------------

def prepare_voter_responses(path: str | Path) -> pd.DataFrame:
    """
    Read a raw voter CSV and pivot from long to wide format.
    Divides positions by 25 to convert from the 0–100 scale to 0–4.
    Returns a wide DataFrame indexed by voteID with one column per item.
    """
    data = pd.read_csv(path)
    pivoted = (
        data[["voteID", "these_id", "voterpos"]]
        .pivot(index="voteID", columns="these_id", values="voterpos")
    )
    pivoted = pivoted.dropna()
    pivoted = pivoted // 25
    return pivoted


def prepare_party_positions(path: str | Path, city_name: str) -> pd.DataFrame:
    """
    Read the combined party positions CSV, filter to one city, pivot to wide,
    and divide by 25 to convert to 0–4 scale.
    Returns a wide DataFrame with a 'party' column and one column per item.
    """
    data = pd.read_csv(path, sep=";")
    data = data[data["gmd_name"] == city_name]
    pivoted = (
        data[["party", "these_id", "partypos"]]
        .pivot(index="party", columns="these_id", values="partypos")
        .reset_index()
    )
    pivoted = pivoted.dropna()
    numeric_cols = pivoted.select_dtypes(include="number").columns
    pivoted[numeric_cols] = pivoted[numeric_cols] // 25
    return pivoted


# ---------------------------------------------------------------------------
# Step 2 — Compute match scores  (Matching.ipynb logic)
# ---------------------------------------------------------------------------

def _transform_value(x: float) -> float:
    """
    Map a combined voter+party position sum to a match-score contribution.
    Implements the voto-vote matching algorithm:
    https://github.com/voto-vote/.github/blob/main/docs/algorithm.md
    """
    if   round(x, 2) == 2:    return  1
    elif round(x, 2) == 3.12: return  0.5
    elif round(x, 2) == 4.11: return  0
    elif round(x, 2) == 4.24: return  0.75
    elif x == 5:               return -0.5
    elif round(x, 2) == 7.12: return -0.5
    elif round(x, 2) == 6.12: return -0.25
    elif round(x, 2) == 6.22: return  0.5
    elif round(x, 2) == 5.23: return  0.25
    elif round(x, 2) == 7.11: return  0.25
    elif round(x, 2) == 8.11: return  0
    elif round(x, 2) == 6:    return -1
    elif round(x, 2) == 8:    return  0.75
    elif round(x, 2) == 9:    return  0.5
    elif round(x, 2) == 10:   return  1
    else:                      return  0


def compute_match_scores(
    voter_df: pd.DataFrame,
    party_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute party match scores for every voter.

    Parameters
    ----------
    voter_df : wide DataFrame of voter responses (0–4 scale, no 'party' column)
    party_df : wide DataFrame of party positions with a 'party' column (0–4 scale)

    Returns
    -------
    voter_df copy with extra columns {party}_res and {party}_acc for each party.
    """
    # Shift to 1–5, then replace neutral/skip values per the matching algorithm
    voters  = voter_df.copy().apply(lambda col: col + 1 if col.dtype != "object" else col)
    parties = party_df.copy().apply(lambda col: col + 1 if col.dtype != "object" else col)
    voters  = voters.replace({3: 3.11, 2: 2.12})
    parties = parties.replace({3: 3.11, 2: 2.12})

    item_cols  = [c for c in voters.columns if c != "party"]
    n_items    = len(item_cols)
    party_list = parties["party"].tolist()

    result = voter_df.copy()  # accumulate _res / _acc columns onto the original scale

    for party in party_list:
        party_row = parties.loc[parties["party"] == party, item_cols].iloc[0]
        party_row.index = item_cols

        combined    = voters[item_cols].add(party_row)
        transformed = combined.map(_transform_value)

        res = transformed.sum(axis=1)
        acc = ((res + n_items) / (2 * n_items)) * 100

        result[f"{party}_res"] = res.values
        result[f"{party}_acc"] = acc.values

    return result


# ---------------------------------------------------------------------------
# Step 3 — Split into experiment-ready files
# ---------------------------------------------------------------------------

SAAR_ITEM_COLS = [
    "bridge_03", "bridge_04", "bridge_05", "bridge_06", "bridge_07",
    "bridge_08", "bridge_09", "bridge_10", "bridge_11", "bridge_12",
    "bridge_13", "bridge_15", "bridge_16", "bridge_17", "bridge_18",
    "bridge_19", "bridge_20", "bridge_21", "bridge_2b", "local_01",
    "local_02", "local_03", "local_04", "local_05", "local_06", "local_07",
    "local_08", "local_09", "local_10", "local_11", "local_12", "local_15",
    "local_16", "local_17", "local_18", "local_19", "local_20",
]

SAAR_PARTY_SCORE_COLS = [
    "CDU_res", "CDU_acc",
    "DIE LINKE_res", "DIE LINKE_acc",
    "Die Grünen_res", "Die Grünen_acc",
    "FDP_res", "FDP_acc",
    "SPD_res", "SPD_acc",
    "bunt.saar_res", "bunt.saar_acc",
]

HALLE_ITEM_COLS = [
    "bridge_01", "bridge_03", "bridge_04", "bridge_05", "bridge_06",
    "bridge_07", "bridge_08", "bridge_09", "bridge_10", "bridge_11",
    "bridge_12", "bridge_14", "bridge_15", "bridge_16", "bridge_17",
    "bridge_18", "bridge_19", "bridge_20", "bridge_21", "bridge_22",
    "bridge_2a", "local_01", "local_02", "local_03", "local_04", "local_05",
    "local_06", "local_07", "local_08", "local_09", "local_10", "local_11",
    "local_12", "local_13", "local_14", "local_15", "local_16",
]

HALLE_PARTY_SCORE_COLS = [
    "AfD_res", "AfD_acc",
    "CDU_res", "CDU_acc",
    "Die Linke Halle_res", "Die Linke Halle_acc",
    "Die PARTEI_res", "Die PARTEI_acc",
    "FDP_res", "FDP_acc",
    "Freie Wähler_res", "Freie Wähler_acc",
    "Grüne_res", "Grüne_acc",
    "MitBürger_res", "MitBürger_acc",
    "SPD_res", "SPD_acc",
    "Volt_res", "Volt_acc",
    "dieBasis_res", "dieBasis_acc",
]


def split_matched(
    matched: pd.DataFrame,
    item_cols: list[str],
    party_score_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add a 1-based 'id' column and split a matched DataFrame into
    (item_df, party_score_df), each starting with 'id'.
    """
    matched = matched.copy()
    matched["id"] = range(1, len(matched) + 1)

    item_df         = matched[["id"] + item_cols].reset_index(drop=True)
    party_score_df  = matched[["id"] + party_score_cols].reset_index(drop=True)
    return item_df, party_score_df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run(
    saar_responses_path:  str | Path,
    halle_responses_path: str | Path,
    party_positions_path: str | Path,
    out_dir:              str | Path = Path("data/"),
    verbose:              bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Saarbrücken ──────────────────────────────────────────────────────────
    log("Preparing Saarbrücken voter responses …")
    saar_voters = prepare_voter_responses(saar_responses_path)
    log(f"  {len(saar_voters)} voters × {len(saar_voters.columns)} items")

    log("Preparing Saarbrücken party positions …")
    saar_parties = prepare_party_positions(party_positions_path, "Saarbrücken")
    log(f"  {len(saar_parties)} parties × {len(saar_parties.columns) - 1} items")

    # Save party positions file (used directly by the experiment pipeline)
    saar_party_pos_path = out_dir / "Saarbru_prepared_Party.csv"
    saar_parties.to_csv(saar_party_pos_path, index=False)
    log(f"  → {saar_party_pos_path}")

    log("Computing Saarbrücken match scores …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        saar_matched = compute_match_scores(saar_voters, saar_parties)

    saar_item, saar_party_scores = split_matched(
        saar_matched, SAAR_ITEM_COLS, SAAR_PARTY_SCORE_COLS
    )
    saar_item_path         = out_dir / "saar_item_ready_for_experiments.csv"
    saar_party_scores_path = out_dir / "saar_party_ready_for_experiments.csv"
    saar_item.to_csv(saar_item_path, index=False)
    saar_party_scores.to_csv(saar_party_scores_path, index=False)
    log(f"  → {saar_item_path}")
    log(f"  → {saar_party_scores_path}")

    # ── Halle ─────────────────────────────────────────────────────────────────
    log("\nPreparing Halle voter responses …")
    halle_voters = prepare_voter_responses(halle_responses_path)
    log(f"  {len(halle_voters)} voters × {len(halle_voters.columns)} items")

    log("Preparing Halle party positions …")
    halle_parties = prepare_party_positions(party_positions_path, "Halle")
    log(f"  {len(halle_parties)} parties × {len(halle_parties.columns) - 1} items")

    log("Computing Halle match scores …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        halle_matched = compute_match_scores(halle_voters, halle_parties)

    halle_item, halle_party_scores = split_matched(
        halle_matched, HALLE_ITEM_COLS, HALLE_PARTY_SCORE_COLS
    )
    halle_item_path         = out_dir / "halle_item_ready_for_experiments.csv"
    halle_party_scores_path = out_dir / "halle_party_ready_for_experiments.csv"
    halle_item.to_csv(halle_item_path, index=False)
    halle_party_scores.to_csv(halle_party_scores_path, index=False)
    log(f"  → {halle_item_path}")
    log(f"  → {halle_party_scores_path}")

    log("\n✓ All files written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare data for adaptive questionnaire experiments.")
    p.add_argument("--saar-responses",   default="Data/Saarbrücken0503.csv",
                   help="Raw Saarbrücken voter responses CSV.")
    p.add_argument("--halle-responses",  default="Data/Halle0503.csv",
                   help="Raw Halle voter responses CSV.")
    p.add_argument("--party-positions",  default="Data/all_partyposthesen.csv",
                   help="Combined party positions CSV (all cities).")
    p.add_argument("--out-dir",          default="data/",
                   help="Output directory (default: data/).")
    args = p.parse_args()

    run(
        saar_responses_path  = args.saar_responses,
        halle_responses_path = args.halle_responses,
        party_positions_path = args.party_positions,
        out_dir              = args.out_dir,
    )


if __name__ == "__main__":
    main()
