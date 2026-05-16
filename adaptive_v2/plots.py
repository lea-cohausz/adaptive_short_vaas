"""
adaptive_v2/src/plots.py
========================
Publication-quality plots for adaptive_v2 results.

All plots include 95% bootstrap confidence intervals.

Plot families
-------------
1. plot_50pct_comparison()      Obs-only vs obs+pred — top-1, top-3, MAE
2. plot_party_breakdown()       Per-party top-1, top-3, MAE (any condition)
3. plot_items_asked()           Items needed distribution — adaptive top-1 & top-3
4. plot_party_items_asked()     Per-party items needed — adaptive top-1 & top-3
5. make_all_plots()             Generate all plots for a set of run results

Usage
-----
    from plots import make_all_plots, PlotConfig
    make_all_plots(all_results, cfg)

    where all_results is a dict:
        {(predictor, strategy): list[PersonResult]}
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import evaluation from same package
sys.path.insert(0, str(Path(__file__).parent))
from evaluation import (
    summarise_50pct, summarise_adaptive,
    party_breakdown, bootstrap_ci, _ci,
)


# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

PREDICTOR_LABELS = {
    "ridge":       "Ridge",
    "naive_bayes": "Naïve Bayes",
}

STRATEGY_LABELS = {
    "entropy":            "Entropy",
    "uncertainty":        "Uncertainty",
    "random":             "Random",
    "variance_reduction": "Variance Reduction",
    "top1_change":        "Top-1 Change",
    "top3_change":        "Top-3 Change",
}

STRATEGY_COLORS = {
    "entropy":            "#2166AC",
    "uncertainty":        "#4DAC26",
    "random":             "#D6604D",
    "variance_reduction": "#762A83",
    "top1_change":        "#E08214",
    "top3_change":        "#B35806",
}

CONDITION_LABELS = {
    "50pct_obs":  "Obs only (50%)",
    "50pct_pred": "Obs + predicted (50%)",
    "top1":       "Adaptive (top-1 stop)",
    "top3":       "Adaptive (top-3 stop)",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.8,
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "savefig.bbox":       "tight",
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
    })


@dataclass
class PlotConfig:
    out_dir: str = "results/plots/"
    dpi: int = 300
    fmt: list[str] = field(default_factory=lambda: ["png", "pdf"])
    show: bool = False


def _save(fig, name, cfg):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    for fmt in cfg.fmt:
        fig.savefig(Path(cfg.out_dir) / f"{name}.{fmt}", dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)


def _run_label(predictor, strategy):
    return (f"{PREDICTOR_LABELS.get(predictor, predictor)} / "
            f"{STRATEGY_LABELS.get(strategy, strategy)}")


def _errorbars(ax, x, mean, lo, hi, **kwargs):
    """Draw symmetric error bars from (lo, hi) around mean."""
    yerr = np.array([[mean - lo], [hi - mean]])
    ax.errorbar(x, mean, yerr=yerr, fmt="none", capsize=4,
                capthick=1.2, elinewidth=1.2, **kwargs)


# ---------------------------------------------------------------------------
# 1. 50% stopping: obs-only vs obs+pred
# ---------------------------------------------------------------------------

def plot_50pct_comparison(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    metric: str = "top1",   # 'top1', 'top3', or 'mae'
    party_for_mae: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing obs-only vs obs+pred at the 50% checkpoint
    across all (predictor, strategy) combinations.

    Parameters
    ----------
    metric : 'top1', 'top3', 'mae'
    party_for_mae : str  Party name for MAE plot. If None, shows mean over all parties.
    """
    _style()
    runs   = list(all_results.keys())
    labels = [_run_label(p, s) for p, s in runs]
    x      = np.arange(len(runs))
    w      = 0.35

    obs_vals, obs_lo, obs_hi   = [], [], []
    pred_vals, pred_lo, pred_hi = [], [], []

    for key in runs:
        s = summarise_50pct(all_results[key])
        if metric == "top1":
            obs_vals.append(s["obs_top1"][0]);  obs_lo.append(s["obs_top1"][1]);  obs_hi.append(s["obs_top1"][2])
            pred_vals.append(s["pred_top1"][0]); pred_lo.append(s["pred_top1"][1]); pred_hi.append(s["pred_top1"][2])
            ylabel, title_sfx = "Accuracy", "Top-1 accuracy"
        elif metric == "top3":
            obs_vals.append(s["obs_top3"][0]);  obs_lo.append(s["obs_top3"][1]);  obs_hi.append(s["obs_top3"][2])
            pred_vals.append(s["pred_top3"][0]); pred_lo.append(s["pred_top3"][1]); pred_hi.append(s["pred_top3"][2])
            ylabel, title_sfx = "Accuracy", "Top-3 accuracy (exact set)"
        elif metric == "mae":
            if party_for_mae:
                om = s["obs_party_mae"].get(party_for_mae, (np.nan, np.nan, np.nan))
                pm = s["pred_party_mae"].get(party_for_mae, (np.nan, np.nan, np.nan))
            else:
                all_obs  = list(s["obs_party_mae"].values())
                all_pred = list(s["pred_party_mae"].values())
                om = (np.mean([v[0] for v in all_obs]),
                      np.mean([v[1] for v in all_obs]),
                      np.mean([v[2] for v in all_obs]))
                pm = (np.mean([v[0] for v in all_pred]),
                      np.mean([v[1] for v in all_pred]),
                      np.mean([v[2] for v in all_pred]))
            obs_vals.append(om[0]); obs_lo.append(om[1]); obs_hi.append(om[2])
            pred_vals.append(pm[0]); pred_lo.append(pm[1]); pred_hi.append(pm[2])
            ylabel    = "MAE (match score points)"
            title_sfx = f"Party MAE ({party_for_mae or 'mean'})"

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.4), 5))
    b1 = ax.bar(x - w/2, obs_vals,  w, label="Obs only",
                color="#2166AC", alpha=0.85, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, pred_vals, w, label="Obs + predicted",
                color="#4DAC26", alpha=0.85, edgecolor="white", linewidth=0.5)

    for i in range(len(runs)):
        _errorbars(ax, x[i] - w/2, obs_vals[i],  obs_lo[i],  obs_hi[i],  color="#1a4f87")
        _errorbars(ax, x[i] + w/2, pred_vals[i], pred_lo[i], pred_hi[i], color="#2d7015")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if metric in ("top1", "top3"):
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(f"Fixed 50% stopping — {title_sfx}\n(with 95% bootstrap CI)",
                 fontweight="bold")

    plt.tight_layout()
    fname = f"50pct_{metric}" + (f"_{party_for_mae}" if party_for_mae else "")
    _save(fig, fname, cfg)
    return fig


# ---------------------------------------------------------------------------
# 2. Party breakdown
# ---------------------------------------------------------------------------

def plot_party_breakdown(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    condition: str = "50pct_obs",
    metric: str = "top1",   # 'top1', 'top3', 'mae'
) -> plt.Figure:
    """
    Per-party grouped bar chart for a given condition and metric,
    with one bar group per (predictor, strategy) combination.
    Parties sorted by frequency (most frequent left).
    """
    _style()
    runs = list(all_results.keys())

    # Collect party order from first run
    pb0    = party_breakdown(list(all_results.values())[0], condition)
    parties = pb0["party"].tolist()
    n_p     = len(parties)
    n_r     = len(runs)

    x      = np.arange(n_p)
    w      = 0.75 / n_r
    offsets = np.linspace(-0.75/2 + w/2, 0.75/2 - w/2, n_r)

    # Metric column mapping
    if metric == "top1":
        col, lo_col, hi_col = "top1_accuracy", "top1_ci_lo", "top1_ci_hi"
        ylabel = "Top-1 Accuracy"
    elif metric == "top3":
        col, lo_col, hi_col = "top3_accuracy", "top3_ci_lo", "top3_ci_hi"
        ylabel = "Top-3 Accuracy (exact set)"
    elif metric == "mae":
        col, lo_col, hi_col = "mean_mae", "mae_ci_lo", "mae_ci_hi"
        ylabel = "MAE (match score points)"

    fig, ax = plt.subplots(figsize=(max(10, n_p * 1.5), 5))

    for i, (pred, strat) in enumerate(runs):
        pb   = party_breakdown(all_results[(pred, strat)], condition)
        pb   = pb.set_index("party").reindex(parties)
        vals = pb[col].fillna(0).values
        los  = pb[lo_col].fillna(0).values if lo_col in pb.columns else vals
        his  = pb[hi_col].fillna(0).values if hi_col in pb.columns else vals
        color = STRATEGY_COLORS.get(strat, "#888888")
        lbl   = _run_label(pred, strat)
        bars  = ax.bar(x + offsets[i], vals, w, label=lbl,
                       color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        for j in range(n_p):
            _errorbars(ax, x[j] + offsets[i], vals[j], los[j], his[j], color=color)

    # Sample sizes from first run
    pb0_idx = pb0.set_index("party").reindex(parties)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(n={int(pb0_idx.loc[p,'n_persons'])})" for p in parties],
        fontsize=8,
    )
    ax.set_ylabel(ylabel)
    if metric in ("top1", "top3"):
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.1)
        ax.axhline(1/6, color="grey", linestyle="--", linewidth=1,
                   alpha=0.5, label="Chance (1/6)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Party breakdown — {CONDITION_LABELS.get(condition, condition)}: "
        f"{ylabel}\n(with 95% bootstrap CI)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, f"party_{condition}_{metric}", cfg)
    return fig


# ---------------------------------------------------------------------------
# 3. Items asked distribution (adaptive)
# ---------------------------------------------------------------------------

def plot_items_asked(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    condition: str = "top1",   # 'top1' or 'top3'
) -> plt.Figure:
    """
    Violin + mean + CI plot showing distribution of items asked
    before the adaptive stopping condition is met.
    One violin per (predictor, strategy) combination.
    """
    _style()
    runs   = list(all_results.keys())
    labels = [_run_label(p, s) for p, s in runs]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.4), 5))

    data_list  = []
    means, los, his = [], [], []

    for pred, strat in runs:
        sad = summarise_adaptive(all_results[(pred, strat)], condition)
        data_list.append(sad["items_distribution"])
        means.append(sad["mean_items_asked"][0])
        los.append(sad["mean_items_asked"][1])
        his.append(sad["mean_items_asked"][2])

    parts = ax.violinplot(data_list, positions=range(len(runs)),
                          showmedians=True, showextrema=True)
    for i, (pc, (pred, strat)) in enumerate(zip(parts["bodies"], runs)):
        pc.set_facecolor(STRATEGY_COLORS.get(strat, "#888888"))
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    for i in range(len(runs)):
        _errorbars(ax, i, means[i], los[i], his[i], color="black")
        ax.scatter(i, means[i], color="black", zorder=5, s=30)

    # 50% reference line
    n_items = all_results[runs[0]][0].n_items_total
    ax.axhline(n_items * 0.5, color="#D6604D", linestyle="--",
               linewidth=1.2, alpha=0.7, label="50% threshold")
    ax.axhline(n_items, color="grey", linestyle=":",
               linewidth=1, alpha=0.5, label="All items (cap)")

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Items asked")
    ax.legend(loc="upper right")
    cond_label = "top-1 correct" if condition == "top1" else "top-3 correct"
    ax.set_title(
        f"Items asked until {cond_label} (adaptive stopping)\n"
        f"Mean ± 95% CI shown; cap at {n_items} if never reached",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, f"items_asked_{condition}", cfg)
    return fig


# ---------------------------------------------------------------------------
# 4. Per-party items asked
# ---------------------------------------------------------------------------

def plot_party_items_asked(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    condition: str = "top1",
) -> plt.Figure:
    """
    Per-party mean items asked (grouped by true best party),
    one bar group per (predictor, strategy) combination.
    """
    _style()
    runs = list(all_results.keys())

    pb0 = party_breakdown(list(all_results.values())[0], condition)
    if "mean_items_asked" not in pb0.columns:
        warnings.warn(f"No items_asked data for condition={condition}", UserWarning)
        return plt.figure()

    parties = pb0["party"].tolist()
    n_p     = len(parties)
    n_r     = len(runs)
    x       = np.arange(n_p)
    w       = 0.75 / n_r
    offsets = np.linspace(-0.75/2 + w/2, 0.75/2 - w/2, n_r)

    fig, ax = plt.subplots(figsize=(max(10, n_p * 1.5), 5))

    for i, (pred, strat) in enumerate(runs):
        pb    = party_breakdown(all_results[(pred, strat)], condition)
        pb    = pb.set_index("party").reindex(parties)
        vals  = pb["mean_items_asked"].fillna(0).values
        los   = pb.get("items_ci_lo", pb["mean_items_asked"]).fillna(0).values
        his   = pb.get("items_ci_hi", pb["mean_items_asked"]).fillna(0).values
        color = STRATEGY_COLORS.get(strat, "#888888")
        ax.bar(x + offsets[i], vals, w, label=_run_label(pred, strat),
               color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        for j in range(n_p):
            _errorbars(ax, x[j] + offsets[i], vals[j], los[j], his[j], color=color)

    n_items = all_results[runs[0]][0].n_items_total
    ax.axhline(n_items * 0.5, color="#D6604D", linestyle="--",
               linewidth=1.2, alpha=0.6, label="50% threshold")

    pb0_idx = pb0.set_index("party").reindex(parties)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(n={int(pb0_idx.loc[p,'n_persons'])})" for p in parties],
        fontsize=8,
    )
    ax.set_ylabel("Mean items asked")
    ax.legend(loc="upper right", fontsize=8)
    cond_label = "top-1" if condition == "top1" else "top-3"
    ax.set_title(
        f"Mean items asked per party — adaptive {cond_label} stop\n"
        f"(with 95% bootstrap CI)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, f"party_items_asked_{condition}", cfg)
    return fig


# ---------------------------------------------------------------------------
# 5. All-party 50% obs+pred results (top-1, top-3, MAE)
# ---------------------------------------------------------------------------

def plot_50pct_pred_by_party(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    metric: str = "top1",   # 'top1', 'top3', or 'mae'
) -> plt.Figure:
    """
    Grouped bar chart: obs+pred metric at the 50% checkpoint,
    one bar group per party, one bar per (predictor, strategy) combination.
    Parties sorted by frequency (most frequent left).
    Includes 95% bootstrap CIs.

    Parameters
    ----------
    metric : 'top1', 'top3', or 'mae'
    """
    _style()
    runs   = list(all_results.keys())
    n_runs = len(runs)

    # Collect party-level data from party_breakdown (50pct_pred condition)
    pb_by_run = {
        key: party_breakdown(all_results[key], "50pct_pred")
        for key in runs
    }

    # Party order from first run (sorted by frequency, most common left)
    pb0     = pb_by_run[runs[0]]
    parties = pb0["party"].tolist()
    n_p     = len(parties)

    # Column mapping
    if metric == "top1":
        col, lo_col, hi_col = "top1_accuracy", "top1_ci_lo", "top1_ci_hi"
        ylabel    = "Top-1 Accuracy"
        pct_axis  = True
    elif metric == "top3":
        col, lo_col, hi_col = "top3_accuracy", "top3_ci_lo", "top3_ci_hi"
        ylabel   = "Top-3 Accuracy (exact set)"
        pct_axis = True
    elif metric == "mae":
        col, lo_col, hi_col = "mean_mae", "mae_ci_lo", "mae_ci_hi"
        ylabel   = "MAE (match score points)"
        pct_axis = False

    x       = np.arange(n_p)
    w       = 0.75 / n_runs
    offsets = np.linspace(-0.75/2 + w/2, 0.75/2 - w/2, n_runs)

    fig, ax = plt.subplots(figsize=(max(10, n_p * 1.5), 5))

    for i, (pred, strat) in enumerate(runs):
        pb   = pb_by_run[(pred, strat)].set_index("party").reindex(parties)
        vals = pb[col].fillna(0).values
        los  = pb[lo_col].fillna(0).values if lo_col in pb.columns else vals
        his  = pb[hi_col].fillna(0).values if hi_col in pb.columns else vals
        color = STRATEGY_COLORS.get(strat, "#888888")
        label = _run_label(pred, strat)

        ax.bar(x + offsets[i], vals, w, label=label,
               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for j in range(n_p):
            _errorbars(ax, x[j] + offsets[i], vals[j], los[j], his[j],
                       color=color)

    # Chance line for accuracy plots
    if pct_axis:
        n_parties = len(parties)
        ax.axhline(1 / n_parties, color="grey", linestyle="--",
                   linewidth=1, alpha=0.5,
                   label=f"Chance (1/{n_parties} = {1/n_parties:.0%})")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.1)

    # Sample sizes on x-axis labels
    pb0_idx = pb0.set_index("party").reindex(parties)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(n={int(pb0_idx.loc[p, 'n_persons'])})" for p in parties],
        fontsize=8,
    )
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)

    metric_labels = {"top1": "Top-1 accuracy", "top3": "Top-3 accuracy (exact set)",
                     "mae": "Party MAE"}
    ax.set_title(
        f"Fixed 50% stopping (obs + predicted) — {metric_labels.get(metric, metric)}"
        f" per party\n(with 95% bootstrap CI)",
        fontweight="bold",
    )

    plt.tight_layout()
    _save(fig, f"50pct_pred_by_party_{metric}", cfg)
    return fig


# ---------------------------------------------------------------------------
# 6. 50% obs+pred only — top-1, top-3, MAE (one clean plot each)
# ---------------------------------------------------------------------------

def _plot_50pct_pred_single(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    metric: str,
) -> plt.Figure:
    """
    Bar chart: obs+pred metric at the 50% checkpoint,
    one bar per (predictor, strategy) combination.
    Bars coloured by strategy. Includes 95% bootstrap CIs.
    """
    _style()
    runs   = list(all_results.keys())
    labels = [_run_label(p, s) for p, s in runs]
    x      = np.arange(len(runs))
    colors = [STRATEGY_COLORS.get(s, "#888888") for _, s in runs]

    vals, los, his = [], [], []
    for key in runs:
        s = summarise_50pct(all_results[key])
        if metric == "top1":
            m, lo, hi = s["pred_top1"]
            ylabel, title = "Accuracy", "Top-1 accuracy"
        elif metric == "top3":
            m, lo, hi = s["pred_top3"]
            ylabel, title = "Accuracy", "Top-3 accuracy (exact set)"
        elif metric == "mae":
            triples = list(s["pred_party_mae"].values())
            m  = float(np.mean([t[0] for t in triples]))
            lo = float(np.mean([t[1] for t in triples]))
            hi = float(np.mean([t[2] for t in triples]))
            ylabel, title = "MAE (match score points)", "Mean party MAE"
        vals.append(m); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(max(7, len(runs) * 1.2), 5))
    bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for i in range(len(runs)):
        _errorbars(ax, x[i], vals[i], los[i], his[i], color="black")

    # Value labels on bars
    for bar, v in zip(bars, vals):
        fmt = f"{v:.1%}" if metric in ("top1", "top3") else f"{v:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.015 if metric != "mae" else 0.05),
                fmt, ha="center", va="bottom", fontsize=8)

    # Legend patches by strategy
    seen = {}
    for (_, strat), color in zip(runs, colors):
        if strat not in seen:
            seen[strat] = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.85)
    ax.legend(seen.values(),
              [STRATEGY_LABELS.get(s, s) for s in seen],
              title="Strategy", loc="lower right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if metric in ("top1", "top3"):
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Fixed 50% stopping (obs + predicted) — {title}\n"
        f"(with 95% bootstrap CI)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, f"50pct_pred_{metric}", cfg)
    return fig


def plot_50pct_pred_top1(all_results, cfg):
    """Top-1 accuracy — obs+pred at 50% — one bar per predictor/strategy."""
    return _plot_50pct_pred_single(all_results, cfg, "top1")

def plot_50pct_pred_top3(all_results, cfg):
    """Top-3 accuracy — obs+pred at 50% — one bar per predictor/strategy."""
    return _plot_50pct_pred_single(all_results, cfg, "top3")

def plot_50pct_pred_mae(all_results, cfg):
    """Mean party MAE — obs+pred at 50% — one bar per predictor/strategy."""
    return _plot_50pct_pred_single(all_results, cfg, "mae")


def make_all_plots(
    all_results: dict[tuple, list],
    cfg: PlotConfig,
    parties: Optional[list[str]] = None,
) -> None:
    """
    Generate all plots for a set of run results.

    Parameters
    ----------
    all_results : dict
        Keys: (predictor_name, strategy)
        Values: list[PersonResult]
    cfg : PlotConfig
    parties : list[str] or None
        Party names for individual MAE plots. If None, inferred from results.
    """
    print(f"Generating plots → {cfg.out_dir}")

    if parties is None:
        first = list(all_results.values())[0][0]
        parties = list(first.true_party_scores.keys())

    figs = [
        # 50% obs+pred only — top-1, top-3, MAE (the three new requested plots)
        ("50pct_pred_top1", lambda: plot_50pct_pred_top1(all_results, cfg)),
        ("50pct_pred_top3", lambda: plot_50pct_pred_top3(all_results, cfg)),
        ("50pct_pred_mae",  lambda: plot_50pct_pred_mae(all_results,  cfg)),
        # All-party obs+pred breakdown at 50%
        ("50pct_pred_by_party_top1", lambda: plot_50pct_pred_by_party(all_results, cfg, "top1")),
        ("50pct_pred_by_party_top3", lambda: plot_50pct_pred_by_party(all_results, cfg, "top3")),
        ("50pct_pred_by_party_mae",  lambda: plot_50pct_pred_by_party(all_results, cfg, "mae")),
        # 50% obs-only vs obs+pred comparison
        ("50pct_top1",  lambda: plot_50pct_comparison(all_results, cfg, "top1")),
        ("50pct_top3",  lambda: plot_50pct_comparison(all_results, cfg, "top3")),
        ("50pct_mae",   lambda: plot_50pct_comparison(all_results, cfg, "mae")),
        # Party breakdowns — all conditions
        ("party_50obs_top1",  lambda: plot_party_breakdown(all_results, cfg, "50pct_obs",  "top1")),
        ("party_50obs_top3",  lambda: plot_party_breakdown(all_results, cfg, "50pct_obs",  "top3")),
        ("party_50obs_mae",   lambda: plot_party_breakdown(all_results, cfg, "50pct_obs",  "mae")),
        ("party_50pred_top1", lambda: plot_party_breakdown(all_results, cfg, "50pct_pred", "top1")),
        ("party_50pred_top3", lambda: plot_party_breakdown(all_results, cfg, "50pct_pred", "top3")),
        ("party_50pred_mae",  lambda: plot_party_breakdown(all_results, cfg, "50pct_pred", "mae")),
        # Adaptive items asked
        ("items_asked_top1",  lambda: plot_items_asked(all_results, cfg, "top1")),
        ("items_asked_top3",  lambda: plot_items_asked(all_results, cfg, "top3")),
        # Per-party items asked
        ("party_items_top1",  lambda: plot_party_items_asked(all_results, cfg, "top1")),
        ("party_items_top3",  lambda: plot_party_items_asked(all_results, cfg, "top3")),
    ]

    # Per-party MAE plots at 50% (obs and pred)
    for party in parties:
        p = party
        figs.append((f"50pct_mae_{p}",
                     lambda p=p: plot_50pct_comparison(all_results, cfg, "mae", p)))

    for name, fn in figs:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            warnings.warn(f"  ✗ {name} failed: {e}", UserWarning)

    print(f"Done. {len(figs)} plots saved to {cfg.out_dir}")
