import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os

# Load and aggregate all per-person files
# Files are expected to be in a "per_person" subfolder relative to this script,
# or fall back to the script's own directory
script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "per_person")
if not os.path.isdir(folder):
    folder = script_dir  # fallback: same directory as script
files = sorted(glob.glob(os.path.join(folder, "n?????.csv")))

records = []
for f in files:
    df = pd.read_csv(f)
    grouped = df.groupby("true_best_party").agg(
        pred_top1=("pred_top1", "mean"),
        pred_top3=("pred_top3", "mean"),
        pred_mae =("pred_mae_mean", "mean"),
    ).reset_index()
    grouped["n_train"] = df["n_train"].iloc[0]
    records.append(grouped)

data = pd.concat(records, ignore_index=True).sort_values(["true_best_party", "n_train"])

parties = sorted(data["true_best_party"].unique())

# Compute party support (share of persons) from the first per-person file
first_df = pd.read_csv(files[0])
party_counts = first_df["true_best_party"].value_counts()
total = party_counts.sum()
party_support = {p: party_counts.get(p, 0) for p in parties}

colors = {
    "FDP":        "gold",
    "CDU":        "black",
    "DIE LINKE":  "hotpink",
    "Die Grünen": "green",
    "SPD":        "red",
    "bunt.saar":  "orange",
}

metrics = [
    {"col": "pred_top1", "ylabel": "Top-1 Accuracy", "title": "Top-1 Accuracy (Pred) vs. Training Size", "pct": True},
    {"col": "pred_top3", "ylabel": "Top-3 Accuracy", "title": "Top-3 Accuracy (Pred) vs. Training Size", "pct": True},
    {"col": "pred_mae",  "ylabel": "MAE",             "title": "MAE (Pred) vs. Training Size",             "pct": False},
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, m in zip(axes, metrics):
    for party in parties:
        sub = data[data["true_best_party"] == party]
        ax.plot(sub["n_train"], sub[m["col"]], marker="o", markersize=4,
                linewidth=1.8, label=f"{party} (n={party_support[party]})",
                color=colors[party], zorder=3)

    ax.set_xlabel("Training size", fontsize=11)
    ax.set_ylabel(m["ylabel"], fontsize=11)
    ax.set_title(m["title"], fontsize=12)

    if m["pct"]:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.set_ylim(0.30, 0.90)

    if not m["pct"]:
        ax.set_ylim(bottom=0)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

# Shared legend outside to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Party (n persons)", fontsize=10, title_fontsize=10,
           loc="center right", bbox_to_anchor=(1.01, 0.5))

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig("summary_metrics_by_party.png", dpi=150, bbox_inches="tight")
print("Saved summary_metrics_by_party.png")