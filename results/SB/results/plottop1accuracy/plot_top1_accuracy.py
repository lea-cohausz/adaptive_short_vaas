import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
entropy = pd.read_csv("ridge__entropy__50pct_pred.csv")
random  = pd.read_csv("ridge__random__50pct_pred.csv")

# Sort parties by n_persons descending (use entropy file as reference)
party_order = entropy.sort_values("n_persons", ascending=False)["party"].tolist()

entropy = entropy.set_index("party").loc[party_order].reset_index()
random  = random.set_index("party").loc[party_order].reset_index()

# --- plot setup ---
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(party_order))
width = 0.35

colors = {"entropy": "#4C72B0", "random": "#DD8452"}

for i, (df, label, color) in enumerate([
    (entropy, "Entropy", colors["entropy"]),
    (random,  "Short VAA",  colors["random"]),
]):
    acc   = df["top1_accuracy"].values
    lo    = acc - df["top1_ci_lo"].values   # lower error bar size
    hi    = df["top1_ci_hi"].values - acc   # upper error bar size
    yerr  = np.array([lo, hi])

    offset = (i - 0.5) * width
    bars = ax.bar(
        x + offset, acc, width,
        label=label, color=color, alpha=0.85,
        yerr=yerr, capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor="black", capthick=1.2),
        zorder=3,
    )

# Connecting curves — one line per strategy through bar centres
for i, (df, label, color) in enumerate([
    (entropy, "Entropy", colors["entropy"]),
    (random,  "Short VAA",  colors["random"]),
]):
    offset = (i - 0.5) * width
    acc = df["top1_accuracy"].values
    ax.plot(x + offset, acc, color=color, linewidth=1.8,
            marker="o", markersize=5, zorder=4, linestyle="-",
            label="_nolegend_")

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{p}\n(n={entropy.loc[entropy.party==p,'n_persons'].iloc[0]})"
     for p in party_order],
    fontsize=10,
)
ax.set_ylabel("Top-1 Accuracy", fontsize=12)
ax.set_title("Top-1 Accuracy per Party\nEntropy vs. Short VAA (50 %)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("top1_accuracy_by_party.png", dpi=150)
print("Saved top1_accuracy_by_party.png")