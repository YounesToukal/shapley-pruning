"""Shared figure style. Each rule keeps the same colour and marker in every figure."""

import matplotlib.pyplot as plt

STYLES = {
    "Shapley": ("#2a78d6", "o"),
    "Random": ("#eb6834", "s"),
    "L1 norm": ("#1baf7a", "^"),
    "Shapley, lowest first": ("#eda100", "v"),
    "Shapley on noisy images": ("#e87ba4", "D"),
    "Logit Shapley": ("#6250d6", "P"),
    "Random active neurons": ("#e34948", "X"),
}
REFERENCE = "#8a8984"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#33322f",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "legend.frameon": False,
    "font.size": 10,
})


def plot_rules(ax, results, x="k", reference=None, reference_label="Full network"):
    """Mean accuracy of each rule against `x`, with a band of ± one standard deviation."""
    for rule, group in results.groupby("rule", sort=False):
        stats = group.groupby(x)["accuracy"].agg(["mean", "std"]).fillna(0)
        color, marker = STYLES[rule]
        ax.plot(stats.index, stats["mean"], color=color, marker=marker,
                linewidth=2, markersize=6, label=rule)
        ax.fill_between(stats.index, stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"], color=color, alpha=0.15, linewidth=0)
    if reference is not None:
        ax.axhline(reference, color=REFERENCE, linestyle="--", linewidth=1.2, label=reference_label)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend()
