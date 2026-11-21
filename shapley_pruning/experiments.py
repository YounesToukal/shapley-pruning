"""Comparisons of unit selection rules, shared by the notebooks.

A rule is given by a score per unit, and keeps the k units with the highest
scores. A random rule draws k units uniformly, possibly from a pool of units,
with one draw per seed.
"""

import pandas as pd
import torch

from .pruning import prune_channels, prune_neurons, random_k, to_mask, top_k
from .training import accuracy, fine_tune

RANDOM = "Random"


def _layer(model, layer):
    if layer == "channels":
        return model.n_channels, prune_channels
    if layer == "neurons":
        return model.n_neurons, prune_neurons
    raise ValueError(f"unknown layer {layer!r}")


def masked_accuracies(game, rankings, sizes, random_trials=20, random_pools=None):
    """Value of the coalition of the k selected units, for a game whose payoff is the accuracy.

    This is the accuracy of the parent network with only those units active,
    without retraining.
    """
    n = game.n_players
    rows = []

    def evaluate(rule, k, seed, keep):
        acc = game.value(to_mask(keep, n)).item()
        rows.append({"rule": rule, "k": k, "seed": seed, "accuracy": acc})

    for k in sizes:
        for rule, scores in rankings.items():
            evaluate(rule, k, 0, top_k(scores, k))
        for rule, pool in (random_pools or {RANDOM: None}).items():
            for seed in range(random_trials):
                evaluate(rule, k, seed, random_k(n, k, seed, pool))
    return pd.DataFrame(rows)


def top_k_coalition_values(game, estimate, sizes, random_trials=20):
    """Value of the coalition of the k highest Shapley values, against v(∅) + Σ_{i∈S} φ_i.

    The two agree when the players do not interact. The mean value of random
    k-coalitions is given for comparison.
    """
    n, phi = game.n_players, estimate.values
    rows = []
    for k in sizes:
        keep = top_k(phi, k)
        random_values = [game.value(to_mask(random_k(n, k, seed), n)) for seed in range(random_trials)]
        rows.append({"k": k,
                     "v(top k)": game.value(to_mask(keep, n)).item(),
                     "v(∅) + Σ φ over the top k": (estimate.empty_value + phi[keep].sum()).item(),
                     "v(random k), mean": torch.stack(random_values).mean().item()})
    return pd.DataFrame(rows).set_index("k")


def fine_tuned_accuracies(parent, data, layer, rankings, sizes, seeds=(0, 1, 2), random_pools=None):
    """Test accuracy after pruning the layer to k units and fine-tuning.

    Every rule is fine-tuned once per seed, so the spread of a deterministic
    rule measures the fine-tuning noise, and that of a random rule also
    includes the choice of units.
    """
    n, prune = _layer(parent, layer)
    rows = []

    def evaluate(rule, k, seed, keep):
        model = fine_tune(prune(parent, keep), data.train, seed)
        rows.append({"rule": rule, "k": k, "seed": seed, "accuracy": accuracy(model, data.test)})

    for k in sizes:
        for seed in seeds:
            for rule, scores in rankings.items():
                evaluate(rule, k, seed, top_k(scores, k))
            for rule, pool in (random_pools or {RANDOM: None}).items():
                evaluate(rule, k, seed, random_k(n, k, seed, pool))
    return pd.DataFrame(rows)


def summarize(results, by="k"):
    """Mean ± standard deviation of the accuracy, one row per value of `by` and one column per rule."""
    stats = results.groupby([by, "rule"], sort=False)["accuracy"].agg(["mean", "std"])
    cells = [f"{m:.2f}" if s != s else f"{m:.2f} ± {s:.2f}"
             for m, s in zip(stats["mean"], stats["std"])]
    table = pd.Series(cells, index=stats.index).unstack("rule")
    return table[list(dict.fromkeys(results["rule"]))]


def rank_correlation(a, b):
    """Spearman correlation between two score vectors without ties."""
    ranks = torch.stack([a.argsort().argsort(), b.argsort().argsort()]).float()
    return torch.corrcoef(ranks)[0, 1].item()
