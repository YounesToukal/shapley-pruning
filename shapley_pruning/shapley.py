"""Cooperative games on the units of a layer, and estimators of their Shapley values.

A coalition S is a binary mask over the units of one layer: the units outside S
output zero. The value of the coalition is a payoff of the masked network on a
fixed set of images, by default the negative mean cross-entropy loss.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def negative_loss(logits, labels):
    return -F.cross_entropy(logits, labels)


def negative_loss_per_class(logits, labels, n_classes=10):
    """One payoff per class: the negative mean loss on the images of that class."""
    losses = F.cross_entropy(logits, labels, reduction="none")
    totals = torch.zeros(n_classes).index_add_(0, labels, losses)
    return -totals / torch.bincount(labels, minlength=n_classes)


def true_class_logit(logits, labels):
    return logits.gather(1, labels[:, None]).mean()


def percent_correct(logits, labels):
    return 100 * (logits.argmax(1) == labels).float().mean()


@dataclass
class Game:
    value: Callable[[torch.Tensor], torch.Tensor]  # binary mask over the players -> payoff
    n_players: int


@torch.no_grad()
def channel_game(model, split, payoff=negative_loss):
    """Players are the conv2 channels.

    The layers before conv2's output do not depend on the coalition, so they run
    once and each evaluation only runs fc1 and fc2.
    """
    model.eval()
    channels = torch.cat([model.channel_activations(batch) for batch in split.images.split(1000)])

    @torch.no_grad()
    def value(mask):
        neurons = model.neuron_activations(channels * mask.view(1, -1, 1, 1))
        return payoff(model.fc2(neurons), split.labels)

    return Game(value, model.n_channels)


@torch.no_grad()
def neuron_game(model, split, channel_mask=None, payoff=negative_loss):
    """Players are the fc1 neurons of the network, optionally with some channels masked."""
    model.eval()
    channels = torch.cat([model.channel_activations(batch) for batch in split.images.split(1000)])
    if channel_mask is not None:
        channels = channels * channel_mask.view(1, -1, 1, 1)
    neurons = model.neuron_activations(channels)

    @torch.no_grad()
    def value(mask):
        return payoff(model.fc2(neurons * mask), split.labels)

    return Game(value, model.n_neurons)


@dataclass
class ShapleyEstimate:
    marginals: torch.Tensor  # (n_permutations, n_players, *payoff_shape)
    empty_value: torch.Tensor  # v(∅)
    full_value: torch.Tensor  # v(N)

    @property
    def values(self):
        return self.marginals.mean(0)

    @property
    def std_error(self):
        return self.marginals.std(0) / len(self.marginals) ** 0.5


@torch.no_grad()
def sample_marginals(game, n_permutations, generator):
    """Marginal contributions v(P ∪ {i}) - v(P) along random orderings of the players.

    Each ordering adds the players one at a time, starting from the empty
    coalition, so it gives one marginal contribution per player for n + 1
    evaluations of the game, and its contributions sum to v(N) - v(∅).
    """
    empty_value = game.value(torch.zeros(game.n_players))
    marginals = torch.empty(n_permutations, game.n_players, *empty_value.shape)
    for t in range(n_permutations):
        mask = torch.zeros(game.n_players)
        previous = empty_value
        for player in torch.randperm(game.n_players, generator=generator):
            mask[player] = 1
            current = game.value(mask)
            marginals[t, player] = current - previous
            previous = current
    return marginals, empty_value, previous


def permutation_shapley(game, n_permutations, seed=0):
    """Monte Carlo estimate of the Shapley values from random player orderings."""
    generator = torch.Generator().manual_seed(seed)
    return ShapleyEstimate(*sample_marginals(game, n_permutations, generator))


@torch.no_grad()
def logit_shapley(model, split):
    """Exact Shapley values of the fc1 neurons for the game v(S) = mean true-class logit.

    The logit z_y = b_y + Σ_{i∈S} W[y, i] a_i is affine in the masked
    activations, so each neuron's marginal contribution is W[y, i] a_i whatever
    the coalition, and φ_i is its mean over the images.
    """
    model.eval()
    neurons = torch.cat([model.neuron_activations(model.channel_activations(batch))
                         for batch in split.images.split(1000)])
    return (neurons * model.fc2.weight[split.labels]).mean(0)
