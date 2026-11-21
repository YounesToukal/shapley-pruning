"""Unit selection rules and structural pruning."""

import copy

import torch

from .model import CNN


def top_k(scores, k):
    """Indices of the k highest scores, in increasing order."""
    return scores.argsort(descending=True)[:k].sort().values


def random_k(n, k, seed, pool=None):
    """k units drawn uniformly at random, in increasing order.

    With a boolean `pool`, the units are drawn from the pool, and from the
    other units only once the pool is exhausted.
    """
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    if pool is not None:
        order = torch.cat([order[pool[order]], order[~pool[order]]])
    return order[:k].sort().values


def to_mask(indices, n):
    mask = torch.zeros(n)
    mask[indices] = 1
    return mask


def l1_channel_scores(model):
    """L1 norm of each conv2 filter."""
    return model.conv2.weight.detach().abs().sum((1, 2, 3))


def l1_neuron_scores(model):
    """L1 norm of the incoming weights of each fc1 neuron."""
    return model.fc1.weight.detach().abs().sum(1)


@torch.no_grad()
def prune_channels(model, keep):
    """A smaller network without the conv2 channels outside `keep`.

    Removes the conv2 filters and the fc1 input columns that read their
    feature maps. Its output equals the parent's with those channels masked.
    """
    pruned = CNN(len(keep), model.n_neurons)
    pruned.conv1 = copy.deepcopy(model.conv1)
    pruned.conv2.weight.copy_(model.conv2.weight[keep])
    pruned.conv2.bias.copy_(model.conv2.bias[keep])
    fc1_weight = model.fc1.weight.view(model.n_neurons, model.n_channels, 8 * 8)
    pruned.fc1.weight.copy_(fc1_weight[:, keep].flatten(1))
    pruned.fc1.bias.copy_(model.fc1.bias)
    pruned.fc2 = copy.deepcopy(model.fc2)
    return pruned.eval()


@torch.no_grad()
def prune_neurons(model, keep):
    """A smaller network without the fc1 neurons outside `keep`.

    Removes their fc1 weight rows and biases, and the fc2 columns that read
    them. Its output equals the parent's with those neurons masked.
    """
    pruned = CNN(model.n_channels, len(keep))
    pruned.conv1 = copy.deepcopy(model.conv1)
    pruned.conv2 = copy.deepcopy(model.conv2)
    pruned.fc1.weight.copy_(model.fc1.weight[keep])
    pruned.fc1.bias.copy_(model.fc1.bias[keep])
    pruned.fc2.weight.copy_(model.fc2.weight[:, keep])
    pruned.fc2.bias.copy_(model.fc2.bias)
    return pruned.eval()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
