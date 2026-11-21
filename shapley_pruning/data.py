"""CIFAR-10 and the train / validation / test split shared by every experiment."""

from typing import NamedTuple

import torch
from torchvision.datasets import CIFAR10

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


class Split(NamedTuple):
    images: torch.Tensor  # (N, 3, 32, 32), values in [0, 1]
    labels: torch.Tensor  # (N,)


class Splits(NamedTuple):
    train: Split
    val: Split
    test: Split


def _to_split(dataset):
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float().div(255)
    return Split(images, torch.tensor(dataset.targets))


def load_cifar10(root="data", val_size=5000, seed=0):
    """Loads CIFAR-10 in memory, downloading it on first use.

    The validation images are held out from the 50 000 training images. Shapley
    values are estimated on them, so the test set is used only for the reported
    accuracies.
    """
    train = _to_split(CIFAR10(root, train=True, download=True))
    test = _to_split(CIFAR10(root, train=False, download=True))
    order = torch.randperm(len(train.labels), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = order[:val_size], order[val_size:]
    return Splits(
        train=Split(train.images[train_idx], train.labels[train_idx]),
        val=Split(train.images[val_idx], train.labels[val_idx]),
        test=test,
    )


def batches(split, batch_size=128, generator=None):
    """Mini-batches of the split, in random order when a generator is given."""
    n = len(split.labels)
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for indices in order.split(batch_size):
        yield split.images[indices], split.labels[indices]
