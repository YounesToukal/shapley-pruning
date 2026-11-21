"""The CNN studied in every experiment, with optional masks on its two player layers."""

import torch
import torch.nn.functional as F
from torch import nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CNN(nn.Module):
    """conv1 (3→32) → pool → conv2 (32→C) → pool → fc1 (C·8·8→K) → fc2 (K→10).

    Every hidden layer is followed by a ReLU. The C output channels of conv2 and
    the K neurons of fc1 are the players. A channel mask is applied after the
    pooling, which for a binary mask is the same as masking the conv2 output.
    """

    def __init__(self, n_channels=32, n_neurons=128):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, n_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(n_channels * 8 * 8, n_neurons)
        self.fc2 = nn.Linear(n_neurons, 10)

    @property
    def n_channels(self):
        return self.conv2.out_channels

    @property
    def n_neurons(self):
        return self.fc1.out_features

    def channel_activations(self, images):
        x = (images - self.mean) / self.std
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return F.max_pool2d(F.relu(self.conv2(x)), 2)

    def neuron_activations(self, channel_activations):
        return F.relu(self.fc1(channel_activations.flatten(1)))

    def forward(self, images, channel_mask=None, neuron_mask=None):
        channels = self.channel_activations(images)
        if channel_mask is not None:
            channels = channels * channel_mask.view(1, -1, 1, 1)
        neurons = self.neuron_activations(channels)
        if neuron_mask is not None:
            neurons = neurons * neuron_mask
        return self.fc2(neurons)
