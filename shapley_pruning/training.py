"""Training, fine-tuning and evaluation."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import batches
from .model import CNN

PARENT_EPOCHS = 10
PARENT_LR = 1e-3
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4


def train(model, split, epochs, lr, seed=0, batch_size=128):
    """Adam on the cross-entropy loss; the seed sets the order of the mini-batches."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for images, labels in batches(split, batch_size, generator):
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()
    return model.eval()


def fine_tune(model, split, seed=0):
    return train(model, split, FINE_TUNE_EPOCHS, FINE_TUNE_LR, seed)


def add_noise(images, noise_std, seed=0):
    """Adds Gaussian noise in pixel space and clips the result to [0, 1]."""
    if noise_std == 0:
        return images
    noise = torch.randn(images.shape, generator=torch.Generator().manual_seed(seed))
    return (images + noise_std * noise).clamp(0, 1)


@torch.no_grad()
def predict(model, split, channel_mask=None, neuron_mask=None, batch_size=1000):
    model.eval()
    return torch.cat([model(batch, channel_mask, neuron_mask).argmax(1)
                      for batch in split.images.split(batch_size)])


def accuracy(model, split, channel_mask=None, neuron_mask=None):
    """Top-1 accuracy in percent."""
    predictions = predict(model, split, channel_mask, neuron_mask)
    return 100 * (predictions == split.labels).float().mean().item()


def get_parent(data, path="checkpoints/parent.pt", seed=0):
    """Loads the trained parent network, training and saving it on first use."""
    torch.manual_seed(seed)
    model = CNN()
    path = Path(path)
    if path.exists():
        model.load_state_dict(torch.load(path))
        return model.eval()
    train(model, data.train, PARENT_EPOCHS, PARENT_LR, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model
