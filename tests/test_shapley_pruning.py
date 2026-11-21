import pytest
import torch

from shapley_pruning.data import Split
from shapley_pruning.model import CNN
from shapley_pruning.pruning import prune_channels, prune_neurons, random_k, to_mask
from shapley_pruning.shapley import (Game, channel_game, logit_shapley, negative_loss_per_class,
                                     neuron_game, permutation_shapley, true_class_logit)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return CNN().eval()


@pytest.fixture
def split():
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(200, 3, 32, 32, generator=generator)
    labels = torch.arange(200) % 10
    return Split(images, labels)


def test_channel_pruning_equals_masking(model, split):
    keep = random_k(32, 9, seed=1)
    pruned = prune_channels(model, keep)
    expected = model(split.images, channel_mask=to_mask(keep, 32))
    assert torch.allclose(pruned(split.images), expected, atol=1e-5)


def test_neuron_pruning_equals_masking(model, split):
    keep = random_k(128, 40, seed=2)
    pruned = prune_neurons(model, keep)
    expected = model(split.images, neuron_mask=to_mask(keep, 128))
    assert torch.allclose(pruned(split.images), expected, atol=1e-5)


def test_estimate_is_efficient(model, split):
    estimate = permutation_shapley(channel_game(model, split), n_permutations=5)
    assert torch.isclose(estimate.values.sum(), estimate.full_value - estimate.empty_value, atol=1e-5)


def test_exact_for_additive_game():
    weights = torch.tensor([0.5, -1.0, 2.0, 0.0])
    game = Game(lambda mask: 3.0 + mask @ weights, 4)
    estimate = permutation_shapley(game, n_permutations=3)
    assert torch.allclose(estimate.values, weights)
    assert torch.allclose(estimate.std_error, torch.zeros(4))


def test_logit_shapley_is_exact(model, split):
    estimate = permutation_shapley(neuron_game(model, split, payoff=true_class_logit), n_permutations=3)
    assert torch.allclose(estimate.values, logit_shapley(model, split), atol=1e-5)


def test_cost_per_unit_shifts_values(model, split):
    game = neuron_game(model, split)
    penalised = Game(lambda mask: game.value(mask) - 0.01 * mask.sum(), game.n_players)
    shifted = permutation_shapley(penalised, n_permutations=3).values
    assert torch.allclose(shifted, permutation_shapley(game, n_permutations=3).values - 0.01, atol=1e-5)


def test_class_games_add_up_to_global_game(model, split):
    per_class = permutation_shapley(neuron_game(model, split, payoff=negative_loss_per_class), 3)
    overall = permutation_shapley(neuron_game(model, split), 3)
    shares = torch.bincount(split.labels).float() / len(split.labels)
    assert torch.allclose(per_class.values @ shares, overall.values, atol=1e-5)


def test_random_units_come_from_the_pool_first():
    pool = torch.zeros(20, dtype=torch.bool)
    pool[[3, 7, 11]] = True
    assert random_k(20, 2, seed=0, pool=pool).tolist() in ([3, 7], [3, 11], [7, 11])
    chosen = random_k(20, 5, seed=0, pool=pool).tolist()
    assert {3, 7, 11} <= set(chosen) and len(set(chosen)) == 5
    assert torch.equal(random_k(20, 5, seed=4), random_k(20, 5, seed=4, pool=None))
