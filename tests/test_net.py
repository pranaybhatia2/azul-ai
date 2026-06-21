"""Tests for the AzulNet model and predict() bridge."""
import pytest

torch = pytest.importorskip("torch")

from azul.net import AzulNet, predict
from azul.encoding import STATE_SIZE, POLICY_SIZE
from azul.state import GameState


def test_forward_shapes():
    net = AzulNet()
    x = torch.zeros((4, STATE_SIZE))
    policy, value = net(x)
    assert policy.shape == (4, POLICY_SIZE)
    assert value.shape == (4, 1)


def test_value_in_range():
    net = AzulNet()
    x = torch.randn((8, STATE_SIZE))
    _, value = net(x)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_predict_returns_priors_over_legal_moves():
    net = AzulNet()
    gs = GameState.new_game(42)
    priors, value = predict(net, gs)
    legal = set(gs.legal_moves())
    assert set(priors.keys()) == legal
    # Priors are a probability distribution over the legal moves.
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert all(p >= 0 for p in priors.values())
    assert -1.0 <= value <= 1.0


def test_predict_is_deterministic_for_fixed_weights():
    net = AzulNet()
    gs = GameState.new_game(42)
    a_priors, a_val = predict(net, gs)
    b_priors, b_val = predict(net, gs)
    assert a_val == b_val
    assert a_priors == b_priors
