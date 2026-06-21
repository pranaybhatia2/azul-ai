"""Tests for the training step and loop."""
import random

import pytest

pytest.importorskip("torch")
import torch

from azul.train import train_step, train, save_net, load_net
from azul.net import AzulNet, predict
from azul.encoding import STATE_SIZE, POLICY_SIZE
from azul.state import GameState


def _synthetic_batch(n=8):
    rng = random.Random(0)
    batch = []
    for _ in range(n):
        enc = [rng.random() for _ in range(STATE_SIZE)]
        # a valid policy distribution over a few indices
        pol = [0.0] * POLICY_SIZE
        for idx in rng.sample(range(POLICY_SIZE), 5):
            pol[idx] = 0.2
        val = rng.uniform(-1, 1)
        batch.append((enc, pol, val))
    return batch


def test_train_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    net = AzulNet(hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _synthetic_batch()
    first, _, _ = train_step(net, opt, batch)
    for _ in range(60):
        last, _, _ = train_step(net, opt, batch)
    assert last < first   # the net overfits the fixed batch


def test_train_loop_runs_and_returns_history():
    torch.manual_seed(0)
    net = AzulNet(hidden=64)
    history = train(net, iterations=1, games_per_iter=1, sp_iterations=10,
                    epochs=1, batch_size=16, rng=random.Random(0))
    assert len(history) == 1
    assert "loss" in history[0]
    assert history[0]["examples"] > 0


def test_eval_fn_is_called():
    torch.manual_seed(0)
    net = AzulNet(hidden=64)
    calls = []
    train(net, iterations=2, games_per_iter=1, sp_iterations=8, epochs=1,
          rng=random.Random(0), eval_fn=lambda n, it: calls.append(it) or it)
    assert calls == [0, 1]


def test_save_and_load_roundtrip(tmp_path):
    net = AzulNet(hidden=64)
    gs = GameState.new_game(42)
    before_priors, before_val = predict(net, gs)
    path = str(tmp_path / "net.pt")
    save_net(net, path)
    loaded = load_net(path, hidden=64)
    after_priors, after_val = predict(loaded, gs)
    assert after_val == before_val
    assert after_priors == before_priors
