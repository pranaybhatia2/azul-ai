"""Tests for supervised warm-start."""
import random

import pytest

pytest.importorskip("torch")
import torch

from azul.warmstart import (
    generate_supervised_examples, _greedy_softmax, warm_start,
    generate_teacher_examples, generate_hybrid_examples, pretrain,
)
from azul.net import AzulNet
from azul.encoding import STATE_SIZE, POLICY_SIZE, move_to_index
from azul.heuristics import evaluate
from azul.state import GameState


def test_greedy_softmax_peaks_on_best_move():
    gs = GameState.new_game(42)
    policy, moves, probs = _greedy_softmax(gs, temp=2.0)
    # The argmax-evaluate move should carry the most probability mass.
    me = gs.current_player
    def val(m):
        nxt = gs.clone(); nxt.apply(m); return evaluate(nxt, me)
    best = max(moves, key=val)
    assert policy[move_to_index(best)] == max(policy)
    assert abs(sum(policy) - 1.0) < 1e-6


def test_supervised_examples_shapes():
    ex = generate_supervised_examples(2, rng=random.Random(0))
    assert len(ex) > 0
    for enc, pol, val in ex:
        assert len(enc) == STATE_SIZE
        assert len(pol) == POLICY_SIZE
        assert abs(sum(pol) - 1.0) < 1e-6
        assert val in (-1.0, 0.0, 1.0)


def test_warm_start_reduces_loss():
    torch.manual_seed(0)
    net = AzulNet(hidden=64)
    losses = warm_start(net, n_games=4, epochs=5, rng=random.Random(0))
    assert losses[-1] < losses[0]   # supervised loss drops


def test_teacher_examples_shapes():
    # Small/fast teacher config — just validate the data plumbing.
    ex = generate_teacher_examples(1, rng=random.Random(0), teacher_iters=20,
                                   rollout_depth=4)
    assert len(ex) > 0
    for enc, pol, val in ex:
        assert len(enc) == STATE_SIZE
        assert len(pol) == POLICY_SIZE
        assert abs(sum(pol) - 1.0) < 1e-6
        assert val in (-1.0, 0.0, 1.0)


def test_hybrid_examples_shapes():
    ex = generate_hybrid_examples(1, rng=random.Random(0), teacher_iters=15,
                                  rollout_depth=3)
    assert len(ex) > 0
    for enc, pol, val in ex:
        assert len(enc) == STATE_SIZE
        assert len(pol) == POLICY_SIZE
        assert abs(sum(pol) - 1.0) < 1e-6
        assert val in (-1.0, 0.0, 1.0)


def test_pretrain_reduces_loss_on_teacher_data():
    torch.manual_seed(0)
    ex = generate_teacher_examples(1, rng=random.Random(0), teacher_iters=20,
                                   rollout_depth=4)
    net = AzulNet(hidden=64)
    losses = pretrain(net, ex, epochs=5, rng=random.Random(0))
    assert losses[-1] < losses[0]
