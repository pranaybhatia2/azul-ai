"""Tests for self-play data generation."""
import random

import pytest

pytest.importorskip("torch")

from azul.selfplay import self_play_game, generate_examples, _sample_move, _value_for
from azul.net import AzulNet
from azul.encoding import STATE_SIZE, POLICY_SIZE
from azul.state import Color, Move


def test_value_for():
    assert _value_for(0, 0) == 1.0
    assert _value_for(0, 1) == -1.0
    assert _value_for(0, None) == 0.0


def test_sample_move_temperature_zero_is_argmax():
    visits = {Move(0, Color.BLUE, 0): 5, Move(1, Color.RED, 2): 20}
    assert _sample_move(visits, 0.0, random.Random(0)) == Move(1, Color.RED, 2)


def test_self_play_game_shapes():
    net = AzulNet()
    examples = self_play_game(net, iterations=15, rng=random.Random(0))
    assert len(examples) > 0
    for enc, pol, val in examples:
        assert len(enc) == STATE_SIZE
        assert len(pol) == POLICY_SIZE
        assert abs(sum(pol) - 1.0) < 1e-6
        assert val in (-1.0, 0.0, 1.0)


def test_self_play_deterministic_with_seed():
    net = AzulNet()
    a = self_play_game(net, iterations=15, rng=random.Random(7))
    b = self_play_game(net, iterations=15, rng=random.Random(7))
    assert a == b


def test_generate_examples_accumulates():
    net = AzulNet()
    ex = generate_examples(net, n_games=2, iterations=12, rng=random.Random(1))
    assert len(ex) > 0
