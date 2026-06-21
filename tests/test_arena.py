"""Tests for the NN-vs-baseline evaluation harness."""
import random

import pytest

pytest.importorskip("torch")

from azul.arena import nn_match
from azul.net import AzulNet
from azul.agent import RandomAgent
from azul.match import MatchResult


def test_nn_match_counts_add_up():
    net = AzulNet()
    res = nn_match(net, lambda i: RandomAgent(random.Random(100 + i)),
                   n_games=2, iterations=12)
    assert isinstance(res, MatchResult)
    assert res.wins_a + res.wins_b + res.ties == 2
