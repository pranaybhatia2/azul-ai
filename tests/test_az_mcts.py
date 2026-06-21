"""Tests for NeuralMCTSAgent (PUCT search)."""
import random

import pytest

pytest.importorskip("torch")

from azul.az_mcts import NeuralMCTSAgent
from azul.net import AzulNet
from azul.agent import RandomAgent
from azul.game import Game
from azul.state import GameState


def test_choose_move_returns_legal():
    net = AzulNet()
    gs = GameState.new_game(42)
    move = NeuralMCTSAgent(net, iterations=30, rng=random.Random(0)).choose_move(gs)
    assert move in gs.legal_moves()


def test_visits_sum_to_iterations():
    from azul.pruning import candidate_moves
    net = AzulNet()
    gs = GameState.new_game(42)
    agent = NeuralMCTSAgent(net, iterations=40, rng=random.Random(0))
    visits = agent.search(gs)
    assert sum(visits.values()) == 40
    # Root children are the pruned candidates (optional floor-dumps dropped).
    assert set(visits.keys()) == set(candidate_moves(gs))


def test_search_deterministic_without_noise():
    net = AzulNet()
    gs = GameState.new_game(42)
    a = NeuralMCTSAgent(net, iterations=40, rng=random.Random(0)).search(gs)
    b = NeuralMCTSAgent(net, iterations=40, rng=random.Random(0)).search(gs)
    assert a == b


def test_dirichlet_noise_changes_search():
    net = AzulNet()
    gs = GameState.new_game(42)
    plain = NeuralMCTSAgent(net, iterations=40, rng=random.Random(0)).search(gs)
    noisy = NeuralMCTSAgent(net, iterations=40, rng=random.Random(0),
                            dirichlet_frac=0.5).search(gs, add_noise=True)
    assert plain != noisy


def test_drives_a_full_game():
    net = AzulNet()
    nn_agent = NeuralMCTSAgent(net, iterations=20, rng=random.Random(1))
    game = Game(agents=[nn_agent, RandomAgent(random.Random(2))], seed=3)
    result = game.play()
    assert game.over
    assert result.scores == [b.score for b in game.state.player_boards]
