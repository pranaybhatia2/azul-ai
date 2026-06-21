"""Tests for MCTSAgent (UCB1, random rollouts).

Note on strength: with RANDOM rollouts, MCTS beats Random decisively but does
NOT beat Greedy (random play in Azul is so weak that rollouts give little
signal). Tests assert what's actually true; the greedy-rollout upgrade is a
separate step.
"""
import random

from azul.mcts import MCTSAgent, _reward
from azul.agent import RandomAgent
from azul.match import play_match
from azul.state import GameState


def test_reward_perspective():
    assert _reward(0, 0) == 1.0      # player 0 won, asking for player 0
    assert _reward(0, 1) == 0.0      # player 0 won, asking for player 1
    assert _reward(None, 0) == 0.5   # tie


def test_returns_a_legal_move():
    gs = GameState.new_game(42)
    move = MCTSAgent(iterations=30, rng=random.Random(0)).choose_move(gs)
    assert move in gs.legal_moves()


def test_deterministic_with_seeded_rng():
    gs = GameState.new_game(42)
    a = MCTSAgent(iterations=50, rng=random.Random(7)).choose_move(gs)
    b = MCTSAgent(iterations=50, rng=random.Random(7)).choose_move(gs)
    assert a == b


def test_root_visits_equal_iterations():
    # Each iteration walks through exactly one root child, so the children's
    # visit counts sum to the iteration budget.
    gs = GameState.new_game(42)
    agent = MCTSAgent(iterations=60, rng=random.Random(1))
    # Re-run the search manually to inspect the tree.
    from azul.mcts import _Node
    root = _Node(gs.clone(), parent=None, move=None)
    for _ in range(agent.iterations):
        node = agent._select(root)
        node = agent._expand(node)
        winner = agent._rollout(node)
        agent._backpropagate(node, winner)
    assert sum(c.visits for c in root.children) == 60
    assert root.visits == 60


def test_beats_random():
    res = play_match(
        make_a=lambda i: MCTSAgent(iterations=40, rng=random.Random(i)),
        make_b=lambda i: RandomAgent(random.Random(i + 9)),
        n_games=3,
        base_seed=0,
    )
    assert res.wins_a >= 2
