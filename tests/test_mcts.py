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


# --- greedy rollout policy ---

def test_invalid_rollout_policy_raises():
    import pytest
    with pytest.raises(ValueError):
        MCTSAgent(rollout="banana")


def test_greedy_rollout_returns_legal_move():
    gs = GameState.new_game(42)
    move = MCTSAgent(iterations=20, rng=random.Random(0), rollout="greedy").choose_move(gs)
    assert move in gs.legal_moves()


def test_greedy_rollout_deterministic():
    gs = GameState.new_game(42)
    a = MCTSAgent(iterations=20, rng=random.Random(3), rollout="greedy").choose_move(gs)
    b = MCTSAgent(iterations=20, rng=random.Random(3), rollout="greedy").choose_move(gs)
    assert a == b


# --- truncated rollouts + eval ---

def test_eval_reward_bounds_and_direction():
    agent = MCTSAgent(eval_scale=10.0)
    neutral = GameState()
    assert abs(agent._eval_reward(neutral) - 0.5) < 1e-9   # equal boards -> 0.5
    neutral.player_boards[0].score = 20                    # P0 ahead
    r = agent._eval_reward(neutral)
    assert 0.5 < r < 1.0

    neutral.player_boards[1].score = 40                    # now P1 ahead
    assert agent._eval_reward(neutral) < 0.5


def test_truncated_rollout_legal_and_deterministic():
    gs = GameState.new_game(42)
    kw = dict(iterations=40, rollout="greedy", rollout_depth=4)
    a = MCTSAgent(rng=random.Random(5), **kw).choose_move(gs)
    b = MCTSAgent(rng=random.Random(5), **kw).choose_move(gs)
    assert a == b
    assert a in gs.legal_moves()
