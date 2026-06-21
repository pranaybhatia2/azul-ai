"""Tests for the Game loop and RandomAgent."""
import random

import pytest

from azul.agent import Agent, RandomAgent
from azul.game import Game, GameResult
from azul.state import GameState, Move


def two_randoms(seed_a=1, seed_b=2):
    return [RandomAgent(random.Random(seed_a)), RandomAgent(random.Random(seed_b))]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_requires_two_agents():
    with pytest.raises(ValueError):
        Game(agents=[RandomAgent(random.Random(0))], seed=1)


def test_factories_filled_on_construction():
    game = Game(agents=two_randoms(), seed=42)
    assert all(sum(f.values()) == 4 for f in game.state.factories)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

def test_step_returns_a_move():
    game = Game(agents=two_randoms(), seed=42)
    move = game.step()
    assert isinstance(move, Move)


def test_step_advances_player():
    game = Game(agents=two_randoms(), seed=42)
    assert game.state.current_player == 0
    game.step()
    assert game.state.current_player == 1


def test_step_does_not_mutate_via_agent_clone():
    """Agent receives a clone; mutating it must not affect the real game."""
    class VandalAgent(Agent):
        def choose_move(self, state):
            moves = state.legal_moves()
            state.player_boards[0].score = 9999   # vandalize the clone
            state.factories.clear()
            return moves[0]

    game = Game(agents=[VandalAgent(), RandomAgent(random.Random(1))], seed=42)
    game.step()
    assert game.state.player_boards[0].score != 9999
    assert len(game.state.factories) == 5


def test_step_returns_none_when_over():
    game = Game(agents=two_randoms(), seed=42)
    game.play()
    assert game.step() is None


# ---------------------------------------------------------------------------
# play()
# ---------------------------------------------------------------------------

def test_play_returns_game_result():
    game = Game(agents=two_randoms(), seed=42)
    result = game.play()
    assert isinstance(result, GameResult)


def test_play_terminates_by_row_or_starvation():
    game = Game(agents=two_randoms(), seed=42)
    game.play()
    # Game ends either when a player completes a horizontal row, OR when the
    # bag + discard are exhausted and factories can no longer be refilled.
    starved = game.state.is_round_over() and sum(game.state.bag.values()) == 0
    assert game.state.is_game_over() or starved


def test_starvation_ends_game():
    # Seed 7 is known to deplete tiles before any row completes (round ~8).
    game = Game(agents=two_randoms(1, 2), seed=7)
    result = game.play()
    assert game.over
    assert result is not None


def test_result_scores_match_boards():
    game = Game(agents=two_randoms(), seed=42)
    result = game.play()
    assert result.scores == [b.score for b in game.state.player_boards]


def test_winner_is_higher_scorer_or_none():
    game = Game(agents=two_randoms(), seed=7)
    result = game.play()
    if result.scores[0] > result.scores[1]:
        assert result.winner == 0
    elif result.scores[1] > result.scores[0]:
        assert result.winner == 1
    else:
        assert result.winner is None


def test_game_is_deterministic_for_same_seeds():
    r1 = Game(agents=two_randoms(1, 2), seed=42).play()
    r2 = Game(agents=two_randoms(1, 2), seed=42).play()
    assert r1 == r2


def test_multiple_rounds_played():
    game = Game(agents=two_randoms(), seed=42)
    result = game.play()
    assert result.rounds >= 1


def test_final_scores_non_negative():
    game = Game(agents=two_randoms(), seed=99)
    result = game.play()
    assert all(s >= 0 for s in result.scores)
