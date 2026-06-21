"""Tests for GreedyAgent and the match harness."""
import random

from azul.agent import GreedyAgent, RandomAgent
from azul.match import play_match, MatchResult
from azul.state import Color, GameState, Move, WALL_PATTERN, CENTER, FLOOR


def test_returns_a_legal_move():
    gs = GameState.new_game(42)
    move = GreedyAgent().choose_move(gs)
    assert move in gs.legal_moves()


def test_deterministic():
    gs = GameState.new_game(42)
    a = GreedyAgent().choose_move(gs)
    b = GreedyAgent().choose_move(gs)
    assert a == b


def test_prefers_completing_a_line_over_dumping_to_floor():
    # One factory with a single blue tile; row 0 (capacity 1) is open for blue.
    # Greedy should place it on line 0 (completes -> +1) rather than the floor.
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 1}
    move = GreedyAgent().choose_move(gs)
    assert move == Move(0, Color.BLUE, 0)


def test_avoids_floor_when_a_pattern_line_fits():
    # 3 blue available; placing on the floor is strictly worse than staging
    # them on a roomy pattern line. Greedy must not choose the floor dump.
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 3}
    move = GreedyAgent().choose_move(gs)
    assert move.dest_line != FLOOR


def test_match_result_counts_add_up():
    res = play_match(
        make_a=lambda i: RandomAgent(random.Random(i)),
        make_b=lambda i: RandomAgent(random.Random(i + 500)),
        n_games=10,
        base_seed=0,
    )
    assert isinstance(res, MatchResult)
    assert res.wins_a + res.wins_b + res.ties == 10


def test_greedy_beats_random_decisively():
    res = play_match(
        make_a=lambda i: GreedyAgent(),
        make_b=lambda i: RandomAgent(random.Random(i + 1000)),
        n_games=40,
        base_seed=0,
    )
    # Greedy should win the large majority of games.
    assert res.win_rate_a > 0.8
