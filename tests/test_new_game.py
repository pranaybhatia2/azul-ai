"""Tests for GameState.new_game(seed)."""
import random
from azul.state import Color, GameState, TILES_PER_COLOR, NUM_FACTORIES


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_returns_game_state():
    gs = GameState.new_game(42)
    assert isinstance(gs, GameState)


def test_factories_filled():
    gs = GameState.new_game(42)
    assert all(sum(f.values()) == 4 for f in gs.factories)


def test_correct_number_of_factories():
    gs = GameState.new_game(42)
    assert len(gs.factories) == NUM_FACTORIES


def test_bag_reduced_by_factory_tiles():
    gs = GameState.new_game(42)
    drawn = sum(sum(f.values()) for f in gs.factories)
    assert sum(gs.bag.values()) == TILES_PER_COLOR * len(Color) - drawn


def test_discard_empty():
    gs = GameState.new_game(42)
    assert sum(gs.discard.values()) == 0


def test_scores_zero():
    gs = GameState.new_game(42)
    assert all(b.score == 0 for b in gs.player_boards)


def test_pattern_lines_empty():
    gs = GameState.new_game(42)
    for board in gs.player_boards:
        for pl in board.pattern_lines:
            assert pl.color is None and pl.count == 0


def test_walls_empty():
    gs = GameState.new_game(42)
    for board in gs.player_boards:
        assert all(cell is None for row in board.wall for cell in row)


def test_current_player_is_zero():
    gs = GameState.new_game(42)
    assert gs.current_player == 0


def test_round_number_is_one():
    gs = GameState.new_game(42)
    assert gs.round_number == 1


def test_first_player_marker_in_center():
    gs = GameState.new_game(42)
    assert gs.first_player_marker_in_center


def test_center_empty_except_marker():
    gs = GameState.new_game(42)
    assert gs.center == {}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_state():
    gs1 = GameState.new_game(7)
    gs2 = GameState.new_game(7)
    assert gs1.factories == gs2.factories
    assert gs1.bag == gs2.bag


def test_different_seeds_produce_different_states():
    gs1 = GameState.new_game(1)
    gs2 = GameState.new_game(2)
    assert gs1.factories != gs2.factories
