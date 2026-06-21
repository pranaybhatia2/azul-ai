"""Phase 1: GameState construction and basic invariants."""
import pytest
from azul.state import GameState, FactoryDisplay, PatternLine, Tile


def test_two_player_game_creates_five_factories():
    gs = GameState(num_players=2)
    assert len(gs.factory_displays) == 5


def test_four_player_game_creates_nine_factories():
    gs = GameState(num_players=4)
    assert len(gs.factory_displays) == 9


def test_player_boards_created_for_each_player():
    gs = GameState(num_players=3)
    assert len(gs.player_boards) == 3


def test_pattern_lines_have_correct_capacities():
    gs = GameState(num_players=2)
    board = gs.player_boards[0]
    capacities = [pl.capacity for pl in board.pattern_lines]
    assert capacities == [1, 2, 3, 4, 5]


def test_initial_scores_are_zero():
    gs = GameState(num_players=2)
    for board in gs.player_boards:
        assert board.score == 0


def test_factory_display_empty_on_init():
    fd = FactoryDisplay()
    assert fd.is_empty()


def test_pattern_line_full_when_capacity_reached():
    pl = PatternLine(capacity=2)
    pl.tiles = [Tile("blue"), Tile("blue")]
    assert pl.is_full()


def test_pattern_line_not_full_when_partial():
    pl = PatternLine(capacity=3)
    pl.tiles = [Tile("red")]
    assert not pl.is_full()
