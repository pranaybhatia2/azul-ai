"""Tests for GameState.apply(move)."""
import pytest
from azul.state import (
    Color, GameState, Move, PATTERN_LINE_CAPACITY,
    CENTER, FLOOR,
)


def make_gs() -> GameState:
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 3, Color.RED: 1}
    gs.factories[1] = {Color.YELLOW: 4}
    gs.center = {Color.BLACK: 2}
    return gs


# ---------------------------------------------------------------------------
# Taking from a factory
# ---------------------------------------------------------------------------

def test_takes_all_of_color_from_factory():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))
    assert Color.BLUE not in gs.factories[0]


def test_factory_leftovers_go_to_center():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))
    assert gs.center.get(Color.RED, 0) == 1


def test_factory_is_empty_after_take():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))
    assert gs.factories[0] == {}


def test_takes_all_tiles_when_factory_is_one_color():
    gs = make_gs()
    gs.apply(Move(source=1, color=Color.YELLOW, dest_line=3))
    assert gs.factories[1] == {}
    assert Color.YELLOW not in gs.center


# ---------------------------------------------------------------------------
# Taking from center
# ---------------------------------------------------------------------------

def test_takes_color_from_center():
    gs = make_gs()
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert Color.BLACK not in gs.center or gs.center[Color.BLACK] == 0


def test_first_player_marker_claimed_from_center():
    gs = make_gs()
    gs.first_player_marker_in_center = True
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert gs.player_boards[0].has_first_player_marker
    assert not gs.first_player_marker_in_center


def test_first_player_marker_not_added_to_floor_count():
    # The marker is tracked by has_first_player_marker, not floor_count.
    # Its penalty is applied at scoring time, so floor_count stays clean here.
    gs = make_gs()
    gs.first_player_marker_in_center = True
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert gs.player_boards[0].floor_count == 0
    assert gs.player_boards[0].has_first_player_marker


def test_no_marker_in_center_does_not_set_flag():
    gs = make_gs()
    gs.first_player_marker_in_center = False
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert not gs.player_boards[0].has_first_player_marker


# ---------------------------------------------------------------------------
# Placing onto a pattern line
# ---------------------------------------------------------------------------

def test_tiles_placed_on_pattern_line():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))  # 3 blue, capacity 3
    pl = gs.player_boards[0].pattern_lines[2]
    assert pl.color == Color.BLUE
    assert pl.count == 3


def test_partial_fill_no_overflow():
    gs = make_gs()
    # 3 blue into row 4 (capacity 5) — fits with room to spare
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=4))
    pl = gs.player_boards[0].pattern_lines[4]
    assert pl.count == 3
    assert gs.player_boards[0].floor_count == 0


def test_overflow_goes_to_floor():
    gs = make_gs()
    # 3 blue into row 0 (capacity 1) — 1 placed, 2 overflow
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=0))
    pl = gs.player_boards[0].pattern_lines[0]
    assert pl.count == 1
    assert gs.player_boards[0].floor_count == 2


def test_dest_floor_sends_all_tiles_to_floor():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=FLOOR))
    assert gs.player_boards[0].floor_count == 3


# ---------------------------------------------------------------------------
# Player advancement
# ---------------------------------------------------------------------------

def test_current_player_advances():
    gs = make_gs()
    assert gs.current_player == 0
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))
    assert gs.current_player == 1


def test_current_player_wraps_back():
    gs = make_gs()
    gs.apply(Move(source=0, color=Color.BLUE, dest_line=2))
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert gs.current_player == 0
