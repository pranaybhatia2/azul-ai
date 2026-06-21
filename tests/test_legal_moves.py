"""Tests for GameState.legal_moves()."""
import pytest
from azul.state import (
    Color, GameState, Move, WALL_PATTERN,
    CENTER, FLOOR,
)


def make_gs() -> GameState:
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 3, Color.RED: 1}
    gs.factories[1] = {Color.YELLOW: 4}
    gs.center = {Color.BLACK: 2}
    return gs


# ---------------------------------------------------------------------------
# Basic enumeration
# ---------------------------------------------------------------------------

def test_returns_list_of_moves():
    gs = make_gs()
    moves = gs.legal_moves()
    assert isinstance(moves, list)
    assert all(isinstance(m, Move) for m in moves)


def test_empty_factories_produce_no_moves():
    gs = GameState()  # all factories empty, center empty
    assert gs.legal_moves() == []


def test_moves_include_all_colors_in_factory():
    gs = make_gs()
    moves = gs.legal_moves()
    sources_colors = {(m.source, m.color) for m in moves}
    assert (0, Color.BLUE) in sources_colors
    assert (0, Color.RED) in sources_colors


def test_moves_include_center_colors():
    gs = make_gs()
    moves = gs.legal_moves()
    sources_colors = {(m.source, m.color) for m in moves}
    assert (CENTER, Color.BLACK) in sources_colors


def test_no_moves_from_empty_factory():
    gs = make_gs()
    # factory 2 is empty
    moves = gs.legal_moves()
    assert not any(m.source == 2 for m in moves)


# ---------------------------------------------------------------------------
# Valid pattern line destinations
# ---------------------------------------------------------------------------

def test_floor_always_a_valid_destination():
    gs = make_gs()
    moves = gs.legal_moves()
    assert any(m.source == 0 and m.color == Color.BLUE and m.dest_line == FLOOR
               for m in moves)


def test_empty_pattern_line_is_valid_dest():
    gs = make_gs()
    moves = gs.legal_moves()
    # row 2 is empty and can accept blue (capacity 3, blue fits)
    assert any(m.source == 0 and m.color == Color.BLUE and m.dest_line == 2
               for m in moves)


def test_full_pattern_line_not_a_valid_dest():
    gs = make_gs()
    board = gs.player_boards[0]
    # Fill row 2 with blue
    board.pattern_lines[2].color = Color.BLUE
    board.pattern_lines[2].count = 3   # capacity 3 — full
    moves = gs.legal_moves()
    assert not any(m.dest_line == 2 and m.color == Color.BLUE for m in moves)


def test_wrong_color_pattern_line_not_valid():
    gs = make_gs()
    board = gs.player_boards[0]
    # Row 2 already has red
    board.pattern_lines[2].color = Color.RED
    board.pattern_lines[2].count = 1
    moves = gs.legal_moves()
    # Blue cannot go into row 2
    assert not any(m.color == Color.BLUE and m.dest_line == 2 for m in moves)


def test_wall_slot_already_filled_blocks_dest():
    gs = make_gs()
    board = gs.player_boards[0]
    # Find which column blue belongs in row 2, fill it
    col = next(c for c in range(5) if WALL_PATTERN[2][c] == Color.BLUE)
    board.wall[2][col] = Color.BLUE
    moves = gs.legal_moves()
    assert not any(m.color == Color.BLUE and m.dest_line == 2 for m in moves)


def test_only_floor_when_all_pattern_lines_blocked():
    gs = GameState()
    board = gs.player_boards[0]
    # Put blue in center
    gs.center = {Color.BLUE: 2}
    # Block every pattern line for blue: fill wall slot or wrong color
    for row in range(5):
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == Color.BLUE)
        board.wall[row][col] = Color.BLUE
    moves = gs.legal_moves()
    blue_moves = [m for m in moves if m.color == Color.BLUE]
    assert blue_moves == [Move(CENTER, Color.BLUE, FLOOR)]


# ---------------------------------------------------------------------------
# Correct player
# ---------------------------------------------------------------------------

def test_legal_moves_for_current_player():
    gs = make_gs()
    # Give player 1 a filled wall slot that would block blue in row 2
    board1 = gs.player_boards[1]
    col = next(c for c in range(5) if WALL_PATTERN[2][c] == Color.BLUE)
    board1.wall[2][col] = Color.BLUE

    gs.current_player = 1
    moves = gs.legal_moves()
    # Row 2 should be blocked for blue from player 1's perspective
    assert not any(m.color == Color.BLUE and m.dest_line == 2 for m in moves)


# ---------------------------------------------------------------------------
# Sorted order and deduplication
# ---------------------------------------------------------------------------

def test_no_duplicate_moves():
    gs = make_gs()
    moves = gs.legal_moves()
    assert len(moves) == len(set(moves))


def test_moves_sorted_by_source_color_dest():
    gs = make_gs()
    moves = gs.legal_moves()
    assert moves == sorted(moves, key=lambda m: (m.source, m.color, m.dest_line))
