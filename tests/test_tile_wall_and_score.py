"""
Tests for GameState.tile_wall_and_score().

Spec (agreed):
- Processes both players in one call
- Mutates in place, returns None
- Complete pattern lines: one tile → wall, extras → discard, line cleared
- Incomplete pattern lines: untouched
- Adjacency scoring:
    - isolated tile: +1
    - has neighbors: +len(horizontal run) + len(vertical run), tile counted in each
- Floor penalty: sum(FLOOR_PENALTIES[:min(floor_count, 7)]), score floor-clamped at 0
- First player marker: returns to center (center[FIRST_PLAYER] flag, not a Color —
  handled by clearing has_first_player_marker on the board)
- Floor cleared after penalty applied
- End-of-game bonuses NOT handled here
"""
import pytest
from azul.state import (
    Color, GameState, PlayerBoard, PatternLine,
    WALL_PATTERN, FLOOR_PENALTIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gs(**kwargs) -> GameState:
    """GameState with two fresh PlayerBoards, keyword-overridable."""
    return GameState(**kwargs)


def place_tile(board: PlayerBoard, row: int, col: int) -> None:
    """Put the correct color on the wall at (row, col)."""
    board.wall[row][col] = WALL_PATTERN[row][col]


def complete_line(board: PlayerBoard, row: int) -> None:
    """Fill pattern line `row` with its wall color."""
    color = WALL_PATTERN[row][row]   # diagonal: column index == row index
    board.pattern_lines[row].color = color
    board.pattern_lines[row].count = row + 1   # capacity = row + 1


# ---------------------------------------------------------------------------
# Wall placement
# ---------------------------------------------------------------------------

def test_complete_line_places_tile_on_wall():
    gs = make_gs()
    complete_line(gs.player_boards[0], 0)   # row 0, capacity 1
    gs.tile_wall_and_score()
    assert gs.player_boards[0].wall[0][0] == WALL_PATTERN[0][0]


def test_complete_line_clears_pattern_line():
    gs = make_gs()
    complete_line(gs.player_boards[0], 1)
    gs.tile_wall_and_score()
    pl = gs.player_boards[0].pattern_lines[1]
    assert pl.color is None and pl.count == 0


def test_incomplete_line_not_cleared():
    gs = make_gs()
    board = gs.player_boards[0]
    board.pattern_lines[2].color = Color.RED
    board.pattern_lines[2].count = 1   # capacity 3, not full
    gs.tile_wall_and_score()
    assert board.pattern_lines[2].color == Color.RED
    assert board.pattern_lines[2].count == 1


def test_excess_tiles_go_to_discard():
    gs = make_gs()
    complete_line(gs.player_boards[0], 2)   # row 2: capacity 3, 1 placed, 2 to discard
    color = WALL_PATTERN[2][2]
    before = gs.discard.get(color, 0)
    gs.tile_wall_and_score()
    assert gs.discard[color] == before + 2


def test_both_players_processed():
    gs = make_gs()
    complete_line(gs.player_boards[0], 0)
    complete_line(gs.player_boards[1], 0)
    gs.tile_wall_and_score()
    assert gs.player_boards[0].wall[0][0] is not None
    assert gs.player_boards[1].wall[0][0] is not None


# ---------------------------------------------------------------------------
# Adjacency scoring
# ---------------------------------------------------------------------------

def test_isolated_tile_scores_one():
    gs = make_gs()
    complete_line(gs.player_boards[0], 0)
    gs.tile_wall_and_score()
    assert gs.player_boards[0].score == 1


def test_one_horizontal_neighbor_scores_two():
    gs = make_gs()
    board = gs.player_boards[0]
    # Pre-fill wall[0][1] then place at [0][0] — run of 2 horizontal, no vertical
    place_tile(board, 0, 1)
    complete_line(board, 0)   # places at [0][0]
    gs.tile_wall_and_score()
    assert board.score == 2


def test_one_vertical_neighbor_scores_two():
    gs = make_gs()
    board = gs.player_boards[0]
    # row-0 diagonal col is 0; row-1 diagonal col is 1 — different columns.
    # Use row 0 col 0, and pre-fill row 1 col 0 to create a vertical neighbor.
    place_tile(board, 1, 0)
    complete_line(board, 0)   # places at [0][0]
    gs.tile_wall_and_score()
    assert board.score == 2


def test_both_horizontal_and_vertical_neighbors_score_both_runs():
    gs = make_gs()
    board = gs.player_boards[0]
    # Place at [0][0]. Pre-fill [0][1] (horizontal) and [1][0] (vertical).
    # Horizontal run = 2, vertical run = 2 → total 4.
    place_tile(board, 0, 1)
    place_tile(board, 1, 0)
    complete_line(board, 0)
    gs.tile_wall_and_score()
    assert board.score == 4


def test_long_horizontal_run():
    gs = make_gs()
    board = gs.player_boards[0]
    # Fill [0][1], [0][2], [0][3], [0][4], place at [0][0] → run of 5
    for col in range(1, 5):
        place_tile(board, 0, col)
    complete_line(board, 0)
    gs.tile_wall_and_score()
    assert board.score == 5


def test_tile_in_middle_of_existing_run():
    gs = make_gs()
    board = gs.player_boards[0]
    # Row 2: WALL_PATTERN[2] = [RED, BLUE, YELLOW, WHITE, BLACK] (formula: (col-2)%5)
    # Place at col 2 (YELLOW). Pre-fill col 1 and col 3 — run of 3.
    place_tile(board, 2, 1)
    place_tile(board, 2, 3)
    board.pattern_lines[2].color = WALL_PATTERN[2][2]
    board.pattern_lines[2].count = 3
    gs.tile_wall_and_score()
    assert board.score == 3


def test_multiple_complete_lines_scores_accumulate():
    gs = make_gs()
    board = gs.player_boards[0]
    # Complete rows 0 and 1; both isolated → 1 + 1 = 2
    complete_line(board, 0)
    complete_line(board, 1)
    gs.tile_wall_and_score()
    assert board.score == 2


# ---------------------------------------------------------------------------
# Floor penalties
# ---------------------------------------------------------------------------

def test_floor_two_tiles_penalty():
    gs = make_gs()
    gs.player_boards[0].floor_count = 2
    gs.player_boards[0].score = 10
    gs.tile_wall_and_score()
    assert gs.player_boards[0].score == 10 + FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1]


def test_floor_penalty_does_not_go_below_zero():
    gs = make_gs()
    gs.player_boards[0].floor_count = 7
    gs.player_boards[0].score = 0
    gs.tile_wall_and_score()
    assert gs.player_boards[0].score == 0


def test_floor_cleared_after_scoring():
    gs = make_gs()
    gs.player_boards[0].floor_count = 3
    gs.player_boards[0].score = 20
    gs.tile_wall_and_score()
    assert gs.player_boards[0].floor_count == 0


def test_floor_penalty_capped_at_seven():
    gs = make_gs()
    gs.player_boards[0].floor_count = 9   # more than max — only 7 penalties apply
    gs.player_boards[0].score = 100
    gs.tile_wall_and_score()
    expected = max(0, 100 + sum(FLOOR_PENALTIES))
    assert gs.player_boards[0].score == expected


def test_first_player_marker_cleared():
    gs = make_gs()
    gs.player_boards[1].has_first_player_marker = True
    gs.tile_wall_and_score()
    assert not gs.player_boards[1].has_first_player_marker


def test_first_player_marker_counts_as_floor_tile():
    gs = make_gs()
    board = gs.player_boards[0]
    board.has_first_player_marker = True
    board.floor_count = 1       # 1 regular + marker = 2 floor tiles total
    board.score = 10
    gs.tile_wall_and_score()
    expected = max(0, 10 + FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1])
    assert board.score == expected


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

def test_returns_none():
    gs = make_gs()
    result = gs.tile_wall_and_score()
    assert result is None
