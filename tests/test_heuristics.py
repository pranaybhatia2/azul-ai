"""Tests for the position evaluation function."""
from azul.heuristics import (
    evaluate, threat_score, threat_aware_evaluate, PARTIAL_WEIGHT, THREAT_WEIGHT,
    _pending_wall_points,
)
from azul.state import Color, GameState, WALL_PATTERN, PATTERN_LINE_CAPACITY


def complete_line(board, row):
    color = WALL_PATTERN[row][row]
    board.pattern_lines[row].color = color
    board.pattern_lines[row].count = row + 1


def test_empty_board_evaluates_to_zero():
    gs = GameState()
    assert evaluate(gs, 0) == 0.0


def test_score_contributes_directly():
    gs = GameState()
    gs.player_boards[0].score = 15
    assert evaluate(gs, 0) == 15.0


def test_floor_penalty_lowers_value():
    gs = GameState()
    gs.player_boards[0].score = 10
    gs.player_boards[0].floor_count = 2   # -1 + -1
    assert evaluate(gs, 0) == 10 - 2


def test_marker_counts_in_floor_penalty():
    gs = GameState()
    gs.player_boards[0].score = 10
    gs.player_boards[0].has_first_player_marker = True   # 1 floor tile -> -1
    assert evaluate(gs, 0) == 10 - 1


def test_completed_line_adds_pending_points():
    gs = GameState()
    complete_line(gs.player_boards[0], 0)   # isolated tile -> +1 pending
    assert evaluate(gs, 0) == 1.0


def test_pending_points_use_real_adjacency():
    gs = GameState()
    board = gs.player_boards[0]
    # Pre-place a horizontal neighbor so the completed line scores a run of 2.
    board.wall[0][1] = WALL_PATTERN[0][1]
    complete_line(board, 0)   # places at [0][0], adjacent to [0][1] -> run of 2
    assert _pending_wall_points(board) == 2


def test_partial_line_adds_progress():
    gs = GameState()
    board = gs.player_boards[0]
    board.pattern_lines[4].color = Color.BLUE
    board.pattern_lines[4].count = 2   # capacity 5
    expected = PARTIAL_WEIGHT * (2 / 5)
    assert abs(evaluate(gs, 0) - expected) < 1e-9


def test_evaluates_the_requested_player():
    gs = GameState()
    gs.player_boards[1].score = 20
    assert evaluate(gs, 1) == 20.0
    assert evaluate(gs, 0) == 0.0


# --- threat-aware evaluation ------------------------------------------------

def test_threat_zero_with_no_partial_lines():
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 3}
    assert threat_score(gs, 0) == 0.0


def test_threat_present_when_needed_tile_is_available():
    # Player has 2/3 of a Blue line (row 2). Wall row 2 col2 = Blue is open.
    # One Blue available -> the line is finishable -> positive threat.
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 1}
    pl = gs.player_boards[0].pattern_lines[2]
    pl.color, pl.count = Color.BLUE, 2
    assert threat_score(gs, 0) > 0.0


def test_threat_vanishes_when_needed_tile_is_unavailable():
    # Same partial line, but no Blue available anywhere -> no imminent threat.
    gs = GameState()
    gs.factories[0] = {Color.RED: 1}
    pl = gs.player_boards[0].pattern_lines[2]
    pl.color, pl.count = Color.BLUE, 2
    assert threat_score(gs, 0) == 0.0


def test_denial_raises_relative_threat_aware_value():
    # OPPONENT (player 1) is one Blue from completing row 2; exactly 1 Blue
    # exists, in factory 0. If I (player 0) take that Blue, the opponent can no
    # longer complete -> my relative threat-aware value should be higher than if
    # I leave the Blue available.
    def make():
        gs = GameState()
        gs.factories[0] = {Color.BLUE: 1}
        pl = gs.player_boards[1].pattern_lines[2]
        pl.color, pl.count = Color.BLUE, 2
        return gs

    leave = make()  # Blue still available to the opponent
    deny = make()
    deny.factories[0] = {}  # I took the Blue -> opponent's threat removed

    def rel(state):
        return (threat_aware_evaluate(state, 0)
                - threat_aware_evaluate(state, 1))

    assert rel(deny) > rel(leave)


def test_threat_aware_equals_evaluate_when_no_threats():
    gs = GameState()
    gs.player_boards[0].score = 12
    assert threat_aware_evaluate(gs, 0) == evaluate(gs, 0)
