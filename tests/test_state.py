"""Phase 1 tests — written before implementation, per the deal."""
import pytest
from azul.state import (
    Color, GameState, Move, PatternLine, PlayerBoard,
    WALL_PATTERN, PATTERN_LINE_CAPACITY, FLOOR_PENALTIES,
    CENTER, FLOOR, NUM_FACTORIES,
)


# ---------------------------------------------------------------------------
# Constants and invariants
# ---------------------------------------------------------------------------

def test_wall_pattern_each_color_appears_once_per_row():
    for row in range(5):
        assert sorted(WALL_PATTERN[row]) == list(range(5))

def test_wall_pattern_each_color_appears_once_per_col():
    for col in range(5):
        col_colors = [WALL_PATTERN[row][col] for row in range(5)]
        assert sorted(col_colors) == list(range(5))

def test_pattern_line_capacities():
    assert PATTERN_LINE_CAPACITY == [1, 2, 3, 4, 5]

def test_floor_penalties_length():
    assert len(FLOOR_PENALTIES) == 7


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_bag_has_20_of_each_color():
    gs = GameState()
    for color in Color:
        assert gs.bag[color] == 20

def test_initial_factories_are_empty():
    gs = GameState()
    for factory in gs.factories:
        assert factory == {}

def test_initial_center_is_empty():
    gs = GameState()
    assert gs.center == {}

def test_initial_scores_are_zero():
    gs = GameState()
    for board in gs.player_boards:
        assert board.score == 0

def test_initial_pattern_lines_are_empty():
    gs = GameState()
    for board in gs.player_boards:
        for pl in board.pattern_lines:
            assert pl.color is None and pl.count == 0

def test_initial_wall_is_empty():
    gs = GameState()
    for board in gs.player_boards:
        for row in board.wall:
            assert all(cell is None for cell in row)

def test_num_factories_for_two_players():
    gs = GameState()
    assert len(gs.factories) == NUM_FACTORIES


# ---------------------------------------------------------------------------
# Move — hashable and immutable
# ---------------------------------------------------------------------------

def test_move_is_hashable():
    m = Move(source=0, color=Color.BLUE, dest_line=2)
    assert hash(m) is not None

def test_moves_with_same_fields_are_equal():
    m1 = Move(source=CENTER, color=Color.RED, dest_line=FLOOR)
    m2 = Move(source=CENTER, color=Color.RED, dest_line=FLOOR)
    assert m1 == m2

def test_move_is_immutable():
    m = Move(source=1, color=Color.YELLOW, dest_line=3)
    with pytest.raises(Exception):
        m.color = Color.BLACK

def test_moves_usable_in_set():
    moves = {
        Move(0, Color.BLUE, 0),
        Move(0, Color.BLUE, 0),   # duplicate
        Move(1, Color.RED, 2),
    }
    assert len(moves) == 2


# ---------------------------------------------------------------------------
# clone()
# ---------------------------------------------------------------------------

def test_clone_is_independent():
    gs = GameState()
    gs2 = gs.clone()
    gs2.player_boards[0].score = 99
    assert gs.player_boards[0].score == 0

def test_clone_bag_is_independent():
    gs = GameState()
    gs2 = gs.clone()
    gs2.bag[Color.BLUE] = 0
    assert gs.bag[Color.BLUE] == 20


# ---------------------------------------------------------------------------
# Stubs raise NotImplementedError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,args", [
    ("legal_moves",         []),
    ("encode",              []),
])
def test_stub_raises(method, args):
    gs = GameState()
    with pytest.raises(NotImplementedError):
        getattr(gs, method)(*args)

