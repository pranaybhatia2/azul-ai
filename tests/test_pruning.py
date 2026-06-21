"""Tests for NN action-space pruning."""
from azul.pruning import candidate_moves
from azul.state import Color, GameState, Move, WALL_PATTERN, FLOOR, CENTER


def test_drops_optional_floor_dumps():
    gs = GameState.new_game(42)
    cands = candidate_moves(gs)
    legal = gs.legal_moves()
    # Fewer than full legal (some optional floors dropped) but non-empty.
    assert 0 < len(cands) < len(legal)
    # No optional floor-dumps remain: every floor move kept must be forced
    # (its (source,color) has no line option among candidates).
    line_pairs = {(m.source, m.color) for m in cands if m.dest_line != FLOOR}
    for m in cands:
        if m.dest_line == FLOOR:
            assert (m.source, m.color) not in line_pairs


def test_keeps_forced_floor():
    gs = GameState()
    gs.center = {Color.BLUE: 2}
    board = gs.player_boards[0]
    for row in range(5):                       # block blue everywhere
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == Color.BLUE)
        board.wall[row][col] = Color.BLUE
    cands = candidate_moves(gs)
    # Blue's only option is the floor -> must be kept.
    assert Move(CENTER, Color.BLUE, FLOOR) in cands


def test_candidates_are_subset_of_legal():
    gs = GameState.new_game(7)
    assert set(candidate_moves(gs)).issubset(set(gs.legal_moves()))
