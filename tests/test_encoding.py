"""Tests for the NN encodings (pure, no torch)."""
from azul.encoding import (
    encode_state, move_to_index, index_to_move, legal_mask,
    POLICY_SIZE, STATE_SIZE,
)
from azul.state import Color, GameState, Move, CENTER, FLOOR


# ---------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------

def test_index_roundtrip_covers_all():
    seen = set()
    for i in range(POLICY_SIZE):
        m = index_to_move(i)
        assert move_to_index(m) == i
        seen.add(i)
    assert len(seen) == POLICY_SIZE == 180


def test_specific_move_indices():
    # factory 0, blue(0), row 0 -> 0
    assert move_to_index(Move(0, Color.BLUE, 0)) == 0
    # center maps to source slot 5, floor to dest slot 5
    assert move_to_index(Move(CENTER, Color.WHITE, FLOOR)) == 5 * 30 + 4 * 6 + 5


def test_move_index_distinct():
    gs = GameState.new_game(42)
    idxs = [move_to_index(m) for m in gs.legal_moves()]
    assert len(idxs) == len(set(idxs))   # no collisions


# ---------------------------------------------------------------------------
# Legal mask
# ---------------------------------------------------------------------------

def test_legal_mask_marks_exactly_legal_moves():
    gs = GameState.new_game(42)
    moves = gs.legal_moves()
    mask = legal_mask(moves)
    assert len(mask) == POLICY_SIZE
    assert sum(mask) == len(moves)
    for m in moves:
        assert mask[move_to_index(m)] == 1.0


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def test_encode_state_length_and_finite():
    gs = GameState.new_game(42)
    v = encode_state(gs)
    assert len(v) == STATE_SIZE
    assert all(isinstance(x, float) for x in v)
    assert all(-1e6 < x < 1e6 for x in v)


def test_encoding_is_canonical_to_player_to_move():
    """P0-to-move with boards [A,B] must encode identically to P1-to-move with
    boards [B,A] — the current player's board is always encoded first."""
    gs = GameState.new_game(42)
    gs.player_boards[0].score = 13
    gs.player_boards[1].score = 27

    mirror = gs.clone()
    mirror.player_boards = [gs.player_boards[1], gs.player_boards[0]]
    mirror.current_player = 1

    assert encode_state(gs) == encode_state(mirror)


def test_encoding_changes_with_my_score():
    gs = GameState.new_game(42)
    before = encode_state(gs)
    gs.player_boards[gs.current_player].score += 10
    assert encode_state(gs) != before
