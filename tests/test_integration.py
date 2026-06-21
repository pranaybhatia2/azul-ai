"""Integration tests — exercise multiple methods composed together.

These catch contradictions that unit tests miss: e.g. apply() and
tile_wall_and_score() disagreeing on how the first-player marker is tracked.
"""
from azul.state import Color, GameState, Move, FLOOR_PENALTIES, CENTER, FLOOR


def test_marker_claim_through_to_scoring_penalizes_one_tile():
    """A player who claims only the marker (no other floor tiles) should
    lose exactly one floor tile's worth of points (-1), not two."""
    gs = GameState()
    gs.center = {Color.BLACK: 2}
    gs.first_player_marker_in_center = True
    gs.player_boards[0].score = 10

    # Player 0 takes the 2 black tiles into row 1 (capacity 2 — both fit,
    # no overflow to floor) and claims the marker.
    gs.apply(Move(source=CENTER, color=Color.BLACK, dest_line=1))
    assert gs.player_boards[0].has_first_player_marker
    assert gs.player_boards[0].floor_count == 0

    gs.tile_wall_and_score()

    # Row 1 isn't complete (2 tiles fit a capacity-2 line — it IS complete).
    # Wait: capacity of row 1 is 2, so it completed and scored +1 (isolated).
    # Floor penalty from the marker alone is FLOOR_PENALTIES[0] == -1.
    expected = 10 + 1 + FLOOR_PENALTIES[0]
    assert gs.player_boards[0].score == expected


def test_marker_plus_overflow_tiles_penalized_together():
    """Marker + 2 overflow tiles = 3 floor tiles total."""
    gs = GameState()
    gs.center = {Color.BLUE: 3}
    gs.first_player_marker_in_center = True
    gs.player_boards[0].score = 20

    # 3 blue into row 0 (capacity 1): 1 placed, 2 overflow to floor.
    gs.apply(Move(source=CENTER, color=Color.BLUE, dest_line=0))
    assert gs.player_boards[0].floor_count == 2
    assert gs.player_boards[0].has_first_player_marker

    gs.tile_wall_and_score()

    # Row 0 completes → +1 (isolated). Floor = 2 overflow + 1 marker = 3 tiles.
    floor_penalty = sum(FLOOR_PENALTIES[:3])
    expected = max(0, 20 + 1 + floor_penalty)
    assert gs.player_boards[0].score == expected


def test_full_round_runs_to_completion():
    """A seeded game can be played move-by-move until the round ends,
    then scored, without crashing or producing illegal states."""
    gs = GameState.new_game(seed=123)

    moves_played = 0
    while not gs.is_round_over():
        moves = gs.legal_moves()
        assert moves, "should always have a legal move until round is over"
        gs.apply(moves[0])
        moves_played += 1
        assert moves_played < 100, "round should terminate"

    gs.tile_wall_and_score()

    # After scoring, all floors are cleared and no marker is held.
    for board in gs.player_boards:
        assert board.floor_count == 0
        assert not board.has_first_player_marker
        assert board.score >= 0
