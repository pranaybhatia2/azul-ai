"""Neural-network encodings for Azul (pure, no torch dependency).

Two mappings the network needs:
  * encode_state(state) -> fixed-length list[float], CANONICAL: the player to
    move is always encoded first ("me"), so the net generalizes across seats.
  * move <-> policy index over a fixed 180-move space, plus a legal-move mask.

Move-index space: 6 sources (factory 0-4, center=5) x 5 colors x 6 dests
(row 0-4, floor=5) = 180.
"""
from __future__ import annotations

from azul.state import (
    Color, GameState, PlayerBoard, Move,
    PATTERN_LINE_CAPACITY, CENTER, FLOOR, NUM_FACTORIES,
)

NUM_SOURCES = 6      # factories 0-4 + center
NUM_DESTS = 6        # rows 0-4 + floor
POLICY_SIZE = NUM_SOURCES * len(Color) * NUM_DESTS   # 180

# State vector layout (see encode_state); kept as a constant for the net.
STATE_SIZE = (
    NUM_FACTORIES * len(Color)   # factories
    + len(Color)                 # center
    + 1                          # marker in center
    + 1                          # round number
    + 2 * (25 + 5 * 6 + 3)       # two boards (wall + lines + floor/marker/score)
)


# ---------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------

def _src_to_idx(source: int) -> int:
    return NUM_FACTORIES if source == CENTER else source


def _idx_to_src(i: int) -> int:
    return CENTER if i == NUM_FACTORIES else i


def _dest_to_idx(dest: int) -> int:
    return 5 if dest == FLOOR else dest


def _idx_to_dest(i: int) -> int:
    return FLOOR if i == 5 else i


def move_to_index(move: Move) -> int:
    return (_src_to_idx(move.source) * (len(Color) * NUM_DESTS)
            + int(move.color) * NUM_DESTS
            + _dest_to_idx(move.dest_line))


def index_to_move(index: int) -> Move:
    src_i, rem = divmod(index, len(Color) * NUM_DESTS)
    color_i, dest_i = divmod(rem, NUM_DESTS)
    return Move(_idx_to_src(src_i), Color(color_i), _idx_to_dest(dest_i))


def legal_mask(moves: list[Move]) -> list[float]:
    """Length-POLICY_SIZE vector, 1.0 at legal move indices else 0.0."""
    mask = [0.0] * POLICY_SIZE
    for m in moves:
        mask[move_to_index(m)] = 1.0
    return mask


# ---------------------------------------------------------------------------
# State -> feature vector
# ---------------------------------------------------------------------------

def _encode_board(b: PlayerBoard) -> list[float]:
    feats: list[float] = []
    # Wall: 25 binary (filled or not).
    for r in range(5):
        for c in range(5):
            feats.append(1.0 if b.wall[r][c] is not None else 0.0)
    # Pattern lines: per row, color one-hot (5) + fill fraction (1).
    for row in range(5):
        pl = b.pattern_lines[row]
        onehot = [0.0] * len(Color)
        if pl.color is not None:
            onehot[int(pl.color)] = 1.0
        feats.extend(onehot)
        feats.append(pl.count / PATTERN_LINE_CAPACITY[row])
    feats.append(b.floor_count / 7.0)
    feats.append(1.0 if b.has_first_player_marker else 0.0)
    feats.append(b.score / 100.0)
    return feats


def encode_state(state: GameState) -> list[float]:
    """Canonical fixed-length encoding from the current player's perspective."""
    me = state.current_player
    opp = 1 - me

    feats: list[float] = []
    for f in state.factories:                       # factories (counts / 4)
        for c in Color:
            feats.append(f.get(c, 0) / 4.0)
    for c in Color:                                 # center (counts / 4)
        feats.append(state.center.get(c, 0) / 4.0)
    feats.append(1.0 if state.first_player_marker_in_center else 0.0)
    feats.append(state.round_number / 10.0)

    feats.extend(_encode_board(state.player_boards[me]))    # me first
    feats.extend(_encode_board(state.player_boards[opp]))   # then opponent
    return feats
