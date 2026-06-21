"""Position evaluation for Azul.

evaluate(state, player) -> float estimates how good `state` is for `player`.
GreedyAgent maximizes it one ply deep; Phase 4 minimax will reuse it as the
leaf evaluator.

The estimate combines:
  - locked-in score
  - pending wall points: the real adjacency points that COMPLETE pattern
    lines will score when tiled at round-end (computed via the engine's own
    scoring, so it's exact, not a guess)
  - a small reward for progress on incomplete lines
  - the floor penalty for tiles currently destined for the floor
"""
from __future__ import annotations

from azul.state import (
    Color, GameState, PlayerBoard, PATTERN_LINE_CAPACITY, WALL_PATTERN,
    FLOOR_PENALTIES,
)

# How much a single staged tile on an incomplete line is worth, as a fraction
# of a point. Small: progress is good but unrealized.
PARTIAL_WEIGHT = 0.5


def _pending_wall_points(board: PlayerBoard) -> float:
    """Adjacency points the board's COMPLETE pattern lines would score if
    tiled now (top-to-bottom, matching tile_wall_and_score)."""
    wall = [row[:] for row in board.wall]
    pts = 0
    for row, pl in enumerate(board.pattern_lines):
        if pl.color is not None and pl.count == PATTERN_LINE_CAPACITY[row]:
            col = next(c for c in range(5) if WALL_PATTERN[row][c] == pl.color)
            if wall[row][col] is None:
                wall[row][col] = pl.color
                pts += GameState._adjacency_score(wall, row, col)
    return pts


def _floor_penalty(board: PlayerBoard) -> int:
    eff = board.floor_count + (1 if board.has_first_player_marker else 0)
    return sum(FLOOR_PENALTIES[:min(eff, len(FLOOR_PENALTIES))])


def evaluate(state: GameState, player: int) -> float:
    board = state.player_boards[player]
    value = float(board.score)
    value += _pending_wall_points(board)
    for row, pl in enumerate(board.pattern_lines):
        cap = PATTERN_LINE_CAPACITY[row]
        if 0 < pl.count < cap:
            value += PARTIAL_WEIGHT * (pl.count / cap)
    value += _floor_penalty(board)
    return value
