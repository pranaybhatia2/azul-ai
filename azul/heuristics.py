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


# Weight on unrealized "threats" — pattern lines a player could still complete
# this round given the tiles currently available to take. Lower than a locked
# point because the tiles must still be acquired and are contested. Used by
# threat-aware MCTS so the leaf eval rewards denying the opponent those tiles.
THREAT_WEIGHT = 0.5


def _available_tiles(state: GameState) -> dict[Color, int]:
    """All tiles a player could take this turn, pooled by color (factories +
    center). Taking tiles removes them here, which is what makes denial show up
    in threat_score at the resulting state."""
    avail = {c: 0 for c in Color}
    for factory in state.factories:
        for c, n in factory.items():
            avail[c] += n
    for c, n in state.center.items():
        avail[c] += n
    return avail


def threat_score(state: GameState, player: int) -> float:
    """Forward-looking completion potential for `player`: for each partially
    filled pattern line, the wall score they'd lock in if they can still get
    the tiles they need from what's currently available. The opponent's
    threat_score drops once you take the tiles they were counting on, so a
    relative threat-aware eval rewards blocking."""
    board = state.player_boards[player]
    avail = _available_tiles(state)
    wall = board.wall
    total = 0.0
    for row, pl in enumerate(board.pattern_lines):
        cap = PATTERN_LINE_CAPACITY[row]
        if pl.color is None or not (0 < pl.count < cap):
            continue  # only a partial line has an imminent completion threat
        col = next(k for k in range(5) if WALL_PATTERN[row][k] == pl.color)
        if wall[row][col] is not None:
            continue  # that color is already walled in this row — dead line
        needed = cap - pl.count
        comp_val = GameState._adjacency_score(wall, row, col)
        # Broadened: credit any developing line whose color is still available,
        # not just ones finishable this turn — closer-to-done lines count more
        # (proximity) and lines they can fully reach count fully (reach). This
        # surfaces 2-away blocking opportunities, not only last-tile ones.
        proximity = pl.count / cap
        reach = min(avail[pl.color], needed) / needed
        if avail[pl.color] >= needed:
            total += comp_val * (0.5 + 0.5 * proximity)        # finishable: near-full
        else:
            total += comp_val * proximity * reach              # developing: partial
    return total


def threat_aware_evaluate(state: GameState, player: int) -> float:
    """evaluate() plus a weighted opponent-/self-threat term. The value
    function for threat-aware MCTS (used relatively: value(0) - value(1))."""
    return evaluate(state, player) + THREAT_WEIGHT * threat_score(state, player)


# Weight on unrealized end-game bonus potential (full column +7, row +2,
# color +10). evaluate() and the minimax search horizon are both within-round
# and never see these, so the ranking search undervalues building toward them.
# Weighted down because they're unrealized; squared fill makes near-complete
# bonuses worth disproportionately more (so the search drives them home).
BONUS_WEIGHT = 0.5


def end_game_bonus_potential(board: PlayerBoard) -> float:
    """Forward-looking value of partial wall structure toward end-game bonuses:
    columns (+7), rows (+2), and color sets (+10), each scaled by (fill/5)^2."""
    wall = board.wall
    total = 0.0
    for c in range(5):
        filled = sum(wall[r][c] is not None for r in range(5))
        total += 7.0 * (filled / 5) ** 2
    for r in range(5):
        filled = sum(wall[r][c] is not None for c in range(5))
        total += 2.0 * (filled / 5) ** 2
    for color in Color:
        placed = sum(wall[r][c] == color for r in range(5) for c in range(5))
        total += 10.0 * (placed / 5) ** 2
    return total


def bonus_aware_evaluate(state: GameState, player: int) -> float:
    """evaluate() plus weighted end-game-bonus potential. Used as the leaf eval
    for the candidate-ranking search so it values column/color building despite
    minimax's within-round horizon."""
    board = state.player_boards[player]
    return evaluate(state, player) + BONUS_WEIGHT * end_game_bonus_potential(board)
