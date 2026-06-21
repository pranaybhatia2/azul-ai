"""Text rendering of a GameState — for the human player and debugging."""
from __future__ import annotations

from azul.state import (
    Color, GameState, Move, PATTERN_LINE_CAPACITY, WALL_PATTERN, CENTER, FLOOR,
)

# Single-letter glyphs for each color; '.' for an empty slot.
GLYPH = {
    Color.BLUE: "B",
    Color.YELLOW: "Y",
    Color.RED: "R",
    Color.BLACK: "K",
    Color.WHITE: "W",
}
EMPTY = "."


def color_name(color: Color) -> str:
    return color.name.capitalize()


def _pool_str(pool: dict[Color, int]) -> str:
    parts = [f"{GLYPH[c]}x{n}" for c, n in sorted(pool.items()) if n]
    return " ".join(parts) if parts else "(empty)"


def render_board(state: GameState, player: int) -> str:
    board = state.player_boards[player]
    lines = [f"Player {player}  score={board.score}"
             + ("  [1st-player marker]" if board.has_first_player_marker else "")]

    for row in range(5):
        pl = board.pattern_lines[row]
        cap = PATTERN_LINE_CAPACITY[row]
        # Pattern line: right-aligned, filled slots then empty slots.
        glyph = GLYPH[pl.color] if pl.color is not None else EMPTY
        staged = glyph * pl.count + EMPTY * (cap - pl.count)
        staged = staged.rjust(5)
        # Wall row.
        wall = "".join(
            GLYPH[board.wall[row][c]] if board.wall[row][c] is not None
            else GLYPH[WALL_PATTERN[row][c]].lower()
            for c in range(5)
        )
        lines.append(f"  {staged} | {wall}")

    lines.append(f"  floor: {board.floor_count} tile(s)")
    return "\n".join(lines)


def render(state: GameState) -> str:
    lines = [f"=== Round {state.round_number} | to move: Player {state.current_player} ==="]
    for i, factory in enumerate(state.factories):
        lines.append(f"Factory {i}: {_pool_str(factory)}")
    center = _pool_str(state.center)
    if state.first_player_marker_in_center:
        center += "  [+1st-player marker]"
    lines.append(f"Center : {center}")
    lines.append("")
    for p in range(len(state.player_boards)):
        lines.append(render_board(state, p))
        lines.append("")
    return "\n".join(lines)


def render_move(move: Move) -> str:
    src = "center" if move.source == CENTER else f"factory {move.source}"
    dest = "floor" if move.dest_line == FLOOR else f"line {move.dest_line}"
    return f"take {color_name(move.color)} from {src} -> {dest}"
