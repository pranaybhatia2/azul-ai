"""Text rendering of a GameState — for the human player and debugging.

Board tiles are drawn with ANSI color when the output is a real terminal
(auto-detected; pass color=True/False to override). Move organization for
the human menu lives here too (organize_moves), so it can be tested apart
from the I/O loop.
"""
from __future__ import annotations

import sys

from azul.state import (
    Color, GameState, Move, PATTERN_LINE_CAPACITY, WALL_PATTERN, CENTER, FLOOR,
)

GLYPH = {
    Color.BLUE: "B",
    Color.YELLOW: "Y",
    Color.RED: "R",
    Color.BLACK: "K",
    Color.WHITE: "W",
}

# (background, foreground) ANSI codes for a filled tile of each color.
_BG = {Color.BLUE: 44, Color.YELLOW: 43, Color.RED: 41, Color.BLACK: 100, Color.WHITE: 47}
_FG = {Color.BLUE: 97, Color.YELLOW: 30, Color.RED: 97, Color.BLACK: 97, Color.WHITE: 30}
# Foreground code for the faint "target color" shown in an empty wall slot.
_DIM_FG = {Color.BLUE: 34, Color.YELLOW: 33, Color.RED: 31, Color.BLACK: 90, Color.WHITE: 37}

RESET = "\033[0m"


def color_name(color: Color) -> str:
    return color.name.capitalize()


def _use_color(color):
    return sys.stdout.isatty() if color is None else color


# ---------------------------------------------------------------------------
# Tile cells (3 chars wide each, for grid alignment)
# ---------------------------------------------------------------------------

def _filled(color: Color, use_color: bool) -> str:
    if use_color:
        return f"\033[{_BG[color]};{_FG[color]}m {GLYPH[color]} {RESET}"
    return f" {GLYPH[color]} "


def _empty_wall(color: Color, use_color: bool) -> str:
    # Faintly shows which color belongs here.
    if use_color:
        return f"\033[2;{_DIM_FG[color]}m {GLYPH[color].lower()} {RESET}"
    return f" {GLYPH[color].lower()} "


def _empty_slot(use_color: bool) -> str:
    return f"\033[2m · {RESET}" if use_color else " · "


def _tile_inline(color: Color, use_color: bool) -> str:
    """Compact colored glyph for factory/center pools (no padding)."""
    if use_color:
        return f"\033[{_BG[color]};{_FG[color]}m{GLYPH[color]}{RESET}"
    return GLYPH[color]


# ---------------------------------------------------------------------------
# Pools and boards
# ---------------------------------------------------------------------------

def _pool_str(pool: dict[Color, int], use_color: bool) -> str:
    parts = [f"{_tile_inline(c, use_color)}x{n}" for c, n in sorted(pool.items()) if n]
    return "  ".join(parts) if parts else "(empty)"


def render_board(state: GameState, player: int, color=None) -> str:
    uc = _use_color(color)
    board = state.player_boards[player]
    header = f"Player {player}  score={board.score}"
    if board.has_first_player_marker:
        header += "  [1st-player marker]"
    lines = [header, "        pattern lines      wall"]

    for row in range(5):
        pl = board.pattern_lines[row]
        cap = PATTERN_LINE_CAPACITY[row]
        # Pattern line: right-aligned within 5 cells; filled then empty.
        cells = [_empty_slot(uc)] * (5 - cap)
        for i in range(cap):
            if pl.color is not None and i < pl.count:
                cells.append(_filled(pl.color, uc))
            else:
                cells.append(_empty_slot(uc))
        staged = "".join(cells)
        # Wall row.
        wall = "".join(
            _filled(board.wall[row][c], uc) if board.wall[row][c] is not None
            else _empty_wall(WALL_PATTERN[row][c], uc)
            for c in range(5)
        )
        lines.append(f"  row {row}: {staged}  | {wall}")

    lines.append(f"  floor: {board.floor_count} tile(s)")
    return "\n".join(lines)


def render(state: GameState, color=None) -> str:
    uc = _use_color(color)
    lines = [f"=== Round {state.round_number} | to move: Player {state.current_player} ==="]
    for i, factory in enumerate(state.factories):
        lines.append(f"Factory {i}: {_pool_str(factory, uc)}")
    center = _pool_str(state.center, uc)
    if state.first_player_marker_in_center:
        center += "  [+1st-player marker]"
    lines.append(f"Center : {center}")
    lines.append("")
    for p in range(len(state.player_boards)):
        lines.append(render_board(state, p, color=uc))
        lines.append("")
    return "\n".join(lines)


def render_move(move: Move) -> str:
    src = "center" if move.source == CENTER else f"factory {move.source}"
    dest = "floor" if move.dest_line == FLOOR else f"line {move.dest_line}"
    return f"take {color_name(move.color)} from {src} -> {dest}"


# ---------------------------------------------------------------------------
# Move organization for the human menu
# ---------------------------------------------------------------------------

def organize_moves(moves: list[Move]) -> tuple[list[Move], list[Move]]:
    """Split moves into (primary, optional_floor).

    A floor-dump (dest == FLOOR) is *forced* when that (source, color) has no
    pattern-line option — those stay in `primary`. All other floor-dumps go to
    `optional_floor` (hidden behind the [f] option in the menu).
    """
    line_moves = [m for m in moves if m.dest_line != FLOOR]
    floor_moves = [m for m in moves if m.dest_line == FLOOR]
    has_line = {(m.source, m.color) for m in line_moves}

    forced_floor = [m for m in floor_moves if (m.source, m.color) not in has_line]
    optional_floor = [m for m in floor_moves if (m.source, m.color) in has_line]

    key = lambda m: (m.source, m.color, m.dest_line)
    primary = sorted(line_moves + forced_floor, key=key)
    optional_floor.sort(key=key)
    return primary, optional_floor


def _source_label(source: int) -> str:
    return "Center" if source == CENTER else f"Factory {source}"


def render_move_menu(moves: list[Move]) -> tuple[str, list[Move]]:
    """Return (menu_text, indexed_moves) for the primary moves, grouped by
    source. indexed_moves[i] is the move chosen by entering i.
    """
    primary, _ = organize_moves(moves)
    out_lines: list[str] = []
    last_source = object()
    for i, m in enumerate(primary):
        if m.source != last_source:
            out_lines.append(f"  {_source_label(m.source)}:")
            last_source = m.source
        dest = "floor (forced)" if m.dest_line == FLOOR else f"line {m.dest_line}"
        out_lines.append(f"    [{i}] {color_name(m.color)} -> {dest}")
    return "\n".join(out_lines), primary
