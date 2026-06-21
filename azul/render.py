"""Text rendering of a GameState — for the human player and debugging.

Board tiles are drawn with ANSI color when the output is a real terminal
(auto-detected; pass color=True/False to override). Move organization for
the human menu lives here too (organize_moves), so it can be tested apart
from the I/O loop.
"""
from __future__ import annotations

import sys
from typing import Optional

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
# Reverse of GLYPH (lowercase) for parsing the move shortcut.
CHAR_TO_COLOR = {"b": Color.BLUE, "y": Color.YELLOW, "r": Color.RED,
                 "k": Color.BLACK, "w": Color.WHITE}

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
    dest = "floor" if move.dest_line == FLOOR else f"row {move.dest_line}"
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


def ordered_sources(moves: list[Move]) -> list[int]:
    """Sources that have at least one legal move, factories ascending then
    Center last."""
    present = {m.source for m in moves}
    factories = sorted(s for s in present if s != CENTER)
    return factories + ([CENTER] if CENTER in present else [])


def render_source_menu(state: GameState, sources: list[int], color=None) -> str:
    """Step 1 of two-step selection: which factory/center to take from."""
    uc = _use_color(color)
    lines = ["Take tiles from:"]
    for i, s in enumerate(sources):
        pool = state.center if s == CENTER else state.factories[s]
        extra = ("  [+1st-player marker]"
                 if s == CENTER and state.first_player_marker_in_center else "")
        lines.append(f"  [{i}] {_source_label(s)}: {_pool_str(pool, uc)}{extra}")
    return "\n".join(lines)


def parse_move_shortcut(text: str) -> Optional[Move]:
    """Parse a one-shot move string '<source><color><row>', e.g. '0y2'
    (factory 0, yellow, row 2) or 'crf' (center, red, floor). Source is a
    digit 0-4 or 'c'; color is b/y/r/k/w; row is a digit 0-4 or 'f'.

    Returns a syntactically-valid Move (legality is the caller's job), or None
    if the string isn't in shortcut form.
    """
    t = text.replace(" ", "").lower()
    if len(t) != 3:
        return None
    s, col, r = t[0], t[1], t[2]

    if s == "c":
        source = CENTER
    elif s.isdigit():
        source = int(s)
    else:
        return None

    if col not in CHAR_TO_COLOR:
        return None
    color = CHAR_TO_COLOR[col]

    if r == "f":
        dest = FLOOR
    elif r.isdigit():
        dest = int(r)
    else:
        return None

    return Move(source, color, dest)


def render_placement_menu(src_moves: list[Move]) -> tuple[list[Move], list[Move], str]:
    """Step 2: placements for the chosen source. Returns
    (primary, optional_floor, menu_text). Floor dumps are hidden unless forced
    (a color with no open pattern line)."""
    primary, optional_floor = organize_moves(src_moves)
    lines = []
    for j, m in enumerate(primary):
        dest = "floor (forced)" if m.dest_line == FLOOR else f"row {m.dest_line}"
        lines.append(f"  [{j}] {color_name(m.color)} -> {dest}")
    return primary, optional_floor, "\n".join(lines)
