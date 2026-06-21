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
# Shortcut input + the move guide
# ---------------------------------------------------------------------------

def ordered_sources(moves: list[Move]) -> list[int]:
    """Sources with at least one legal move: factories ascending, Center last."""
    present = {m.source for m in moves}
    factories = sorted(s for s in present if s != CENTER)
    return factories + ([CENTER] if CENTER in present else [])


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


def move_to_shortcut(move: Move) -> str:
    """Inverse of parse_move_shortcut: render a Move as its 3-char code, e.g.
    Move(0, YELLOW, 2) -> '0y2', Move(CENTER, RED, FLOOR) -> 'crf'."""
    src = "c" if move.source == CENTER else str(move.source)
    color = GLYPH[move.color].lower()
    row = "f" if move.dest_line == FLOOR else str(move.dest_line)
    return f"{src}{color}{row}"


def _compress_rows(rows: list[int]) -> str:
    """[0,1,2,3,4] -> '0-4'; [0,2,3,4] -> '0,2-4'."""
    if not rows:
        return ""
    parts, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = r
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def render_move_guide(state: GameState, moves: list[Move], color=None) -> str:
    """Shortcut-style guide: each source by its code (0-4 / c) with its colors
    and the rows each may go to. Floor is always available via row 'f'."""
    uc = _use_color(color)
    # source -> {color: [valid pattern rows]}
    by_src: dict[int, dict] = {}
    for m in moves:
        cols = by_src.setdefault(m.source, {})
        cols.setdefault(m.color, [])
        if m.dest_line != FLOOR:
            cols[m.color].append(m.dest_line)

    lines = [
        "Your move — type <source><color><row>",
        "  source: 0-4 or c (center) · color: b y r k w · row: 0-4 or f (floor)",
    ]
    for s in ordered_sources(moves):
        code = "c" if s == CENTER else str(s)
        marker = (" +marker" if s == CENTER
                  and state.first_player_marker_in_center else "")
        specs = []
        for col, rows in sorted(by_src[s].items()):
            rows = sorted(set(rows))
            spec = _compress_rows(rows) if rows else "f"
            specs.append(f"{_tile_inline(col, uc)}:{spec}")
        lines.append(f"  {code}  " + "   ".join(specs) + marker)
    return "\n".join(lines)
