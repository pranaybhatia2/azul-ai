from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Color(IntEnum):
    BLUE   = 0
    YELLOW = 1
    RED    = 2
    BLACK  = 3
    WHITE  = 4


NUM_COLORS = len(Color)
TILES_PER_COLOR = 20
NUM_FACTORIES = 5          # 2-player game

PATTERN_LINE_CAPACITY = [1, 2, 3, 4, 5]  # indexed by row

# WALL_PATTERN[row][col] gives the Color that belongs in that cell.
# Formula: Color((col - row) % 5)
WALL_PATTERN: list[list[Color]] = [
    [Color((col - row) % NUM_COLORS) for col in range(5)]
    for row in range(5)
]

FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]  # up to 7 tiles


# ---------------------------------------------------------------------------
# Move — immutable, hashable
# ---------------------------------------------------------------------------

CENTER = -1   # sentinel for source
FLOOR  = -1   # sentinel for dest_line


@dataclass(frozen=True)
class Move:
    source: int      # factory index 0–4, or CENTER (-1)
    color: Color
    dest_line: int   # pattern row 0–4, or FLOOR (-1)


# ---------------------------------------------------------------------------
# Per-player data
# ---------------------------------------------------------------------------

@dataclass
class PatternLine:
    """Stores (color, count) only. Capacity is PATTERN_LINE_CAPACITY[row]."""
    color: Optional[Color] = None
    count: int = 0


@dataclass
class PlayerBoard:
    pattern_lines: list[PatternLine] = field(
        default_factory=lambda: [PatternLine() for _ in range(5)]
    )
    # wall[row][col] is Color if filled, None if empty
    wall: list[list[Optional[Color]]] = field(
        default_factory=lambda: [[None] * 5 for _ in range(5)]
    )
    floor_count: int = 0
    has_first_player_marker: bool = False
    score: int = 0


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    # factories[i] is a dict[Color, int] — tile counts per factory display
    factories: list[dict[Color, int]] = field(
        default_factory=lambda: [{} for _ in range(NUM_FACTORIES)]
    )
    center: dict[Color, int] = field(default_factory=dict)
    player_boards: list[PlayerBoard] = field(
        default_factory=lambda: [PlayerBoard(), PlayerBoard()]
    )
    bag: dict[Color, int] = field(
        default_factory=lambda: {c: TILES_PER_COLOR for c in Color}
    )
    discard: dict[Color, int] = field(
        default_factory=lambda: {c: 0 for c in Color}
    )
    current_player: int = 0
    round_number: int = 1

    # ------------------------------------------------------------------
    # Infrastructure (implemented)
    # ------------------------------------------------------------------

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Game logic stubs — Pranay implements these, driven by tests
    # ------------------------------------------------------------------

    def tile_wall_and_score(self) -> None:
        """End-of-round: move complete pattern lines to wall, score adjacency."""
        raise NotImplementedError

    def is_round_over(self) -> bool:
        """True when all factories and the center pool are empty."""
        raise NotImplementedError

    def is_game_over(self) -> bool:
        """True when any player has completed a full horizontal wall row."""
        raise NotImplementedError

    def apply(self, move: Move) -> None:
        """Apply move in place. Mutates self."""
        raise NotImplementedError

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for the current player."""
        raise NotImplementedError

    def refill_factories(self, rng) -> None:
        """Draw 4 tiles per factory from bag; reshuffle discard into bag if needed."""
        raise NotImplementedError

    def encode(self):
        """Return a compact numeric representation (Phase 4+)."""
        raise NotImplementedError
