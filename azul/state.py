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
    first_player_marker_in_center: bool = False

    # ------------------------------------------------------------------
    # Infrastructure (implemented)
    # ------------------------------------------------------------------

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Game logic stubs — Pranay implements these, driven by tests
    # ------------------------------------------------------------------

    def tile_wall_and_score(self) -> None:
        """End-of-round: move complete pattern lines to wall, score adjacency,
        apply floor penalties, clear floors. Processes both players."""
        for board in self.player_boards:
            for row, pl in enumerate(board.pattern_lines):
                capacity = PATTERN_LINE_CAPACITY[row]
                if pl.count < capacity:
                    continue

                # Find which column this color belongs in for this row
                col = next(
                    c for c in range(5) if WALL_PATTERN[row][c] == pl.color
                )
                board.wall[row][col] = pl.color

                # Extras go to discard (one tile was placed, rest discarded)
                self.discard[pl.color] = self.discard.get(pl.color, 0) + (capacity - 1)

                # Score adjacency
                board.score += self._adjacency_score(board.wall, row, col)

                # Clear pattern line
                pl.color = None
                pl.count = 0

            # Floor penalty — marker counts as an extra floor tile
            effective_floor = board.floor_count + (1 if board.has_first_player_marker else 0)
            penalty = sum(FLOOR_PENALTIES[:min(effective_floor, len(FLOOR_PENALTIES))])
            board.score = max(0, board.score + penalty)

            board.floor_count = 0
            board.has_first_player_marker = False

    @staticmethod
    def _adjacency_score(wall: list[list[Optional[Color]]], row: int, col: int) -> int:
        """Score for placing a tile at (row, col) based on contiguous runs."""
        def run_length(dr: int, dc: int) -> int:
            length = 0
            r, c = row + dr, col + dc
            while 0 <= r < 5 and 0 <= c < 5 and wall[r][c] is not None:
                length += 1
                r += dr
                c += dc
            return length

        h = run_length(0, -1) + 1 + run_length(0, 1)
        v = run_length(-1, 0) + 1 + run_length(1, 0)

        # Isolated tile: score 1. Otherwise score each run that has length > 1.
        if h == 1 and v == 1:
            return 1
        score = 0
        if h > 1:
            score += h
        if v > 1:
            score += v
        return score

    def is_round_over(self) -> bool:
        """True when all factories and the center pool are empty."""
        return all(not f for f in self.factories) and not self.center

    def is_game_over(self) -> bool:
        """True when any player has completed a full horizontal wall row."""
        return any(
            all(cell is not None for cell in board.wall[row])
            for board in self.player_boards
            for row in range(5)
        )

    def apply(self, move: Move) -> None:
        """Apply move in place. Mutates self. Trusts caller for legality."""
        board = self.player_boards[self.current_player]

        # --- Take tiles from source ---
        if move.source == CENTER:
            count = self.center.pop(move.color, 0)
            if self.first_player_marker_in_center:
                board.has_first_player_marker = True
                board.floor_count += 1
                self.first_player_marker_in_center = False
        else:
            factory = self.factories[move.source]
            count = factory.pop(move.color, 0)
            # Leftovers go to center
            for color, n in factory.items():
                if n:
                    self.center[color] = self.center.get(color, 0) + n
            factory.clear()

        # --- Place tiles ---
        if move.dest_line == FLOOR:
            board.floor_count += count
        else:
            pl = board.pattern_lines[move.dest_line]
            capacity = PATTERN_LINE_CAPACITY[move.dest_line]
            space = capacity - pl.count
            pl.color = move.color
            pl.count += min(count, space)
            overflow = count - space
            if overflow > 0:
                board.floor_count += overflow

        self.current_player = 1 - self.current_player

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for the current player, sorted by (source, color, dest)."""
        board = self.player_boards[self.current_player]
        moves: list[Move] = []

        sources: list[tuple[int, dict[Color, int]]] = (
            [(i, f) for i, f in enumerate(self.factories) if f]
            + ([(CENTER, self.center)] if self.center else [])
        )

        for source, pool in sources:
            for color, count in pool.items():
                if count == 0:
                    continue
                valid_dests = [
                    row for row in range(5)
                    if self._can_place(board, row, color)
                ]
                # Floor is always valid
                for dest in valid_dests:
                    moves.append(Move(source, color, dest))
                moves.append(Move(source, color, FLOOR))

        moves.sort(key=lambda m: (m.source, m.color, m.dest_line))
        return moves

    @staticmethod
    def _can_place(board: PlayerBoard, row: int, color: Color) -> bool:
        """True if `color` can legally be staged on pattern line `row`."""
        pl = board.pattern_lines[row]
        if pl.count == PATTERN_LINE_CAPACITY[row]:
            return False
        if pl.color is not None and pl.color != color:
            return False
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == color)
        if board.wall[row][col] is not None:
            return False
        return True

    def refill_factories(self, rng) -> None:
        """Draw 4 tiles per factory from bag; reshuffle discard into bag if needed."""
        raise NotImplementedError

    def encode(self):
        """Return a compact numeric representation (Phase 4+)."""
        raise NotImplementedError
