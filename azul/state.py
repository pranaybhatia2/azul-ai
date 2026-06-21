"""
Core state management for Azul AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tile:
    color: str  # 'blue', 'yellow', 'red', 'black', 'white', 'first'


@dataclass
class FactoryDisplay:
    tiles: list[Tile] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.tiles) == 0


@dataclass
class PatternLine:
    capacity: int
    tiles: list[Tile] = field(default_factory=list)

    def is_full(self) -> bool:
        return len(self.tiles) == self.capacity

    def is_empty(self) -> bool:
        return len(self.tiles) == 0


@dataclass
class Wall:
    grid: list[list[Optional[Tile]]] = field(
        default_factory=lambda: [[None] * 5 for _ in range(5)]
    )


@dataclass
class FloorLine:
    tiles: list[Tile] = field(default_factory=list)
    MAX_PENALTIES: int = field(default=7, init=False, repr=False)


@dataclass
class PlayerBoard:
    pattern_lines: list[PatternLine] = field(
        default_factory=lambda: [PatternLine(capacity=i + 1) for i in range(5)]
    )
    wall: Wall = field(default_factory=Wall)
    floor_line: FloorLine = field(default_factory=FloorLine)
    score: int = 0


@dataclass
class GameState:
    num_players: int
    factory_displays: list[FactoryDisplay] = field(default_factory=list)
    center_pool: list[Tile] = field(default_factory=list)
    player_boards: list[PlayerBoard] = field(default_factory=list)
    bag: list[Tile] = field(default_factory=list)
    lid: list[Tile] = field(default_factory=list)  # discard box
    current_player: int = 0
    round_number: int = 1

    def __post_init__(self) -> None:
        if not self.player_boards:
            self.player_boards = [PlayerBoard() for _ in range(self.num_players)]
        if not self.factory_displays:
            num_factories = self.num_players * 2 + 1
            self.factory_displays = [FactoryDisplay() for _ in range(num_factories)]
