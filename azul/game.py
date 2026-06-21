"""The game loop — drives a full Azul game with pluggable agents.

Game owns the GameState, the agents, and the rng. It exposes:
- step(): advance exactly one ply (one agent move), running end-of-round
  scoring and refills automatically at round boundaries.
- play(): step() until the game is over, then return a GameResult.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from azul.agent import Agent
from azul.state import GameState, Move


@dataclass
class GameResult:
    scores: list[int]
    winner: Optional[int]   # None on a tie
    rounds: int


class Game:
    def __init__(self, agents: list[Agent], seed: int):
        if len(agents) != 2:
            raise ValueError("Azul (this implementation) is 2-player")
        self.agents = agents
        self.rng = random.Random(seed)
        self.state = GameState()
        self.state.refill_factories(self.rng)
        self.over = False
        self.result: Optional[GameResult] = None

    def step(self) -> Optional[Move]:
        """Advance one ply. Returns the move played, or None if game is over."""
        if self.over:
            return None

        agent = self.agents[self.state.current_player]
        move = agent.choose_move(self.state.clone())
        self.state.apply(move)

        if self.state.is_round_over():
            self._end_round()

        return move

    def play(self) -> GameResult:
        """Run to completion and return the result."""
        while not self.over:
            self.step()
        return self.result

    # ------------------------------------------------------------------

    def _end_round(self) -> None:
        # Capture the marker holder BEFORE scoring clears the flag — they
        # lead the next round (Decision 3a). None if marker was unclaimed.
        marker_holder = next(
            (i for i, b in enumerate(self.state.player_boards)
             if b.has_first_player_marker),
            None,
        )

        self.state.tile_wall_and_score()

        if self.state.is_game_over():
            self._finalize()
            return

        self.state.refill_factories(self.rng)
        self.state.round_number += 1
        self.state.current_player = marker_holder if marker_holder is not None else 0

        # Tile starvation: bag + discard exhausted, so refill dealt nothing and
        # no player can move. This ends the game just like a completed row.
        if self.state.is_round_over():
            self._finalize()

    def _finalize(self) -> None:
        self.state.end_game_bonus()
        self.over = True
        self.result = self._build_result()

    def _build_result(self) -> GameResult:
        scores = [b.score for b in self.state.player_boards]
        if scores[0] > scores[1]:
            winner = 0
        elif scores[1] > scores[0]:
            winner = 1
        else:
            winner = None   # tie (tiebreaker left unimplemented for now)
        return GameResult(scores=scores, winner=winner, rounds=self.state.round_number)
