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


def winner_of(scores: list[int]) -> Optional[int]:
    """Index of the higher scorer, or None on a tie."""
    if scores[0] > scores[1]:
        return 0
    if scores[1] > scores[0]:
        return 1
    return None


def advance_round_if_over(state: GameState, rng) -> Optional[list[int]]:
    """If the round is over, run end-of-round processing IN PLACE: capture the
    marker holder, tile + score, then either end the game (returning final
    scores) or refill for the next round. Returns final scores if the game
    ended (completed row OR tile starvation), else None.

    Shared by the Game loop and MCTS rollouts so both progress identically.
    """
    if not state.is_round_over():
        return None

    # Marker holder leads next round; capture before scoring clears the flag.
    marker_holder = next(
        (i for i, b in enumerate(state.player_boards) if b.has_first_player_marker),
        None,
    )

    state.tile_wall_and_score()

    if state.is_game_over():
        state.end_game_bonus()
        return [b.score for b in state.player_boards]

    state.refill_factories(rng)
    state.round_number += 1
    state.current_player = marker_holder if marker_holder is not None else 0

    # Tile starvation: refill dealt nothing, nobody can move -> game ends.
    if state.is_round_over():
        state.end_game_bonus()
        return [b.score for b in state.player_boards]

    return None


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

        scores = advance_round_if_over(self.state, self.rng)
        if scores is not None:
            self.over = True
            self.result = GameResult(
                scores=scores,
                winner=winner_of(scores),
                rounds=self.state.round_number,
            )

        return move

    def play(self) -> GameResult:
        """Run to completion and return the result."""
        while not self.over:
            self.step()
        return self.result
