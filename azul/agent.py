"""Agent interface for Azul.

The game loop hands each agent a *clone* of the current GameState (see
Decision 1a), so agents may read it freely without corrupting the real game.
Agents return a single Move chosen from state.legal_moves().
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from azul.state import GameState, Move


class Agent(ABC):
    @abstractmethod
    def choose_move(self, state: GameState) -> Move:
        """Return one legal move for state.current_player.

        `state` is a clone — mutating it has no effect on the real game.
        """
        ...


class RandomAgent(Agent):
    """Picks uniformly at random from the legal moves. Seeded for reproducibility."""

    def __init__(self, rng):
        self.rng = rng

    def choose_move(self, state: GameState) -> Move:
        return self.rng.choice(state.legal_moves())
