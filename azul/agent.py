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


class HumanAgent(Agent):
    """Renders the board and reads a move choice from stdin.

    `input_fn` and `output_fn` are injectable so the agent is testable
    without real stdin/stdout.
    """

    def __init__(self, input_fn=input, output_fn=print):
        self._input = input_fn
        self._output = output_fn

    def choose_move(self, state: GameState) -> Move:
        # Imported here to avoid a hard dependency for headless agents.
        from azul.render import render, render_move

        moves = state.legal_moves()
        self._output(render(state))
        self._output("Legal moves:")
        for i, m in enumerate(moves):
            self._output(f"  [{i}] {render_move(m)}")

        while True:
            raw = self._input(f"Choose a move [0-{len(moves) - 1}]: ")
            try:
                idx = int(raw)
            except (ValueError, TypeError):
                self._output("Please enter a number.")
                continue
            if 0 <= idx < len(moves):
                return moves[idx]
            self._output("Out of range.")
