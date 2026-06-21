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

    def __init__(self, input_fn=None, output_fn=None):
        # Resolved lazily at call time (see choose_move) so that tests which
        # monkeypatch builtins.input/print take effect.
        self._input = input_fn
        self._output = output_fn

    def choose_move(self, state: GameState) -> Move:
        # Imported here to avoid a hard dependency for headless agents.
        from azul.render import (
            render, render_move, organize_moves, render_move_menu,
        )

        inp = self._input if self._input is not None else input
        out = self._output if self._output is not None else print

        moves = state.legal_moves()
        _, optional_floor = organize_moves(moves)
        menu_text, indexed = render_move_menu(moves)

        out(render(state))
        out("Your moves:")
        out(menu_text)
        if optional_floor:
            out("    [f] dump a color to the floor (penalty)")

        while True:
            raw = inp(f"Choose [0-{len(indexed) - 1}]"
                      + (" or f: " if optional_floor else ": "))
            raw = raw.strip().lower()

            if raw == "f" and optional_floor:
                return self._choose_floor(optional_floor, inp, out)

            try:
                idx = int(raw)
            except (ValueError, TypeError):
                out("Please enter a number" + (" or 'f'." if optional_floor else "."))
                continue
            if 0 <= idx < len(indexed):
                return indexed[idx]
            out("Out of range.")

    @staticmethod
    def _choose_floor(floor_moves, inp, out):
        from azul.render import render_move

        out("Floor dumps:")
        for i, m in enumerate(floor_moves):
            out(f"    [{i}] {render_move(m)}")
        while True:
            raw = inp(f"Choose [0-{len(floor_moves) - 1}]: ").strip()
            try:
                idx = int(raw)
            except (ValueError, TypeError):
                out("Please enter a number.")
                continue
            if 0 <= idx < len(floor_moves):
                return floor_moves[idx]
            out("Out of range.")
