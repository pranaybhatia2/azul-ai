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


class GreedyAgent(Agent):
    """One-ply lookahead: applies each legal move to a clone, scores the
    resulting position with evaluate(), and picks the best. Ties go to the
    first move in legal_moves()'s sorted order (deterministic)."""

    def choose_move(self, state: GameState) -> Move:
        from azul.heuristics import evaluate

        me = state.current_player
        best_move = None
        best_value = float("-inf")
        for move in state.legal_moves():
            nxt = state.clone()
            nxt.apply(move)
            value = evaluate(nxt, me)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move


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
            render, ordered_sources, render_source_menu, render_placement_menu,
        )

        inp = self._input if self._input is not None else input
        out = self._output if self._output is not None else print

        moves = state.legal_moves()
        out(render(state))

        while True:  # outer loop lets the player back out of a source
            sources = ordered_sources(moves)
            out(render_source_menu(state, sources))
            source = self._read_index(inp, out, len(sources), "Choose source")
            source = sources[source]

            src_moves = [m for m in moves if m.source == source]
            primary, optional_floor, menu = render_placement_menu(src_moves)
            out(f"Place tiles (b = back to sources):")
            out(menu)
            if optional_floor:
                out("  [f] dump a color to the floor (penalty)")

            chosen = self._read_placement(inp, out, primary, optional_floor)
            if chosen is not None:
                return chosen
            # chosen is None -> player typed 'b'; re-show source menu.

    def _read_index(self, inp, out, n, prompt):
        """Read a valid index in [0, n)."""
        while True:
            raw = inp(f"{prompt} [0-{n - 1}]: ").strip()
            try:
                idx = int(raw)
            except (ValueError, TypeError):
                out("Please enter a number.")
                continue
            if 0 <= idx < n:
                return idx
            out("Out of range.")

    def _read_placement(self, inp, out, primary, optional_floor):
        """Return chosen Move, or None if the player typed 'b' (go back)."""
        while True:
            raw = inp(f"Choose [0-{len(primary) - 1}]"
                      + (", f" if optional_floor else "")
                      + ", or b: ").strip().lower()
            if raw == "b":
                return None
            if raw == "f" and optional_floor:
                return self._choose_floor(optional_floor, inp, out)
            try:
                idx = int(raw)
            except (ValueError, TypeError):
                out("Please enter a number, 'b'"
                    + (", or 'f'." if optional_floor else "."))
                continue
            if 0 <= idx < len(primary):
                return primary[idx]
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
