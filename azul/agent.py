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
        from azul.render import render, render_move_guide, parse_move_shortcut

        inp = self._input if self._input is not None else input
        out = self._output if self._output is not None else print

        moves = state.legal_moves()
        legal = set(moves)

        out(render(state))
        out(render_move_guide(state, moves))

        while True:
            raw = inp("Your move: ")
            move = parse_move_shortcut(raw)
            if move is None:
                out("Format: <source><color><row>, e.g. 0y2, crf, 0bf.")
                continue
            if move not in legal:
                out("Not a legal move — see the options above.")
                continue
            return move
