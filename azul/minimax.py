"""Depth-limited alpha-beta minimax for Azul.

Search horizon is the current round: we never expand past a refill (the
stochastic part of Azul). A node is a leaf when the depth limit is hit, or
when the round is over — in the latter case we apply the deterministic
end-of-round tiling, then evaluate, so round-end leaves are exact.

Leaf value is RELATIVE: evaluate(me) - evaluate(opponent), so the opponent
genuinely plays to maximise its own lead (proper zero-sum minimax).

A transposition table (keyed on state.encode()) caches results with bound
flags (EXACT / LOWER / UPPER) so it stays correct under alpha-beta windows.
"""
from __future__ import annotations

from azul.agent import Agent
from azul.heuristics import evaluate
from azul.state import GameState, Move

# Transposition-table bound flags.
EXACT, LOWER, UPPER = 0, -1, 1

INF = float("inf")


class MinimaxAgent(Agent):
    def __init__(self, depth: int = 2, use_tt: bool = True):
        self.depth = depth
        self.use_tt = use_tt
        self.nodes = 0   # populated each choose_move, for measurement

    def choose_move(self, state: GameState) -> Move:
        self.nodes = 0
        me = state.current_player
        tt: dict | None = {} if self.use_tt else None

        best_move = None
        best_value = -INF
        alpha = -INF
        for move, child in self._children(state, me, maximizing=True):
            value = self._search(child, self.depth - 1, alpha, INF, me, tt)
            if best_move is None or value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
        return best_move

    def move_values(self, state: GameState) -> list[tuple[Move, float]]:
        """Every legal root move with its depth-limited minimax value (from the
        mover's perspective), best-first. Same search as choose_move, but each
        root move is searched with a FULL window (no rising alpha across the
        root) so all values are exact and comparable for ranking — used to feed
        the LLM a set of lookahead-vetted candidates."""
        self.nodes = 0
        me = state.current_player
        tt: dict | None = {} if self.use_tt else None
        out = []
        for move, child in self._children(state, me, maximizing=True):
            value = self._search(child, self.depth - 1, -INF, INF, me, tt)
            out.append((move, value))
        out.sort(key=lambda mv: mv[1], reverse=True)
        return out

    # ------------------------------------------------------------------

    def _leaf_value(self, state: GameState, me: int) -> float:
        return evaluate(state, me) - evaluate(state, 1 - me)

    def _children(self, state: GameState, me: int, maximizing: bool):
        """All (move, resulting_state) pairs, ordered best-first for the side
        to move (improves alpha-beta pruning)."""
        pairs = []
        for move in state.legal_moves():
            child = state.clone()
            child.apply(move)
            pairs.append((move, child))
        pairs.sort(key=lambda mc: self._leaf_value(mc[1], me), reverse=maximizing)
        return pairs

    def _search(self, state, depth, alpha, beta, me, tt) -> float:
        self.nodes += 1

        if state.is_round_over():
            scored = state.clone()
            scored.tile_wall_and_score()
            return self._leaf_value(scored, me)
        if depth == 0:
            return self._leaf_value(state, me)

        alpha_orig, beta_orig = alpha, beta
        key = None
        if tt is not None:
            key = state.encode()
            entry = tt.get(key)
            if entry is not None and entry[0] >= depth:
                _, val, flag = entry
                if flag == EXACT:
                    return val
                if flag == LOWER:
                    alpha = max(alpha, val)
                elif flag == UPPER:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        maximizing = (state.current_player == me)
        children = self._children(state, me, maximizing)

        if maximizing:
            value = -INF
            for _, child in children:
                value = max(value, self._search(child, depth - 1, alpha, beta, me, tt))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = INF
            for _, child in children:
                value = min(value, self._search(child, depth - 1, alpha, beta, me, tt))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        if tt is not None:
            if value <= alpha_orig:
                flag = UPPER
            elif value >= beta_orig:
                flag = LOWER
            else:
                flag = EXACT
            prev = tt.get(key)
            if prev is None or prev[0] <= depth:
                tt[key] = (depth, value, flag)
        return value
