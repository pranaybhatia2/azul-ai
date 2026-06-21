"""Monte Carlo Tree Search for Azul (UCB1 selection, random rollouts).

Unlike minimax, MCTS needs no heuristic and no depth limit: it estimates a
move's value by playing many random games to the end and averaging outcomes,
growing the tree toward promising lines via UCB1. It also handles Azul's
stochastic refill naturally — rollouts just play through refills with the rng.

Reward convention (avoids perspective bugs): each node stores `value` =
sum over rollouts of the reward FROM THE PERSPECTIVE OF THE PLAYER WHO MOVED
INTO IT (i.e. its parent's mover). So when a parent selects among its children
by value/visits, that average is already from the parent-mover's viewpoint —
no negation anywhere.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from azul.agent import Agent
from azul.game import advance_round_if_over, winner_of
from azul.heuristics import evaluate
from azul.state import GameState, Move

SQRT2 = math.sqrt(2)


def _reward(winner: Optional[int], player: int) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == player else 0.0


class _Node:
    __slots__ = ("state", "parent", "move", "player", "terminal", "winner",
                 "children", "untried", "visits", "value")

    def __init__(self, state, parent, move, terminal=False, winner=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.player = state.current_player
        self.terminal = terminal
        self.winner = winner
        self.children: list[_Node] = []
        self.untried: list[Move] = [] if terminal else state.legal_moves()
        self.visits = 0
        self.value = 0.0


class MCTSAgent(Agent):
    def __init__(self, iterations: int = 200, rng=None, c: float = SQRT2,
                 rollout: str = "random", rollout_depth: Optional[int] = None,
                 eval_scale: float = 10.0):
        if rollout not in ("random", "greedy"):
            raise ValueError("rollout must be 'random' or 'greedy'")
        self.iterations = iterations
        self.rng = rng if rng is not None else random.Random()
        self.c = c
        self.rollout = rollout
        # rollout_depth=None plays to game end; an int truncates after that
        # many moves and scores with evaluate() (squashed to [0,1] by eval_scale).
        self.rollout_depth = rollout_depth
        self.eval_scale = eval_scale

    def choose_move(self, state: GameState) -> Move:
        root = _Node(state.clone(), parent=None, move=None)
        for _ in range(self.iterations):
            node = self._select(root)
            node = self._expand(node)
            r0 = self._rollout(node)          # reward for player 0, in [0,1]
            self._backpropagate(node, r0)
        # Robust child: most-visited root move.
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    # ------------------------------------------------------------------

    def _select(self, node: _Node) -> _Node:
        # Descend by UCB1 while node is fully expanded and not terminal.
        while not node.terminal and not node.untried:
            node = self._best_uct_child(node)
        return node

    def _best_uct_child(self, node: _Node) -> _Node:
        log_n = math.log(node.visits)

        def uct(child: _Node) -> float:
            if child.visits == 0:
                return math.inf
            exploit = child.value / child.visits
            explore = self.c * math.sqrt(log_n / child.visits)
            return exploit + explore

        return max(node.children, key=uct)

    def _expand(self, node: _Node) -> _Node:
        if node.terminal or not node.untried:
            return node
        move = node.untried.pop()
        child_state = node.state.clone()
        child_state.apply(move)
        # Normalize to a decision state: resolve any round end (which may end
        # the game). One sampled refill is fixed per edge; rollouts average.
        scores = advance_round_if_over(child_state, self.rng)
        terminal = scores is not None
        winner = winner_of(scores) if terminal else None
        child = _Node(child_state, node, move, terminal, winner)
        node.children.append(child)
        return child

    def _rollout(self, node: _Node) -> float:
        """Return the rollout reward FROM PLAYER 0's PERSPECTIVE, in [0,1].
        Terminal -> win/loss/tie. Truncated -> squashed evaluate()."""
        if node.terminal:
            return _reward(node.winner, 0)
        s = node.state.clone()
        steps = 0
        while True:
            if self.rollout_depth is not None and steps >= self.rollout_depth:
                return self._eval_reward(s)
            moves = s.legal_moves()
            if self.rollout == "greedy":
                move = self._greedy_move(s, moves)
            else:
                move = self.rng.choice(moves)
            s.apply(move)
            steps += 1
            scores = advance_round_if_over(s, self.rng)
            if scores is not None:
                return _reward(winner_of(scores), 0)

    def _eval_reward(self, state) -> float:
        """Squash the relative heuristic value into a [0,1] reward for P0."""
        v = evaluate(state, 0) - evaluate(state, 1)
        return 1.0 / (1.0 + math.exp(-v / self.eval_scale))

    def _greedy_move(self, state, moves):
        """Best move for the side to move by one-ply evaluate() (same rule as
        GreedyAgent). Makes rollouts realistic instead of random."""
        from azul.heuristics import evaluate
        me = state.current_player
        best_move, best_val = None, float("-inf")
        for move in moves:
            nxt = state.clone()
            nxt.apply(move)
            val = evaluate(nxt, me)
            if val > best_val:
                best_val, best_move = val, move
        return best_move

    def _backpropagate(self, node: _Node, r0: float) -> None:
        # r0 is the reward for player 0; the value stored at each node is from
        # the perspective of the player who moved into it (its parent's mover).
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                node.value += r0 if node.parent.player == 0 else (1.0 - r0)
            node = node.parent
