"""Neural-network-guided MCTS (PUCT) — the AlphaZero search.

Differences from Phase 5's MCTS:
  * No rollouts. A leaf is evaluated by the network's value head.
  * Selection uses PUCT with the network's policy as priors:
        score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
  * On expansion, ALL legal children are created at once with their priors.

Value convention (keyed on player identity, robust to Azul's non-alternating
round boundaries): a node's W/N is the value from the perspective of the
player to move at that node. A parent uses -child.Q (opponent's view negated).
Backprop adds +v to nodes whose player matches the evaluated leaf's player,
-v otherwise.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from azul.agent import Agent
from azul.game import advance_round_if_over, winner_of
from azul.net import AzulNet, predict
from azul.state import GameState, Move


class _AZNode:
    __slots__ = ("state", "parent", "move", "player", "terminal", "winner",
                 "children", "prior", "N", "W", "expanded")

    def __init__(self, state, parent, move, prior=0.0, terminal=False, winner=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.player = state.current_player
        self.terminal = terminal
        self.winner = winner
        self.children: list[_AZNode] = []
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class NeuralMCTSAgent(Agent):
    def __init__(self, net: AzulNet, iterations: int = 100, c_puct: float = 1.5,
                 rng=None, dirichlet_frac: float = 0.0, dirichlet_alpha: float = 0.5):
        self.net = net
        self.iterations = iterations
        self.c_puct = c_puct
        self.rng = rng if rng is not None else random.Random()
        self.dirichlet_frac = dirichlet_frac
        self.dirichlet_alpha = dirichlet_alpha

    def choose_move(self, state: GameState) -> Move:
        visits = self.search(state)
        return max(visits, key=visits.get)

    def search(self, state: GameState, add_noise: bool = False) -> dict[Move, int]:
        """Run the PUCT search and return {move: visit_count} at the root."""
        root = _AZNode(state.clone(), parent=None, move=None)
        self._expand(root)
        if add_noise and self.dirichlet_frac > 0 and root.children:
            self._add_dirichlet_noise(root)
        for _ in range(self.iterations):
            self._simulate(root)
        return {c.move: c.N for c in root.children}

    # ------------------------------------------------------------------

    def _simulate(self, root: _AZNode) -> None:
        path = [root]
        node = root
        while node.expanded and not node.terminal:
            node = self._select_child(node)
            path.append(node)

        if node.terminal:
            value = self._terminal_value(node)
        else:
            value = self._expand(node)

        leaf_player = node.player
        for n in path:
            n.N += 1
            n.W += value if n.player == leaf_player else -value

    def _select_child(self, node: _AZNode) -> _AZNode:
        sqrt_n = math.sqrt(max(1, node.N))
        best, best_score = None, -math.inf
        for c in node.children:
            q = -c.Q   # child.Q is from the opponent's view; negate for parent
            u = self.c_puct * c.prior * sqrt_n / (1 + c.N)
            score = q + u
            if score > best_score:
                best_score, best = score, c
        return best

    def _expand(self, node: _AZNode) -> float:
        """Create all children with net priors; return the net value (for the
        player to move at `node`)."""
        priors, value = predict(self.net, node.state)
        for move, p in priors.items():
            child_state = node.state.clone()
            child_state.apply(move)
            scores = advance_round_if_over(child_state, self.rng)
            terminal = scores is not None
            winner = winner_of(scores) if terminal else None
            node.children.append(
                _AZNode(child_state, node, move, prior=p,
                        terminal=terminal, winner=winner))
        node.expanded = True
        return value

    def _terminal_value(self, node: _AZNode) -> float:
        if node.winner is None:
            return 0.0
        return 1.0 if node.winner == node.player else -1.0

    def _add_dirichlet_noise(self, root: _AZNode) -> None:
        # Gamma-sample a Dirichlet without numpy: Dir(a) ~ normalized Gamma(a,1).
        gammas = [self.rng.gammavariate(self.dirichlet_alpha, 1.0)
                  for _ in root.children]
        total = sum(gammas) or 1.0
        noise = [g / total for g in gammas]
        f = self.dirichlet_frac
        for child, n in zip(root.children, noise):
            child.prior = (1 - f) * child.prior + f * n
