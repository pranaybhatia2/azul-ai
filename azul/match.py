"""A small match harness for comparing two agents over many seeded games.

This is the seed that Phase 7's full tournament system grows from. Agents are
built per game via factory callables (taking the game index) so each game gets
fresh, independently-seeded agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from azul.agent import Agent
from azul.game import Game

AgentFactory = Callable[[int], Agent]


@dataclass
class MatchResult:
    games: int
    wins_a: int
    wins_b: int
    ties: int

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0


def play_match(
    make_a: AgentFactory,
    make_b: AgentFactory,
    n_games: int,
    base_seed: int = 0,
) -> MatchResult:
    """Play n_games of make_a (seat 0) vs make_b (seat 1).

    Note: seats are fixed (a is always Player 0). Azul has a slight
    first-player advantage; alternate seats if you need a stricter comparison.
    """
    wins = [0, 0]
    ties = 0
    for i in range(n_games):
        game = Game(agents=[make_a(i), make_b(i)], seed=base_seed + i)
        result = game.play()
        if result.winner is None:
            ties += 1
        else:
            wins[result.winner] += 1
    return MatchResult(games=n_games, wins_a=wins[0], wins_b=wins[1], ties=ties)
