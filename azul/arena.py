"""Evaluation: pit a trained net's NeuralMCTSAgent against baseline agents."""
from __future__ import annotations

import random

from azul.az_mcts import NeuralMCTSAgent
from azul.match import play_match, MatchResult
from azul.net import AzulNet


def nn_match(net: AzulNet, make_opponent, n_games: int, *, iterations: int = 100,
             c_puct: float = 1.5, base_seed: int = 0, agent_seed: int = 0) -> MatchResult:
    """Play the net's NeuralMCTSAgent (seat 0) vs make_opponent (seat 1)."""
    return play_match(
        make_a=lambda i: NeuralMCTSAgent(net, iterations=iterations,
                                         c_puct=c_puct,
                                         rng=random.Random(agent_seed + i)),
        make_b=make_opponent,
        n_games=n_games,
        base_seed=base_seed,
    )
