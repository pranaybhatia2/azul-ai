"""Self-play data generation for AlphaZero-lite training.

Each game produces training examples (encoded_state, policy_target,
value_target):
  * policy_target: the MCTS visit-count distribution over the 180-move space
    (the "improved policy" the net should learn toward).
  * value_target: +1 / -1 / 0 from THAT example's player's perspective, set
    once the game's winner is known.

Early moves are sampled with temperature for exploration; later moves are
greedy on visit counts.
"""
from __future__ import annotations

import random
from typing import Optional

from azul.az_mcts import NeuralMCTSAgent
from azul.encoding import encode_state, move_to_index, POLICY_SIZE
from azul.game import advance_round_if_over, winner_of
from azul.net import AzulNet
from azul.state import GameState, Move

# One training example before/after labelling.
Example = tuple   # (encoding: list[float], policy: list[float], value: float)


def _sample_move(visits: dict[Move, int], temperature: float, rng) -> Move:
    if temperature <= 0:
        return max(visits, key=visits.get)
    moves = list(visits.keys())
    weights = [visits[m] ** (1.0 / temperature) for m in moves]
    return rng.choices(moves, weights=weights, k=1)[0]


def _value_for(player: int, winner: Optional[int]) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def self_play_game(net: AzulNet, *, iterations: int = 50, rng=None,
                   temp_cutoff: int = 10, c_puct: float = 1.5,
                   dirichlet_frac: float = 0.25, dirichlet_alpha: float = 0.5):
    """Play one self-play game; return its list of labelled examples."""
    rng = rng if rng is not None else random.Random()
    agent = NeuralMCTSAgent(net, iterations=iterations, c_puct=c_puct, rng=rng,
                            dirichlet_frac=dirichlet_frac,
                            dirichlet_alpha=dirichlet_alpha)

    state = GameState()
    state.refill_factories(rng)

    pending = []   # (encoding, policy, player) awaiting the final outcome
    move_num = 0
    while True:
        visits = agent.search(state, add_noise=True)

        policy = [0.0] * POLICY_SIZE
        total = sum(visits.values()) or 1
        for m, n in visits.items():
            policy[move_to_index(m)] = n / total
        pending.append((encode_state(state), policy, state.current_player))

        temperature = 1.0 if move_num < temp_cutoff else 0.0
        move = _sample_move(visits, temperature, rng)
        state.apply(move)
        scores = advance_round_if_over(state, rng)
        move_num += 1

        if scores is not None:
            winner = winner_of(scores)
            return [(enc, pol, _value_for(player, winner))
                    for (enc, pol, player) in pending]


def generate_examples(net: AzulNet, n_games: int, *, iterations: int = 50,
                      rng=None, **kwargs):
    """Play n_games of self-play and return all examples flattened."""
    rng = rng if rng is not None else random.Random()
    examples = []
    for _ in range(n_games):
        examples.extend(self_play_game(net, iterations=iterations, rng=rng, **kwargs))
    return examples
