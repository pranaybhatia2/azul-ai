"""Supervised warm-start: distill the Greedy/heuristic into the network.

AlphaZero self-play is starved on a laptop (Azul's ~90-move branching needs far
more sims than we can afford), so the net never gets a useful policy target
from scratch. Warm-start fixes that: we generate states by playing with a
greedy-softmax policy and label each state with
  * policy target = softmax over evaluate() of each legal move (peaked on the
    greedy move) — a rich target, not just one-hot, and
  * value target = the actual game outcome from that state's player.
Then we train the net on this supervised data. After warm-start the net's
priors mimic Greedy, so even a small-sim PUCT search plays well; AZ self-play
can then refine from a strong base.
"""
from __future__ import annotations

import math
import random

import torch

from azul.encoding import encode_state, move_to_index, POLICY_SIZE
from azul.game import advance_round_if_over, winner_of
from azul.heuristics import evaluate
from azul.net import AzulNet
from azul.selfplay import _value_for
from azul.state import GameState
from azul.train import train_step


def _greedy_softmax(state: GameState, temp: float):
    """Return (policy_vector, moves, probs): a softmax over evaluate() of each
    legal move's resulting position, from the mover's perspective."""
    me = state.current_player
    moves = state.legal_moves()
    vals = []
    for m in moves:
        nxt = state.clone()
        nxt.apply(m)
        vals.append(evaluate(nxt, me))
    hi = max(vals)
    exps = [math.exp((v - hi) / temp) for v in vals]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    policy = [0.0] * POLICY_SIZE
    for m, p in zip(moves, probs):
        policy[move_to_index(m)] = p
    return policy, moves, probs


def generate_supervised_examples(n_games: int, *, rng=None, temp: float = 2.0):
    """Play greedy-softmax self-play games; label states with the greedy-softmax
    policy and the Monte-Carlo game outcome."""
    rng = rng if rng is not None else random.Random()
    examples = []
    for _ in range(n_games):
        state = GameState()
        state.refill_factories(rng)
        pending = []
        while True:
            policy, moves, probs = _greedy_softmax(state, temp)
            pending.append((encode_state(state), policy, state.current_player))
            move = rng.choices(moves, weights=probs, k=1)[0]
            state.apply(move)
            scores = advance_round_if_over(state, rng)
            if scores is not None:
                winner = winner_of(scores)
                examples.extend((enc, pol, _value_for(p, winner))
                                for enc, pol, p in pending)
                break
    return examples


def warm_start(net: AzulNet, *, n_games: int = 60, epochs: int = 8,
               batch_size: int = 64, lr: float = 1e-3, temp: float = 2.0,
               rng=None) -> list[float]:
    """Pretrain `net` on supervised greedy data. Returns per-epoch avg loss."""
    rng = rng if rng is not None else random.Random()
    examples = generate_supervised_examples(n_games, rng=rng, temp=temp)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(epochs):
        rng.shuffle(examples)
        batch_losses = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            if batch:
                batch_losses.append(train_step(net, opt, batch)[0])
        losses.append(sum(batch_losses) / len(batch_losses))
    return losses
