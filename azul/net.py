"""The policy + value network (AlphaZero-lite) and a state-prediction bridge.

A small MLP: state vector -> shared body -> two heads:
  * policy: POLICY_SIZE logits (one per move in the fixed 180-move space)
  * value:  scalar in [-1, 1] (expected outcome for the player to move)

predict() bridges net <-> game: encode a GameState, run the net, mask illegal
moves, and return {move: prior} plus the value estimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from azul.encoding import (
    encode_state, move_to_index, STATE_SIZE, POLICY_SIZE,
)
from azul.state import GameState, Move


class AzulNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, POLICY_SIZE)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        """x: (batch, STATE_SIZE) -> (policy_logits (batch, POLICY_SIZE),
        value (batch, 1) in [-1, 1])."""
        h = self.body(x)
        return self.policy_head(h), torch.tanh(self.value_head(h))


@torch.no_grad()
def predict(net: AzulNet, state: GameState) -> tuple[dict[Move, float], float]:
    """Return ({legal_move: prior_prob}, value) for `state`. Priors are a
    softmax over the legal moves only; value is for the player to move."""
    net.eval()
    x = torch.tensor([encode_state(state)], dtype=torch.float32)
    logits, value = net(x)
    logits = logits[0]

    moves = state.legal_moves()
    idxs = [move_to_index(m) for m in moves]
    legal_logits = logits[idxs]
    probs = F.softmax(legal_logits, dim=0)
    priors = {m: probs[i].item() for i, m in enumerate(moves)}
    return priors, value.item()
