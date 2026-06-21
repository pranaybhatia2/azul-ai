"""Training loop for AlphaZero-lite.

Loss = policy cross-entropy (predicted log-softmax vs MCTS visit distribution)
     + value MSE (predicted value vs game outcome).

train() runs the full iterate: each iteration generates self-play games with
the current net, then trains on those examples. Stronger net -> stronger
self-play -> stronger net.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from azul.net import AzulNet
from azul.selfplay import generate_examples


def train_step(net: AzulNet, optimizer, batch) -> tuple[float, float, float]:
    """One gradient step on a batch of (encoding, policy_target, value_target).
    Returns (total_loss, policy_loss, value_loss)."""
    states = torch.tensor([e[0] for e in batch], dtype=torch.float32)
    target_policy = torch.tensor([e[1] for e in batch], dtype=torch.float32)
    target_value = torch.tensor([[e[2]] for e in batch], dtype=torch.float32)

    net.train()
    logits, value = net(states)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(target_policy * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()


def train(net: AzulNet, *, iterations: int = 5, games_per_iter: int = 10,
          sp_iterations: int = 50, epochs: int = 4, batch_size: int = 64,
          lr: float = 1e-3, weight_decay: float = 1e-4, rng=None,
          eval_fn=None) -> list[dict]:
    """Run the self-play -> train iterate loop. Returns per-iteration history.

    eval_fn(net, iteration) is called after each iteration (e.g. to measure
    strength vs a baseline) and its return value is stored under 'eval'.
    """
    rng = rng if rng is not None else random.Random()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for it in range(iterations):
        examples = generate_examples(net, games_per_iter,
                                     iterations=sp_iterations, rng=rng)
        losses = []
        for _ in range(epochs):
            rng.shuffle(examples)
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                if batch:
                    losses.append(train_step(net, optimizer, batch))

        avg = [sum(c) / len(losses) for c in zip(*losses)] if losses else [0, 0, 0]
        entry = {"iteration": it, "examples": len(examples),
                 "loss": avg[0], "policy_loss": avg[1], "value_loss": avg[2]}
        if eval_fn is not None:
            entry["eval"] = eval_fn(net, it)
        history.append(entry)

    return history


def save_net(net: AzulNet, path: str) -> None:
    torch.save(net.state_dict(), path)


def load_net(path: str, hidden: int = 256) -> AzulNet:
    net = AzulNet(hidden=hidden)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net
