"""Distill the strong Phase-5 MCTS teacher (beats Greedy 8-0) into the net.

    python -m scripts.train_teacher

The net's ceiling is its teacher's strength, so distilling the MCTS agent
(not Greedy) is the principled path to a net that beats Greedy.
"""
import random

import torch

from azul.net import AzulNet
from azul.warmstart import generate_teacher_examples, pretrain
from azul.train import save_net
from azul.arena import nn_match
from azul.agent import RandomAgent, GreedyAgent


def _vs(net, name, make_opp, n, iters, seed):
    r = nn_match(net, make_opp, n_games=n, iterations=iters, agent_seed=seed)
    print(f"  vs {name} ({n}): {r.wins_a}-{r.wins_b}-{r.ties}  ({r.win_rate_a:.0%})")
    return r.win_rate_a


def main():
    torch.manual_seed(0)
    rng = random.Random(0)

    print("=== generating teacher (MCTS) data ===")
    examples = generate_teacher_examples(30, rng=rng, teacher_iters=120,
                                         rollout_depth=6, temp_cutoff=8)
    print(f"  {len(examples)} examples")

    print("=== distilling into net ===")
    net = AzulNet()
    losses = pretrain(net, examples, epochs=12, batch_size=64, rng=rng)
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    save_net(net, "azul_net.pt")
    print("  saved -> azul_net.pt")

    print("\n=== evaluation ===")
    _vs(net, "Random", lambda i: RandomAgent(random.Random(1234 + i)), 12, 120, 555)
    for iters in (120, 300):
        _vs(net, f"Greedy@{iters}sims", lambda i: GreedyAgent(), 12, iters, 700 + iters)


if __name__ == "__main__":
    main()
