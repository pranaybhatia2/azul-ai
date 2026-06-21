"""Hybrid training: supervised warm-start (distill Greedy) then AZ self-play.

    python -m scripts.train_hybrid

Goal: clear the Phase 6 exit bar — the trained NeuralMCTSAgent beats Greedy.
"""
import random

import torch

from azul.net import AzulNet
from azul.warmstart import warm_start
from azul.train import train, save_net
from azul.arena import nn_match
from azul.agent import RandomAgent, GreedyAgent


def _vs(net, name, make_opp, n, iters, seed):
    r = nn_match(net, make_opp, n_games=n, iterations=iters, agent_seed=seed)
    print(f"  vs {name} ({n}): {r.wins_a}-{r.wins_b}-{r.ties}  ({r.win_rate_a:.0%})")
    return r.win_rate_a


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    net = AzulNet()

    print("=== warm-start (distill Greedy) ===")
    losses = warm_start(net, n_games=80, epochs=10, rng=rng)
    print(f"  warm-start loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print("  post-warm-start strength:")
    _vs(net, "Random", lambda i: RandomAgent(random.Random(7000 + i)), 10, 80, 11)
    _vs(net, "Greedy", lambda i: GreedyAgent(), 10, 80, 22)

    print("\n=== AZ self-play refine ===")
    def eval_fn(net, it):
        return {"vs_greedy": _vs(net, f"Greedy@it{it}", lambda i: GreedyAgent(),
                                 6, 80, it * 50 + 3)}
    train(net, iterations=4, games_per_iter=12, sp_iterations=80, epochs=4,
          batch_size=64, rng=rng, eval_fn=eval_fn)

    save_net(net, "azul_net.pt")
    print("\nsaved -> azul_net.pt")

    print("\n=== final evaluation ===")
    _vs(net, "Random", lambda i: RandomAgent(random.Random(1234 + i)), 12, 120, 555)
    _vs(net, "Greedy", lambda i: GreedyAgent(), 12, 120, 777)


if __name__ == "__main__":
    main()
