"""Train an AlphaZero-lite net for Azul and report the learning curve.

Run from the repo root:
    python -m scripts.train_demo

Demonstrates the Phase 6 exit condition: the net improves across iterations
and the trained NeuralMCTSAgent beats Random and (the goal) Greedy.
"""
import random

import torch

from azul.net import AzulNet
from azul.train import train, save_net
from azul.arena import nn_match
from azul.agent import RandomAgent, GreedyAgent


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    net = AzulNet()

    def eval_fn(net, it):
        r = nn_match(net, lambda i: RandomAgent(random.Random(9000 + i)),
                     n_games=6, iterations=60, agent_seed=it * 100 + 1)
        return {"vs_random": r.win_rate_a}

    history = train(net, iterations=6, games_per_iter=16, sp_iterations=50,
                    epochs=4, batch_size=64, rng=rng, eval_fn=eval_fn)

    print("\n=== learning curve ===")
    for h in history:
        print(f"iter {h['iteration']}: loss={h['loss']:.3f} "
              f"(policy {h['policy_loss']:.3f}, value {h['value_loss']:.3f}) "
              f"examples={h['examples']}  vs_random={h['eval']['vs_random']:.0%}")

    save_net(net, "azul_net.pt")
    print("\nsaved -> azul_net.pt")

    print("\n=== final evaluation ===")
    vr = nn_match(net, lambda i: RandomAgent(random.Random(1234 + i)),
                  n_games=10, iterations=120, agent_seed=555)
    print(f"vs Random (10): {vr.wins_a}-{vr.wins_b}-{vr.ties}  ({vr.win_rate_a:.0%})")
    vg = nn_match(net, lambda i: GreedyAgent(), n_games=10, iterations=120,
                  agent_seed=777)
    print(f"vs Greedy (10): {vg.wins_a}-{vg.wins_b}-{vg.ties}  ({vg.win_rate_a:.0%})")


if __name__ == "__main__":
    main()
