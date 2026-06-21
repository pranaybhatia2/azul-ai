"""Heavy run: best-shot at beating Greedy with the NN.

    python -m scripts.train_heavy

Strategy (targets the value-head bottleneck): generate states from STRONG MCTS
self-play (depth-8 rollouts, 250 sims), label policy with the clean
greedy-softmax and value with the strong-play outcome, distill hard, evaluate
at high play-time sims. Long run (~an hour on CPU).
"""
import random

import torch

from azul.net import AzulNet
from azul.warmstart import generate_hybrid_examples, pretrain
from azul.train import save_net
from azul.arena import nn_match
from azul.agent import RandomAgent, GreedyAgent


def _vs(net, name, make_opp, n, iters, seed):
    r = nn_match(net, make_opp, n_games=n, iterations=iters, agent_seed=seed)
    print(f"  vs {name} ({n}): {r.wins_a}-{r.wins_b}-{r.ties}  ({r.win_rate_a:.0%})",
          flush=True)
    return r.win_rate_a


def main():
    torch.manual_seed(0)
    rng = random.Random(0)

    print("=== generating strong-teacher hybrid data (this is the slow part) ===",
          flush=True)
    examples = generate_hybrid_examples(50, rng=rng, teacher_iters=250,
                                        rollout_depth=8, temp=1.5, temp_cutoff=8)
    print(f"  {len(examples)} examples", flush=True)

    print("=== distilling ===", flush=True)
    net = AzulNet(hidden=512)
    losses = pretrain(net, examples, epochs=20, batch_size=128, lr=1e-3, rng=rng)
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
    save_net(net, "azul_net.pt")
    print("  saved -> azul_net.pt", flush=True)

    print("\n=== evaluation ===", flush=True)
    _vs(net, "Random", lambda i: RandomAgent(random.Random(1234 + i)), 12, 200, 555)
    for iters in (200, 400):
        _vs(net, f"Greedy@{iters}sims", lambda i: GreedyAgent(), 12, iters, 700 + iters)


if __name__ == "__main__":
    main()
