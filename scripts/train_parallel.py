"""Parallel high-sim self-play refinement on the Mac's cores.

    python -m scripts.train_parallel

Strategy: start from the distilled net (azul_net.pt, strong vs Random) and run
AlphaZero self-play with a high simulation count, parallelized across all CPU
cores. Unlike distillation, self-play improvement is NOT capped at Greedy — if
the search out-plays the current net, the visit-count targets pull it past
Greedy. This is the genuine path; it just needs the sim budget that
parallelism buys.
"""
import multiprocessing as mp
import os
import random
import time

import torch

from azul.net import AzulNet
from azul.train import train_step, save_net, load_net
from azul.parallel_selfplay import generate_examples_parallel
from azul.arena import nn_match
from azul.agent import GreedyAgent, RandomAgent

HIDDEN = 512          # matches the heavy distilled checkpoint
SP_ITERS = 400        # self-play simulations per move
GAMES_PER_ITER = None # default: 2x workers
ITERATIONS = 8
EPOCHS = 4
BATCH = 128
CKPT = "azul_net.pt"


def _vs(net, name, make_opp, n, iters, seed):
    r = nn_match(net, make_opp, n_games=n, iterations=iters, agent_seed=seed)
    print(f"    vs {name}: {r.wins_a}-{r.wins_b}-{r.ties} ({r.win_rate_a:.0%})",
          flush=True)
    return r.win_rate_a


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    workers = mp.cpu_count()
    games = GAMES_PER_ITER or 2 * workers
    print(f"cores={workers}  games/iter={games}  sims/move={SP_ITERS}", flush=True)

    if os.path.exists(CKPT):
        net = load_net(CKPT, hidden=HIDDEN)
        print(f"loaded {CKPT} (hidden={HIDDEN})", flush=True)
    else:
        net = AzulNet(hidden=HIDDEN)
        print("fresh net", flush=True)

    print("baseline:", flush=True)
    _vs(net, "Greedy", lambda i: GreedyAgent(), 8, 200, 1)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for it in range(ITERATIONS):
        t = time.time()
        examples = generate_examples_parallel(
            net, total_games=games, n_workers=workers, sp_iterations=SP_ITERS,
            base_seed=it * 1_000_000 + 1, dirichlet_frac=0.25)
        gen_t = time.time() - t

        losses = []
        for _ in range(EPOCHS):
            rng.shuffle(examples)
            for i in range(0, len(examples), BATCH):
                batch = examples[i:i + BATCH]
                if batch:
                    losses.append(train_step(net, opt, batch)[0])
        avg = sum(losses) / len(losses) if losses else 0.0
        print(f"iter {it}: {len(examples)} examples, gen {gen_t:.0f}s, "
              f"loss {avg:.3f}", flush=True)
        _vs(net, "Greedy", lambda i: GreedyAgent(), 8, 200, it + 2)
        save_net(net, CKPT)

    print("\n=== final ===", flush=True)
    _vs(net, "Random", lambda i: RandomAgent(random.Random(1234 + i)), 12, 200, 99)
    _vs(net, "Greedy", lambda i: GreedyAgent(), 16, 300, 77)


if __name__ == "__main__":
    main()
