"""Parallel self-play across CPU cores.

The bottleneck in self-play is CPU-side game simulation, so the lever is more
cores: generate games in worker processes, each rebuilding the net from a
shared state_dict, and pool the examples. Each worker pins torch to 1 thread
so workers don't oversubscribe the cores.
"""
from __future__ import annotations

import multiprocessing as mp
import random

import torch

from azul.net import AzulNet
from azul.selfplay import self_play_game


def _worker(task):
    (sd, hidden, n_games, sp_iterations, seed, c_puct,
     dirichlet_frac, dirichlet_alpha, temp_cutoff) = task
    torch.set_num_threads(1)
    net = AzulNet(hidden=hidden)
    net.load_state_dict(sd)
    net.eval()
    rng = random.Random(seed)
    examples = []
    for _ in range(n_games):
        examples.extend(self_play_game(
            net, iterations=sp_iterations, rng=rng, c_puct=c_puct,
            dirichlet_frac=dirichlet_frac, dirichlet_alpha=dirichlet_alpha,
            temp_cutoff=temp_cutoff))
    return examples


def _hidden_of(net: AzulNet) -> int:
    return net.body[0].out_features


def generate_examples_parallel(net: AzulNet, total_games: int, *,
                               n_workers: int | None = None,
                               sp_iterations: int = 200, base_seed: int = 0,
                               c_puct: float = 1.5, dirichlet_frac: float = 0.25,
                               dirichlet_alpha: float = 0.5, temp_cutoff: int = 10):
    """Generate self-play examples across n_workers processes."""
    n_workers = n_workers or mp.cpu_count()
    hidden = _hidden_of(net)
    sd = {k: v.cpu() for k, v in net.state_dict().items()}

    # Split games across workers.
    per = [total_games // n_workers] * n_workers
    for i in range(total_games % n_workers):
        per[i] += 1
    tasks = [
        (sd, hidden, per[i], sp_iterations, base_seed + i * 100003, c_puct,
         dirichlet_frac, dirichlet_alpha, temp_cutoff)
        for i in range(n_workers) if per[i] > 0
    ]
    if len(tasks) == 1:           # avoid pool overhead for a single chunk
        return _worker(tasks[0])

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        results = pool.map(_worker, tasks)
    return [e for chunk in results for e in chunk]
