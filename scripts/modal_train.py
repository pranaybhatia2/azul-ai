"""Cloud-scale AlphaZero self-play for Azul on Modal (deployed fan-out).

Robust launch (survives client disconnect because a DEPLOYED app is persistent
server-side; an ephemeral `modal run` app is not):

    modal deploy scripts/modal_train.py
    python -c "import modal; modal.Function.from_name('azul-az','run_training').spawn(10,30,2,800,4,128)"
    # args: iterations, tasks, games_per_task, sp_iters, epochs, batch

Self-play is fanned out via Modal containers (generate.starmap) — Modal's own
parallelism, not in-container multiprocessing. Progress + results stream to the
`azul-checkpoints` Volume:

    modal volume get azul-checkpoints metrics.txt .
    modal volume get azul-checkpoints azul_net.pt .

Resumes from the checkpoint if present.
"""
import io

import modal

app = modal.App("azul-az")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_python_source("azul")
)
vol = modal.Volume.from_name("azul-checkpoints", create_if_missing=True)
CKPT_DIR = "/ckpt"
HIDDEN = 512


@app.function(image=image, cpu=4.0, timeout=3600)
def generate(weights: bytes, n_games: int, sp_iters: int, seed: int):
    import random
    import torch
    from azul.net import AzulNet
    from azul.selfplay import self_play_game

    torch.set_num_threads(4)
    net = AzulNet(hidden=HIDDEN)
    net.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
    net.eval()
    rng = random.Random(seed)
    examples = []
    for _ in range(n_games):
        examples.extend(self_play_game(net, iterations=sp_iters, rng=rng,
                                       dirichlet_frac=0.25))
    return examples


@app.function(image=image, cpu=4.0, timeout=86400, volumes={CKPT_DIR: vol})
def run_training(iterations: int, tasks: int, games_per_task: int,
                 sp_iters: int, epochs: int, batch: int,
                 buffer_iters: int = 6, eval_games: int = 24, fresh: bool = False):
    """AlphaZero loop with a REPLAY BUFFER: each iteration trains on the last
    `buffer_iters` iterations' games (not just the newest), which prevents the
    forgetting/oscillation seen when training on one iteration at a time.
    `fresh=True` ignores any checkpoint and starts from a warm-started net."""
    import os
    import random
    from collections import deque
    import torch
    from azul.net import AzulNet
    from azul.train import train_step
    from azul.warmstart import warm_start
    from azul.arena import nn_match
    from azul.agent import GreedyAgent, RandomAgent

    rng = random.Random(0)
    ckpt = os.path.join(CKPT_DIR, "azul_net.pt")
    log_path = os.path.join(CKPT_DIR, "metrics.txt")
    log = []

    def flush():
        with open(log_path, "w") as f:
            f.write("\n".join(log) + "\n")
        vol.commit()

    net = AzulNet(hidden=HIDDEN)
    if os.path.exists(ckpt) and not fresh:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        log.append("resumed from checkpoint")
    else:
        log.append("warm-start (distill Greedy)...")
        flush()
        warm_start(net, n_games=80, epochs=10, rng=rng)
    log.append(f"replay buffer = last {buffer_iters} iters, eval = {eval_games} games")
    flush()

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_iters)   # each entry = one iteration's examples

    def vs(name, make_opp, n, sims, seed):
        r = nn_match(net, make_opp, n_games=n, iterations=sims, agent_seed=seed)
        log.append(f"  vs {name}: {r.wins_a}-{r.wins_b}-{r.ties} ({r.win_rate_a:.0%})")
        return r.win_rate_a

    n_games = tasks * games_per_task
    for it in range(iterations):
        log.append(f"iter {it}: generating {n_games} games @ {sp_iters} sims...")
        flush()

        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        w = buf.getvalue()
        args = [(w, games_per_task, sp_iters, it * 1_000_000 + i)
                for i in range(tasks)]
        new_examples = [e for chunk in generate.starmap(args) for e in chunk]
        buffer.append(new_examples)
        train_data = [e for it_ex in buffer for e in it_ex]   # replay buffer

        log[-1] = (f"iter {it}: +{len(new_examples)} new, "
                   f"training on {len(train_data)} (buffer)...")
        flush()
        for _ in range(epochs):
            rng.shuffle(train_data)
            for i in range(0, len(train_data), batch):
                b = train_data[i:i + batch]
                if b:
                    train_step(net, opt, b)

        log[-1] = f"iter {it}: +{len(new_examples)} new, buffer {len(train_data)}"
        vs("Greedy", lambda i: GreedyAgent(), eval_games, 200, it + 1)
        torch.save(net.state_dict(), ckpt)
        flush()

    log.append("=== final ===")
    vs("Random", lambda i: RandomAgent(random.Random(i)), 24, 200, 999)
    vs("Greedy", lambda i: GreedyAgent(), 40, 400, 1000)
    flush()
    return "\n".join(log)


@app.local_entrypoint()
def main(iterations: int = 12, tasks: int = 30, games_per_task: int = 2,
         sp_iters: int = 800, epochs: int = 4, batch: int = 128,
         fresh: bool = True):
    # For ad-hoc ephemeral runs. For a durable run, use `modal deploy` + spawn
    # (see module docstring) so it survives this client disconnecting.
    print(run_training.remote(iterations, tasks, games_per_task, sp_iters,
                              epochs, batch, fresh=fresh))
