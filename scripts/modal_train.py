"""Cloud-scale AlphaZero self-play for Azul on Modal.

This workload is CPU-bound (tiny net, Python game logic), so Modal's win is
FAN-OUT: many CPU containers generating self-play games in parallel, far more
than one machine's cores. A GPU isn't needed — the net is tiny.

Run (needs a Modal account: `pip install modal` then `modal token new`):

    modal run scripts/modal_train.py \
        --iterations 20 --tasks 50 --games-per-task 2 --sp-iters 800

It runs entirely in the cloud and writes checkpoints + a metrics log to the
`azul-checkpoints` Volume. Fetch them with:

    modal volume get azul-checkpoints azul_net.pt .
    modal volume get azul-checkpoints metrics.txt .

Strategy: start from a warm (Greedy-distilled) net, then high-sim self-play
refinement at a scale that's infeasible on a laptop — the genuine path to a
net that surpasses Greedy.
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


# --- self-play worker: one container generates a few games ---
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


# --- orchestrator: fans out self-play, trains, evaluates, checkpoints ---
@app.function(image=image, cpu=8.0, timeout=86400, volumes={CKPT_DIR: vol})
def run_training(iterations: int, tasks: int, games_per_task: int,
                 sp_iters: int, epochs: int, batch: int):
    import os
    import random
    import torch
    from azul.net import AzulNet
    from azul.train import train_step
    from azul.warmstart import warm_start
    from azul.arena import nn_match
    from azul.agent import GreedyAgent, RandomAgent

    rng = random.Random(0)
    ckpt = os.path.join(CKPT_DIR, "azul_net.pt")
    log_path = os.path.join(CKPT_DIR, "metrics.txt")

    net = AzulNet(hidden=HIDDEN)
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        log = ["resumed from checkpoint"]
    else:
        log = ["warm-start (distill Greedy)"]
        warm_start(net, n_games=80, epochs=10, rng=rng)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    def vs(name, make_opp, n, iters, seed):
        r = nn_match(net, make_opp, n_games=n, iterations=iters, agent_seed=seed)
        line = f"  vs {name}: {r.wins_a}-{r.wins_b}-{r.ties} ({r.win_rate_a:.0%})"
        log.append(line)
        return r.win_rate_a

    for it in range(iterations):
        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        w = buf.getvalue()
        args = [(w, games_per_task, sp_iters, it * 1_000_000 + i)
                for i in range(tasks)]
        examples = [e for chunk in generate.starmap(args) for e in chunk]

        for _ in range(epochs):
            rng.shuffle(examples)
            for i in range(0, len(examples), batch):
                b = examples[i:i + batch]
                if b:
                    train_step(net, opt, b)

        log.append(f"iter {it}: {len(examples)} examples")
        vs("Greedy", lambda i: GreedyAgent(), 12, 200, it + 1)

        torch.save(net.state_dict(), ckpt)
        with open(log_path, "w") as f:
            f.write("\n".join(log) + "\n")
        vol.commit()

    log.append("=== final ===")
    vs("Random", lambda i: RandomAgent(random.Random(i)), 16, 200, 999)
    vs("Greedy", lambda i: GreedyAgent(), 24, 400, 1000)
    with open(log_path, "w") as f:
        f.write("\n".join(log) + "\n")
    vol.commit()
    return "\n".join(log)


@app.local_entrypoint()
def main(iterations: int = 20, tasks: int = 50, games_per_task: int = 2,
         sp_iters: int = 800, epochs: int = 4, batch: int = 128):
    result = run_training.remote(iterations, tasks, games_per_task,
                                 sp_iters, epochs, batch)
    print(result)
