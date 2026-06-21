# Cloud-scale self-play on Modal

This workload is **CPU-bound** (tiny network, Python game logic), so the win on
Modal is **fan-out** — dozens of CPU containers generating self-play games in
parallel — not a GPU. The orchestrator runs the whole AlphaZero loop in the
cloud and checkpoints to a persistent Volume.

## One-time setup

```bash
pip install modal        # only needed locally to launch; not a repo dependency
modal token new          # authenticates to your Modal account (opens browser)
```

## Run

```bash
modal run scripts/modal_train.py \
    --iterations 20 --tasks 50 --games-per-task 2 --sp-iters 800
```

- `--tasks 50 --games-per-task 2` → 100 self-play games per iteration, fanned
  out across 50 containers (so ~one container's worth of wall-clock).
- `--sp-iters 800` → 800 MCTS simulations per move (vs 400 we could afford on
  the laptop). More sims = a stronger policy-improvement target — the lever.
- The job starts from a Greedy-distilled warm net, then refines via self-play.
  It resumes from the checkpoint if one exists in the Volume.

It runs entirely in the cloud (your laptop can disconnect). Progress + results
are written to the `azul-checkpoints` Volume.

## Fetch results

```bash
modal volume get azul-checkpoints azul_net.pt .     # the trained net
modal volume get azul-checkpoints metrics.txt .     # per-iteration vs-Greedy log
```

Then evaluate locally:

```bash
python3 -c "
import random
from azul.train import load_net
from azul.arena import nn_match
from azul.agent import GreedyAgent
net = load_net('azul_net.pt', hidden=512)
r = nn_match(net, lambda i: GreedyAgent(), n_games=24, iterations=400, agent_seed=1)
print(f'vs Greedy: {r.wins_a}-{r.wins_b}-{r.ties} ({r.win_rate_a:.0%})')
"
```

## Cost / scaling notes

- Cost scales with `tasks × games_per_task × sp_iters × iterations`. Start
  small (e.g. `--iterations 3 --tasks 20`) to gauge wall-clock and spend before
  a long run.
- To push harder, raise `--tasks` (more parallel containers) and `--sp-iters`.
- This is the "more meaningful sims" experiment: does enough self-play at high
  simulation count let the net surpass Greedy? Combined with the action-space
  pruning (drops dominated floor-dumps), each simulation counts for more.
- If you later want true GPU-native scale (thousands of games stepping on the
  accelerator), that's the JAX + `mctx` engine rewrite — a separate project.
