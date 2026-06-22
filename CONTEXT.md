# Azul AI — Project Context
## What this is
A 2-player implementation of the board game Azul, with progressively smarter AI agents trained to play it. Built as a learning project: the goal is to understand CS principles (state representation, TDD, game-tree search, Monte Carlo methods, self-play RL) — not to vibe-code a finished product. AI is used heavily for scaffolding and explanation, but Pranay writes the load-bearing logic.
## The deal
- Pranay reviews and co-creates key design decisions, specs, and direction
- Claude writes all the code — logic, tests, scaffolding, everything
- Nothing moves to the next phase until tests pass
- The tests are the discipline
## Stack
- Python, dataclasses, enums
- pytest for tests
- NumPy later (Phase 4+ optimization)
- PyTorch in Phase 6 (self-play NN)
---
## Roadmap
| Phase | What | Key concepts |
|-------|------|-------------|
| 1 | Game engine (state, move gen, scoring) | TDD, invariants, state representation |
| 2 | Play loop + agent interface | Separation of concerns, abstractions |
| 3 | Random + greedy agents | Evaluation functions |
| 4 | Minimax + alpha-beta + transposition table | Game trees, pruning, Zobrist hashing |
| 5 | MCTS + UCB | Monte Carlo, exploration/exploitation |
| 6 | Self-play NN (AlphaZero-lite) | RL, policy/value networks, self-play |
| 7 | Evaluation harness | Tournaments, reproducibility |
Currently: **Phase 6 complete — roadmap done** (Phases 1–6 built & tested).
Scaled cloud self-play has the NN climbing past Greedy (0→33% over 4 iters) —
the laptop plateau was a compute wall, broken on Modal.

- Phase 1: engine complete — state, scoring, move gen, refill, new_game. All TDD'd.
- Phase 2: `Agent` ABC, `end_game_bonus()`, `Game` loop (`play()`/`step()`),
  `RandomAgent`, `HumanAgent`, text `render()`. Loop handles two game-ending
  conditions: completed wall row, and tile starvation (bag+discard exhausted).
  Agents receive a `clone()` of the state (safe, optimize later).
- Phase 3: `evaluate()` heuristic (azul/heuristics.py), `GreedyAgent`,
  `play_match` harness (azul/match.py). Greedy beats Random 100/100.
- Phase 4: `encode()` (TT key), `MinimaxAgent` (azul/minimax.py) — alpha-beta,
  round-end horizon (no expansion past the stochastic refill), relative
  zero-sum leaf eval, greedy move ordering, flagged transposition table.
  Minimax(d2) beats Greedy 8-0.
  **Measured finding:** deepcopy `clone()` is ~94% of search time, and the TT
  gets ~0 hits — transpositions are rare in Azul (moves consume specific tiles
  + players alternate, so positions diverge). So Zobrist (planned for Phase 4)
  was DEFERRED: it speeds the TT key, but the TT barely helps here. The real
  bottleneck is `clone()`. Zobrist still deferred (TT barely helps here).
- clone() optimization (done in Phase 5): hand-written copy replaced deepcopy,
  ~60x faster (240us->4us; minimax d3 1726ms->331ms). Behind the stable
  boundary — no API/behavior change. Enabled practical greedy rollouts.
- Phase 5: `MCTSAgent` (azul/mcts.py) — UCB1 select/expand/rollout/backprop,
  most-visits choice, seeded rng. Round transitions in rollouts via shared
  `advance_round_if_over` (azul/game.py). Findings:
    * random rollouts: beat Random 5-0, LOSE to Greedy 0-4 (random Azul play
      too weak to give signal).
    * greedy rollouts: improve to 2-4 vs Greedy but iteration-starved.
    * truncated rollouts (play N moves then score with evaluate(), squashed to
      [0,1]): depth-8 greedy @400it beats Greedy 8-0 — matches Minimax.
      depth-0 (eval node immediately) LOSES 1-7, so a few rollout moves matter.
  Reward stored per-node from player-0's perspective (flip for P1).
- Phase 6: AlphaZero-lite. `encoding.py` (state->148 floats canonical to the
  player to move; move<->index over 180; legal mask), `net.py` (MLP, policy +
  value heads, `predict` bridge), `az_mcts.py` (`NeuralMCTSAgent` — PUCT, net
  value at leaves, no rollouts; value stored by player identity), `selfplay.py`,
  `train.py` (policy CE + value MSE, iterate loop, checkpoints), `arena.py`
  (NN vs baseline), `warmstart.py` (distill Greedy / MCTS / hybrid). PyTorch CPU.

  **Exit condition was "learns + beats Greedy." Result: LEARNS (emphatically),
  does NOT beat Greedy — and we proved why.** Training results vs baselines:
    * pure self-play:            0% vs Random, 0-10 vs Greedy
    * greedy-softmax distill:   75% vs Random, 0-12 vs Greedy
    * MCTS-teacher distill:     42% vs Random, 0-12 vs Greedy (visit-count
      target is itself starved at affordable sims → noisier than greedy-softmax)
    * heavy hybrid (strong-MCTS states, clean greedy-softmax policy targets,
      strong-play value targets, 512-wide net, 2739 examples, loss 5.57->2.68):
      **100% vs Random**, still 0-12 vs Greedy at 200 AND 400 play-time sims.

  On a LAPTOP the Greedy bar held: more play-time sims didn't help (0-6 at 200
  and 400) and every distillation asymptoted at ~Greedy strength. It looked
  like a method wall. It wasn't — it was compute starvation.

  **BREAKTHROUGH — scaled self-play breaks the wall (Modal cloud).** With
  C (action-space pruning, `pruning.py` — drops dominated optional floor-dumps)
  + scaled self-play (800 sims/move, ~3200 examples/iter, fanned across
  cloud containers), genuine AlphaZero refinement of the distilled net CLIMBED
  vs Greedy:
        iter 0: 0%  →  iter 1: 8%  →  iter 2: 17%  →  iter 3: 33%
  First-ever NN wins vs Greedy — the policy-improvement loop working once fed
  enough simulations. Self-play improvement is open-ended (NOT capped at the
  teacher, unlike distillation). The laptop plateau was the branching-factor
  compute wall (same as Phases 4-5); cloud scale clears it.

  **Final scaled result (replay buffer + 24-game evals, 12 iters):** training
  on only the newest iteration's games caused oscillation (17/0/25%); a REPLAY
  BUFFER (last 6 iters) fixed it → steady climb 0→...→30% vs Greedy, **100% vs
  Random, 30% vs Greedy (12-28, 40-game eval)**. Still gently rising at iter 12.
  Honest bottom line: cloud scale broke the 0% wall and the net climbs steadily
  to ~30% vs Greedy, but did NOT surpass Greedy (>50%) on this budget (12 iters
  / 60 games / 800 sims) — slow diminishing returns. Beating Greedy outright
  with the net would need many more iterations / more sims+games per iter.

## Scaling infra (scripts/modal_train.py, docs/MODAL.md)
  Workload is CPU-bound (tiny net) → scale = many CPU containers, not a GPU.
  Lesson: a remote orchestrator fanning out via nested starmap does NOT survive
  the launcher disconnecting (children get cancelled). Robust form: one
  self-contained detached `run_training` (cpu=16) doing self-play via internal
  multiprocessing, `.spawn()`-launched, resuming from the Volume checkpoint.
  Progress: `modal volume get azul-checkpoints metrics.txt .`

## Final agent ladder
  Random  <  Greedy  <  { NN-after-scaled-self-play (climbing past Greedy),
                          Minimax, MCTS (both 8-0 vs Greedy) }
  All behind one `Agent` interface; comparable via `play_match`.
---
## State representation decisions
These were made deliberately — don't change without discussion.
- **Colors**: `Color(IntEnum)` with values 0–4. IntEnum so colors can double as array indices later.
- **Wall**: stores `Color | None` per cell. Each cell's color is fixed by position (`WALL_PATTERN[row][col]`), derived from the formula `Color((col - row) % 5)`.
- **Pattern lines**: store only `(color, count)`. Capacity is positional — `PATTERN_LINE_CAPACITY[row_index]`, not stored on the line itself.
- **Bag / discard / factories / center**: `dict[Color, int]` (count per color). Not ordered lists.
- **Floor line**: `floor_count: int` + `has_first_player_marker: bool`. Not a list of tiles.
- **Move**: `frozen=True` dataclass — hashable, immutable. Fields: `source` (factory index or `CENTER = -1`), `color`, `dest_line` (pattern row or `FLOOR = -1`).
- **Copying**: `clone()` uses `deepcopy` for now. This is flagged as the Phase 4 optimization point. The search interface (`legal_moves / apply / clone / encode`) is the stable boundary — guts can be swapped to a flat array later without touching agent code.
---
## File structure
```
azul-ai/
  azul/
    __init__.py
    state.py        ← data model + stubbed logic (Phase 1 target)
  tests/
    test_state.py   ← Phase 1 tests (to be written)
  CONTEXT.md        ← this file
  README.md
```
---
## Phase 1 — where we are
`azul/state.py` is scaffolded:
- All dataclasses and constants are defined and correct
- Every method with real game logic raises `NotImplementedError`
- `clone()` is the only implemented method (it's infrastructure, not game logic)
**The first task**: write `tile_wall_and_score()`, driven by tests. This is the trickiest rule — adjacency scoring — and the best TDD target. Start with tests for adjacency scoring edge cases before writing a single line of implementation.
### Adjacency scoring rules (for reference)
When a tile is placed on the wall at end of round:
- If it has no filled neighbors (horizontally or vertically): score 1 point
- If it has filled neighbors: score the length of its contiguous horizontal run + the length of its contiguous vertical run (counting the tile itself in each)
- A tile in both a horizontal and vertical run scores both
### Methods still to implement (in rough order)
1. `tile_wall_and_score()` — end-of-round scoring. Start here.
2. `is_round_over()` — True when all factories + center are empty
3. `is_game_over()` — True when any player has a complete horizontal wall row
4. `apply(move)` — apply a move in place
5. `legal_moves()` — all legal moves for current player
6. `refill_factories()` — draw from bag, handle bag running dry
7. `new_game(seed)` — fresh game state, calls refill_factories
### Known design tension to revisit in Phase 4
`PatternLine` doesn't know its own capacity — callers need the row index. This is correct but worth flagging when writing `apply()`.
