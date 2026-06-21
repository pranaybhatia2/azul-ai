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
Currently: **Phase 6 complete — roadmap done** (Phases 1–6 built & tested)

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

  **Why the Greedy bar held (well-evidenced, not a bug):**
    1. More play-time sims don't help (0-6 at 200 and 400) → the bottleneck is
       the learned VALUE head, not search budget.
    2. On CPU the net can only be distilled from the heuristic (or from
       search whose visit-count signal is starved by Azul's ~90-move
       branching), so it asymptotes at ~Greedy strength — a fuzzy copy of
       Greedy. Beating a sharp 1-ply heuristic needs genuine multi-step
       planning, which Minimax/MCTS get from DEPTH (both beat Greedy 8-0) but
       a learned value can't reveal at this simulation budget.
    3. Same branching-factor wall as Phases 4-5, now shown to also starve the
       NN's training signal. Clearing it needs AlphaZero-scale self-play
       (thousands of sims/move) → a GPU, not a code change.

## Final agent ladder
  Random  <  Greedy ≈ NN (100% vs Random)  <  Minimax ≈ MCTS (both 8-0 vs Greedy)
  All behind one `Agent` interface; comparable via `play_match`. The search
  agents are the Greedy-beaters; the NN matches Greedy on CPU compute.

## How to actually beat Greedy with the NN (GPU-scale self-play)
  See the "GPU self-play" notes below — the method is right, only the compute
  budget (sims/move × games × iterations) is short.
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
