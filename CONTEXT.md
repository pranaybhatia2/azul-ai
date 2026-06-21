# Azul AI — Project Context
## What this is
A 2-player implementation of the board game Azul, with progressively smarter AI agents trained to play it. Built as a learning project: the goal is to understand CS principles (state representation, TDD, game-tree search, Monte Carlo methods, self-play RL) — not to vibe-code a finished product. AI is used heavily for scaffolding and explanation, but Pranay writes the load-bearing logic.
## The deal
- Claude scaffolds structure, explains concepts, writes tests alongside Pranay
- Pranay writes the game logic (scoring, move generation, etc.)
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
Currently: **Phase 1**
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
