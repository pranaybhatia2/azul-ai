"""An LLM-driven Azul agent (Anthropic Claude).

`LLMAgent` plays Azul by describing the position to a Claude model in plain
text, asking for a short rationale plus a move code, and parsing the reply back
into a `Move`. It is a drop-in `Agent`: the game loop calls `choose_move()` and
gets one legal move, exactly like Greedy/MCTS.

Design notes (see CONTEXT.md / project memory):
- The model is told the rules + strategy and is shown the current state and the
  *exact* legal moves with their shortcut codes (e.g. ``0y2``). It replies with
  brief reasoning then a final ``MOVE: <code>`` line, reusing the same shortcut
  grammar the human CLI uses (azul.render.parse_move_shortcut).
- Robustness: an illegal or unparseable reply is retried with a corrective
  nudge (up to ``max_move_retries``); if the model still won't produce a legal
  move, we fall back to a deterministic agent (Greedy by default) so a game
  never crashes on a bad generation.
- The network call is injected as ``complete`` so tests run without an API key
  or any network; the default implementation uses the official ``anthropic``
  SDK and resolves credentials from the environment (ANTHROPIC_API_KEY /
  ANTHROPIC_BASE_URL / an ``ant auth login`` profile).
"""
from __future__ import annotations

from typing import Callable, Optional

from azul.agent import Agent, GreedyAgent
from azul.render import (
    color_name,
    move_to_shortcut,
    parse_move_shortcut,
    render_move,
)
from azul.state import (
    CENTER,
    FLOOR,
    PATTERN_LINE_CAPACITY,
    Color,
    GameState,
    Move,
    WALL_PATTERN,
)

# Type of the injectable completion function: given a system prompt and the
# running message list ([{"role", "content"}, ...]), return the model's text.
CompleteFn = Callable[[str, list[dict]], str]

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_EFFORT = "low"  # GA effort param; low = fast/cheap, fits a per-move call

SYSTEM_PROMPT = """\
You are an expert Azul player. You will be given the full state of a 2-player \
Azul game and the list of every legal move available to you on this turn. Pick \
the single best move.

RULES (this implementation):
- Each turn you take ALL tiles of one color from one factory OR from the center, \
then place them on one of your five pattern lines, or onto your floor line.
- Taking from a factory pushes that factory's other tiles into the center. \
Taking from the center for the first time in a round also takes the \
first-player marker: you lead next round, but the marker sits on your floor and \
costs you a floor penalty at round end.
- Pattern line `row r` holds r+1 tiles (row 0 holds 1 ... row 4 holds 5) and may \
only ever hold a single color. Tiles that don't fit the chosen line overflow \
onto your floor.
- At round end, each COMPLETELY FILLED pattern line moves one tile to its wall \
cell (the rest are discarded) and scores; partially filled lines stay for next \
round. Each wall row/column has a fixed color layout — a line can only be filled \
with a color whose wall cell in that row is still empty.
- Wall scoring on placement: a lone tile scores 1; otherwise it scores the \
length of its horizontal run plus the length of its vertical run (a tile that \
extends both rows and columns scores for both).
- Floor tiles are penalties: the floor slots cost -1,-1,-2,-2,-2,-3,-3.
- End-of-game bonuses: +2 per complete wall row, +7 per complete wall column, \
+10 per color placed all 5 times. The game ends the round any player completes \
a full horizontal wall row. Highest score wins.

STRATEGY TIPS:
- Build out from tiles you already have: adjacency (both a horizontal and a \
vertical run at once) is where the points are. Favor the central columns early \
to keep future placement options open.
- END-GAME BONUSES DECIDE CLOSE GAMES. A completed COLUMN is +7 and is the \
single biggest swing — actively drive your near-complete columns to 5. Use the \
'Wall progress' counts for both players: a column at 4/5, or a color at 4/5 \
(+10), is worth chasing. The 5-tile bottom row and full color sets are hard in \
a 2-player game (tiles usually don't recycle), so weigh them against easier \
columns.
- ORCHESTRATE THE WALL, don't just grab the easiest line each turn. The big \
scores come from bonuses COMPOUNDING: aim to complete rows, columns, and color \
sets together rather than one isolated line at a time. Prefer a placement that \
advances a column AND a row you are already building. A common losing pattern is \
to keep completing small lines for a few points each while the opponent quietly \
assembles a wall that erupts for column/color bonuses in the final round — your \
mid-game tempo means nothing if their end-game bonuses dwarf it.
- DON'T SCATTER LATE. Placing single tiles across many different rows wastes \
turns; concentrating tiles to actually COMPLETE a line — especially to finish a \
column — is what scores. Late in the game, prefer the move that completes (or \
directly sets up completing) a line or column over one that merely starts a new \
line.
- USE THE MOVE ANNOTATIONS. Each legal move is labeled with its computed effect \
(overflow, whether it completes a line, the wall cell, adjacency points, and how \
it advances each bonus). Read those numbers and choose strategically — do not \
re-derive wall geometry or adjacency yourself; the annotations are authoritative.
- The top rows (smaller pattern lines) complete fastest — the 1-tile row 0 is \
the quickest path and watch for the opponent racing it to end the game early.
- Only stage what you can finish: tiles left in a pattern line at game end score \
nothing, and overfilling dumps tiles onto your floor for penalties. Most games \
last ~5 rounds — don't plan long chains in the big bottom rows.
- Avoid dumping tiles to the floor unless you truly have no better placement, \
and especially avoid floor dumps on the last round where they are pure loss.
- TIME THE GAME END. The game ends the round any player completes a full wall \
row. If you are ahead on the board but behind on pending end-game bonuses \
(near-complete columns/colors), don't be the one to end it early; if you are \
ahead overall, ending it can deny the opponent their bonuses.
- Order matters within a round: scoring happens top-to-bottom, so completing a \
line adjacent to one you'll also complete this round compounds the score.
- BLOCK THE OPPONENT. Anticipate their next move: if they have a line one or \
two tiles from completing and the tiles they need are available, TAKING those \
tiles yourself denies them — often worth more than a small gain of your own. \
The 'OPPONENT IMMINENT THREATS' section lists exactly what they can complete and \
for how much; candidate moves marked 'DENIES' take tiles they were counting on. \
A move that blocks a big completion can swing the game even if it parks a tile on \
your floor. Also: if the opponent can't place a color (their only open line for \
it is taken or walled), forcing 3+ of that color onto them floods their floor \
with penalties. Watch which colors will be left at round end so you don't get \
stuck taking penalties yourself.
- The first-player marker is worth a small penalty when it secures the exact \
tiles you need to complete a key line.

OUTPUT FORMAT:
Think briefly (one or two short sentences), then end your reply with a line of \
exactly the form:
MOVE: <code>
where <code> is one of the move codes listed for this turn (e.g. `MOVE: 0y2`). \
Output the MOVE line last and nothing after it."""


def _describe_pool(pool: dict[Color, int]) -> str:
    parts = [f"{n} {color_name(c)}" for c, n in sorted(pool.items()) if n]
    return ", ".join(parts) if parts else "empty"


def _describe_wall_row(wall_row: list[Optional[Color]], row: int) -> str:
    cells = []
    for col in range(5):
        target = WALL_PATTERN[row][col]
        if wall_row[col] is not None:
            cells.append(f"{color_name(target)}[placed]")
        else:
            cells.append(color_name(target))
    return ", ".join(cells)


def _wall_progress(board) -> list[str]:
    """Counts toward the end-game bonuses, so the model doesn't have to tally
    the wall itself: per-column (5 -> +7), per-row (5 -> +2 and ENDS the game),
    per-color (5 -> +10)."""
    wall = board.wall
    col_counts = [
        sum(wall[r][c] is not None for r in range(5)) for c in range(5)
    ]
    row_counts = [
        sum(wall[r][c] is not None for c in range(5)) for r in range(5)
    ]
    color_counts = {
        color: sum(
            wall[r][c] == color for r in range(5) for c in range(5)
        )
        for color in Color
    }
    cols = ", ".join(f"col{c}={col_counts[c]}/5" for c in range(5))
    rows = ", ".join(f"row{r}={row_counts[r]}/5" for r in range(5))
    colors = ", ".join(
        f"{color_name(c)}={color_counts[c]}/5" for c in Color
    )
    return [
        "  Wall progress toward end-game bonuses:",
        f"    columns (5 = +7): {cols}",
        f"    rows (5 = +2, and completing one ENDS the game): {rows}",
        f"    colors (5 = +10): {colors}",
    ]


def _describe_board(state: GameState, player: int, label: str) -> str:
    board = state.player_boards[player]
    lines = [f"{label} (Player {player}) — score {board.score}:"]

    lines.append("  Pattern lines:")
    for row in range(5):
        pl = board.pattern_lines[row]
        cap = PATTERN_LINE_CAPACITY[row]
        if pl.count:
            state_str = f"{pl.count}/{cap} {color_name(pl.color)}"
        else:
            state_str = f"0/{cap} (empty)"
        lines.append(f"    row {row} (holds {cap}): {state_str}")

    lines.append("  Wall (each cell's fixed target color; [placed] = filled):")
    for row in range(5):
        lines.append(f"    row {row}: {_describe_wall_row(board.wall[row], row)}")

    lines.extend(_wall_progress(board))

    floor = f"  Floor: {board.floor_count} tile(s)"
    if board.has_first_player_marker:
        floor += " + first-player marker"
    lines.append(floor)
    return "\n".join(lines)


def describe_state(state: GameState) -> str:
    """Render the position as plain text aimed at an LLM reader. The current
    player is framed as YOU; the other player as the OPPONENT."""
    me = state.current_player
    opp = 1 - me
    parts = [
        f"=== Azul — Round {state.round_number} ===",
        f"You are Player {me}. It is your turn.",
        "",
        "FACTORIES:",
    ]
    for i, factory in enumerate(state.factories):
        parts.append(f"  Factory {i}: {_describe_pool(factory)}")
    center = f"CENTER: {_describe_pool(state.center)}"
    if state.first_player_marker_in_center:
        center += "  (first-player marker still here)"
    parts.append(center)
    parts.append("")
    parts.append(_describe_board(state, me, "YOUR BOARD"))
    parts.append("")
    parts.append(_describe_board(state, opp, "OPPONENT BOARD"))
    threats = _imminent_completions(state, opp)
    if threats:
        parts.append("")
        parts.append("OPPONENT THREATS (lines they are developing whose tiles are "
                     "still available — taking those tiles blocks them; ✓ = they "
                     "could finish it right now):")
        parts.extend(f"  {t}" for t in threats)
    return "\n".join(parts)


def _imminent_completions(state: GameState, player: int) -> list[str]:
    """Partial pattern lines `player` is developing whose color is still
    available — both finishable-now lines and near-complete ones (within 2 of
    the end), with the wall score completing would lock in. Surfaces blocking
    opportunities for the opponent of `player`, biggest threat first."""
    board = state.player_boards[player]
    avail = {c: 0 for c in Color}
    for factory in state.factories:
        for c, n in factory.items():
            avail[c] += n
    for c, n in state.center.items():
        avail[c] += n
    rows = []
    for row, pl in enumerate(board.pattern_lines):
        cap = PATTERN_LINE_CAPACITY[row]
        if pl.color is None or not (0 < pl.count < cap):
            continue
        col = next(k for k in range(5) if WALL_PATTERN[row][k] == pl.color)
        if board.wall[row][col] is not None:
            continue
        needed = cap - pl.count
        # Show developing lines that are close (<=2 away) AND have tiles to grab.
        if avail[pl.color] == 0 or needed > 2:
            continue
        pts = GameState._adjacency_score(board.wall, row, col)
        finishable = avail[pl.color] >= needed
        rows.append((needed, -pts, finishable, row, pl.count, cap, pl.color, avail[pl.color]))
    rows.sort()  # nearest-to-done, then highest-scoring, first
    out = []
    for needed, negpts, finishable, row, count, cap, color, navail in rows:
        mark = "✓ " if finishable else ""
        out.append(f"{mark}row {row}: {count}/{cap} {color_name(color)} — "
                   f"needs {needed}, {navail} available → ~{-negpts} pts")
    return out


def _move_annotation(state: GameState, board, move: Move) -> str:
    """Compute the mechanical consequence of `move` so the model doesn't have to
    (and doesn't get it wrong): tiles taken, pattern-line fill / floor overflow,
    and — if the move completes a line — the wall cell, the adjacency points it
    would score against the current wall, and how it advances the column / row /
    color toward their end-game bonuses."""
    pool = state.center if move.source == CENTER else state.factories[move.source]
    n = pool.get(move.color, 0)

    if move.dest_line == FLOOR:
        return f"takes {n}; all {n} -> FLOOR (penalty, no scoring)"

    cap = PATTERN_LINE_CAPACITY[move.dest_line]
    pl = board.pattern_lines[move.dest_line]
    space = cap - pl.count
    placed = min(n, space)
    new_count = pl.count + placed
    overflow = n - space

    seg = f"takes {n}; row {move.dest_line}: {pl.count}/{cap}->{new_count}/{cap}"
    if overflow > 0:
        seg += f" ({overflow} overflow -> floor)"

    if new_count != cap:
        return seg + "  (does not complete)"

    # Completes the line -> a tile lands on the wall this round end.
    wall = board.wall
    col = next(c for c in range(5) if WALL_PATTERN[move.dest_line][c] == move.color)
    adj = GameState._adjacency_score(wall, move.dest_line, col)
    col_n = sum(wall[r][col] is not None for r in range(5))
    row_n = sum(wall[move.dest_line][c] is not None for c in range(5))
    color_n = sum(
        wall[r][c] == move.color for r in range(5) for c in range(5)
    )

    def prog(now, label, bonus):
        s = f"{label} {now}/5->{now + 1}/5"
        return s + (f" ({bonus}!)" if now + 1 == 5 else "")

    bonus_bits = [
        prog(col_n, f"col{col}", "+7"),
        prog(row_n, f"row{move.dest_line}", "+2 ENDS GAME"),
        prog(color_n, color_name(move.color), "+10"),
    ]
    return (
        seg + f"  COMPLETES -> wall r{move.dest_line}c{col}, "
        f"~{adj} adjacency pts now; " + ", ".join(bonus_bits)
    )


def rank_moves(state: GameState, k: Optional[int] = None,
               opponent_aware: bool = False):
    """Score every legal move and return [(move, score), ...] best-first
    (truncated to top k if given).

    opponent_aware=False: GreedyAgent's one-ply self score, evaluate(nxt, me).
    opponent_aware=True: a relative, threat-aware score —
        threat_aware_evaluate(nxt, me) - threat_aware_evaluate(nxt, opp).
    The relative form makes DENIAL surface: taking tiles the opponent needs
    drops their threat at nxt, raising the move's score, so blocking moves climb
    into the top-k instead of being pruned by the self-only ranking.
    """
    from azul.heuristics import evaluate, threat_aware_evaluate

    me = state.current_player
    opp = 1 - me
    scored = []
    for move in state.legal_moves():
        nxt = state.clone()
        nxt.apply(move)
        if opponent_aware:
            s = threat_aware_evaluate(nxt, me) - threat_aware_evaluate(nxt, opp)
        else:
            s = evaluate(nxt, me)
        scored.append((move, s))
    scored.sort(key=lambda ms: ms[1], reverse=True)
    return scored if k is None else scored[:k]


def _denial_note(state: GameState, move: Move) -> str:
    """How much this move cuts the opponent's imminent completion threat — its
    blocking value. Empty string if it denies nothing meaningful."""
    from azul.heuristics import threat_score

    opp = 1 - state.current_player
    before = threat_score(state, opp)
    nxt = state.clone()
    nxt.apply(move)
    delta = before - threat_score(nxt, opp)
    if delta > 0.05:
        return f"  DENIES opponent ~{delta:.1f} (takes tiles they needed to complete)"
    return ""


def describe_candidate_moves(state: GameState, scored: list) -> str:
    """Render a pre-ranked candidate set (move, score) with annotations and a
    denial note. The score is a one-ply estimate (relative + threat-aware when
    the ranking is opponent-aware) — useful but shallow, which the header flags."""
    board = state.player_boards[state.current_player]
    lines = [
        "CANDIDATE MOVES — pre-ranked by a one-ply evaluator that accounts for "
        "both your gains AND denying the opponent (higher = better for you). It "
        "is still shallow: when scores are close, prefer the move that best "
        "builds toward a column/color bonus or blocks the opponent's biggest "
        "threat. 'DENIES' marks blocking moves. Choose exactly one code:",
    ]
    for rank, (move, score) in enumerate(scored, 1):
        lines.append(
            f"  {rank}. {move_to_shortcut(move)} (score {score:+.1f})  — "
            f"{render_move(move)}  [{_move_annotation(state, board, move)}]"
            f"{_denial_note(state, move)}"
        )
    return "\n".join(lines)


def describe_legal_moves(state: GameState, moves: list[Move]) -> str:
    """List every legal move with its shortcut code, a readable description, and
    a computed annotation of its consequence (see _move_annotation)."""
    board = state.player_boards[state.current_player]
    lines = [
        "LEGAL MOVES (choose exactly one code). Each line is annotated with its "
        "computed effect — trust these numbers instead of recomputing:",
    ]
    for move in moves:
        lines.append(
            f"  {move_to_shortcut(move)}  — {render_move(move)}  "
            f"[{_move_annotation(state, board, move)}]"
        )
    return "\n".join(lines)


def _extract_move(text: str, legal: dict[str, Move]) -> Optional[Move]:
    """Pull a legal move out of the model's reply. Prefers the `MOVE:` line;
    falls back to any legal code appearing as a whole token in the text."""
    codes_by_line = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.upper().startswith("MOVE:"):
            codes_by_line.append(line.split(":", 1)[1].strip())
    # MOVE: line(s) first (last one wins if several), then a scan of all tokens.
    for code in reversed(codes_by_line):
        token = code.replace(" ", "").lower().strip(".`*")
        if token in legal:
            return legal[token]
    # Fallback: scan every whitespace token for an exact legal code.
    for raw_token in text.replace("`", " ").split():
        token = raw_token.lower().strip(".,:*()[]")
        if token in legal:
            return legal[token]
    return None


class LLMAgent(Agent):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        top_k: Optional[int] = 12,
        opponent_aware: bool = True,
        client=None,
        complete: Optional[CompleteFn] = None,
        max_move_retries: int = 2,
        max_tokens: int = 1024,
        fallback: Optional[Agent] = None,
        verbose: bool = False,
    ):
        """Args:
        model: Claude model id (default Sonnet 4.6).
        effort: output_config effort — "low" (default) | "medium" | "high" | "max".
        top_k: show only the top-k moves ranked by the evaluator (hybrid mode —
            tactically vetted candidates + LLM judgment). None = show all legal
            moves annotated (no ranking).
        opponent_aware: rank candidates by a relative, threat-aware score so
            blocking/denial moves surface into the top-k (default True). Only
            applies in top_k mode.
        client: an anthropic.Anthropic instance; created lazily if omitted.
        complete: override the network call entirely — (system, messages) -> text.
            Used by tests so no API key or network is needed.
        max_move_retries: extra attempts after an illegal/unparseable reply.
        max_tokens: cap on the model's reply length.
        fallback: agent used if the model never yields a legal move (default Greedy).
        verbose: print the model's raw reply per turn (debugging / watching play).
        """
        self.model = model
        self.effort = effort
        self.top_k = top_k
        self.opponent_aware = opponent_aware
        self._client = client
        self._complete = complete
        self.max_move_retries = max_move_retries
        self.max_tokens = max_tokens
        self.fallback = fallback if fallback is not None else GreedyAgent()
        self.verbose = verbose
        # Surfaced for debugging: the last raw reply and whether we fell back.
        self.last_reply: Optional[str] = None
        self.used_fallback: bool = False

    def _default_complete(self, system: str, messages: list[dict]) -> str:
        if self._client is None:
            import anthropic

            # Bounded per-request timeout so a slow/hung call fails fast and is
            # retried, rather than blocking on the SDK's 10-min default — which
            # otherwise stalls concurrent eval runs for tens of minutes. The SDK
            # auto-retries timeouts and 429s with backoff.
            self._client = anthropic.Anthropic(timeout=90.0, max_retries=4)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            # Reasoning goes in the visible reply (brief, then MOVE:), so we run
            # without extended thinking; effort tunes depth/cost (low by default).
            thinking={"type": "disabled"},
            output_config={"effort": self.effort},
            messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def choose_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        legal = {move_to_shortcut(m): m for m in moves}
        self.used_fallback = False

        complete = self._complete or self._default_complete
        if self.top_k is not None:
            ranked = rank_moves(state, self.top_k, opponent_aware=self.opponent_aware)
            moves_text = describe_candidate_moves(state, ranked)
        else:
            moves_text = describe_legal_moves(state, moves)
        user = f"{describe_state(state)}\n\n{moves_text}\n\nChoose your move."
        messages: list[dict] = [{"role": "user", "content": user}]

        for attempt in range(self.max_move_retries + 1):
            try:
                reply = complete(SYSTEM_PROMPT, messages)
            except Exception as exc:  # network/API error — don't crash the game
                if self.verbose:
                    print(f"[LLMAgent] completion failed ({exc}); using fallback")
                break
            self.last_reply = reply
            if self.verbose:
                print(f"[LLMAgent] reply (attempt {attempt}):\n{reply}")

            move = _extract_move(reply, legal)
            if move is not None:
                return move

            # Steer the model with a corrective message and try again.
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not a valid move code. Reply with exactly one "
                        "code from the LEGAL MOVES list above, ending with a line "
                        "'MOVE: <code>'."
                    ),
                }
            )

        # Exhausted retries (or the API failed): fall back to a safe agent.
        self.used_fallback = True
        return self.fallback.choose_move(state)
