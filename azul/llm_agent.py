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
- Columns (+7) are very valuable and achievable; the 5-tile bottom row and a \
full color set are hard in a 2-player game (tiles usually don't recycle), so \
don't over-invest in them.
- The top rows (smaller pattern lines) complete fastest — the 1-tile row 0 is \
the quickest path and watch for the opponent racing it to end the game early.
- Only stage what you can finish: tiles left in a pattern line at game end score \
nothing, and overfilling dumps tiles onto your floor for penalties. Most games \
last ~5 rounds — don't plan long chains in the big bottom rows.
- Order matters within a round: scoring happens top-to-bottom, so completing a \
line adjacent to one you'll also complete this round compounds the score.
- Denial matters: if the opponent can't place a color (their only open line for \
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
    return "\n".join(parts)


def describe_legal_moves(moves: list[Move]) -> str:
    """List every legal move with its shortcut code and a readable description."""
    lines = ["LEGAL MOVES (choose exactly one code):"]
    for move in moves:
        lines.append(f"  {move_to_shortcut(move)}  — {render_move(move)}")
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

            self._client = anthropic.Anthropic()
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
        user = (
            f"{describe_state(state)}\n\n{describe_legal_moves(moves)}\n\n"
            "Choose your move."
        )
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
