"""Tests for render() and HumanAgent (with injected I/O)."""
import random

from azul.agent import HumanAgent, RandomAgent
from azul.game import Game
from azul.render import (
    render, render_move, render_board,
    ordered_sources, render_move_guide, parse_move_shortcut, _compress_rows,
)
from azul.state import Color, GameState, Move, WALL_PATTERN, CENTER, FLOOR


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_returns_string():
    gs = GameState.new_game(42)
    out = render(gs)
    assert isinstance(out, str) and out


def test_render_mentions_round_and_player():
    gs = GameState.new_game(42)
    out = render(gs)
    assert "Round 1" in out
    assert "Player 0" in out


def test_render_board_shows_score():
    gs = GameState()
    gs.player_boards[0].score = 17
    assert "score=17" in render_board(gs, 0)


def test_render_move_readable():
    assert render_move(Move(CENTER, Color.RED, FLOOR)) == "take Red from center -> floor"
    assert render_move(Move(2, Color.BLUE, 3)) == "take Blue from factory 2 -> row 3"


# ---------------------------------------------------------------------------
# HumanAgent with injected I/O
# ---------------------------------------------------------------------------

def test_parse_move_shortcut_factory():
    assert parse_move_shortcut("0y2") == Move(0, Color.YELLOW, 2)
    assert parse_move_shortcut("3k4") == Move(3, Color.BLACK, 4)


def test_parse_move_shortcut_center_and_floor():
    assert parse_move_shortcut("crf") == Move(CENTER, Color.RED, FLOOR)
    assert parse_move_shortcut("0bf") == Move(0, Color.BLUE, FLOOR)


def test_parse_move_shortcut_tolerates_spaces_and_case():
    assert parse_move_shortcut(" 0 Y 2 ") == Move(0, Color.YELLOW, 2)


def test_parse_move_shortcut_rejects_garbage():
    assert parse_move_shortcut("zz") is None      # too short
    assert parse_move_shortcut("0z2") is None      # bad color
    assert parse_move_shortcut("0y") is None       # too short
    assert parse_move_shortcut("xy2") is None      # bad source


def test_human_agent_accepts_legal_shortcut():
    gs = GameState.new_game(42)
    # Factory 1 has yellow (Yx3); yellow -> row 2 is legal on a fresh board.
    target = Move(1, Color.YELLOW, 2)
    assert target in gs.legal_moves()
    agent = HumanAgent(input_fn=lambda _: "1y2", output_fn=lambda _: None)
    assert agent.choose_move(gs) == target


def test_human_agent_reprompts_on_bad_format_then_illegal():
    gs = GameState.new_game(42)
    # garbage -> illegal (factory 0 has no yellow) -> legal shortcut.
    answers = iter(["nonsense", "0y2", "0b0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    assert move == Move(0, Color.BLUE, 0)
    assert any("Format" in o for o in outputs)
    assert any("Not a legal move" in o for o in outputs)


def test_human_agent_shows_move_guide():
    gs = GameState.new_game(42)
    outputs = []
    agent = HumanAgent(input_fn=lambda _: "0b0", output_fn=outputs.append)
    agent.choose_move(gs)
    text = "\n".join(outputs)
    assert "<source><color><row>" in text


# ---------------------------------------------------------------------------
# move guide
# ---------------------------------------------------------------------------

def test_compress_rows():
    assert _compress_rows([0, 1, 2, 3, 4]) == "0-4"
    assert _compress_rows([0, 2, 3, 4]) == "0,2-4"
    assert _compress_rows([1]) == "1"
    assert _compress_rows([]) == ""


def test_ordered_sources_factories_then_center():
    gs = GameState()
    gs.factories[2] = {Color.BLUE: 1}
    gs.factories[0] = {Color.RED: 1}
    gs.center = {Color.WHITE: 1}
    srcs = ordered_sources(gs.legal_moves())
    assert srcs[-1] == CENTER          # center last
    assert srcs[:-1] == sorted(srcs[:-1])  # factories ascending


def test_move_guide_uses_source_codes_not_indices():
    gs = GameState.new_game(42)
    text = render_move_guide(gs, gs.legal_moves(), color=False)
    assert "<source><color><row>" in text
    # Source codes are the factory numbers themselves (0..4), not [n] indices.
    assert "[0]" not in text
    # A factory line starts with its own number.
    assert any(line.strip().startswith("0") for line in text.splitlines())


def test_move_guide_marks_forced_floor():
    gs = GameState()
    gs.center = {Color.BLUE: 2}
    board = gs.player_boards[0]
    for row in range(5):
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == Color.BLUE)
        board.wall[row][col] = Color.BLUE
    text = render_move_guide(gs, gs.legal_moves(), color=False)
    # Blue has no open rows -> shown as floor-only.
    assert "B:f" in text


def test_human_agent_drives_a_game():
    """A HumanAgent fed a legal shortcut each turn completes a game."""
    from azul.render import GLYPH

    holder = {}

    def input_fn(_):
        m = holder["moves"][0]
        src = "c" if m.source == CENTER else str(m.source)
        row = "f" if m.dest_line == FLOOR else str(m.dest_line)
        return f"{src}{GLYPH[m.color].lower()}{row}"

    human = HumanAgent(input_fn=input_fn, output_fn=lambda _: None)
    orig = human.choose_move

    def choose(state):           # stash this turn's legal moves for input_fn
        holder["moves"] = state.legal_moves()
        return orig(state)

    human.choose_move = choose
    game = Game(agents=[human, RandomAgent(random.Random(5))], seed=3)
    result = game.play()
    assert game.over
    assert result.scores == [b.score for b in game.state.player_boards]
