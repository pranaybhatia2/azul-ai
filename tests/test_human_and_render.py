"""Tests for render() and HumanAgent (with injected I/O)."""
import random

from azul.agent import HumanAgent, RandomAgent
from azul.game import Game
from azul.render import (
    render, render_move, render_board, organize_moves, render_move_menu,
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

def test_human_agent_picks_indexed_move():
    gs = GameState.new_game(42)
    outputs = []
    agent = HumanAgent(input_fn=lambda _: "0", output_fn=outputs.append)
    move = agent.choose_move(gs)
    # Index maps into the primary (grouped) move list, not raw legal_moves.
    primary, _ = organize_moves(gs.legal_moves())
    assert move == primary[0]
    assert any("Your moves" in o for o in outputs)


def test_human_agent_retries_on_bad_input():
    gs = GameState.new_game(42)
    answers = iter(["nonsense", "999", "1"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    primary, _ = organize_moves(gs.legal_moves())
    assert move == primary[1]


def test_human_agent_f_opens_floor_submenu():
    gs = GameState.new_game(42)
    answers = iter(["f", "0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    _, optional_floor = organize_moves(gs.legal_moves())
    assert move == optional_floor[0]
    assert move.dest_line == FLOOR


# ---------------------------------------------------------------------------
# organize_moves
# ---------------------------------------------------------------------------

def test_organize_hides_optional_floor_dumps():
    gs = GameState.new_game(42)
    moves = gs.legal_moves()
    primary, optional_floor = organize_moves(moves)
    # On a fresh board every color has a valid line, so all floor dumps are optional.
    assert all(m.dest_line != FLOOR for m in primary)
    assert all(m.dest_line == FLOOR for m in optional_floor)
    assert optional_floor  # there are some


def test_organize_keeps_forced_floor_in_primary():
    gs = GameState()
    gs.center = {Color.BLUE: 2}
    board = gs.player_boards[0]
    # Block blue on every pattern line by filling its wall slot.
    for row in range(5):
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == Color.BLUE)
        board.wall[row][col] = Color.BLUE
    primary, optional_floor = organize_moves(gs.legal_moves())
    # Blue's only option is the floor — it must be forced into primary.
    assert Move(CENTER, Color.BLUE, FLOOR) in primary
    assert optional_floor == []


def test_menu_groups_by_source():
    gs = GameState.new_game(42)
    text, indexed = render_move_menu(gs.legal_moves())
    assert "Factory 0:" in text
    assert len(indexed) > 0


def test_human_agent_drives_a_game():
    """A HumanAgent that always picks move 0 can complete a game."""
    human = HumanAgent(input_fn=lambda _: "0", output_fn=lambda _: None)
    game = Game(agents=[human, RandomAgent(random.Random(5))], seed=3)
    result = game.play()
    assert game.over
    assert result.scores == [b.score for b in game.state.player_boards]
