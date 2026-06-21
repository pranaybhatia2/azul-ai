"""Tests for render() and HumanAgent (with injected I/O)."""
import random

from azul.agent import HumanAgent, RandomAgent
from azul.game import Game
from azul.render import render, render_move, render_board
from azul.state import Color, GameState, Move, CENTER, FLOOR


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
    assert render_move(Move(2, Color.BLUE, 3)) == "take Blue from factory 2 -> line 3"


# ---------------------------------------------------------------------------
# HumanAgent with injected I/O
# ---------------------------------------------------------------------------

def test_human_agent_picks_indexed_move():
    gs = GameState.new_game(42)
    outputs = []
    agent = HumanAgent(input_fn=lambda _: "0", output_fn=outputs.append)
    move = agent.choose_move(gs)
    assert move == gs.legal_moves()[0]
    assert any("Legal moves" in o for o in outputs)


def test_human_agent_retries_on_bad_input():
    gs = GameState.new_game(42)
    answers = iter(["nonsense", "999", "1"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    assert move == gs.legal_moves()[1]


def test_human_agent_drives_a_game():
    """A HumanAgent that always picks move 0 can complete a game."""
    human = HumanAgent(input_fn=lambda _: "0", output_fn=lambda _: None)
    game = Game(agents=[human, RandomAgent(random.Random(5))], seed=3)
    result = game.play()
    assert game.over
    assert result.scores == [b.score for b in game.state.player_boards]
