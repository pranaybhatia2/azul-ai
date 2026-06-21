"""Tests for render() and HumanAgent (with injected I/O)."""
import random

from azul.agent import HumanAgent, RandomAgent
from azul.game import Game
from azul.render import (
    render, render_move, render_board, organize_moves,
    ordered_sources, render_source_menu, render_placement_menu,
    parse_move_shortcut,
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

def _src_primary(gs, source_pick):
    """Helper: the primary placements for the source at index `source_pick`."""
    moves = gs.legal_moves()
    sources = ordered_sources(moves)
    source = sources[source_pick]
    src_moves = [m for m in moves if m.source == source]
    primary, optional_floor, _ = render_placement_menu(src_moves)
    return primary, optional_floor


def test_human_agent_two_step_selection():
    gs = GameState.new_game(42)
    # Pick source 0, then placement 0.
    answers = iter(["0", "0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    primary, _ = _src_primary(gs, 0)
    assert move == primary[0]
    assert any("Take tiles from" in o for o in outputs)


def test_human_agent_retries_on_bad_input():
    gs = GameState.new_game(42)
    # source "0", then a bad placement, then placement "1".
    answers = iter(["0", "nonsense", "1"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    primary, _ = _src_primary(gs, 0)
    assert move == primary[1]


def test_human_agent_back_navigation():
    gs = GameState.new_game(42)
    # Enter source 1, back out ('b'), then source 0 placement 0.
    answers = iter(["1", "b", "0", "0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    primary, _ = _src_primary(gs, 0)
    assert move == primary[0]


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


def test_human_agent_rejects_illegal_shortcut_then_falls_back():
    gs = GameState.new_game(42)
    # Factory 0 has no yellow, so '0y2' is illegal -> error -> two-step 0,0.
    answers = iter(["0y2", "0", "0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    primary, _ = _src_primary(gs, 0)
    assert move == primary[0]
    assert any("Not a legal move" in o for o in outputs)


def test_human_agent_f_opens_floor_submenu():
    gs = GameState.new_game(42)
    # source 0, then 'f', then floor option 0.
    answers = iter(["0", "f", "0"])
    outputs = []
    agent = HumanAgent(input_fn=lambda _: next(answers), output_fn=outputs.append)
    move = agent.choose_move(gs)
    _, optional_floor = _src_primary(gs, 0)
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


def test_ordered_sources_factories_then_center():
    gs = GameState()
    gs.factories[2] = {Color.BLUE: 1}
    gs.factories[0] = {Color.RED: 1}
    gs.center = {Color.WHITE: 1}
    srcs = ordered_sources(gs.legal_moves())
    assert srcs[-1] == CENTER          # center last
    assert srcs[:-1] == sorted(srcs[:-1])  # factories ascending


def test_render_source_menu_lists_sources():
    gs = GameState.new_game(42)
    sources = ordered_sources(gs.legal_moves())
    text = render_source_menu(gs, sources)
    assert "Take tiles from" in text
    assert "Factory 0" in text


def test_render_placement_menu_for_single_source():
    gs = GameState.new_game(42)
    src_moves = [m for m in gs.legal_moves() if m.source == 0]
    primary, optional_floor, text = render_placement_menu(src_moves)
    assert all(m.source == 0 for m in primary)
    assert "row" in text


def test_human_agent_drives_a_game():
    """A HumanAgent that always picks move 0 can complete a game."""
    human = HumanAgent(input_fn=lambda _: "0", output_fn=lambda _: None)
    game = Game(agents=[human, RandomAgent(random.Random(5))], seed=3)
    result = game.play()
    assert game.over
    assert result.scores == [b.score for b in game.state.player_boards]
