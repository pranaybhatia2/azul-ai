"""Tests for the LLM agent — all offline (the network call is injected)."""
from azul.agent import Agent
from azul.llm_agent import (
    LLMAgent,
    describe_legal_moves,
    describe_state,
    _extract_move,
)
from azul.render import move_to_shortcut
from azul.state import Color, GameState, Move, CENTER, FLOOR


# --- move_to_shortcut <-> parse_move_shortcut round-trip --------------------

def test_move_shortcut_roundtrip():
    from azul.render import parse_move_shortcut
    moves = [
        Move(0, Color.YELLOW, 2),
        Move(CENTER, Color.RED, FLOOR),
        Move(4, Color.WHITE, 0),
        Move(CENTER, Color.BLACK, 4),
    ]
    for m in moves:
        assert parse_move_shortcut(move_to_shortcut(m)) == m


# --- state / move description ----------------------------------------------

def test_describe_state_frames_current_player_and_lists_factories():
    gs = GameState.new_game(42)
    text = describe_state(gs)
    assert f"You are Player {gs.current_player}." in text
    assert "FACTORIES:" in text
    assert "YOUR BOARD" in text
    assert "OPPONENT BOARD" in text


def test_describe_legal_moves_includes_every_code():
    gs = GameState.new_game(42)
    moves = gs.legal_moves()
    text = describe_legal_moves(moves)
    for m in moves:
        assert move_to_shortcut(m) in text


# --- reply parsing ----------------------------------------------------------

def _legal(gs):
    return {move_to_shortcut(m): m for m in gs.legal_moves()}


def test_extract_move_from_move_line():
    gs = GameState.new_game(42)
    legal = _legal(gs)
    code = next(iter(legal))
    text = f"I'll build toward the center.\nMOVE: {code}"
    assert _extract_move(text, legal) == legal[code]


def test_extract_move_tolerates_formatting():
    gs = GameState.new_game(42)
    legal = _legal(gs)
    code = next(iter(legal))
    text = f"Reasoning here.\n**MOVE:** `{code}`."
    assert _extract_move(text, legal) == legal[code]


def test_extract_move_fallback_scans_tokens():
    gs = GameState.new_game(42)
    legal = _legal(gs)
    code = next(iter(legal))
    # No MOVE: line, but the code appears as a bare token.
    text = f"The best play is {code} because it scores adjacency."
    assert _extract_move(text, legal) == legal[code]


def test_extract_move_returns_none_when_absent():
    gs = GameState.new_game(42)
    legal = _legal(gs)
    assert _extract_move("I cannot decide.", legal) is None


# --- agent behavior ---------------------------------------------------------

def test_returns_a_legal_move():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])

    def fake_complete(system, messages):
        return f"Taking the central tiles.\nMOVE: {legal_code}"

    move = LLMAgent(complete=fake_complete).choose_move(gs)
    assert move in gs.legal_moves()
    assert move == gs.legal_moves()[0]


def test_system_prompt_and_state_reach_the_model():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])
    seen = {}

    def fake_complete(system, messages):
        seen["system"] = system
        seen["user"] = messages[0]["content"]
        return f"MOVE: {legal_code}"

    LLMAgent(complete=fake_complete).choose_move(gs)
    assert "Azul" in seen["system"]
    assert "LEGAL MOVES" in seen["user"]
    assert "YOUR BOARD" in seen["user"]


def test_retries_after_illegal_move_then_succeeds():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])
    calls = {"n": 0}

    def fake_complete(system, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "MOVE: 9z9"  # syntactically parseable but not legal
        return f"MOVE: {legal_code}"

    agent = LLMAgent(complete=fake_complete)
    move = agent.choose_move(gs)
    assert calls["n"] == 2
    assert move == gs.legal_moves()[0]
    assert agent.used_fallback is False


def test_corrective_message_is_appended_on_retry():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])
    lengths = []

    def fake_complete(system, messages):
        lengths.append(len(messages))
        if len(messages) == 1:
            return "no idea"
        return f"MOVE: {legal_code}"

    LLMAgent(complete=fake_complete).choose_move(gs)
    # First call sees 1 message; after a bad reply, assistant + corrective user
    # are appended, so the retry sees 3.
    assert lengths == [1, 3]


class _StubAgent(Agent):
    def __init__(self, move):
        self.move = move

    def choose_move(self, state):
        return self.move


def test_falls_back_when_model_never_yields_a_legal_move():
    gs = GameState.new_game(42)
    fallback_move = gs.legal_moves()[0]

    def fake_complete(system, messages):
        return "I refuse to pick a code."

    agent = LLMAgent(
        complete=fake_complete,
        max_move_retries=1,
        fallback=_StubAgent(fallback_move),
    )
    move = agent.choose_move(gs)
    assert agent.used_fallback is True
    assert move == fallback_move


def test_falls_back_when_completion_raises():
    gs = GameState.new_game(42)
    fallback_move = gs.legal_moves()[0]

    def boom(system, messages):
        raise RuntimeError("network down")

    agent = LLMAgent(complete=boom, fallback=_StubAgent(fallback_move))
    move = agent.choose_move(gs)
    assert agent.used_fallback is True
    assert move == fallback_move


def test_default_fallback_is_greedy_and_plays_legally():
    gs = GameState.new_game(7)

    def fake_complete(system, messages):
        return "nope"

    agent = LLMAgent(complete=fake_complete, max_move_retries=0)
    move = agent.choose_move(gs)
    assert move in gs.legal_moves()
    assert agent.used_fallback is True
