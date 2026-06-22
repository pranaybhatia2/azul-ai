"""Tests for the LLM agent — all offline (the network call is injected)."""
from azul.agent import Agent
from azul.llm_agent import (
    LLMAgent,
    describe_candidate_moves,
    describe_legal_moves,
    describe_state,
    rank_moves,
    _denial_note,
    _imminent_completions,
    _extract_move,
)
from azul.render import move_to_shortcut
from azul.state import Color, GameState, Move, WALL_PATTERN, CENTER, FLOOR


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
    text = describe_legal_moves(gs, moves)
    for m in moves:
        assert move_to_shortcut(m) in text


def test_legal_move_annotations_flag_completion_and_bonus():
    # 4 blue available, col 0 already at 4/5 -> taking blue to row 0 completes
    # the line and finishes column 0 for +7. The annotation must say so.
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 1}
    b = gs.player_boards[0]
    # Fill col 0 in rows 1-4 so row 0 (Blue at col 0) would complete the column.
    for r in range(1, 5):
        b.wall[r][0] = WALL_PATTERN[r][0]
    text = describe_legal_moves(gs, gs.legal_moves())
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("0b0"))
    assert "COMPLETES" in line
    assert "+7" in line  # finishing column 0


def test_rank_moves_matches_greedy_top_choice():
    # The top-ranked candidate must be the move GreedyAgent would pick.
    from azul.agent import GreedyAgent
    gs = GameState.new_game(42)
    ranked = rank_moves(gs)
    assert ranked[0][0] == GreedyAgent().choose_move(gs)
    # Scores are sorted descending.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_moves_truncates_to_k():
    gs = GameState.new_game(42)
    assert len(rank_moves(gs, 5)) == 5


def test_describe_candidate_moves_shows_eval_and_codes():
    gs = GameState.new_game(42)
    scored = rank_moves(gs, 8)
    text = describe_candidate_moves(gs, scored)
    assert "score" in text
    for move, _ in scored:
        assert move_to_shortcut(move) in text


def test_legal_move_annotations_flag_overflow():
    # 4 blue, row 0 holds 1 -> 3 overflow to floor; annotation must warn.
    gs = GameState()
    gs.factories[0] = {Color.BLUE: 4}
    text = describe_legal_moves(gs, gs.legal_moves())
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("0b0"))
    assert "overflow" in line.lower()


# --- opponent-aware ranking / denial ----------------------------------------

def _blocking_position():
    """P0 to move. Opponent (P1) is one RED from completing row 4 for ~5 pts;
    the only RED is in factory 0. P0's RED wall cells are filled so RED can only
    go to the floor — making the block a pure denial (bad for P0's own board)."""
    gs = GameState()
    gs.current_player = 0
    gs.factories[0] = {Color.RED: 1, Color.BLUE: 2}
    opp = gs.player_boards[1]
    opp.pattern_lines[4].color, opp.pattern_lines[4].count = Color.RED, 4
    for r in range(4):
        opp.wall[r][1] = WALL_PATTERN[r][1]   # vertical run -> completion worth ~5
    me = gs.player_boards[0]
    for r in range(5):
        me.wall[r][(r + 2) % 5] = Color.RED   # RED only placeable on the floor
    return gs


def _took_red(move):
    return move.source == 0 and move.color == Color.RED


def test_opponent_aware_ranking_surfaces_the_block():
    gs = _blocking_position()
    plain_top = rank_moves(gs, opponent_aware=False)[0][0]
    oa_top = rank_moves(gs, opponent_aware=True)[0][0]
    assert not _took_red(plain_top)   # self-only ranking ignores the block
    assert _took_red(oa_top)          # opponent-aware ranking blocks


def test_denial_note_flags_blocking_move():
    gs = _blocking_position()
    block = next(m for m in gs.legal_moves() if _took_red(m))
    own = next(m for m in gs.legal_moves()
               if m.color == Color.BLUE and m.dest_line == 1)
    assert "DENIES" in _denial_note(gs, block)
    assert _denial_note(gs, own) == ""   # taking blue denies nothing


def test_imminent_completions_lists_opponent_threat():
    gs = _blocking_position()
    threats = _imminent_completions(gs, 1)
    assert len(threats) == 1 and "row 4" in threats[0] and "Red" in threats[0]


def test_describe_state_surfaces_opponent_threats():
    gs = _blocking_position()
    assert "OPPONENT THREATS" in describe_state(gs)


def test_opponent_aware_is_the_default():
    assert LLMAgent(complete=lambda s, m: "x").opponent_aware is True


def test_search_depth_ranking_uses_minimax():
    from azul.minimax import MinimaxAgent
    gs = GameState.new_game(42)
    ranked = rank_moves(gs, search_depth=2)
    assert ranked[0][0] == MinimaxAgent(depth=2).choose_move(gs)


def test_search_depth_is_default_three():
    assert LLMAgent(complete=lambda s, m: "x").search_depth == 3


def test_selective_deepening_depth4_returns_legal_topk():
    gs = GameState.new_game(42)
    ranked = rank_moves(gs, 12, search_depth=4)  # selective: depth-2 prune -> depth-4
    assert len(ranked) == 12
    assert all(m in gs.legal_moves() for m, _ in ranked)


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

    move = LLMAgent(complete=fake_complete, search_depth=1).choose_move(gs)
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

    LLMAgent(complete=fake_complete, search_depth=1).choose_move(gs)
    assert "Azul" in seen["system"]
    assert "CANDIDATE MOVES" in seen["user"]  # default hybrid mode (top_k)
    assert "YOUR BOARD" in seen["user"]


def test_all_moves_mode_when_top_k_none():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])
    seen = {}

    def fake_complete(system, messages):
        seen["user"] = messages[0]["content"]
        return f"MOVE: {legal_code}"

    LLMAgent(complete=fake_complete, top_k=None).choose_move(gs)
    assert "LEGAL MOVES" in seen["user"]


def test_retries_after_illegal_move_then_succeeds():
    gs = GameState.new_game(42)
    legal_code = move_to_shortcut(gs.legal_moves()[0])
    calls = {"n": 0}

    def fake_complete(system, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "MOVE: 9z9"  # syntactically parseable but not legal
        return f"MOVE: {legal_code}"

    agent = LLMAgent(complete=fake_complete, search_depth=1)
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

    LLMAgent(complete=fake_complete, search_depth=1).choose_move(gs)
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
        search_depth=1,
    )
    move = agent.choose_move(gs)
    assert agent.used_fallback is True
    assert move == fallback_move


def test_falls_back_when_completion_raises():
    gs = GameState.new_game(42)
    fallback_move = gs.legal_moves()[0]

    def boom(system, messages):
        raise RuntimeError("network down")

    agent = LLMAgent(complete=boom, fallback=_StubAgent(fallback_move), search_depth=1)
    move = agent.choose_move(gs)
    assert agent.used_fallback is True
    assert move == fallback_move


def test_default_fallback_is_greedy_and_plays_legally():
    gs = GameState.new_game(7)

    def fake_complete(system, messages):
        return "nope"

    agent = LLMAgent(complete=fake_complete, max_move_retries=0, search_depth=1)
    move = agent.choose_move(gs)
    assert move in gs.legal_moves()
    assert agent.used_fallback is True
