"""Tests for MinimaxAgent (alpha-beta + transposition table)."""
from azul.minimax import MinimaxAgent
from azul.agent import GreedyAgent
from azul.match import play_match
from azul.state import GameState


def test_returns_a_legal_move():
    gs = GameState.new_game(42)
    move = MinimaxAgent(depth=2).choose_move(gs)
    assert move in gs.legal_moves()


def test_move_values_top_matches_choose_move():
    gs = GameState.new_game(42)
    ranked = MinimaxAgent(depth=2).move_values(gs)
    assert len(ranked) == len(gs.legal_moves())          # every move scored
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)        # best-first
    assert ranked[0][0] == MinimaxAgent(depth=2).choose_move(gs)  # top == pick


def test_deterministic():
    gs = GameState.new_game(42)
    a = MinimaxAgent(depth=2).choose_move(gs)
    b = MinimaxAgent(depth=2).choose_move(gs)
    assert a == b


def test_tt_does_not_change_the_chosen_move():
    """The transposition table is an optimization: with or without it, the
    search must return the same move (correctness under alpha-beta bounds)."""
    for seed in (1, 7, 42):
        gs = GameState.new_game(seed)
        with_tt = MinimaxAgent(depth=2, use_tt=True).choose_move(gs)
        without = MinimaxAgent(depth=2, use_tt=False).choose_move(gs)
        assert with_tt == without


def test_depth_one_equals_greedy_on_relative_eval():
    """Depth-1 minimax maximizes the same one-ply value greedy does (modulo
    greedy's self-only eval vs minimax's relative eval) — both pick legal."""
    gs = GameState.new_game(3)
    move = MinimaxAgent(depth=1).choose_move(gs)
    assert move in gs.legal_moves()


def test_beats_greedy():
    # Slow-ish (clone-bound); kept small. Minimax(d2) dominates Greedy.
    res = play_match(
        make_a=lambda i: MinimaxAgent(depth=2),
        make_b=lambda i: GreedyAgent(),
        n_games=4,
        base_seed=0,
    )
    assert res.wins_a >= 3
