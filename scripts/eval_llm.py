"""Evaluate the LLM agent against the existing baselines.

Plays the LLMAgent against Greedy and/or MCTS over several seeded games and
reports the win rate, mirroring the "beats Greedy N-0" checkpoints from earlier
phases. Seats are alternated so first-player advantage doesn't skew the result.

This makes REAL Claude API calls (one per LLM move, ~tens per game), so it
costs tokens and takes a while — especially against MCTS. Start with a small
--games count.

Run:
    python -m scripts.eval_llm --games 4 --opponent greedy
    python -m scripts.eval_llm --games 2 --opponent mcts --mcts-iterations 200
    python -m scripts.eval_llm --games 4 --opponent both --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from azul.agent import GreedyAgent
from azul.game import Game
from azul.llm_agent import DEFAULT_EFFORT, DEFAULT_MODEL, LLMAgent


def make_opponent(name: str, seed: int):
    if name == "greedy":
        return GreedyAgent()
    if name == "mcts":
        from azul.mcts import MCTSAgent

        return MCTSAgent(
            iterations=MCTS_ITERATIONS, rng=random.Random(seed),
            rollout="greedy", rollout_depth=8,
        )
    raise ValueError(name)


_print_lock = threading.Lock()


def _say(*args) -> None:
    """Print and flush immediately so a redirected stream shows live progress.
    Lock-guarded so concurrent games don't interleave a line mid-write."""
    with _print_lock:
        print(*args, flush=True)


def play_one_game(llm, opponent_name, opp, llm_seat, game_index, progress=True):
    """Play a single game move-by-move. With progress=True, emit the LLM's
    chosen move each turn and the running scores at each round boundary (only
    legible when one game runs at a time). Returns (GameResult, fallback_moves)."""
    from azul.render import move_to_shortcut

    agents = [None, None]
    agents[llm_seat] = llm
    agents[1 - llm_seat] = opp
    game = Game(agents=agents, seed=game_index)

    if progress:
        _say(f"\n  game {game_index} starting  "
             f"(LLM=P{llm_seat} vs {opponent_name}=P{1 - llm_seat})")
    fallback_moves = 0
    while not game.over:
        mover = game.state.current_player
        prev_round = game.state.round_number
        move = game.step()

        if mover == llm_seat and getattr(llm, "used_fallback", False):
            fallback_moves += 1
        if progress and mover == llm_seat:
            tag = " [FALLBACK]" if getattr(llm, "used_fallback", False) else ""
            _say(f"    r{prev_round} LLM -> {move_to_shortcut(move)}{tag}")

        if progress and not game.over and game.state.round_number != prev_round:
            s = [b.score for b in game.state.player_boards]
            _say(f"    -- round {prev_round} scored: P0={s[0]}  P1={s[1]} --")

    return game.result, fallback_moves


def _report_game(i, llm_seat, opponent, result, fallback_moves, tally) -> None:
    """Update the shared tally and print one game's result. Returns nothing;
    `tally` is a dict mutated under the print lock so concurrent games are safe."""
    with _print_lock:
        if result.winner is None:
            tally["ties"] += 1
            outcome = "tie"
        elif result.winner == llm_seat:
            tally["llm"] += 1
            outcome = "LLM win"
        else:
            tally["opp"] += 1
            outcome = f"{opponent} win"
        if fallback_moves:
            tally["fb_games"] += 1
        fb = f"  ({fallback_moves} fallback move(s))" if fallback_moves else ""
        print(
            f"  game {i} done (LLM=P{llm_seat}): scores={result.scores}  "
            f"-> {outcome}  (rounds={result.rounds}){fb}",
            flush=True,
        )
        print(
            f"  running tally: LLM {tally['llm']} - {tally['opp']} {opponent} "
            f"(ties {tally['ties']})  [{tally['done']+1}/{tally['n']} done]",
            flush=True,
        )
        tally["done"] += 1


def eval_vs(opponent: str, n_games: int, model: str, effort: str,
            concurrency: int, verbose: bool, start: int = 0) -> None:
    _say(f"\n=== LLM ({model}, effort={effort}) vs {opponent} — "
         f"games {start}..{start + n_games - 1}, concurrency={concurrency} ===")
    tally = {"llm": 0, "opp": 0, "ties": 0, "fb_games": 0, "done": 0, "n": n_games}
    game_indices = range(start, start + n_games)

    def run_game(i):
        llm = LLMAgent(model=model, effort=effort, top_k=TOP_K,
                       search_depth=SEARCH_DEPTH, bonus_aware=BONUS_AWARE,
                       verbose=verbose)
        opp = make_opponent(opponent, seed=1000 + i)
        llm_seat = i % 2  # alternate seats (by absolute game index)
        result, fb = play_one_game(
            llm, opponent, opp, llm_seat, i, progress=(concurrency == 1)
        )
        return i, llm_seat, result, fb

    if concurrency == 1:
        for i in game_indices:
            i, llm_seat, result, fb = run_game(i)
            _report_game(i, llm_seat, opponent, result, fb, tally)
    else:
        _say(f"  launching {n_games} games, up to {concurrency} at a time...")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(run_game, i) for i in game_indices]
            for fut in as_completed(futures):
                i, llm_seat, result, fb = fut.result()
                _report_game(i, llm_seat, opponent, result, fb, tally)

    _say(
        f"\n--- FINAL: LLM {tally['llm']} - {tally['opp']} {opponent}  "
        f"(ties {tally['ties']}); win rate {tally['llm'] / n_games:.0%} ---"
    )
    if tally["fb_games"]:
        _say(f"    note: {tally['fb_games']} game(s) had at least one fallback "
             "move (API error or unparseable reply — lower --concurrency if "
             "this is from rate limits)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="scripts.eval_llm")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--opponent", choices=["greedy", "mcts", "both"],
                        default="greedy")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--mcts-iterations", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=12,
                        help="show the LLM only the top-k ranked moves (hybrid "
                        "mode). 0 = show all moves.")
    parser.add_argument("--search-depth", type=int, default=3,
                        help="rank the LLM's candidates by alpha-beta minimax to "
                        "this depth (default 3). 0/1 = 1-ply ranking.")
    parser.add_argument("--no-bonus-aware", action="store_true",
                        help="disable the end-game-bonus-aware leaf eval in the "
                        "ranking search (on by default).")
    parser.add_argument("--start", type=int, default=0,
                        help="first game index (offsets seeds + seat alternation "
                        "so batches cover distinct games)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="games to run in parallel (independent games; "
                        "calls are network-bound so this ~N-x's throughput). "
                        "Per-round progress is shown only at concurrency=1.")
    parser.add_argument("--verbose", action="store_true",
                        help="print the model's reply each turn")
    args = parser.parse_args(argv)

    from azul.envfile import load_env
    load_env()  # pick up ANTHROPIC_API_KEY from a local .env

    global MCTS_ITERATIONS, TOP_K, SEARCH_DEPTH, BONUS_AWARE
    MCTS_ITERATIONS = args.mcts_iterations
    TOP_K = args.top_k if args.top_k > 0 else None  # 0 -> show all moves
    SEARCH_DEPTH = args.search_depth if args.search_depth >= 2 else None
    BONUS_AWARE = not args.no_bonus_aware

    opponents = ["greedy", "mcts"] if args.opponent == "both" else [args.opponent]
    for opp in opponents:
        eval_vs(opp, args.games, args.model, args.effort,
                args.concurrency, args.verbose, start=args.start)


if __name__ == "__main__":
    main()
