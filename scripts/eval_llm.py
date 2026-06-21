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


def eval_vs(opponent: str, n_games: int, model: str, effort: str,
            verbose: bool) -> None:
    print(f"\n=== LLM ({model}, effort={effort}) vs {opponent} — "
          f"{n_games} games ===")
    llm_wins = opp_wins = ties = 0
    fallbacks = 0

    for i in range(n_games):
        llm = LLMAgent(model=model, effort=effort, verbose=verbose)
        opp = make_opponent(opponent, seed=1000 + i)
        # Alternate seats: LLM is Player 0 on even games, Player 1 on odd.
        llm_seat = i % 2
        agents = [None, None]
        agents[llm_seat] = llm
        agents[1 - llm_seat] = opp
        game = Game(agents=agents, seed=i)
        result = game.play()

        if result.winner is None:
            ties += 1
            outcome = "tie"
        elif result.winner == llm_seat:
            llm_wins += 1
            outcome = "LLM win"
        else:
            opp_wins += 1
            outcome = f"{opponent} win"

        # Count any turns where the model failed to produce a legal move.
        if getattr(llm, "used_fallback", False):
            fallbacks += 1

        print(
            f"  game {i}: LLM=P{llm_seat}  scores={result.scores}  "
            f"-> {outcome}  (rounds={result.rounds})"
        )

    print(
        f"--- LLM {llm_wins} - {opp_wins} {opponent}  "
        f"(ties {ties}); win rate {llm_wins / n_games:.0%} ---"
    )
    if fallbacks:
        print(f"    note: {fallbacks} game(s) had at least one fallback move")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="scripts.eval_llm")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--opponent", choices=["greedy", "mcts", "both"],
                        default="greedy")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--mcts-iterations", type=int, default=200)
    parser.add_argument("--verbose", action="store_true",
                        help="print the model's reply each turn")
    args = parser.parse_args(argv)

    from azul.envfile import load_env
    load_env()  # pick up ANTHROPIC_API_KEY from a local .env

    global MCTS_ITERATIONS
    MCTS_ITERATIONS = args.mcts_iterations

    opponents = ["greedy", "mcts"] if args.opponent == "both" else [args.opponent]
    for opp in opponents:
        eval_vs(opp, args.games, args.model, args.effort, args.verbose)


if __name__ == "__main__":
    main()
