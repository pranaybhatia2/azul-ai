"""Interactive CLI: play Azul as a human against a bot.

Run:
    python -m azul.play                      # vs Greedy, random seed
    python -m azul.play --seed 42            # reproducible
    python -m azul.play --opponent random    # vs RandomAgent
    python -m azul.play 42 --opponent greedy # seed positional + opponent

You are Player 0. The board is shown on your turn; the opponent's move is
printed after it plays. Round scores and the final result are announced.
"""
from __future__ import annotations

import argparse
import random
import sys

from azul.agent import HumanAgent, RandomAgent, GreedyAgent
from azul.game import Game
from azul.render import render, render_move

HUMAN_SEAT = 0
OPP_SEAT = 1


def _make_opponent(name: str, seed: int):
    if name == "random":
        return RandomAgent(random.Random(seed))
    if name == "greedy":
        return GreedyAgent()
    if name == "mcts":
        from azul.mcts import MCTSAgent
        # Tuned for interactive speed (~2-4s/move) while still strong.
        return MCTSAgent(iterations=150, rng=random.Random(seed),
                         rollout="greedy", rollout_depth=6)
    raise ValueError(f"unknown opponent: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="azul.play")
    parser.add_argument("seed", nargs="?", type=int, default=None,
                        help="random seed (reproducible game)")
    parser.add_argument("--seed", dest="seed_flag", type=int, default=None,
                        help="random seed (alternative to positional)")
    parser.add_argument("--opponent", choices=["greedy", "random", "mcts"],
                        default="greedy")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    seed = args.seed if args.seed is not None else args.seed_flag
    if seed is None:
        seed = random.randrange(1_000_000)

    print(f"=== Azul — you are Player {HUMAN_SEAT} vs {args.opponent} "
          f"(seed={seed}) ===\n")

    agents = [None, None]
    agents[HUMAN_SEAT] = HumanAgent()
    agents[OPP_SEAT] = _make_opponent(args.opponent, seed)
    game = Game(agents=agents, seed=seed)

    while not game.over:
        mover = game.state.current_player
        prev_round = game.state.round_number

        move = game.step()

        if mover == OPP_SEAT:
            print(f"\n>>> Player {OPP_SEAT} ({args.opponent}): {render_move(move)}\n")

        if game.over:
            break
        if game.state.round_number != prev_round:
            scores = [b.score for b in game.state.player_boards]
            print(f"--- Round {prev_round} scored | "
                  f"P0={scores[0]}  P1={scores[1]} ---\n")

    r = game.result
    print("\n================ GAME OVER ================")
    print("Final board (after end-of-game bonuses):\n")
    print(render(game.state))
    print(f"Final: Player 0 = {r.scores[0]}   Player 1 = {r.scores[1]}")
    if r.winner is None:
        print("Result: tie")
    elif r.winner == HUMAN_SEAT:
        print("Result: you win! 🎉")
    else:
        print(f"Result: {args.opponent} agent wins.")
    print(f"(rounds played: {r.rounds})")


if __name__ == "__main__":
    main()
