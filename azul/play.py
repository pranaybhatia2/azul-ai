"""Interactive CLI: play Azul as a human against the RandomAgent.

Run:
    python -m azul.play            # random seed
    python -m azul.play 42         # fixed seed (reproducible)

You are Player 0. The board is shown on your turn; the random agent's
move is printed after it plays. Round scores and the final result are
announced at the boundaries.
"""
from __future__ import annotations

import random
import sys

from azul.agent import HumanAgent, RandomAgent
from azul.game import Game
from azul.render import render_move

HUMAN_SEAT = 0
RANDOM_SEAT = 1


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    seed = int(argv[0]) if argv else random.randrange(1_000_000)
    print(f"=== Azul — you are Player {HUMAN_SEAT} vs RandomAgent (seed={seed}) ===\n")

    agents = [None, None]
    agents[HUMAN_SEAT] = HumanAgent()
    agents[RANDOM_SEAT] = RandomAgent(random.Random(seed))
    game = Game(agents=agents, seed=seed)

    while not game.over:
        mover = game.state.current_player
        prev_round = game.state.round_number

        move = game.step()

        if mover == RANDOM_SEAT:
            print(f"\n>>> Player {RANDOM_SEAT} (random): {render_move(move)}\n")

        # A round boundary (or game end) just happened inside step().
        if game.over:
            break
        if game.state.round_number != prev_round:
            scores = [b.score for b in game.state.player_boards]
            print(f"--- Round {prev_round} scored | "
                  f"P0={scores[0]}  P1={scores[1]} ---\n")

    r = game.result
    print("\n================ GAME OVER ================")
    print(f"Final: Player 0 = {r.scores[0]}   Player 1 = {r.scores[1]}")
    if r.winner is None:
        print("Result: tie")
    elif r.winner == HUMAN_SEAT:
        print("Result: you win! 🎉")
    else:
        print("Result: random agent wins.")
    print(f"(rounds played: {r.rounds})")


if __name__ == "__main__":
    main()
