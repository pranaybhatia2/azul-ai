"""Action-space pruning for the NN search.

Drops *optional* floor-dumps — taking tiles straight to the floor when a
pattern line could accept that color. Those are almost always dominated (a
free penalty), and they inflate the branching factor (~7 of ~26 moves on
average, more early-game), starving each MCTS simulation. Forced floor-dumps
(a color with no open line) are kept.

Used ONLY by the neural search (az_mcts / self-play). The engine, Minimax, and
Greedy keep full legal_moves() so their established results are unchanged.
"""
from __future__ import annotations

from azul.state import GameState, Move, FLOOR


def candidate_moves(state: GameState) -> list[Move]:
    moves = state.legal_moves()
    line = [m for m in moves if m.dest_line != FLOOR]
    has_line = {(m.source, m.color) for m in line}
    forced_floor = [m for m in moves
                    if m.dest_line == FLOOR and (m.source, m.color) not in has_line]
    return line + forced_floor
