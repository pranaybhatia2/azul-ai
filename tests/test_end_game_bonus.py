"""Tests for GameState.end_game_bonus().

Bonuses (applied once, when the game ends):
- +2 for each complete horizontal row
- +7 for each complete vertical column
- +10 for each color placed all 5 times on the wall
Processes both players, mutates score in place, returns None.
"""
from azul.state import Color, GameState, WALL_PATTERN


def fill_row(board, row):
    for col in range(5):
        board.wall[row][col] = WALL_PATTERN[row][col]


def fill_col(board, col):
    for row in range(5):
        board.wall[row][col] = WALL_PATTERN[row][col]


def fill_color(board, color):
    # Place `color` in its slot on every row.
    for row in range(5):
        col = next(c for c in range(5) if WALL_PATTERN[row][c] == color)
        board.wall[row][col] = color


def test_no_bonus_on_empty_wall():
    gs = GameState()
    gs.player_boards[0].score = 5
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 5


def test_complete_row_scores_two():
    gs = GameState()
    fill_row(gs.player_boards[0], 0)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 2


def test_complete_column_scores_seven():
    gs = GameState()
    fill_col(gs.player_boards[0], 0)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 7


def test_complete_color_scores_ten():
    gs = GameState()
    fill_color(gs.player_boards[0], Color.BLUE)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 10


def test_two_complete_rows():
    gs = GameState()
    fill_row(gs.player_boards[0], 0)
    fill_row(gs.player_boards[0], 1)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 4


def test_full_wall_scores_all_bonuses():
    gs = GameState()
    board = gs.player_boards[0]
    for row in range(5):
        fill_row(board, row)
    # 5 rows (+10), 5 cols (+35), 5 colors (+50) = 95
    gs.end_game_bonus()
    assert board.score == 5 * 2 + 5 * 7 + 5 * 10


def test_bonus_added_to_existing_score():
    gs = GameState()
    gs.player_boards[0].score = 30
    fill_row(gs.player_boards[0], 2)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 32


def test_both_players_get_bonuses():
    gs = GameState()
    fill_row(gs.player_boards[0], 0)
    fill_col(gs.player_boards[1], 0)
    gs.end_game_bonus()
    assert gs.player_boards[0].score == 2
    assert gs.player_boards[1].score == 7


def test_returns_none():
    gs = GameState()
    assert gs.end_game_bonus() is None
