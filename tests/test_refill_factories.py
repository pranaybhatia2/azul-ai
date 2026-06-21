"""Tests for GameState.refill_factories(rng)."""
import random
from azul.state import Color, GameState, TILES_PER_COLOR


def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Basic refill
# ---------------------------------------------------------------------------

def test_each_factory_gets_four_tiles():
    gs = GameState()
    gs.refill_factories(make_rng())
    for factory in gs.factories:
        assert sum(factory.values()) == 4


def test_tiles_drawn_from_bag():
    gs = GameState()
    gs.refill_factories(make_rng())
    total_in_factories = sum(sum(f.values()) for f in gs.factories)
    total_in_bag = sum(gs.bag.values())
    assert total_in_bag == sum(TILES_PER_COLOR for _ in Color) - total_in_factories


def test_factory_tiles_are_valid_colors():
    gs = GameState()
    gs.refill_factories(make_rng())
    for factory in gs.factories:
        assert all(isinstance(c, Color) for c in factory)


def test_refill_is_deterministic_with_same_seed():
    gs1, gs2 = GameState(), GameState()
    gs1.refill_factories(random.Random(99))
    gs2.refill_factories(random.Random(99))
    assert gs1.factories == gs2.factories


def test_refill_differs_with_different_seeds():
    gs1, gs2 = GameState(), GameState()
    gs1.refill_factories(random.Random(1))
    gs2.refill_factories(random.Random(2))
    assert gs1.factories != gs2.factories


# ---------------------------------------------------------------------------
# Center and marker
# ---------------------------------------------------------------------------

def test_first_player_marker_placed_in_center():
    gs = GameState()
    gs.refill_factories(make_rng())
    assert gs.first_player_marker_in_center


def test_center_cleared_before_refill():
    gs = GameState()
    gs.center = {Color.BLUE: 3}
    gs.refill_factories(make_rng())
    assert Color.BLUE not in gs.center or gs.center[Color.BLUE] == 0


# ---------------------------------------------------------------------------
# Bag exhaustion — discard reshuffled in
# ---------------------------------------------------------------------------

def test_discard_reshuffled_when_bag_runs_dry():
    gs = GameState()
    # Nearly empty bag — only 5 tiles left (not enough for 5 factories × 4)
    gs.bag = {c: 0 for c in Color}
    gs.bag[Color.BLUE] = 5
    # Discard has plenty
    gs.discard = {c: 10 for c in Color}
    gs.refill_factories(make_rng())
    total = sum(sum(f.values()) for f in gs.factories)
    assert total == 20   # 5 factories × 4 tiles


def test_discard_reset_after_reshuffling():
    gs = GameState()
    gs.bag = {c: 0 for c in Color}
    gs.bag[Color.BLUE] = 5
    gs.discard = {c: 10 for c in Color}
    gs.refill_factories(make_rng())
    assert sum(gs.discard.values()) == 0


def test_partial_fill_when_bag_and_discard_exhausted():
    gs = GameState()
    # Only 6 tiles total across bag and discard
    gs.bag = {c: 0 for c in Color}
    gs.bag[Color.RED] = 6
    gs.discard = {c: 0 for c in Color}
    gs.refill_factories(make_rng())
    total = sum(sum(f.values()) for f in gs.factories)
    assert total == 6


def test_round_number_not_changed():
    gs = GameState()
    gs.round_number = 3
    gs.refill_factories(make_rng())
    assert gs.round_number == 3
