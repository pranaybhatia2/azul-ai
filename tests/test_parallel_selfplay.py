"""Tests for parallel self-play."""
import pytest

pytest.importorskip("torch")

from azul.parallel_selfplay import _worker, generate_examples_parallel, _hidden_of
from azul.net import AzulNet
from azul.encoding import STATE_SIZE, POLICY_SIZE


def _valid(examples):
    assert len(examples) > 0
    for enc, pol, val in examples:
        assert len(enc) == STATE_SIZE
        assert len(pol) == POLICY_SIZE
        assert abs(sum(pol) - 1.0) < 1e-6
        assert val in (-1.0, 0.0, 1.0)


def test_hidden_inference():
    assert _hidden_of(AzulNet(hidden=128)) == 128


def test_worker_direct():
    # Call the worker directly (no Pool) — fast, deterministic logic check.
    net = AzulNet(hidden=64)
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    task = (sd, 64, 1, 10, 0, 1.5, 0.25, 0.5, 10)
    _valid(_worker(task))


def test_single_chunk_avoids_pool():
    net = AzulNet(hidden=64)
    ex = generate_examples_parallel(net, total_games=1, n_workers=1,
                                    sp_iterations=10)
    _valid(ex)


@pytest.mark.slow
def test_parallel_pool_runs():
    # Spawns processes; slower. Verifies the multiprocessing path works.
    net = AzulNet(hidden=64)
    ex = generate_examples_parallel(net, total_games=2, n_workers=2,
                                    sp_iterations=8)
    _valid(ex)
