"""Smoke test for the interactive CLI driver."""
import builtins

from azul import play


def test_cli_runs_to_completion(monkeypatch, capsys):
    # Always choose move 0; the driver should run a full game and finish.
    monkeypatch.setattr(builtins, "input", lambda *_: "0")
    play.main(["42"])
    out = capsys.readouterr().out
    assert "GAME OVER" in out
    assert "Final: Player 0" in out
