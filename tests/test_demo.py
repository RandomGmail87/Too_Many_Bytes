"""Tests for the command-line demonstration script in AI Mayor City Sim."""

from ai_mayor.demo import main, run_demo


def test_run_demo_executes_successfully(capsys):
    """Verify that run_demo executes a full 12-month simulation and prints output."""
    status = run_demo(seed=42)
    assert status == 0

    captured = capsys.readouterr()
    assert "AI MAYOR CITY SIM — MILESTONE 1 DEMONSTRATION" in captured.out
    assert "DISCLAIMER:" in captured.out
    assert "synthetic simulation outputs" in captured.out
    assert "MONTH 01 / 12" in captured.out
    assert "MONTH 12 / 12" in captured.out
    assert "12-MONTH SIMULATION COMPLETED SUCCESSFULLY" in captured.out
    assert "Total Budget Appropriated (12 x $100k): $1,200,000.00" in captured.out


def test_main_cli_entry_point(capsys):
    """Verify that main() handles CLI arguments and exits cleanly."""
    status = main(["--seed", "100"])
    assert status == 0

    captured = capsys.readouterr()
    assert "[INIT] Simulation Seed: 100" in captured.out
    assert "12-MONTH SIMULATION COMPLETED SUCCESSFULLY" in captured.out

