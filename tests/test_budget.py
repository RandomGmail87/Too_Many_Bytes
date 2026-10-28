"""Tests for budget constraints, accounting, and action validation in AI Mayor City Sim."""

import numpy as np
import pytest

from ai_mayor.env.city_env import CityEnvironment


@pytest.fixture
def env():
    """Create a default simulation environment."""
    e = CityEnvironment()
    e.reset(seed=42)
    return e


def test_spending_exceeding_budget_rejected(env):
    """Verify that allocating more than the monthly budget raises ValueError."""
    # Monthly budget is 100,000. Try spending 100,001.
    overspending_action = np.array([25000.0, 25000.0, 25000.0, 25000.0, 1.0])
    with pytest.raises(ValueError, match="exceeds renewed monthly budget"):
        env.step(overspending_action)


def test_negative_allocation_rejected(env):
    """Verify that negative allocations to any neighborhood are rejected."""
    negative_action = np.array([-5000.0, 20000.0, 20000.0, 10000.0, 10000.0])
    with pytest.raises(ValueError, match="Allocation cannot be negative"):
        env.step(negative_action)


def test_nan_or_infinite_allocation_rejected(env):
    """Verify that NaN or Infinite values in actions are strictly rejected."""
    nan_action = np.array([np.nan, 10000.0, 10000.0, 10000.0, 10000.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        env.step(nan_action)

    inf_action = np.array([np.inf, 10000.0, 10000.0, 10000.0, 10000.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        env.step(inf_action)


def test_invalid_action_dimensions_rejected(env):
    """Verify that action vectors of wrong length are rejected."""
    short_action = np.array([20000.0, 20000.0, 20000.0])
    with pytest.raises(ValueError, match="1D vector of length 5"):
        env.step(short_action)

    long_action = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    with pytest.raises(ValueError, match="1D vector of length 5"):
        env.step(long_action)


def test_exact_budget_allocation_accepted(env):
    """Verify that allocating exactly 100% of the renewed monthly budget succeeds."""
    exact_action = np.array([20000.0, 20000.0, 20000.0, 20000.0, 20000.0])
    state, reward, done, info = env.step(exact_action)

    assert info["step_spending"] == pytest.approx(100000.0)
    assert info["cumulative_spending"] == pytest.approx(100000.0)
    assert info["unspent_monthly_budget"] == pytest.approx(0.0)


def test_zero_spending_allowed(env):
    """Verify that allocating $0 is accepted and tracked cleanly."""
    zero_action = np.zeros(5)
    state, reward, done, info = env.step(zero_action)

    assert info["step_spending"] == 0.0
    assert info["cumulative_spending"] == 0.0
    assert info["unspent_monthly_budget"] == pytest.approx(100000.0)


def test_cumulative_and_renewed_spending_across_steps(env):
    """Verify that monthly budget renews each step while cumulative spending aggregates."""
    action = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0])  # $50,000 per step

    # Step 1
    _, _, _, info1 = env.step(action)
    assert info1["step_spending"] == pytest.approx(50000.0)
    assert info1["cumulative_spending"] == pytest.approx(50000.0)
    assert info1["unspent_monthly_budget"] == pytest.approx(50000.0)

    # Step 2: budget renewed to $100,000, step spending is $50,000, cumulative is $100,000
    _, _, _, info2 = env.step(action)
    assert info2["step_spending"] == pytest.approx(50000.0)
    assert info2["cumulative_spending"] == pytest.approx(100000.0)
    assert info2["unspent_monthly_budget"] == pytest.approx(50000.0)

