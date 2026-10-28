"""Tests for simulation dynamics, state transitions, and metrics in AI Mayor City Sim."""

import numpy as np
import pytest

from ai_mayor.data.synthetic import load_synthetic_city
from ai_mayor.env.city_env import CityEnvironment, compute_gini


def test_synthetic_city_loads_correctly():
    """Verify that synthetic fixture loads and validates the 5-neighborhood schema."""
    data = load_synthetic_city()
    assert data["dataset_metadata"]["type"] == "synthetic_test_fixture"
    assert len(data["neighborhoods"]) == 5

    for n in data["neighborhoods"]:
        assert "id" in n
        assert "name" in n
        assert n["service_backlog"] >= 0
        assert 0.0 <= n["traffic_safety_risk"] <= 1.0
        assert n["monthly_inflow_rate"] >= 0
        assert n["vulnerability_weight"] > 0.0


def test_reset_reproducibility():
    """Verify that reset(seed) produces deterministic initial state across separate runs."""
    env1 = CityEnvironment()
    state1, info1 = env1.reset(seed=123)

    env2 = CityEnvironment()
    state2, info2 = env2.reset(seed=123)

    np.testing.assert_array_equal(state1["neighborhood_backlogs"], state2["neighborhood_backlogs"])
    np.testing.assert_array_equal(state1["traffic_safety_risks"], state2["traffic_safety_risks"])
    np.testing.assert_array_equal(state1["vulnerability_weights"], state2["vulnerability_weights"])
    assert state1["step"] == 0
    assert info1["initial_total_backlog"] == info2["initial_total_backlog"]


def test_step_backlog_reduction_and_nonnegativity():
    """Verify that service funding resolves backlogs and that backlogs never become negative."""
    env = CityEnvironment()
    env.reset(seed=42)

    initial_backlog_0 = env.backlogs[0]
    # Allocate $40,000 to neighborhood 0. Service portion is 80% = $32,000.
    # At $100/ticket, this resolves up to 320 tickets.
    action = np.array([40000.0, 0.0, 0.0, 0.0, 0.0])
    state, reward, done, info = env.step(action)

    resolved_n0 = info["resolved_backlogs_step"][0]
    expected_resolved = min(initial_backlog_0, np.floor(32000.0 / 100.0))
    assert resolved_n0 == expected_resolved

    # Backlog must be non-negative
    assert np.all(state["neighborhood_backlogs"] >= 0.0)


def test_safety_improvement_depends_strictly_on_safety_funding():
    """Verify that safety improvement depends exclusively on the designated safety portion."""
    env = CityEnvironment()
    env.reset(seed=42)

    initial_risk_0 = env.safety_risks[0]
    # Allocation of $50,000 to neighborhood 0
    # Safety fraction = 20% -> $10,000 designated for safety
    action = np.array([50000.0, 0.0, 0.0, 0.0, 0.0])
    _, _, _, info = env.step(action)

    safety_funding = 50000.0 * 0.20
    expected_reduction = min(initial_risk_0, safety_funding / env.safety_ref_budget)
    actual_reduction = info["safety_reductions_step"][0]

    assert actual_reduction == pytest.approx(expected_reduction)
    # Neighborhoods with $0 allocation must have 0 safety reduction
    assert info["safety_reductions_step"][1] == 0.0


def test_twelve_step_horizon():
    """Verify that episode runs for exactly 12 steps and terminates."""
    env = CityEnvironment()
    env.reset(seed=42)

    action = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    for step_num in range(1, 13):
        state, reward, done, info = env.step(action)
        assert state["step"] == step_num
        if step_num < 12:
            assert done is False
        else:
            assert done is True

    # 13th step must raise RuntimeError
    with pytest.raises(RuntimeError, match="Episode has already finished"):
        env.step(action)


def test_compute_gini_numerical_stability():
    """Verify Gini calculation handles zero vectors, uniform vectors, and extreme inequality."""
    # Zero vector: all backlogs resolved
    assert compute_gini(np.zeros(5)) == 0.0

    # Perfectly equal distribution
    assert compute_gini(np.array([100.0, 100.0, 100.0, 100.0, 100.0])) == 0.0

    # Single-element vector
    assert compute_gini(np.array([42.0])) == 0.0

    # Extreme inequality: only one neighborhood has backlog
    extreme_gini = compute_gini(np.array([500.0, 0.0, 0.0, 0.0, 0.0]))
    # For N=5 with 1 non-zero: theoretical Gini is (5-1)/5 = 0.8
    assert extreme_gini == pytest.approx(0.8, abs=1e-4)


def test_reward_numerical_stability_with_zero_initial_backlog():
    """Verify reward function produces finite, stable output even if initial backlogs are zero."""
    env = CityEnvironment()
    env.reset(seed=42)
    env.initial_total_backlog = 0.0
    env.backlogs = np.zeros(5)

    action = np.zeros(5)
    state, reward, done, info = env.step(action)

    assert not np.isnan(reward)
    assert not np.isinf(reward)
    assert isinstance(reward, float)

