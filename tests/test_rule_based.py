"""Tests for the rule-based baseline agent in AI Mayor City Sim."""

import numpy as np
import pytest

from ai_mayor.agents.rule_based import RuleBasedAgent
from ai_mayor.env.city_env import CityEnvironment


def test_rule_based_action_validity():
    """Verify that rule-based agent outputs valid actions satisfying budget and caps."""
    env = CityEnvironment()
    state, _ = env.reset(seed=42)

    agent = RuleBasedAgent(max_neighborhood_fraction=0.40)
    action = agent.select_action(state)

    assert isinstance(action, np.ndarray)
    assert action.shape == (5,)
    assert np.all(action >= 0.0)
    assert np.sum(action) <= state["monthly_budget"] + 1e-6

    # 40% per-neighborhood allocation cap verification
    cap = state["monthly_budget"] * 0.40
    for alloc in action:
        assert alloc <= cap + 1e-6


def test_rule_based_full_12_month_episode():
    """Verify that rule-based agent completes an entire 12-month simulation episode cleanly."""
    env = CityEnvironment()
    state, _ = env.reset(seed=42)
    agent = RuleBasedAgent()

    total_reward = 0.0
    total_tickets_resolved = 0.0

    done = False
    step_count = 0
    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_tickets_resolved += info["total_resolved_backlogs_step"]
        step_count += 1

    assert step_count == 12
    assert done is True
    assert total_tickets_resolved > 0.0
    assert not np.isnan(total_reward)


def test_rule_based_outperforms_zero_allocation():
    """Verify that active rule-based allocation resolves more backlog than passive zero spending."""
    # Run active agent
    env_active = CityEnvironment()
    state, _ = env_active.reset(seed=42)
    agent = RuleBasedAgent()
    active_resolved = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        state, _, done, info = env_active.step(action)
        active_resolved += info["total_resolved_backlogs_step"]

    # Run passive policy ($0 allocation every month)
    env_passive = CityEnvironment()
    state, _ = env_passive.reset(seed=42)
    passive_resolved = 0.0
    done = False
    while not done:
        action = np.zeros(5)
        state, _, done, info = env_passive.step(action)
        passive_resolved += info["total_resolved_backlogs_step"]

    assert passive_resolved == 0.0
    assert active_resolved > 500.0


def test_rule_based_determinism():
    """Verify that identical seeds yield bitwise identical simulation trajectories."""
    def run_simulation(seed: int):
        env = CityEnvironment()
        state, _ = env.reset(seed=seed)
        agent = RuleBasedAgent()
        actions = []
        rewards = []
        done = False
        while not done:
            action = agent.select_action(state)
            actions.append(action.copy())
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        return actions, rewards

    actions1, rewards1 = run_simulation(seed=101)
    actions2, rewards2 = run_simulation(seed=101)

    for a1, a2 in zip(actions1, actions2):
        np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(rewards1, rewards2)

