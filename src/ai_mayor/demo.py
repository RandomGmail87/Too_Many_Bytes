"""Command-line demonstration script for AI Mayor City Sim (Milestone 1).

This script executes a complete 12-month simulation episode using the synthetic
five-neighborhood city fixture and the rule-based baseline agent.

DISCLAIMER:
All inputs, dynamics, and outputs in this demonstration are synthetic simulation
values. They do not represent empirical observations from Chicago Open Data and
do not establish real-world causal effects.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from ai_mayor.agents.rule_based import RuleBasedAgent
from ai_mayor.env.city_env import CityEnvironment


def format_currency(amount: float) -> str:
    """Format float as USD currency string."""
    return f"${amount:,.2f}"


def run_demo(seed: int = 42) -> int:
    """Run a complete 12-month simulation episode and display formatted progress.

    Args:
        seed: Random seed for deterministic simulation.

    Returns:
        int: Exit status code (0 for success).
    """
    env = CityEnvironment()
    state, info = env.reset(seed=seed)
    agent = RuleBasedAgent()

    print("=" * 78)
    print("AI MAYOR CITY SIM — MILESTONE 1 DEMONSTRATION")
    print("Course: CS 3793 | Team: Too Many Bytes")
    print("=" * 78)
    print("DISCLAIMER:")
    print("  All numbers displayed below are synthetic simulation outputs generated")
    print("  from placeholder test fixtures. They do NOT represent actual Chicago")
    print("  Open Data measurements and do not establish causal real-world claims.")
    print("=" * 78)

    names = env.neighborhood_names
    initial_backlogs = state["neighborhood_backlogs"]
    initial_risks = state["traffic_safety_risks"]
    vulnerability = state["vulnerability_weights"]

    print(f"\n[INIT] Simulation Seed: {seed}")
    print(f"[INIT] Renewed Monthly Budget: {format_currency(env.monthly_budget)}")
    print(f"[INIT] Episode Horizon: {env.episode_steps} months")
    print(f"[INIT] Initial Total Backlog: {int(info['initial_total_backlog'])} service tickets\n")

    print("-" * 78)
    print(f"{'#':<3} {'Neighborhood':<35} {'Backlog':<10} {'Safety Risk':<12} {'Vulnerability':<14}")
    print("-" * 78)
    for i, name in enumerate(names):
        print(
            f"{i:<3} {name:<35} {int(initial_backlogs[i]):<10} "
            f"{initial_risks[i]:<12.2f} {vulnerability[i]:<14.2f}"
        )
    print("-" * 78)

    print("\nStarting 12-Month Fiscal Year Simulation...\n")

    cumulative_reward = 0.0
    cumulative_resolved_tickets = 0.0

    for month in range(1, env.episode_steps + 1):
        action = agent.select_action(state)
        next_state, reward, done, step_info = env.step(action)
        cumulative_reward += reward
        cumulative_resolved_tickets += step_info["total_resolved_backlogs_step"]

        step_spend = step_info["step_spending"]
        cum_spend = step_info["cumulative_spending"]
        unspent = step_info["unspent_monthly_budget"]
        gini = step_info["gini_index"]
        resolved_step = step_info["total_resolved_backlogs_step"]

        print(f"--- MONTH {month:02d} / {env.episode_steps:02d} ---")
        print(
            f"Monthly Spending: {format_currency(step_spend)} | "
            f"Unspent: {format_currency(unspent)} | "
            f"Cumulative Spending: {format_currency(cum_spend)}"
        )
        print(
            f"Tickets Resolved This Month: {int(resolved_step)} | "
            f"Gini Inequality: {gini:.4f} | "
            f"Step Reward: {reward:+.4f}"
        )

        print("Neighborhood Allocations & Post-Transition Status:")
        print(f"  {'Index':<6} {'Allocated ($)':<16} {'Backlog':<12} {'Safety Risk':<12}")
        for idx in range(env.neighborhood_count):
            alloc_val = action[idx]
            rem_b = next_state["neighborhood_backlogs"][idx]
            rem_r = next_state["traffic_safety_risks"][idx]
            print(
                f"  [{idx}]    {format_currency(alloc_val):<16} "
                f"{int(rem_b):<12} {rem_r:<12.3f}"
            )
        print()
        state = next_state

    # Final summary
    print("=" * 78)
    print("12-MONTH SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print("SUMMARY PERFORMANCE METRICS:")
    total_budget_appropriated = env.monthly_budget * env.episode_steps
    print(f"  Total Budget Appropriated (12 x $100k): {format_currency(total_budget_appropriated)}")
    print(f"  Total Cumulative Expenditure:           {format_currency(env.cumulative_spending)}")
    print(f"  Total Budget Utilization:               {(env.cumulative_spending / total_budget_appropriated) * 100:.2f}%")
    print(f"  Total Service Tickets Resolved:         {int(cumulative_resolved_tickets)} tickets")
    final_backlog = float(np.sum(state["neighborhood_backlogs"]))
    print(f"  Final Total City Backlog:               {int(final_backlog)} tickets")
    mean_final_risk = float(np.mean(state["traffic_safety_risks"]))
    print(f"  Final Mean Traffic Safety Risk:         {mean_final_risk:.3f}")
    final_gini = step_info["gini_index"]
    print(f"  Final Backlog Gini Inequality:          {final_gini:.4f}")
    print(f"  Cumulative Episode Reward:              {cumulative_reward:+.4f}")
    print("=" * 78)
    print("[NOTE] All outputs are synthetic simulation results for CS 3793 research evaluation.\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and run demonstration."""
    parser = argparse.ArgumentParser(
        description="Run 12-month synthetic simulation demonstration for AI Mayor City Sim."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)
    return run_demo(seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())

