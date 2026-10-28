"""Rule-based baseline agent for AI Mayor City Sim.

This agent implements a transparent, priority-queue-based heuristic policy.
It prioritizes neighborhoods with higher service backlogs, compounded by
traffic-safety risk and socioeconomic vulnerability, subject to the renewed
monthly budget constraint.

Design Choice Note on Allocation Cap:
The agent enforces a maximum allocation cap of 40% of the monthly budget per
individual neighborhood (defaulting to $40,000 out of $100,000). This constraint
is an intentional policy design heuristic to prevent any single high-backlog
district from consuming the entire monthly appropriation and starving all other
neighborhoods. It represents a practical municipal budget safeguard rather than
a formal mathematical proof of social fairness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class RuleBasedAgent:
    """Heuristic priority-based resource allocation policy.

    Attributes:
        max_neighborhood_fraction: Maximum fraction of monthly budget allowed
            for any single neighborhood in a step (default 0.40).
        service_fraction: Fraction of neighborhood funds used for service backlogs (0.80).
        backlog_unit_cost: Cost to resolve one service backlog item ($100).
    """

    def __init__(
        self,
        max_neighborhood_fraction: float = 0.40,
        service_fraction: float = 0.80,
        backlog_unit_cost: float = 100.0,
    ) -> None:
        """Initialize rule-based baseline agent.

        Args:
            max_neighborhood_fraction: Cap per neighborhood (default 0.40).
            service_fraction: Expected fraction allocated to service backlog.
            backlog_unit_cost: Cost in dollars per backlog ticket resolved.
        """
        if not (0.0 < max_neighborhood_fraction <= 1.0):
            raise ValueError("max_neighborhood_fraction must be in (0.0, 1.0].")

        self.max_neighborhood_fraction = float(max_neighborhood_fraction)
        self.service_fraction = float(service_fraction)
        self.backlog_unit_cost = float(backlog_unit_cost)

    def select_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute budget allocation vector for all neighborhoods based on current state.

        Urgency Priority Metric:
            Urgency_i = Backlog_i * (1.0 + Risk_i) * Vulnerability_i

        The policy allocates funding greedily in descending order of urgency:
        1. Calculate target funding needed to clear neighborhood backlog.
        2. Cap allocation at max_neighborhood_fraction * monthly_budget.
        3. Allocate available funds without exceeding total monthly budget.
        4. Distribute any residual budget to remaining neighborhoods with unmet backlog.

        Args:
            state: Environment state dictionary containing 'monthly_budget',
                'neighborhood_backlogs', 'traffic_safety_risks', and 'vulnerability_weights'.

        Returns:
            np.ndarray: Non-negative 1D float array of allocations summing to <= monthly_budget.
        """
        monthly_budget = float(state.get("monthly_budget", 100000.0))
        backlogs = np.asarray(state["neighborhood_backlogs"], dtype=np.float64)
        risks = np.asarray(state["traffic_safety_risks"], dtype=np.float64)
        weights = np.asarray(state.get("vulnerability_weights", np.ones_like(backlogs)), dtype=np.float64)

        n = len(backlogs)
        action = np.zeros(n, dtype=np.float64)

        if monthly_budget <= 0.0 or n == 0:
            return action

        per_neighborhood_cap = monthly_budget * self.max_neighborhood_fraction

        # Urgency score: backlog scaled by safety risk and vulnerability weight
        urgency = backlogs * (1.0 + risks) * weights

        # Priority ranking: descending order of urgency
        priority_order = np.argsort(-urgency)

        remaining_budget = monthly_budget

        # First pass: Allocate funds to resolve backlogs in order of priority up to cap
        for idx in priority_order:
            if remaining_budget <= 1e-6:
                break

            b = backlogs[idx]
            if b <= 0.0:
                continue

            # Funding needed to clear this backlog: b * unit_cost / service_fraction
            needed_funding = (b * self.backlog_unit_cost) / self.service_fraction

            # Allocatable amount bounded by remaining budget and per-neighborhood cap
            alloc = min(remaining_budget, per_neighborhood_cap, needed_funding)
            action[idx] = alloc
            remaining_budget -= alloc

        # Second pass: If budget remains and some neighborhoods still have unaddressed backlog
        # (because they were capped in pass 1), allocate up to cap
        if remaining_budget > 1e-6:
            for idx in priority_order:
                if remaining_budget <= 1e-6:
                    break

                b = backlogs[idx]
                needed_funding = (b * self.backlog_unit_cost) / self.service_fraction
                current_alloc = action[idx]

                if current_alloc < per_neighborhood_cap and current_alloc < needed_funding:
                    extra_needed = min(
                        remaining_budget,
                        per_neighborhood_cap - current_alloc,
                        needed_funding - current_alloc,
                    )
                    action[idx] += extra_needed
                    remaining_budget -= extra_needed

        # Final safety check: ensure strictly non-negative and total <= monthly_budget
        action = np.maximum(0.0, action)
        total_spent = float(np.sum(action))
        if total_spent > monthly_budget:
            # Scale down if minor numerical precision issue occurs
            action = action * (monthly_budget / total_spent)

        return action

