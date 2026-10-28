"""Synthetic fixture loader and validator for AI Mayor City Sim.

This module provides deterministic synthetic neighborhood fixtures used to test
the simulation engine and baseline algorithms without relying on external network
access or large raw datasets.

IMPORTANT: All data produced by or loaded from this module consists solely of
synthetic testing values. They are not real Chicago Open Data observations and
must not be presented as empirical measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def get_default_fixture_path() -> Path:
    """Return the default path to the synthetic city sample fixture."""
    # Resolve relative to package directory or workspace root
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    candidate = base_dir / "data" / "samples" / "synthetic_city.json"
    if candidate.exists():
        return candidate
    # Fallback if installed in different directory context
    return Path("data/samples/synthetic_city.json")


def validate_synthetic_data(data: Dict[str, Any]) -> None:
    """Validate that synthetic city data matches the required schema and constraints.

    Raises:
        ValueError: If data structure is invalid or values violate bounds.
    """
    if not isinstance(data, dict):
        raise ValueError("Synthetic city data must be a JSON object (dictionary).")

    metadata = data.get("dataset_metadata")
    if not isinstance(metadata, dict) or metadata.get("type") != "synthetic_test_fixture":
        raise ValueError(
            "Missing or invalid 'dataset_metadata.type'. Expected 'synthetic_test_fixture'."
        )

    neighborhoods = data.get("neighborhoods")
    if not isinstance(neighborhoods, list) or len(neighborhoods) != 5:
        raise ValueError("Synthetic city must contain exactly 5 neighborhoods.")

    seen_ids = set()
    for idx, n in enumerate(neighborhoods):
        if not isinstance(n, dict):
            raise ValueError(f"Neighborhood at index {idx} must be a dictionary.")

        n_id = n.get("id")
        if n_id is None or n_id in seen_ids:
            raise ValueError(f"Neighborhood at index {idx} has invalid or duplicate id: {n_id}")
        seen_ids.add(n_id)

        backlog = n.get("service_backlog")
        if backlog is None or backlog < 0:
            raise ValueError(f"Neighborhood {n_id} must have non-negative service_backlog.")

        safety_risk = n.get("traffic_safety_risk")
        if safety_risk is None or not (0.0 <= safety_risk <= 1.0):
            raise ValueError(
                f"Neighborhood {n_id} traffic_safety_risk must be in range [0.0, 1.0]."
            )

        inflow = n.get("monthly_inflow_rate")
        if inflow is None or inflow < 0:
            raise ValueError(f"Neighborhood {n_id} monthly_inflow_rate must be non-negative.")

        weight = n.get("vulnerability_weight")
        if weight is None or weight <= 0.0:
            raise ValueError(f"Neighborhood {n_id} vulnerability_weight must be strictly positive.")


def load_synthetic_city(path: Path | str | None = None) -> Dict[str, Any]:
    """Load and validate the synthetic 5-neighborhood scenario.

    Args:
        path: Optional path to the JSON fixture. If None, uses default fixture location.

    Returns:
        Dict[str, Any]: Validated synthetic city specification.
    """
    if path is None:
        target_path = get_default_fixture_path()
    else:
        target_path = Path(path)

    if not target_path.exists():
        raise FileNotFoundError(f"Synthetic fixture file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_synthetic_data(data)
    return data

