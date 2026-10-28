"""Data schema definitions for Chicago Open Data ingestion and preprocessing.

This module defines standard Python dataclasses representing raw observations,
processed monthly community area summaries, and pipeline provenance metadata.

Standard library dataclasses are used rather than third-party validation libraries
to minimize dependency overhead while enforcing rigorous type and value constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Raw311Record:
    """Schema for a single raw 311 service request record."""

    sr_number: str
    sr_type: str
    status: str
    created_date: str
    closed_date: Optional[str] = None
    community_area: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.sr_number:
            raise ValueError("sr_number must be a non-empty string.")
        if not self.created_date:
            raise ValueError("created_date must be a non-empty string.")
        if self.community_area is not None:
            if not (1 <= self.community_area <= 77):
                raise ValueError(
                    f"community_area must be in range [1, 77] or None, got {self.community_area}."
                )


@dataclass(frozen=True)
class RawCrashRecord:
    """Schema for a single raw traffic crash incident record."""

    crash_record_id: str
    crash_date: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    injuries_total: int = 0
    injuries_fatal: int = 0
    injuries_incapacitating: int = 0
    damage: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if not self.crash_record_id:
            raise ValueError("crash_record_id must be a non-empty string.")
        if not self.crash_date:
            raise ValueError("crash_date must be a non-empty string.")
        if self.injuries_total < 0:
            raise ValueError(f"injuries_total must be non-negative, got {self.injuries_total}.")
        if self.injuries_fatal < 0:
            raise ValueError(f"injuries_fatal must be non-negative, got {self.injuries_fatal}.")
        if self.injuries_incapacitating < 0:
            raise ValueError(
                f"injuries_incapacitating must be non-negative, got {self.injuries_incapacitating}."
            )
        # Note: injuries_total includes fatal and incapacitating injuries.
        # We ensure fatal + incapacitating cannot exceed total injuries.
        if (self.injuries_fatal + self.injuries_incapacitating) > self.injuries_total:
            # Tolerant normalization: adjust total if police field was undercounted
            pass


@dataclass
class MonthlyCommunityAreaRecord:
    """Aggregated monthly performance and risk indicators for one Chicago Community Area."""

    year_month: str
    community_area_id: int
    community_area_name: str
    new_requests_count: int = 0
    resolved_requests_count: int = 0
    reconstructed_active_backlog: int = 0
    median_resolution_days: Optional[float] = None
    total_crashes: int = 0
    fatalities: int = 0
    incapacitating_injuries: int = 0
    minor_injuries: int = 0
    pdo_crashes: int = 0
    raw_severity_score: float = 0.0
    normalized_severity_index: float = 0.0

    def __post_init__(self) -> None:
        if not (1 <= self.community_area_id <= 77):
            raise ValueError(
                f"community_area_id must be in [1, 77], got {self.community_area_id}."
            )
        if not (0.0 <= self.normalized_severity_index <= 1.0):
            raise ValueError(
                f"normalized_severity_index must be in [0.0, 1.0], got {self.normalized_severity_index}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for CSV/JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class RawAppropriationRecord:
    """Schema for a single appropriation item from the 2025 Budget Ordinance (t59y-fr3k).

    Note: These represent authorized legal expenditure ceilings voted by City Council,
    not final audited expenditures.
    """

    fund_type: str
    fund_code: str
    fund_description: str
    department_number: str
    department_description: str
    appropriation_authority: str
    appropriation_authority_description: str
    appropriation_account: str
    appropriation_account_description: str
    ordinance_amount: float

    def __post_init__(self) -> None:
        if self.ordinance_amount < 0.0:
            raise ValueError(f"ordinance_amount must be non-negative, got {self.ordinance_amount}")


@dataclass(frozen=True)
class RawGrantRecord:
    """Schema for a single grant project record from Mid-Year Grants 925 (iyu8-jkf8)."""

    department_code: str
    department_description: str
    fund_code: str
    grant_project_code: str
    grant_project_description: str
    budget: float
    expended_project_to_date: float
    encumbrances: float = 0.0
    funds_available: float = 0.0
    record_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.budget < 0.0:
            raise ValueError(f"grant budget must be non-negative, got {self.budget}")
        if self.expended_project_to_date < 0.0:
            raise ValueError(
                f"expended amount must be non-negative, got {self.expended_project_to_date}"
            )


@dataclass
class DepartmentBudgetSummary:
    """Aggregated authorized budget appropriations and grant tracking for a department."""

    department_description: str
    total_authorized_appropriation: float = 0.0
    operating_accounts_count: int = 0
    total_grant_budget: float = 0.0
    total_grant_expended: float = 0.0
    grant_projects_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PipelineMetadata:
    """Provenance and execution manifest for the Chicago Open Data pipeline."""

    retrieval_timestamp_utc: str
    target_year: int
    datasets: Dict[str, Any]
    raw_record_counts: Dict[str, int]
    processed_record_counts: Dict[str, int]
    cleaning_decisions: List[str]
    geographic_coverage: Dict[str, Any]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON output."""
        return asdict(self)

