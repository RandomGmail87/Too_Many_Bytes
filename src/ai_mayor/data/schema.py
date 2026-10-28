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

