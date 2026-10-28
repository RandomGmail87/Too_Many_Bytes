"""Data ingestion, fixture loading, and preprocessing modules for AI Mayor City Sim."""

from ai_mayor.data.schema import (
    DepartmentBudgetSummary,
    MonthlyCommunityAreaRecord,
    PipelineMetadata,
    Raw311Record,
    RawAppropriationRecord,
    RawCrashRecord,
    RawGrantRecord,
)
from ai_mayor.data.spatial import COMMUNITY_AREA_NAMES, SpatialCommunityMapper
from ai_mayor.data.synthetic import load_synthetic_city

__all__ = [
    "load_synthetic_city",
    "Raw311Record",
    "RawCrashRecord",
    "RawAppropriationRecord",
    "RawGrantRecord",
    "DepartmentBudgetSummary",
    "MonthlyCommunityAreaRecord",
    "PipelineMetadata",
    "COMMUNITY_AREA_NAMES",
    "SpatialCommunityMapper",
    "INFRASTRUCTURE_DEPARTMENTS",
    "parse_appropriations",
    "parse_grants",
    "aggregate_department_budgets",
    "filter_infrastructure_budgets",
    "process_budget_datasets",
]


def __getattr__(name: str):
    """Lazy import for budget loader functions to avoid runpy module import warnings."""
    if name in [
        "INFRASTRUCTURE_DEPARTMENTS",
        "parse_appropriations",
        "parse_grants",
        "aggregate_department_budgets",
        "filter_infrastructure_budgets",
        "process_budget_datasets",
    ]:
        import ai_mayor.data.budget_loader as bl
        return getattr(bl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
