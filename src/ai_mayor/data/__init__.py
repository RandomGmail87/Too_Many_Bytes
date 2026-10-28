"""Data ingestion, fixture loading, and preprocessing modules for AI Mayor City Sim."""

from ai_mayor.data.schema import (
    MonthlyCommunityAreaRecord,
    PipelineMetadata,
    Raw311Record,
    RawCrashRecord,
)
from ai_mayor.data.spatial import COMMUNITY_AREA_NAMES, SpatialCommunityMapper
from ai_mayor.data.synthetic import load_synthetic_city

__all__ = [
    "load_synthetic_city",
    "Raw311Record",
    "RawCrashRecord",
    "MonthlyCommunityAreaRecord",
    "PipelineMetadata",
    "COMMUNITY_AREA_NAMES",
    "SpatialCommunityMapper",
]
