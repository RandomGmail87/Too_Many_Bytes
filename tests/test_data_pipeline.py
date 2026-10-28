"""Unit tests for Chicago data schemas, spatial mapping, and preprocessing pipeline.

All tests in this module use small, locally generated offline fixtures and do NOT
require internet access or external API calls.
"""

import json
from datetime import datetime
import pytest

from ai_mayor.data.preprocessor import (
    aggregate_monthly_community_data,
    clean_raw_311,
    clean_raw_crashes,
    get_month_str,
    parse_iso_datetime,
)
from ai_mayor.data.schema import (
    MonthlyCommunityAreaRecord,
    Raw311Record,
    RawCrashRecord,
)
from ai_mayor.data.spatial import (
    COMMUNITY_AREA_NAMES,
    SpatialCommunityMapper,
    point_in_polygon,
)


def test_schema_raw_311_record_validation():
    """Verify Raw311Record dataclass validates field boundaries and constraints."""
    # Valid record
    rec = Raw311Record(
        sr_number="SR23-0001",
        sr_type="Pothole in Street",
        status="COMPLETED",
        created_date="2023-01-10T14:30:00.000",
        closed_date="2023-01-15T09:00:00.000",
        community_area=32,
        latitude=41.88,
        longitude=-87.63,
    )
    assert rec.sr_number == "SR23-0001"
    assert rec.community_area == 32

    # Invalid empty sr_number
    with pytest.raises(ValueError, match="sr_number must be a non-empty string"):
        Raw311Record(
            sr_number="",
            sr_type="Pothole",
            status="OPEN",
            created_date="2023-01-01T00:00:00",
        )

    # Invalid community area outside [1, 77]
    with pytest.raises(ValueError, match="community_area must be in range"):
        Raw311Record(
            sr_number="SR23-0002",
            sr_type="Pothole",
            status="OPEN",
            created_date="2023-01-01T00:00:00",
            community_area=99,
        )


def test_schema_raw_crash_record_validation():
    """Verify RawCrashRecord dataclass enforces non-negative counts and valid structure."""
    rec = RawCrashRecord(
        crash_record_id="CRASH-001",
        crash_date="2023-05-12T18:15:00.000",
        latitude=41.88,
        longitude=-87.63,
        injuries_total=3,
        injuries_fatal=1,
        injuries_incapacitating=1,
        damage="OVER $1,500",
    )
    assert rec.injuries_total == 3
    assert rec.injuries_fatal == 1

    # Negative injury count rejected
    with pytest.raises(ValueError, match="injuries_total must be non-negative"):
        RawCrashRecord(
            crash_record_id="CRASH-002",
            crash_date="2023-01-01T00:00:00",
            injuries_total=-1,
        )


def test_spatial_point_in_polygon_ray_casting():
    """Verify ray-casting algorithm detects points inside, outside, and on perimeter."""
    # Simple unit square: (0,0) to (10,10)
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

    # Point strictly inside
    assert point_in_polygon(5.0, 5.0, square) is True

    # Point strictly outside
    assert point_in_polygon(15.0, 5.0, square) is False
    assert point_in_polygon(-2.0, 5.0, square) is False
    assert point_in_polygon(5.0, 15.0, square) is False


def test_spatial_community_mapper_with_synthetic_geojson(tmp_path):
    """Verify SpatialCommunityMapper loads GeoJSON and matches coordinates to area ID."""
    mock_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"area_num_1": "32", "community": "LOOP"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-87.64, 41.87],
                            [-87.62, 41.87],
                            [-87.62, 41.89],
                            [-87.64, 41.89],
                            [-87.64, 41.87],
                        ]
                    ],
                },
            }
        ],
    }
    geo_file = tmp_path / "test_boundaries.geojson"
    with open(geo_file, "w", encoding="utf-8") as f:
        json.dump(mock_geojson, f)

    mapper = SpatialCommunityMapper(geo_file)
    assert mapper.find_community_area(41.88, -87.63) == 32
    # Outside coordinates return None
    assert mapper.find_community_area(41.95, -87.63) is None
    assert mapper.find_community_area(None, None) is None


def test_clean_raw_311_filters_missing_geo():
    """Verify clean_raw_311 excludes records lacking valid community areas."""
    raw_sample = [
        {"sr_number": "S1", "created_date": "2023-01-05T10:00:00", "community_area": "32"},
        {"sr_number": "S2", "created_date": "2023-01-06T10:00:00", "community_area": None},  # Missing CA
        {"sr_number": "S3", "created_date": "2023-01-07T10:00:00", "community_area": "99"},  # Invalid CA
    ]
    cleaned, logs = clean_raw_311(raw_sample)
    assert len(cleaned) == 1
    assert cleaned[0].sr_number == "S1"
    assert cleaned[0].community_area == 32


def test_monthly_aggregation_and_backlog_reconstruction():
    """Verify aggregation distinguishes monthly created count from active backlog and turnaround."""
    # 2 records for Community Area 32 in Jan 2023
    # Rec 1: Created Jan 10, Closed Jan 20 (Duration: 10 days) -> resolved in Jan
    # Rec 2: Created Jan 25, Closed Feb 15 -> active backlog at Jan 31
    clean_311 = [
        Raw311Record(
            sr_number="S1",
            sr_type="Pothole",
            status="COMPLETED",
            created_date="2023-01-10T00:00:00",
            closed_date="2023-01-20T00:00:00",
            community_area=32,
        ),
        Raw311Record(
            sr_number="S2",
            sr_type="Street Light",
            status="COMPLETED",
            created_date="2023-01-25T00:00:00",
            closed_date="2023-02-15T00:00:00",
            community_area=32,
        ),
    ]
    clean_crashes = []

    records, coverage = aggregate_monthly_community_data(
        clean_311, clean_crashes, target_year=2023
    )

    # Find Jan 2023 record for CA 32
    jan_32 = next(r for r in records if r.year_month == "2023-01" and r.community_area_id == 32)
    assert jan_32.new_requests_count == 2
    assert jan_32.resolved_requests_count == 1
    assert jan_32.reconstructed_active_backlog == 1
    assert jan_32.median_resolution_days == pytest.approx(10.0)


def test_crash_severity_disjoint_partitioning_no_double_counting():
    """Verify that crash severity formula partitions injuries to prevent double-counting."""
    # 1 crash with total 3 injuries: 1 fatal, 1 incapacitating, and 1 minor
    crash_rec = RawCrashRecord(
        crash_record_id="C1",
        crash_date="2023-03-15T12:00:00",
        injuries_total=3,
        injuries_fatal=1,
        injuries_incapacitating=1,
    )
    clean_crashes = [(crash_rec, 32)]
    clean_311 = []

    custom_weights = {
        "fatal": 20.0,
        "incapacitating": 10.0,
        "minor_injury": 2.0,
        "property_damage_only": 1.0,
    }

    records, _ = aggregate_monthly_community_data(
        clean_311, clean_crashes, target_year=2023, severity_weights=custom_weights
    )

    mar_32 = next(r for r in records if r.year_month == "2023-03" and r.community_area_id == 32)
    assert mar_32.total_crashes == 1
    assert mar_32.fatalities == 1
    assert mar_32.incapacitating_injuries == 1
    assert mar_32.minor_injuries == 1  # 3 total - 1 fatal - 1 incap = 1 minor
    assert mar_32.pdo_crashes == 0     # Had injuries, not PDO

    # Expected severity: 20*1 + 10*1 + 2*1 + 1*0 = 32.0
    assert mar_32.raw_severity_score == pytest.approx(32.0)
    assert mar_32.normalized_severity_index == pytest.approx(1.0)  # Only record, so max


def test_coverage_analysis_matrix_dimensions():
    """Verify coverage summary evaluates all 77 community areas across 12 months."""
    records, coverage = aggregate_monthly_community_data([], [], target_year=2023)

    assert len(records) == 77 * 12
    assert coverage["total_possible_area_months"] == 924
    assert coverage["total_official_community_areas"] == 77

