"""Preprocessor for Chicago Open Data.

Aggregates raw 311 Service Requests and Traffic Crashes into standardized
monthly community area indicators, evaluates geographic coverage across all 77
community areas, and generates provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ai_mayor.data.schema import (
    MonthlyCommunityAreaRecord,
    PipelineMetadata,
    Raw311Record,
    RawCrashRecord,
)
from ai_mayor.data.spatial import COMMUNITY_AREA_NAMES, SpatialCommunityMapper


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string from Socrata format."""
    if not dt_str:
        return None
    # Strip fractional seconds or Z if present
    cleaned = dt_str.split(".")[0].replace("Z", "")
    try:
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def get_month_str(dt: Optional[datetime]) -> Optional[str]:
    """Extract 'YYYY-MM' string from datetime."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m")


def clean_raw_311(raw_list: List[Dict[str, Any]]) -> Tuple[List[Raw311Record], List[str]]:
    """Clean and validate raw 311 records.

    Returns:
        Tuple of (valid_records, cleaning_log_entries)
    """
    cleaned: List[Raw311Record] = []
    dropped_missing_geo = 0
    dropped_invalid_date = 0

    for r in raw_list:
        sr_num = r.get("sr_number")
        if not sr_num:
            continue

        created_dt = parse_iso_datetime(r.get("created_date"))
        if created_dt is None:
            dropped_invalid_date += 1
            continue

        ca_raw = r.get("community_area")
        ca_val: Optional[int] = None
        if ca_raw is not None:
            try:
                ca_val = int(float(ca_raw))
                if not (1 <= ca_val <= 77):
                    ca_val = None
            except (ValueError, TypeError):
                ca_val = None

        if ca_val is None:
            dropped_missing_geo += 1
            continue

        lat = float(r["latitude"]) if r.get("latitude") else None
        lon = float(r["longitude"]) if r.get("longitude") else None

        record = Raw311Record(
            sr_number=str(sr_num),
            sr_type=str(r.get("sr_type", "OTHER")),
            status=str(r.get("status", "OPEN")),
            created_date=r["created_date"],
            closed_date=r.get("closed_date"),
            community_area=ca_val,
            latitude=lat,
            longitude=lon,
        )
        cleaned.append(record)

    logs = [
        f"311: Parsed {len(raw_list):,} raw records.",
        f"311: Valid records retained: {len(cleaned):,}.",
        f"311: Dropped {dropped_missing_geo:,} records lacking valid community_area.",
        f"311: Dropped {dropped_invalid_date:,} records with unparseable created_date.",
    ]
    return cleaned, logs


def clean_raw_crashes(
    raw_list: List[Dict[str, Any]], mapper: SpatialCommunityMapper
) -> Tuple[List[Tuple[RawCrashRecord, int]], List[str]]:
    """Clean raw crash records and map coordinates to Community Area IDs.

    Returns:
        Tuple of (list_of_tuples(RawCrashRecord, community_area_id), cleaning_log_entries)
    """
    cleaned: List[Tuple[RawCrashRecord, int]] = []
    dropped_missing_geo = 0
    dropped_invalid_date = 0

    for r in raw_list:
        cid = r.get("crash_record_id")
        if not cid:
            continue

        crash_dt = parse_iso_datetime(r.get("crash_date"))
        if crash_dt is None:
            dropped_invalid_date += 1
            continue

        lat = float(r["latitude"]) if r.get("latitude") else None
        lon = float(r["longitude"]) if r.get("longitude") else None

        ca_id = mapper.find_community_area(lat, lon)
        if ca_id is None:
            dropped_missing_geo += 1
            continue

        def safe_int(val: Any) -> int:
            try:
                return int(float(val)) if val is not None else 0
            except (ValueError, TypeError):
                return 0

        inj_total = safe_int(r.get("injuries_total"))
        inj_fatal = safe_int(r.get("injuries_fatal"))
        inj_incap = safe_int(r.get("injuries_incapacitating"))

        rec = RawCrashRecord(
            crash_record_id=str(cid),
            crash_date=r["crash_date"],
            latitude=lat,
            longitude=lon,
            injuries_total=inj_total,
            injuries_fatal=inj_fatal,
            injuries_incapacitating=inj_incap,
            damage=str(r.get("damage", "UNKNOWN")),
        )
        cleaned.append((rec, ca_id))

    logs = [
        f"Crashes: Parsed {len(raw_list):,} raw records.",
        f"Crashes: Spatially aligned records retained: {len(cleaned):,}.",
        f"Crashes: Dropped {dropped_missing_geo:,} records unmapped to Chicago community areas.",
        f"Crashes: Dropped {dropped_invalid_date:,} records with invalid crash_date.",
    ]
    return cleaned, logs


def aggregate_monthly_community_data(
    records_311: List[Raw311Record],
    records_crashes: List[Tuple[RawCrashRecord, int]],
    target_year: int = 2023,
    severity_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[MonthlyCommunityAreaRecord], Dict[str, Any]]:
    """Aggregate cleaned 311 and crash records by (year_month, community_area_id).

    Returns:
        Tuple of (list_of_MonthlyCommunityAreaRecord, coverage_summary)
    """
    if severity_weights is None:
        severity_weights = {
            "fatal": 20.0,
            "incapacitating": 10.0,
            "minor_injury": 2.0,
            "property_damage_only": 1.0,
        }

    # Generate all expected months in target year
    all_months = [f"{target_year}-{m:02d}" for m in range(1, 13)]

    # Containers for grouping
    # Key: (year_month, ca_id)
    cell_311_created: Dict[Tuple[str, int], List[Raw311Record]] = {}
    cell_311_closed: Dict[Tuple[str, int], List[Raw311Record]] = {}
    cell_crashes: Dict[Tuple[str, int], List[RawCrashRecord]] = {}

    for r in records_311:
        created_dt = parse_iso_datetime(r.created_date)
        m_str = get_month_str(created_dt)
        if m_str and m_str in all_months and r.community_area:
            cell_311_created.setdefault((m_str, r.community_area), []).append(r)

        closed_dt = parse_iso_datetime(r.closed_date)
        if closed_dt:
            cl_m_str = get_month_str(closed_dt)
            if cl_m_str and cl_m_str in all_months and r.community_area:
                cell_311_closed.setdefault((cl_m_str, r.community_area), []).append(r)

    for rec, ca_id in records_crashes:
        c_dt = parse_iso_datetime(rec.crash_date)
        m_str = get_month_str(c_dt)
        if m_str and m_str in all_months:
            cell_crashes.setdefault((m_str, ca_id), []).append(rec)

    # First pass: compute raw metrics across all 77 community areas x 12 months
    raw_records: List[MonthlyCommunityAreaRecord] = []
    max_observed_severity = 0.0

    observed_pairs: Set[Tuple[str, int]] = set()

    for ca_id in range(1, 78):
        ca_name = COMMUNITY_AREA_NAMES.get(ca_id, f"Community Area {ca_id}")
        for m_str in all_months:
            pair = (m_str, ca_id)

            created_list = cell_311_created.get(pair, [])
            closed_list = cell_311_closed.get(pair, [])
            crash_list = cell_crashes.get(pair, [])

            if created_list or closed_list or crash_list:
                observed_pairs.add(pair)

            new_reqs = len(created_list)
            resolved_reqs = len(closed_list)

            # Reconstruct active backlog at month's end
            # End of month date:
            y, m = map(int, m_str.split("-"))
            month_end = datetime(y, m + 1, 1, 0, 0, 0) if m < 12 else datetime(y + 1, 1, 1, 0, 0, 0)

            active_backlog = 0
            for r in records_311:
                if r.community_area == ca_id:
                    c_dt = parse_iso_datetime(r.created_date)
                    cl_dt = parse_iso_datetime(r.closed_date)
                    if c_dt and c_dt < month_end:
                        if cl_dt is None or cl_dt >= month_end:
                            active_backlog += 1

            # Median turnaround days for tickets closed in this month
            turnarounds: List[float] = []
            for r in closed_list:
                c_dt = parse_iso_datetime(r.created_date)
                cl_dt = parse_iso_datetime(r.closed_date)
                if c_dt and cl_dt and cl_dt >= c_dt:
                    turnarounds.append((cl_dt - c_dt).total_seconds() / 86400.0)

            med_days = statistics.median(turnarounds) if turnarounds else None

            # Crash severity partitioning (disjoint categories)
            tot_crashes = len(crash_list)
            fatalities = sum(c.injuries_fatal for c in crash_list)
            incapacitating = sum(c.injuries_incapacitating for c in crash_list)
            # Minor injuries: total injuries minus fatal minus incapacitating
            minor = sum(
                max(0, c.injuries_total - c.injuries_fatal - c.injuries_incapacitating)
                for c in crash_list
            )
            pdo = sum(1 for c in crash_list if c.injuries_total == 0)

            raw_sev = float(
                (severity_weights["fatal"] * fatalities)
                + (severity_weights["incapacitating"] * incapacitating)
                + (severity_weights["minor_injury"] * minor)
                + (severity_weights["property_damage_only"] * pdo)
            )

            if raw_sev > max_observed_severity:
                max_observed_severity = raw_sev

            record = MonthlyCommunityAreaRecord(
                year_month=m_str,
                community_area_id=ca_id,
                community_area_name=ca_name,
                new_requests_count=new_reqs,
                resolved_requests_count=resolved_reqs,
                reconstructed_active_backlog=active_backlog,
                median_resolution_days=med_days,
                total_crashes=tot_crashes,
                fatalities=fatalities,
                incapacitating_injuries=incapacitating,
                minor_injuries=minor,
                pdo_crashes=pdo,
                raw_severity_score=raw_sev,
                normalized_severity_index=0.0,  # Will normalize in second pass
            )
            raw_records.append(record)

    # Second pass: normalize severity indicator to [0.0, 1.0]
    for r in raw_records:
        if max_observed_severity > 0.0:
            norm_val = r.raw_severity_score / max_observed_severity
            r.normalized_severity_index = float(round(min(1.0, max(0.0, norm_val)), 4))
        else:
            r.normalized_severity_index = 0.0

    # Geographic Coverage Analysis
    total_possible_cells = 77 * 12
    observed_cells_count = len(observed_pairs)
    unique_areas_observed = len({ca for _, ca in observed_pairs})

    coverage = {
        "total_possible_area_months": total_possible_cells,
        "observed_area_months": observed_cells_count,
        "missing_area_months": total_possible_cells - observed_cells_count,
        "coverage_percentage": round((observed_cells_count / total_possible_cells) * 100, 2),
        "unique_community_areas_observed": unique_areas_observed,
        "total_official_community_areas": 77,
        "max_observed_monthly_severity_score": max_observed_severity,
    }

    return raw_records, coverage


def process_datasets(
    raw_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    target_year: int = 2023,
) -> PipelineMetadata:
    """Run end-to-end preprocessing pipeline on raw data files.

    Returns:
        PipelineMetadata: Detailed provenance and summary manifest.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    in_dir = Path(raw_dir) if raw_dir else base_dir / "data" / "raw"
    out_dir = Path(output_dir) if output_dir else base_dir / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    sr_file = in_dir / "service_requests_311.json"
    crash_file = in_dir / "traffic_crashes.json"
    geo_file = in_dir / "community_boundaries.geojson"

    raw_311_data: List[Dict[str, Any]] = []
    raw_crash_data: List[Dict[str, Any]] = []

    if sr_file.exists():
        with open(sr_file, "r", encoding="utf-8") as f:
            raw_311_data = json.load(f)

    if crash_file.exists():
        with open(crash_file, "r", encoding="utf-8") as f:
            raw_crash_data = json.load(f)

    # Initialize spatial mapper
    mapper = SpatialCommunityMapper(geo_file if geo_file.exists() else None)

    # Clean records
    clean_311, logs_311 = clean_raw_311(raw_311_data)
    clean_crashes, logs_crashes = clean_raw_crashes(raw_crash_data, mapper)

    # Aggregate
    processed_records, coverage = aggregate_monthly_community_data(
        clean_311, clean_crashes, target_year=target_year
    )

    # Write CSV output
    csv_path = out_dir / "chicago_monthly_community_areas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year_month",
            "community_area_id",
            "community_area_name",
            "new_requests_count",
            "resolved_requests_count",
            "reconstructed_active_backlog",
            "median_resolution_days",
            "total_crashes",
            "fatalities",
            "incapacitating_injuries",
            "minor_injuries",
            "pdo_crashes",
            "raw_severity_score",
            "normalized_severity_index",
        ])
        for r in processed_records:
            writer.writerow([
                r.year_month,
                r.community_area_id,
                r.community_area_name,
                r.new_requests_count,
                r.resolved_requests_count,
                r.reconstructed_active_backlog,
                f"{r.median_resolution_days:.2f}" if r.median_resolution_days else "",
                r.total_crashes,
                r.fatalities,
                r.incapacitating_injuries,
                r.minor_injuries,
                r.pdo_crashes,
                f"{r.raw_severity_score:.1f}",
                f"{r.normalized_severity_index:.4f}",
            ])

    # Write JSON output
    json_path = out_dir / "chicago_monthly_community_areas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in processed_records], f, indent=2)

    # Write metadata manifest
    metadata = PipelineMetadata(
        retrieval_timestamp_utc=timestamp_utc,
        target_year=target_year,
        datasets={
            "service_requests_311": {"socrata_id": "v6vf-nfxy", "file": sr_file.name},
            "traffic_crashes": {"socrata_id": "85ca-t3if", "file": crash_file.name},
            "community_boundaries": {"socrata_id": "igwz-8jzy", "file": geo_file.name},
        },
        raw_record_counts={
            "raw_311_records": len(raw_311_data),
            "raw_crash_records": len(raw_crash_data),
        },
        processed_record_counts={
            "retained_311_records": len(clean_311),
            "retained_crash_records": len(clean_crashes),
            "monthly_community_area_aggregates": len(processed_records),
        },
        cleaning_decisions=logs_311 + logs_crashes,
        geographic_coverage=coverage,
        limitations=[
            "Sample dataset size: Initial batches validate the pipeline; larger datasets required for full research.",
            "Backlog reconstruction assumes tickets remained continuously open between created_date and closed_date; intermediate status updates are unrecorded.",
            "Crashes lacking geographic coordinates cannot be mapped to community areas and are excluded from neighborhood metrics.",
            "Traffic safety metric represents normalized Equivalent Property Damage Only (EPDO) severity, not a crash probability.",
        ],
    )

    meta_path = out_dir / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    print(f"[SUCCESS] Processed {len(processed_records):,} monthly records.")
    print(f"  CSV:  {csv_path.name}")
    print(f"  JSON: {json_path.name}")
    print(f"  Meta: {meta_path.name}")
    print(f"  Coverage: {coverage['observed_area_months']}/{coverage['total_possible_area_months']} "
          f"({coverage['coverage_percentage']}% of all 77 areas x 12 months)\n")

    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for preprocessor."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw Chicago Open Data into monthly community area indicators."
    )
    parser.add_argument("--year", type=int, default=2023, help="Target calendar year (default: 2023)")
    args = parser.parse_args(argv)

    try:
        process_datasets(target_year=args.year)
        return 0
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

