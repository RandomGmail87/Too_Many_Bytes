"""Data fetcher module for Chicago Open Data SODA API.

This module retrieves raw sample batches of 311 Service Requests, Traffic Crashes,
and Community Area Boundaries directly from data.cityofchicago.org.

Uses standard library urllib.request to avoid external HTTP dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def fetch_soda_resource(
    endpoint: str,
    query_params: Dict[str, str],
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Fetch JSON or GeoJSON resource from a Socrata SODA endpoint.

    Args:
        endpoint: Base Socrata resource URL.
        query_params: Dictionary of SODA query parameters (e.g. $where, $limit, $order).
        timeout: Request timeout in seconds.
        headers: Optional custom HTTP headers.

    Returns:
        Parsed JSON content (List or Dict).
    """
    req_headers = {"User-Agent": "AIMayorCitySim-CS3793/1.0"}
    if headers:
        req_headers.update(headers)

    url = endpoint
    if query_params:
        encoded_query = urllib.parse.urlencode(query_params)
        url = f"{endpoint}?{encoded_query}"

    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def fetch_chicago_datasets(
    year: int = 2023,
    sample_limit: int = 5000,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Retrieve sample datasets for 311 requests, crashes, and boundaries.

    Args:
        year: Calendar year to query.
        sample_limit: Maximum number of rows to retrieve per dataset.
        output_dir: Directory where raw JSON files should be stored.

    Returns:
        Dict[str, Any]: Retrieval summary metadata.
    """
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        target_dir = base_dir / "data" / "raw"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    start_date = f"{year}-01-01T00:00:00"
    end_date = f"{year}-12-31T23:59:59"

    print(f"Fetching Chicago Open Data samples for Year: {year} (Limit: {sample_limit:,} rows)...")

    # 1. Fetch 311 Service Requests
    print("  -> Downloading 311 Service Requests (v6vf-nfxy)...")
    sr_params = {
        "$where": f"created_date between '{start_date}' and '{end_date}'",
        "$limit": str(sample_limit),
        "$order": "created_date ASC",
    }
    sr_data = fetch_soda_resource(
        "https://data.cityofchicago.org/resource/v6vf-nfxy.json",
        sr_params,
    )
    sr_path = target_dir / "service_requests_311.json"
    with open(sr_path, "w", encoding="utf-8") as f:
        json.dump(sr_data, f)
    print(f"     Saved {len(sr_data):,} 311 records to {sr_path.name}")

    time.sleep(1.0)  # Polite API pacing

    # 2. Fetch Traffic Crashes
    print("  -> Downloading Traffic Crashes (85ca-t3if)...")
    crash_params = {
        "$where": f"crash_date between '{start_date}' and '{end_date}'",
        "$limit": str(sample_limit),
        "$order": "crash_date ASC",
    }
    crash_data = fetch_soda_resource(
        "https://data.cityofchicago.org/resource/85ca-t3if.json",
        crash_params,
    )
    crash_path = target_dir / "traffic_crashes.json"
    with open(crash_path, "w", encoding="utf-8") as f:
        json.dump(crash_data, f)
    print(f"     Saved {len(crash_data):,} crash records to {crash_path.name}")

    time.sleep(1.0)

    # 3. Fetch Boundaries GeoJSON (static polygons)
    print("  -> Downloading Boundaries - Community Areas (igwz-8jzy)...")
    geo_path = target_dir / "community_boundaries.geojson"
    geo_data = fetch_soda_resource(
        "https://data.cityofchicago.org/resource/igwz-8jzy.geojson",
        {},
    )
    with open(geo_path, "w", encoding="utf-8") as f:
        json.dump(geo_data, f)
    print(f"     Saved {len(geo_data.get('features', [])):,} community polygons to {geo_path.name}")

    # Write retrieval manifest
    manifest = {
        "retrieval_timestamp_utc": timestamp_utc,
        "target_year": year,
        "sample_limit_applied": sample_limit,
        "datasets": {
            "service_requests_311": {
                "socrata_id": "v6vf-nfxy",
                "retrieved_count": len(sr_data),
                "query": sr_params,
                "file": sr_path.name,
            },
            "traffic_crashes": {
                "socrata_id": "85ca-t3if",
                "retrieved_count": len(crash_data),
                "query": crash_params,
                "file": crash_path.name,
            },
            "community_boundaries": {
                "socrata_id": "igwz-8jzy",
                "retrieved_count": len(geo_data.get("features", [])),
                "file": geo_path.name,
            },
        },
    }
    meta_path = target_dir / "retrieval_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Download completed. Metadata manifest written to {meta_path.name}\n")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for data fetcher."""
    parser = argparse.ArgumentParser(
        description="Retrieve Chicago Open Data samples from the Socrata SODA API."
    )
    parser.add_argument("--year", type=int, default=2023, help="Target calendar year (default: 2023)")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5000,
        help="Max sample rows per dataset (default: 5000)",
    )
    args = parser.parse_args(argv)

    try:
        fetch_chicago_datasets(year=args.year, sample_limit=args.sample_limit)
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to fetch Chicago Open Data: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

