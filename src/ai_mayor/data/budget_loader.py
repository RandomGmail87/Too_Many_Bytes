"""Budget loader and parser for Chicago Municipal Appropriations and Grants.

This module ingests and aggregates:
1. Chicago 2025 Budget Ordinance - Appropriations (t59y-fr3k):
   Represents authorized legal expenditure ceilings approved by City Council.
2. Mid-Year Grants 925 (iyu8-jkf8):
   Tracks active grant projects, committed encumbrances, and project-to-date expenditures.

RESEARCH METHODOLOGY CAVEAT:
Appropriations represent legal spending ceilings (authorized limits), not final audited
expenditures. This loader explicitly separates authorized appropriations from grant
expenditures and documents that distinction for downstream budget allocation models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_mayor.data.schema import (
    DepartmentBudgetSummary,
    RawAppropriationRecord,
    RawGrantRecord,
)

# Core municipal infrastructure departments directly aligned with 311 services
INFRASTRUCTURE_DEPARTMENTS = [
    "Chicago Department of Transportation",
    "Department of Streets and Sanitation",
    "Department of Water Management",
]


def parse_appropriations(raw_data: List[Dict[str, Any]]) -> Tuple[List[RawAppropriationRecord], List[str]]:
    """Parse and validate raw records from 2025 Budget Ordinance (t59y-fr3k).

    Returns:
        Tuple of (list_of_RawAppropriationRecord, parsing_logs)
    """
    parsed: List[RawAppropriationRecord] = []
    dropped_invalid_amount = 0

    for idx, r in enumerate(raw_data):
        raw_amt = r.get("_ordinance_amount_")
        if raw_amt is None:
            dropped_invalid_amount += 1
            continue

        try:
            amt = float(raw_amt)
            if amt < 0.0:
                dropped_invalid_amount += 1
                continue
        except (ValueError, TypeError):
            dropped_invalid_amount += 1
            continue

        record = RawAppropriationRecord(
            fund_type=str(r.get("fund_type", "UNKNOWN")),
            fund_code=str(r.get("fund_code", "")),
            fund_description=str(r.get("fund_description", "")),
            department_number=str(r.get("department_number", "")),
            department_description=str(r.get("department_description", "UNKNOWN")),
            appropriation_authority=str(r.get("appropriation_authority", "")),
            appropriation_authority_description=str(
                r.get("appropriation_authority_description", "")
            ),
            appropriation_account=str(r.get("appropriation_account", "")),
            appropriation_account_description=str(
                r.get("appropriation_account_description", "")
            ),
            ordinance_amount=amt,
        )
        parsed.append(record)

    logs = [
        f"Appropriations: Parsed {len(raw_data):,} raw records.",
        f"Appropriations: Retained {len(parsed):,} valid line items.",
        f"Appropriations: Dropped {dropped_invalid_amount:,} records with invalid/negative amounts.",
    ]
    return parsed, logs


def parse_grants(raw_data: List[Dict[str, Any]]) -> Tuple[List[RawGrantRecord], List[str]]:
    """Parse and validate raw records from Mid-Year Grants 925 (iyu8-jkf8).

    Returns:
        Tuple of (list_of_RawGrantRecord, parsing_logs)
    """
    parsed: List[RawGrantRecord] = []
    dropped_invalid = 0

    def safe_float(val: Any) -> float:
        if val is None or val == "":
            return 0.0
        try:
            f = float(val)
            return max(0.0, f)
        except (ValueError, TypeError):
            return 0.0

    for r in raw_data:
        budget_val = safe_float(r.get("budget"))
        expended_val = safe_float(r.get("expended_project_to_date"))
        encumbered_val = safe_float(r.get("encumbrances"))
        avail_val = safe_float(r.get("funds_available_project_to_date"))

        dept_desc = r.get("department_description") or "UNKNOWN"

        rec = RawGrantRecord(
            department_code=str(r.get("department_code", "")),
            department_description=str(dept_desc),
            fund_code=str(r.get("fund_code", "")),
            grant_project_code=str(r.get("grant_project_code", "")),
            grant_project_description=str(r.get("grant_project_description", "")),
            budget=budget_val,
            expended_project_to_date=expended_val,
            encumbrances=encumbered_val,
            funds_available=avail_val,
            record_id=r.get("record_id"),
        )
        parsed.append(rec)

    logs = [
        f"Grants: Parsed {len(raw_data):,} raw records.",
        f"Grants: Retained {len(parsed):,} grant project records.",
    ]
    return parsed, logs


def aggregate_department_budgets(
    appropriations: List[RawAppropriationRecord],
    grants: Optional[List[RawGrantRecord]] = None,
) -> Dict[str, DepartmentBudgetSummary]:
    """Aggregate authorized appropriations and grant finances by department.

    Args:
        appropriations: List of validated 2025 Budget Ordinance line items.
        grants: Optional list of validated Mid-Year Grant records.

    Returns:
        Dict[str, DepartmentBudgetSummary]: Summaries keyed by department description.
    """
    summaries: Dict[str, DepartmentBudgetSummary] = {}

    # 1. Aggregate Ordinance Appropriations
    for app in appropriations:
        d = app.department_description
        if d not in summaries:
            summaries[d] = DepartmentBudgetSummary(department_description=d)

        summaries[d].total_authorized_appropriation += app.ordinance_amount
        summaries[d].operating_accounts_count += 1

    # 2. Aggregate Grant Projects (if provided)
    if grants:
        for g in grants:
            d = g.department_description
            if d not in summaries:
                summaries[d] = DepartmentBudgetSummary(department_description=d)

            summaries[d].total_grant_budget += g.budget
            summaries[d].total_grant_expended += g.expended_project_to_date
            summaries[d].grant_projects_count += 1

    return summaries


def filter_infrastructure_budgets(
    summaries: Dict[str, DepartmentBudgetSummary],
    departments: Optional[List[str]] = None,
) -> Dict[str, DepartmentBudgetSummary]:
    """Filter department budget summaries to key infrastructure departments.

    Args:
        summaries: Dictionary of department budget summaries.
        departments: Optional list of target departments. If None, uses INFRASTRUCTURE_DEPARTMENTS.

    Returns:
        Dict[str, DepartmentBudgetSummary]: Filtered dictionary.
    """
    target_depts = set(departments if departments is not None else INFRASTRUCTURE_DEPARTMENTS)
    return {
        d: summary for d, summary in summaries.items() if d in target_depts
    }


def process_budget_datasets(
    raw_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Load, validate, aggregate, and export Chicago budget data.

    Returns:
        Dict[str, Any]: Budget processing results and summary statistics.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    in_dir = Path(raw_dir) if raw_dir else base_dir / "data" / "raw"
    out_dir = Path(output_dir) if output_dir else base_dir / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)

    app_file = in_dir / "budget_appropriations_2025.json"
    grant_file = in_dir / "mid_year_grants_925.json"

    raw_apps: List[Dict[str, Any]] = []
    raw_grants: List[Dict[str, Any]] = []

    if app_file.exists():
        with open(app_file, "r", encoding="utf-8") as f:
            raw_apps = json.load(f)

    if grant_file.exists():
        with open(grant_file, "r", encoding="utf-8") as f:
            raw_grants = json.load(f)

    # Parse records
    clean_apps, logs_apps = parse_appropriations(raw_apps)
    clean_grants, logs_grants = parse_grants(raw_grants)

    # Aggregate by department
    all_dept_summaries = aggregate_department_budgets(clean_apps, clean_grants)
    infra_summaries = filter_infrastructure_budgets(all_dept_summaries)

    # Total city-wide authorized budget
    total_city_authorized = sum(s.total_authorized_appropriation for s in all_dept_summaries.values())
    total_infra_authorized = sum(s.total_authorized_appropriation for s in infra_summaries.values())

    # Export CSV summary
    csv_path = out_dir / "chicago_department_budgets_2025.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "department_description",
            "total_authorized_appropriation",
            "operating_accounts_count",
            "total_grant_budget",
            "total_grant_expended",
            "grant_projects_count",
            "is_infrastructure_service",
        ])
        for d, s in sorted(all_dept_summaries.items()):
            is_infra = d in INFRASTRUCTURE_DEPARTMENTS
            writer.writerow([
                s.department_description,
                f"{s.total_authorized_appropriation:.2f}",
                s.operating_accounts_count,
                f"{s.total_grant_budget:.2f}",
                f"{s.total_grant_expended:.2f}",
                s.grant_projects_count,
                "YES" if is_infra else "NO",
            ])

    # Export JSON summary
    json_path = out_dir / "chicago_department_budgets_2025.json"
    export_payload = {
        "metadata": {
            "source_datasets": {
                "appropriations_2025": "t59y-fr3k",
                "mid_year_grants_925": "iyu8-jkf8",
            },
            "total_citywide_authorized_appropriation": total_city_authorized,
            "total_infrastructure_authorized_appropriation": total_infra_authorized,
            "total_departments": len(all_dept_summaries),
            "disclaimer": "Appropriations represent legal authorized expenditure limits enacted in the 2025 Budget Ordinance, not actual audited spending.",
        },
        "infrastructure_departments": {
            d: s.to_dict() for d, s in infra_summaries.items()
        },
        "all_departments": {
            d: s.to_dict() for d, s in all_dept_summaries.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    return {
        "total_city_authorized": total_city_authorized,
        "total_infra_authorized": total_infra_authorized,
        "department_count": len(all_dept_summaries),
        "infrastructure_departments": infra_summaries,
        "logs": logs_apps + logs_grants,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for budget loader."""
    try:
        res = process_budget_datasets()
        print("=" * 78)
        print("CHICAGO 2025 BUDGET ORDINANCE & GRANTS PROCESSING")
        print("=" * 78)
        print(f"Total Citywide Authorized Appropriations: ${res['total_city_authorized']:,.2f}")
        print(f"Total Infrastructure Authorized Ceilings: ${res['total_infra_authorized']:,.2f}")
        print(f"Total Municipal Departments Processed:   {res['department_count']}")
        print("\nInfrastructure Departments Aligned with Municipal Services:")
        for d, s in res["infrastructure_departments"].items():
            print(
                f"  - {d:<42} ${s.total_authorized_appropriation:>16,.2f} "
                f"({s.operating_accounts_count} accounts)"
            )
        print("\nOutputs Generated:")
        print(f"  CSV:  {res['csv_path']}")
        print(f"  JSON: {res['json_path']}")
        print("=" * 78)
        return 0
    except Exception as e:
        print(f"[ERROR] Budget processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

