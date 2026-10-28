"""Unit tests for Chicago budget appropriations and grants ingestion.

All tests use local offline mock fixtures and require zero network access.
"""

import json
from pathlib import Path
import pytest

from ai_mayor.data.budget_loader import (
    INFRASTRUCTURE_DEPARTMENTS,
    aggregate_department_budgets,
    filter_infrastructure_budgets,
    parse_appropriations,
    parse_grants,
    process_budget_datasets,
)
from ai_mayor.data.schema import (
    DepartmentBudgetSummary,
    RawAppropriationRecord,
    RawGrantRecord,
)


def test_parse_appropriations_valid():
    """Verify parse_appropriations correctly extracts and converts line items."""
    raw_data = [
        {
            "fund_type": "LOCAL",
            "fund_code": "0100",
            "fund_description": "Corporate Fund",
            "department_number": "84",
            "department_description": "Chicago Department of Transportation",
            "appropriation_authority": "2005",
            "appropriation_authority_description": "In-House Construction",
            "appropriation_account": "0005",
            "appropriation_account_description": "Salaries and Wages - on Payroll",
            "_ordinance_amount_": "12500000",
        },
        {
            "fund_type": "LOCAL",
            "fund_code": "0100",
            "fund_description": "Corporate Fund",
            "department_number": "81",
            "department_description": "Department of Streets and Sanitation",
            "appropriation_authority": "2005",
            "appropriation_authority_description": "Bureau of Sanitation",
            "appropriation_account": "0140",
            "appropriation_account_description": "Professional and Technical Services",
            "_ordinance_amount_": "350000.50",
        },
    ]

    records, logs = parse_appropriations(raw_data)
    assert len(records) == 2
    assert records[0].department_description == "Chicago Department of Transportation"
    assert records[0].ordinance_amount == 12500000.0
    assert records[1].ordinance_amount == 350000.50


def test_parse_appropriations_handles_invalid_or_negative_amounts():
    """Verify parse_appropriations skips records with negative or unparseable amounts."""
    raw_data = [
        {
            "department_description": "CDOT",
            "_ordinance_amount_": "1000",
        },
        {
            "department_description": "CDOT",
            "_ordinance_amount_": "-500",  # Negative
        },
        {
            "department_description": "CDOT",
            "_ordinance_amount_": "INVALID_TEXT",  # Unparseable
        },
        {
            "department_description": "CDOT",
            "_ordinance_amount_": None,  # None
        },
    ]

    records, logs = parse_appropriations(raw_data)
    assert len(records) == 1
    assert records[0].ordinance_amount == 1000.0


def test_parse_grants_valid():
    """Verify parse_grants correctly extracts multi-year grant commitments and expenditures."""
    raw_data = [
        {
            "department_code": "D84",
            "department_description": "Chicago Department of Transportation",
            "fund_code": "F0G12",
            "grant_project_code": "G8401",
            "grant_project_description": "Vision Zero Traffic Calming",
            "budget": "500000.00",
            "expended_project_to_date": "120000.00",
            "encumbrances": "30000.00",
            "funds_available_project_to_date": "350000.00",
            "record_id": "REC-001",
        }
    ]

    records, logs = parse_grants(raw_data)
    assert len(records) == 1
    assert records[0].budget == 500000.00
    assert records[0].expended_project_to_date == 120000.00
    assert records[0].encumbrances == 30000.00


def test_distinguish_authorized_appropriations_from_actual_expenditures():
    """Verify that authorized appropriations and grant expenditures are strictly separated."""
    apps = [
        RawAppropriationRecord(
            fund_type="LOCAL",
            fund_code="0100",
            fund_description="Corporate",
            department_number="84",
            department_description="Chicago Department of Transportation",
            appropriation_authority="2005",
            appropriation_authority_description="Operations",
            appropriation_account="0005",
            appropriation_account_description="Payroll",
            ordinance_amount=1000000.0,
        )
    ]
    grants = [
        RawGrantRecord(
            department_code="D84",
            department_description="Chicago Department of Transportation",
            fund_code="F01",
            grant_project_code="GP01",
            grant_project_description="Bridge Repair",
            budget=500000.0,
            expended_project_to_date=200000.0,
        )
    ]

    summaries = aggregate_department_budgets(apps, grants)
    cdot = summaries["Chicago Department of Transportation"]

    # Authorized legal ceiling from Budget Ordinance
    assert cdot.total_authorized_appropriation == 1000000.0
    # Grant commitments & expenditures are tracked distinctly, NOT merged into authorized ceiling
    assert cdot.total_grant_budget == 500000.0
    assert cdot.total_grant_expended == 200000.0


def test_filter_infrastructure_budgets():
    """Verify filter_infrastructure_budgets retains only relevant municipal service departments."""
    summaries = {
        "Chicago Department of Transportation": DepartmentBudgetSummary(
            department_description="Chicago Department of Transportation",
            total_authorized_appropriation=5000000.0,
        ),
        "Department of Streets and Sanitation": DepartmentBudgetSummary(
            department_description="Department of Streets and Sanitation",
            total_authorized_appropriation=2000000.0,
        ),
        "Department of Cultural Affairs": DepartmentBudgetSummary(
            department_description="Department of Cultural Affairs",
            total_authorized_appropriation=800000.0,
        ),
    }

    filtered = filter_infrastructure_budgets(summaries)
    assert len(filtered) == 2
    assert "Chicago Department of Transportation" in filtered
    assert "Department of Streets and Sanitation" in filtered
    assert "Department of Cultural Affairs" not in filtered


def test_process_budget_datasets_offline(tmp_path):
    """Verify process_budget_datasets reads raw JSON files, aggregates, and writes CSV/JSON."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    out_dir.mkdir()

    raw_apps = [
        {
            "department_description": "Chicago Department of Transportation",
            "_ordinance_amount_": "5000000",
        },
        {
            "department_description": "Department of Streets and Sanitation",
            "_ordinance_amount_": "2000000",
        },
    ]
    raw_grants = [
        {
            "department_description": "Chicago Department of Transportation",
            "budget": "100000",
            "expended_project_to_date": "50000",
        }
    ]

    with open(raw_dir / "budget_appropriations_2025.json", "w", encoding="utf-8") as f:
        json.dump(raw_apps, f)
    with open(raw_dir / "mid_year_grants_925.json", "w", encoding="utf-8") as f:
        json.dump(raw_grants, f)

    result = process_budget_datasets(raw_dir=raw_dir, output_dir=out_dir)

    assert result["total_city_authorized"] == 7000000.0
    assert result["total_infra_authorized"] == 7000000.0
    assert Path(result["csv_path"]).exists()
    assert Path(result["json_path"]).exists()

    # Verify JSON content
    with open(result["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["metadata"]["total_citywide_authorized_appropriation"] == 7000000.0
        assert "Chicago Department of Transportation" in data["infrastructure_departments"]

