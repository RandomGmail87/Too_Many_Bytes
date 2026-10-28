# AI Mayor City Sim: Research Engine Foundation

**Course:** CS 3793  
**Team:** Too Many Bytes  
**Project:** AI Mayor City Sim  
**Milestone:** 1 — Research Engine Foundation

---

## 1. Project Overview

**AI Mayor City Sim** is a computer science research simulation designed to benchmark how distinct artificial intelligence decision-making paradigms solve complex municipal resource allocation problems. 

Municipal decision-makers must balance trade-offs among fiscal constraints, public safety, service operational backlogs, and equitable resource distribution across geographically diverse neighborhoods. The completed project will evaluate four distinct decision methods:
1. **Rule-Based Baseline (Implemented in Milestone 1):** Transparent heuristic priority queue with budget-safeguard caps.
2. **Random Forest Prediction (Future Milestone):** Supervised policy trained on empirical feature targets derived from Chicago Open Data.
3. **Tabular Q-Learning (Future Milestone):** Discrete state-action reinforcement learning policy using action binning.
4. **Deep Q-Network (DQN) (Future Milestone):** Value-function neural network reinforcement learning agent.

---

## 2. Core Simulation Architecture & Assumptions

### A. Non-Causality & Research Scope Disclaimer
The simulator models city dynamics as a stylized Markov Decision Process (MDP). **Simulated state transitions do not establish causal effects in the real world.** The model provides an objective, reproducible benchmark environment to evaluate decision algorithms under controlled constraints.

### B. Synthetic Fixture vs. Empirical Data Separation
- **Synthetic Test Data (`data/samples/synthetic_city.json`):** Contains a deterministic 5-neighborhood scenario created solely to verify simulator mechanics, financial accounting, and test suites. These values do **not** represent actual measurements from the City of Chicago.
- **Empirical Chicago Data (Future Milestones):** Chicago 311 Service Requests and Traffic Crash datasets will be ingested and stored separately in future milestones.

### C. Fiscal Budget Model & Renewal
- **Renewed Monthly Budget:** A fixed municipal budget of **\$100,000** is appropriated at each monthly step ($t \in \{0, 1, \dots, 11\}$) across a 12-month fiscal year episode.
- **No Roll-Over:** Unspent funds do not roll over to subsequent months unless explicitly configured.
- **Financial Tracking:** The environment tracks both single-step expenditure (`step_spending`) and cumulative episode expenditure (`cumulative_spending`).
- **Strict Budget Validation:** Any action attempting to allocate negative funds, non-numeric values, or total funds exceeding the renewed monthly budget ($> \$100,000$) is strictly rejected with a `ValueError`.

### D. Funding Allocation Split & State Transitions
For any neighborhood allocation $a_i \ge 0$:
1. **Service Backlog Portion ($a_{i, \text{service}} = 0.80 \cdot a_i$):**
   - Resolves pending 311 service tickets at a documented unit cost of **\$100 / ticket**.
   - Resolved tickets:
     $$\Delta B_{i, t}^{\text{resolved}} = \min\left(B_{i, t}, \left\lfloor \frac{a_{i, \text{service}}}{\$100} \right\rfloor\right)$$
   - Incoming monthly demand ($\delta_{i, t}$) is added deterministically based on neighborhood baseline rates.
   - Non-negativity guarantee: Residual backlogs are bounded by $B_{i, t+1} \ge 0$.
2. **Traffic Safety Portion ($a_{i, \text{safety}} = 0.20 \cdot a_i$):**
   - **Transition Rule:** Traffic-safety improvement depends strictly on the portion of funding designated for safety ($a_{i, \text{safety}}$), **not** total neighborhood funding.
   - Reduction scale:
     $$\Delta R_{i, t}^{\text{reduced}} = \min\left(R_{i, t}, \frac{a_{i, \text{safety}}}{\$50,000}\right)$$
   - Natural traffic wear and roadway hazard drift adds a baseline drift ($\omega = +0.02$), clipped strictly to $[0.0, 1.0]$.

### E. Numerically Stable Composite Reward Function
The reward $R_t$ balances operational efficiency, traffic safety, and equity:
$$R_t = w_{\text{backlog}} \cdot \text{Reward}_{\text{backlog}} + w_{\text{safety}} \cdot \text{Reward}_{\text{safety}} - w_{\text{equity}} \cdot \text{Gini}(\mathbf{B}_{t+1})$$

- **Normalized Backlog Component:**
  - If initial total backlog $\sum B_{j, 0} > 0$: $\frac{\sum_i \Delta B_{i, t}^{\text{resolved}}}{\sum_j B_{j, 0}}$.
  - If initial total backlog is 0: returns $1.0$ if residual backlogs remain 0, otherwise $0.0$.
- **Normalized Safety Component:** Mean risk reduction across neighborhoods: $\frac{1}{N} \sum_i \Delta R_{i, t}^{\text{reduced}} \in [0.0, 1.0]$.
- **Equity Penalty (Gini Coefficient):**
  $$\text{Gini}(\mathbf{x}) = \begin{cases} 0.0 & \text{if } \sum x_i = 0 \\ \frac{\sum_i \sum_j |x_i - x_j|}{2 N \sum_i x_i} & \text{if } \sum x_i > 0 \end{cases}$$
  - Returns $0.0$ when all residual backlogs are 0 or perfectly equal.
  - Penalizes disproportionate neglect of any single neighborhood.

### F. Rule-Based Baseline Policy & 40% Allocation Cap
- **Urgency Metric:**
  $$U_{i, t} = B_{i, t} \cdot (1.0 + R_{i, t}) \cdot W_i$$
- **Design Choice Note on Allocation Cap:**
  The baseline policy caps spending at **40% of the monthly budget (\$40,000)** for any single neighborhood. This constraint is an intentional municipal engineering safeguard to prevent any single high-demand area from monopolizing the entire monthly appropriation. **It is a heuristic design choice, not a formal mathematical proof of fairness.**

### G. Future Action Space Discretization for Tabular Q-Learning
The current simulation accepts continuous dollar allocations $\mathbf{a} \in \mathbb{R}^5_{\ge 0}$. In Milestone 2, Tabular Q-learning will require a discrete action set or action-binning layer (e.g. discrete crew deployments of \$20,000 chunks).

---

## 3. Project Structure

```text
Too_Many_Bytes/
├── .gitignore                          # Git exclusions (.venv, caches, logs)
├── README.md                           # Documentation, mathematical definitions, and run guide
├── pyproject.toml                      # Packaging metadata and pytest configuration
├── requirements.txt                    # Milestone 1 dependencies (numpy, pytest)
├── configs/
│   └── env_config.json                 # Simulation hyperparameters and weights
├── data/
│   └── samples/
│       └── synthetic_city.json         # Labeled 5-neighborhood synthetic test fixture
├── src/
│   └── ai_mayor/
│       ├── __init__.py                 # Package version
│       ├── data/
│       │   ├── __init__.py
│       │   └── synthetic.py            # Fixture loader and schema validation
│       ├── env/
│       │   ├── __init__.py
│       │   └── city_env.py             # CityEnvironment MDP and Gini calculation
│       └── agents/
│           ├── __init__.py
│           └── rule_based.py           # Priority-queue rule-based agent
└── tests/
    ├── test_budget.py                  # Strict budget enforcement tests
    ├── test_environment.py             # Dynamics, horizon, and numerical stability tests
    └── test_rule_based.py              # Rule-based policy validation tests
```

---

## 4. Setup and Installation

### Prerequisites
- Python **3.10 to 3.14** (tested on Python 3.14.2)
- Windows PowerShell, macOS, or Linux terminal

### Setup Virtual Environment
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# (On macOS/Linux: source .venv/bin/activate)

# 3. Upgrade pip and install minimal dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install local package in editable mode
pip install -e .
```

---

## 5. Running the 12-Month CLI Demonstration

To execute a complete 12-month simulation run using the synthetic five-neighborhood city fixture and the rule-based agent:

```powershell
python -m ai_mayor
# or explicitly specify seed:
python -m ai_mayor --seed 42
```

The demonstration displays the initial city conditions, monthly neighborhood allocations and state transitions, cumulative financial expenditure, and final performance metrics.

---

## 6. Real Chicago Data Pipeline (Milestone 2)

Milestone 2 establishes a reproducible data pipeline ingesting official Chicago Open Data:
1. **311 Service Requests (`v6vf-nfxy`):** Ingests service tickets, timestamps, request types, and community areas.
2. **Traffic Crashes (`85ca-t3if`):** Ingests incident reports, dates, injury breakdowns, and coordinates.
3. **Boundaries - Community Areas (`igwz-8jzy`):** Polygons for all 77 Chicago Community Areas used by the ray-casting spatial mapper to assign crash coordinates to community areas.

### Data Fetching and Processing Commands

```powershell
# Fetch Chicago Open Data sample batch (default: 2023, limit: 1,000 or custom)
python -m ai_mayor.data.fetcher --year 2023 --sample-limit 1000

# Preprocess raw records into monthly community area indicators
python -m ai_mayor.data.preprocessor --year 2023
```

Outputs generated:
- `data/processed/chicago_monthly_community_areas.csv`: Standardized monthly aggregates across all 77 areas x 12 months.
- `data/processed/chicago_monthly_community_areas.json`: JSON format for downstream models.
- `data/processed/dataset_metadata.json`: Immutable provenance manifest recording retrieval timestamps, Socrata IDs, cleaning logs, and geographic coverage.

### Methodological & Modeling Notes:
- **Historical Backlog vs. Inflow:** Reconstructs monthly backlog using `created_date` and `closed_date`. Avoids assuming current status reflects past months.
- **Traffic Safety Severity (EPDO):** Partitions injuries into disjoint categories (`fatalities`, `incapacitating`, `minor`, `pdo`) to prevent double-counting. Weighted using standard municipal safety weights and normalized to $[0.0, 1.0]$.
- **Offline Test Integrity:** The test suite uses small local mock fixtures and requires zero network access.

---

## 7. Chicago Budget Ingestion (Pivot Phase 1)

Ingests authorized spending ceilings enacted by City Council and active grant commitments:
1. **2025 Budget Ordinance Appropriations (`t59y-fr3k`):** Complete ordinance with 3,477 appropriation line items across 47 city departments ($18.9B citywide, $2.8B municipal infrastructure services).
2. **Mid-Year Grants 925 (`iyu8-jkf8`):** Project-to-date grant budgets, encumbrances, and expenditures across 789 projects.

### Budget Commands

```powershell
# Fetch Budget Ordinance and Grants
python -m ai_mayor.data.fetcher --dataset budget

# Process, validate, and aggregate departmental budgets
python -m ai_mayor.data.budget_loader
```

Outputs generated:
- `data/processed/chicago_department_budgets_2025.csv`
- `data/processed/chicago_department_budgets_2025.json`

---

## 8. Running Automated Tests

Run the complete test suite via `pytest`:

```powershell
pytest tests/ -v
```

Expected output:
```text
tests/test_budget.py::test_spending_exceeding_budget_rejected PASSED
tests/test_budget.py::test_negative_allocation_rejected PASSED
tests/test_budget.py::test_nan_or_infinite_allocation_rejected PASSED
tests/test_budget.py::test_invalid_action_dimensions_rejected PASSED
tests/test_budget.py::test_exact_budget_allocation_accepted PASSED
tests/test_budget.py::test_zero_spending_allowed PASSED
tests/test_budget.py::test_cumulative_and_renewed_spending_across_steps PASSED
tests/test_budget_loader.py::test_parse_appropriations_valid PASSED
tests/test_budget_loader.py::test_parse_appropriations_handles_invalid_or_negative_amounts PASSED
tests/test_budget_loader.py::test_parse_grants_valid PASSED
tests/test_budget_loader.py::test_distinguish_authorized_appropriations_from_actual_expenditures PASSED
tests/test_budget_loader.py::test_filter_infrastructure_budgets PASSED
tests/test_budget_loader.py::test_process_budget_datasets_offline PASSED
tests/test_data_pipeline.py::test_schema_raw_311_record_validation PASSED
tests/test_data_pipeline.py::test_schema_raw_crash_record_validation PASSED
tests/test_data_pipeline.py::test_spatial_point_in_polygon_ray_casting PASSED
tests/test_data_pipeline.py::test_spatial_community_mapper_with_synthetic_geojson PASSED
tests/test_data_pipeline.py::test_clean_raw_311_filters_missing_geo PASSED
tests/test_data_pipeline.py::test_monthly_aggregation_and_backlog_reconstruction PASSED
tests/test_data_pipeline.py::test_crash_severity_disjoint_partitioning_no_double_counting PASSED
tests/test_data_pipeline.py::test_coverage_analysis_matrix_dimensions PASSED
tests/test_demo.py::test_run_demo_executes_successfully PASSED
tests/test_demo.py::test_main_cli_entry_point PASSED
tests/test_environment.py::test_synthetic_city_loads_correctly PASSED
tests/test_environment.py::test_reset_reproducibility PASSED
tests/test_environment.py::test_step_backlog_reduction_and_nonnegativity PASSED
tests/test_environment.py::test_safety_improvement_depends_strictly_on_safety_funding PASSED
tests/test_environment.py::test_twelve_step_horizon PASSED
tests/test_environment.py::test_compute_gini_numerical_stability PASSED
tests/test_environment.py::test_reward_numerical_stability_with_zero_initial_backlog PASSED
tests/test_rule_based.py::test_rule_based_action_validity PASSED
tests/test_rule_based.py::test_rule_based_full_12_month_episode PASSED
tests/test_rule_based.py::test_rule_based_outperforms_zero_allocation PASSED
tests/test_rule_based.py::test_rule_based_determinism PASSED

34 passed in 0.25s
```
