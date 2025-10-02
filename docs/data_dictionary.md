# EVM Data Dictionary

**Project:** Portfolio Management Suite
**Version:** Beta 0.9
**Last Updated:** 2025-10-02

---

## Core EVM Metrics

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `bac` | Budget_at_Completion, budget | float | Total project budget at completion | > 0 | All EVM modules |
| `ac` | Actual_Cost, actual_cost | float | Total actual cost incurred to date | ≥ 0 | All EVM modules |
| `ev` | Earned_Value, earned_value | float | Budgeted cost of work performed | ≥ 0, ≤ bac | All EVM modules |
| `pv` | Planned_Value, planned_value | float | Budgeted cost of work scheduled | ≥ 0, ≤ bac | All EVM modules |
| `cpi` | Cost_Performance_Index, cost_performance_index | float | Cost efficiency ratio (EV/AC) | > 0, typically 0.5-2.0 | All EVM modules |
| `spi` | Schedule_Performance_Index, schedule_performance_index | float | Schedule efficiency ratio (EV/PV) | > 0, typically 0.5-2.0 | All EVM modules |
| `cv` | Cost_Variance, cost_variance | float | Cost variance (EV - AC) | Any, positive is under budget | All EVM modules |
| `sv` | Schedule_Variance, schedule_variance | float | Schedule variance (EV - PV) | Any, positive is ahead | All EVM modules |
| `eac` | Estimate_at_Completion, estimate_at_completion | float | Forecast total project cost (BAC/CPI) | > 0 | All EVM modules |
| `etc` | Estimate_to_Complete, estimate_to_complete | float | Estimated remaining cost ((BAC-EV)/CPI) | ≥ 0 | All EVM modules |
| `vac` | Variance_at_Completion, variance_at_completion | float | Forecast variance at completion (EAC - BAC) | Any | All EVM modules |
| `tcpi_bac` | TCPI_BAC, tcpi | float | Performance needed to complete on budget ((BAC-EV)/(BAC-AC)) | > 0 | pages/7_EVM_Simulator |
| `tcpi_eac` | TCPI_EAC | float | Performance needed to complete at EAC ((BAC-EV)/(EAC-AC)) | > 0 | Advanced EVM analysis |

---

## Earned Schedule Metrics

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `es` | Earned_Schedule, earned_schedule, Earned_Duration, earned_duration | float | Time at which PV equals current EV | ≥ 0, ≤ od | core/evm_engine, pages/3_Project_Analysis, pages/7_EVM_Simulator |
| `spie` | Schedule_Performance_Index_ES, SPI(e) | float | Earned schedule efficiency (ES/AD) | > 0, typically 0.5-2.0 | core/evm_engine, pages/3_Project_Analysis, pages/4_Portfolio_Analysis |
| `tve` | Time_Variance_ES, TV(e) | float | Time variance (ES - AD) in months | Any | core/evm_engine |
| `ld` | Likely_Duration, likely_duration | float | Forecast project duration (OD/SPIe) | ≥ 0, ≤ 2.5*OD | core/evm_engine |
---

## Date & Duration Variables

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `plan_start` | Plan_Start, ps, project_start_date | datetime | Project planned start date | Valid date | All EVM modules |
| `plan_finish` | Plan_Finish, pf, project_finish_date | datetime | Project planned finish date | > plan_start | All EVM modules |
| `data_date` | Data_Date, dd, as_of_date, status_date | datetime | Current reporting/analysis date | ≥ plan_start | All EVM modules |
| `od` | Original_Duration, original_duration | float | Planned project duration (months) | > 0 | All EVM modules |
| `ad` | Actual_Duration, actual_duration, duration_to_date, used_duration | float | Elapsed duration to data date (months) | ≥ 0 | All EVM modules |
---

## Financial Metrics (Inflation-Adjusted)

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `present_value` | PV_Progress, pv_progress | float | Present value of progress (inflation-adjusted) | ≥ 0 | core/evm_engine |
| `annual_inflation_rate` | Inflation_Rate, inflation | float | Annual inflation rate (decimal) | 0.0-1.0 | All EVM modules |
| `monthly_rate` | Monthly_Inflation_Rate, monthly_inflation | float | Monthly compound inflation rate | ≥ 0 | core/evm_engine |
| `planned_value_project` | PV_Project, pvp | float | Present value of entire project | ≥ 0 | core/evm_engine |
| `likely_value_project` | LV_Project, lvp | float | Likely present value of project | ≥ 0 | core/evm_engine |

---

## Progress & Percentage Metrics

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `percent_complete` | Completion_Percentage, pct_complete, % Complete | float | Percentage of work completed | 0.0-100.0 | All EVM modules |
| `percent_budget_used` | Budget_Utilized_Pct | float | Percentage of budget spent (AC/BAC*100) | 0.0-∞ | core/evm_engine |
| `percent_time_used` | Time_Elapsed_Pct | float | Percentage of time elapsed (AD/OD*100) | 0.0-∞ | core/evm_engine |
| `percent_present_value_project` | PV_Project_Pct | float | Present value % of BAC | 0.0-100.0 | core/evm_engine |
| `percent_likely_value_project` | LV_Project_Pct | float | Likely value % of BAC | 0.0-∞ | core/evm_engine |
| `completion_efficiency` | Efficiency_Ratio | float | Completion vs time ratio | > 0 | pages/7_EVM_Simulator |

---

## Manual Override Variables

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `manual_pv` | Manual_PV, pv_manual | float | User-entered planned value override | ≥ 0 | pages/2_Manual_Data_Entry, core/evm_engine |
| `use_manual_pv` | Use_Manual_PV, pv_override_flag | bool | Flag to use manual PV instead of calculated | True/False | pages/2_Manual_Data_Entry, core/evm_engine |
| `manual_ev` | Manual_EV, ev_manual | float | User-entered earned value override | ≥ 0 | pages/2_Manual_Data_Entry, core/evm_engine |
| `use_manual_ev` | Use_Manual_EV, ev_override_flag | bool | Flag to use manual EV instead of calculated | True/False | pages/2_Manual_Data_Entry, core/evm_engine |

---

## S-Curve Parameters

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `curve_type` | Curve_Type, pv_curve, distribution_type | str | Curve type for PV calculation | 'linear', 's-curve' | All EVM modules |
| `alpha` | S_Curve_Alpha, scurve_alpha, α | float | S-curve beta distribution alpha parameter | 0.1-10.0 | core/evm_engine, pages/3_Project_Analysis |
| `beta` | S_Curve_Beta, scurve_beta, β | float | S-curve beta distribution beta parameter | 0.1-10.0 | core/evm_engine, pages/3_Project_Analysis |

---

## Project Metadata

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `project_id` | Project_ID, pid, ID | str | Unique project identifier | Non-empty string | All modules |
| `project_name` | Project, Name, project | str | Project name/title | Non-empty string | All modules |
| `organization` | Organization, Org, org | str | Owning organization | Any string | All modules |
| `project_manager` | Project_Manager, PM, pm | str | Project manager name | Any string | All modules |

---

---

## Constants & Configuration

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `DAYS_PER_MONTH` | Days_Per_Month | float | Standard days in a month (constant) | 30.44 | core/evm_engine |
| `INTEGRATION_STEPS` | Integration_Steps | int | S-curve numerical integration steps | 200 | core/evm_engine |
| `EXCEL_ORDINAL_BASE` | Excel_Base_Date | datetime | Excel date ordinal base (1899-12-30) | Fixed | core/evm_engine |

---

## Currency & Display Variables

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `currency_symbol` | Currency_Symbol, symbol | str | Currency symbol for display | Any string | All pages |
| `currency_postfix` | Currency_Postfix, postfix | str | Currency unit (e.g., "Million") | Any string | All pages |
| `date_format` | Date_Format | str | Date display format | 'YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY' | pages/1_File_Management |

---

## Budget Tier Configuration

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `tier_config` | Tier_Configuration | dict | Budget tier configuration object | Valid config dict | pages/1_File_Management |
| `cutoff_points` | Tier_Cutoffs | list[float] | Budget cutoff values for tiers | 3 ascending values | pages/1_File_Management |
| `tier_names` | Tier_Names | list[str] | Names for each tier | 4 strings | pages/1_File_Management |

---

## Cash Flow Modeling Variables

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `budget` | Total_Budget | float | Total project budget (millions) | > 0 | pages/6_Cash_Flow_Simulator |
| `duration` | Project_Duration | int | Project duration (months) | ≥ 1 | pages/6_Cash_Flow_Simulator |
| `cashflow_type` | Pattern, Flow_Type | str | Cash flow pattern type | 'Linear', 'Highway', 'Building', 'S-Curve' | pages/6_Cash_Flow_Simulator |
| `start_delay` | Start_Delay | int | Delay before project starts (months) | ≥ 0 | pages/6_Cash_Flow_Simulator |
| `project_delay` | Project_Delay | int | Additional project duration (months) | ≥ 0 | pages/6_Cash_Flow_Simulator |
| `inflation` | Inflation_Pct | float | Annual inflation rate (percentage) | 0.0-99.9 | pages/6_Cash_Flow_Simulator |

---

## Performance Thresholds & Health Indicators

| Standard_Name | Alternative_Names | Type | Description | Valid_Range | Used_In |
|--------------|-------------------|------|-------------|-------------|---------|
| `health_score` | Project_Health | float | Overall health score ((CPI+SPI)/2) | > 0 | pages/7_EVM_Simulator |
| `health_status` | Status | str | Health status label | 'Excellent', 'Good', 'Fair', 'Poor' | pages/7_EVM_Simulator |

---

## Performance Interpretation Guide

### Cost Performance Index (CPI)
- **CPI > 1.0**: Under budget (good)
- **CPI = 1.0**: On budget
- **CPI < 1.0**: Over budget (concern)

### Schedule Performance Index (SPI / SPIe)
- **SPI > 1.0**: Ahead of schedule (good)
- **SPI = 1.0**: On schedule
- **SPI < 1.0**: Behind schedule (concern)

### Cost Variance (CV)
- **CV > 0**: Under budget
- **CV = 0**: On budget
- **CV < 0**: Over budget

### Schedule Variance (SV)
- **SV > 0**: Ahead of schedule
- **SV = 0**: On schedule
- **SV < 0**: Behind schedule

### To-Complete Performance Index (TCPI)
- **TCPI ≤ 1.0**: Achievable with current performance
- **1.0 < TCPI ≤ 1.2**: Requires improvement (challenging)
- **TCPI > 1.2**: Requires significant improvement (very difficult)

---

## Data Types Reference

| Type | Python Type | Description | Example |
|------|-------------|-------------|---------|
| float | `float` | Floating-point number | `1000.50` |
| int | `int` | Integer number | `12` |
| str | `str` | String/text | `"Project Alpha"` |
| bool | `bool` | Boolean true/false | `True`, `False` |
| datetime | `datetime.datetime` | Date and time | `datetime(2025, 1, 1)` |
| dict | `dict` | Dictionary/object | `{'key': 'value'}` |
| list | `list` | Array/list | `[1, 2, 3]` |

---

## Column Mapping Reference

For CSV imports, use these standard column names:

| Standard Column | Alternative Names | Required | Default |
|----------------|-------------------|----------|---------|
| `Project ID` | `pid`, `ID`, `project_id` | ✅ Yes | - |
| `Project` | `Project Name`, `Name`, `project_name` | ✅ Yes | - |
| `Organization` | `Org`, `org` | ✅ Yes | - |
| `Project Manager` | `PM`, `pm`, `Manager` | ✅ Yes | - |
| `BAC` | `Budget`, `budget` | ✅ Yes | - |
| `AC` | `Actual Cost`, `actual_cost` | ✅ Yes | - |
| `Plan Start` | `Start Date`, `plan_start` | ✅ Yes | - |
| `Plan Finish` | `Finish Date`, `plan_finish` | ✅ Yes | - |
| `PV` | `Planned Value`, `Manual_PV` | ❌ No | Calculated |
| `EV` | `Earned Value`, `Manual_EV` | ❌ No | Calculated |

---

## Notes

1. **Null/NA Handling**: Division by zero returns `float('nan')` for display as 'N/A'
2. **Infinity Values**: EAC/ETC may be `float('inf')` when AC=0
3. **Constraint**: Likely Duration (LD) is capped at 2.5 × Original Duration (OD)
4. **Date Formats**: Supports DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, Excel ordinals
5. **Currency**: All financial values in millions unless otherwise specified

---

**End of Data Dictionary**
For questions or clarifications, refer to: `core/evm_engine.py` (source of truth)
