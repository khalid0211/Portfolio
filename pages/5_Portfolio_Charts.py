from typing import List, Dict

from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Portfolio Gantt Chart",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {
    "Progress": "#40b57b",  # lighter green
    "Planned": "#4389d1",   # lighter blue
    "Predicted": "#6366f1", # purple for predicted completion
    "Overrun": "#d55454"    # lighter red
}

YEAR_LINE_COLOR = "#374151"  # dark grey for year boundaries
QUARTER_LINE_COLOR = "#9CA3AF"  # light grey for quarter boundaries

PERIOD_OPTIONS = {
    "Month": {"dtick": "M1", "delta": pd.Timedelta(days=30)},
    "Quarter": {"dtick": "M3", "delta": pd.Timedelta(days=91)},
    "Year": {"dtick": "M12", "delta": pd.Timedelta(days=365)}
}


def load_portfolio_dataframe() -> pd.DataFrame | None:
    """Return the latest batch results DataFrame if available."""
    df = None
    if hasattr(st.session_state, "batch_results") and st.session_state.batch_results is not None:
        df = st.session_state.batch_results.copy()
    elif hasattr(st.session_state, "dashboard_data") and st.session_state.dashboard_data is not None:
        # dashboard_data is formatted for display (strings), but use as last resort
        df = st.session_state.dashboard_data.copy()

    if df is None or df.empty:
        return None

    if "error" in df.columns:
        df = df[df["error"].isna()]

    if df.empty:
        return None

    return df


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Convert series to datetime with proper handling of different formats."""
    # First try standard conversion
    converted = pd.to_datetime(series, errors="coerce")

    # For any that failed, try specific date formats that EVM engine uses
    mask = pd.isna(converted) & pd.notna(series)
    if mask.any():
        # Try the EVM engine format: 'dd/mm/yyyy'
        for idx in series[mask].index:
            try:
                if isinstance(series[idx], str) and series[idx].strip():
                    # Try specific format used by EVM engine
                    converted.loc[idx] = pd.to_datetime(series[idx], format='%d/%m/%Y', errors='coerce')
                    if pd.isna(converted.loc[idx]):
                        # Try other common formats
                        converted.loc[idx] = pd.to_datetime(series[idx], format='%Y-%m-%d', errors='coerce')
            except:
                continue

    return converted


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_valid_finite_number(x):
    """Check if a number is valid and finite."""
    try:
        return pd.notna(x) and np.isfinite(float(x)) and not pd.isna(x)
    except (ValueError, TypeError, OverflowError):
        return False


def format_currency(amount: float, symbol: str = '$', postfix: str = "", decimals: int = 2, thousands: bool = False) -> str:
    """Enhanced currency formatting with comma separators and postfix options."""
    if not is_valid_finite_number(amount):
        return "—"

    # Handle thousands parameter (for cash flow chart compatibility)
    if thousands:
        formatted_amount = f"{float(amount)/1000:.0f}K"
        if postfix:
            return f"{symbol} {formatted_amount} {postfix}"
        else:
            return f"{symbol} {formatted_amount}"

    # Format with comma separators and specified decimal places
    formatted_amount = f"{float(amount):,.{decimals}f}"

    # Map postfix to abbreviations
    postfix_map = {
        "Thousand": "K",
        "Million": "M",
        "Billion": "B"
    }

    if postfix in postfix_map:
        return f"{symbol}{formatted_amount} {postfix_map[postfix]}"
    elif postfix:
        return f"{symbol}{formatted_amount} {postfix}"
    else:
        return f"{symbol}{formatted_amount}"



def apply_filters(df: pd.DataFrame, start_col: str = None, finish_col: str = None) -> pd.DataFrame:
    """Render filter widgets and return the filtered DataFrame with control values."""
    organizations = sorted({str(org) for org in df.get("organization", pd.Series()).dropna() if str(org).strip()})

    numeric_bac = df["bac"].dropna().astype(float) if "bac" in df.columns else pd.Series(dtype=float)
    min_budget = float(numeric_bac.min()) if not numeric_bac.empty else 0.0
    max_budget = float(numeric_bac.max()) if not numeric_bac.empty else min_budget

    numeric_od = df["original_duration_months"].dropna().astype(float) if "original_duration_months" in df.columns else pd.Series(dtype=float)
    min_od = float(numeric_od.min()) if not numeric_od.empty else 0.0
    max_od = float(numeric_od.max()) if not numeric_od.empty else min_od

    min_start = df[start_col].min() if start_col and start_col in df.columns else pd.NaT
    max_start = df[start_col].max() if start_col and start_col in df.columns else pd.NaT
    min_finish = df[finish_col].min() if finish_col and finish_col in df.columns else pd.NaT
    max_finish = df[finish_col].max() if finish_col and finish_col in df.columns else pd.NaT

    st.markdown("### 🔍 Filters")

    # 1. Organization (single line)
    org_toggle = st.toggle("Filter by Organization", value=False, key="gantt_org_toggle")
    if org_toggle:
        org_selection = st.multiselect(
            "Organization",
            options=organizations,
            default=organizations,
            placeholder="Select organization(s)" if organizations else "No organizations available"
        ) if organizations else []
    else:
        org_selection = organizations

    # 3. Plan Start Date (single line)
    col1, col2 = st.columns([1, 1])
    with col1:
        plan_start_later_toggle = st.toggle("Plan Start Later Than", value=False, key="gantt_plan_start_later_toggle") if pd.notna(min_start) else False
        if plan_start_later_toggle and pd.notna(min_start):
            plan_start_later_value = st.date_input(
                "Plan Start Later Than",
                value=min_start.date(),
                min_value=min_start.date(),
                max_value=max_start.date() if pd.notna(max_start) else min_start.date(),
                key="gantt_plan_start_later_value"
            )
        else:
            plan_start_later_value = None
    with col2:
        plan_start_earlier_toggle = st.toggle("Plan Start Earlier Than", value=False, key="gantt_plan_start_earlier_toggle") if pd.notna(max_start) else False
        if plan_start_earlier_toggle and pd.notna(max_start):
            plan_start_earlier_value = st.date_input(
                "Plan Start Earlier Than",
                value=max_start.date(),
                min_value=min_start.date() if pd.notna(min_start) else max_start.date(),
                max_value=max_start.date(),
                key="gantt_plan_start_earlier_value"
            )
        else:
            plan_start_earlier_value = None

    # 4. Plan Finish Date (single line)
    col1, col2 = st.columns([1, 1])
    with col1:
        plan_finish_later_toggle = st.toggle("Plan Finish Later Than", value=False, key="gantt_plan_finish_later_toggle") if pd.notna(min_finish) else False
        if plan_finish_later_toggle and pd.notna(min_finish):
            plan_finish_later_value = st.date_input(
                "Plan Finish Later Than",
                value=min_finish.date(),
                min_value=min_finish.date(),
                max_value=max_finish.date() if pd.notna(max_finish) else min_finish.date(),
                key="gantt_plan_finish_later_value"
            )
        else:
            plan_finish_later_value = None
    with col2:
        plan_finish_earlier_toggle = st.toggle("Plan Finish Earlier Than", value=False, key="gantt_plan_finish_earlier_toggle") if pd.notna(max_finish) else False
        if plan_finish_earlier_toggle and pd.notna(max_finish):
            plan_finish_earlier_value = st.date_input(
                "Plan Finish Earlier Than",
                value=max_finish.date(),
                min_value=min_finish.date() if pd.notna(min_finish) else max_finish.date(),
                max_value=max_finish.date(),
                key="gantt_plan_finish_earlier_value"
            )
        else:
            plan_finish_earlier_value = None

    # 5. Original Duration (single line)
    col1, col2 = st.columns([1, 1])
    with col1:
        od_min_toggle = st.toggle("Set Min OD", value=False, key="gantt_od_min_toggle", help="Filter by minimum Original Duration (months)")
        if od_min_toggle:
            od_min_value = st.number_input(
                "Min OD (months)",
                value=min_od,
                step=max(1.0, (max_od - min_od) / 10) if max_od > min_od else 1.0,
                min_value=0.0,
                key="gantt_od_min_value"
            )
        else:
            od_min_value = min_od
    with col2:
        od_max_toggle = st.toggle("Set Max OD", value=False, key="gantt_od_max_toggle", help="Filter by maximum Original Duration (months)")
        if od_max_toggle:
            od_max_value = st.number_input(
                "Max OD (months)",
                value=max_od,
                step=max(1.0, (max_od - min_od) / 10) if max_od > min_od else 1.0,
                min_value=0.0,
                key="gantt_od_max_value"
            )
        else:
            od_max_value = max_od

    # 6. Budget (single line)
    col1, col2 = st.columns([1, 1])
    with col1:
        min_budget_toggle = st.toggle("Set Min Budget", value=False, key="gantt_min_budget_toggle")
        if min_budget_toggle:
            min_budget_value = st.number_input(
                "Min Budget",
                value=min_budget,
                step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                min_value=0.0,
                key="gantt_min_budget_value"
            )
        else:
            min_budget_value = min_budget
    with col2:
        max_budget_toggle = st.toggle("Set Max Budget", value=False, key="gantt_max_budget_toggle")
        if max_budget_toggle:
            max_budget_value = st.number_input(
                "Max Budget",
                value=max_budget,
                step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                min_value=0.0,
                key="gantt_max_budget_value"
            )
        else:
            max_budget_value = max_budget

    # Apply filters
    filtered = df.copy()

    # Organization filter
    if org_toggle and organizations and org_selection:
        filtered = filtered[filtered["organization"].isin(org_selection)]

    # Plan Start date filters
    if plan_start_later_value is not None and start_col:
        plan_start_later_dt = pd.to_datetime(plan_start_later_value)
        filtered = filtered[filtered[start_col] >= plan_start_later_dt]

    if plan_start_earlier_value is not None and start_col:
        plan_start_earlier_dt = pd.to_datetime(plan_start_earlier_value)
        filtered = filtered[filtered[start_col] <= plan_start_earlier_dt]

    # Plan Finish date filters
    if plan_finish_later_value is not None and finish_col:
        plan_finish_later_dt = pd.to_datetime(plan_finish_later_value)
        filtered = filtered[filtered[finish_col] >= plan_finish_later_dt]

    if plan_finish_earlier_value is not None and finish_col:
        plan_finish_earlier_dt = pd.to_datetime(plan_finish_earlier_value)
        filtered = filtered[filtered[finish_col] <= plan_finish_earlier_dt]

    # Original Duration filters
    if "original_duration_months" in filtered.columns:
        if od_min_toggle:
            filtered = filtered[filtered["original_duration_months"] >= od_min_value]
        if od_max_toggle:
            filtered = filtered[filtered["original_duration_months"] <= od_max_value]

    # Budget filters
    if "bac" in filtered.columns:
        if min_budget_toggle:
            filtered = filtered[filtered["bac"] >= min_budget_value]
        if max_budget_toggle:
            filtered = filtered[filtered["bac"] <= max_budget_value]

    return filtered



def build_segments(df: pd.DataFrame, show_predicted: bool) -> List[Dict]:
    segments: List[Dict] = []
    for _, row in df.iterrows():
        start = row.get("plan_start", row.get("Plan Start"))
        finish = row.get("plan_finish", row.get("Plan Finish"))
        if pd.isna(start) or pd.isna(finish):
            continue

        project_id = str(row.get("project_id", row.get("Project ID", ""))) or "Unknown"
        project_name = row.get("project_name", row.get("Project Name", ""))
        organization = row.get("organization", row.get("Organization", ""))

        bac = row.get("bac", row.get("BAC", 0.0))
        ac = row.get("ac", row.get("AC", 0.0))
        earned_value = row.get("ev", row.get("EV", 0.0))
        cpi = row.get("cpi", row.get("CPI", 0.0))
        spi = row.get("spi", row.get("SPI", 0.0))
        actual_duration = row.get("actual_duration_months", 0.0)
        original_duration = row.get("original_duration_months", 0.0)

        # Calculate percentages
        percent_budget_used = (ac / bac * 100) if is_valid_finite_number(bac) and bac > 0 else 0.0
        percent_time_used = (actual_duration / original_duration * 100) if is_valid_finite_number(original_duration) and original_duration > 0 else 0.0
        percent_work_completed = (earned_value / bac * 100) if is_valid_finite_number(bac) and bac > 0 else 0.0

        # Get currency settings from session state
        currency_symbol = (
            getattr(st.session_state, 'dashboard_currency_symbol', None) or
            getattr(st.session_state, 'currency_symbol', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
        )
        currency_postfix = (
            getattr(st.session_state, 'dashboard_currency_postfix', None) or
            getattr(st.session_state, 'currency_postfix', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
        )

        # Format values for tooltip
        bac_formatted = format_currency(bac, currency_symbol, currency_postfix)
        plan_start_str = start.strftime('%Y-%m-%d') if pd.notna(start) else "N/A"
        plan_finish_str = finish.strftime('%Y-%m-%d') if pd.notna(finish) else "N/A"
        if pd.isna(bac) or bac <= 0:
            progress_ratio = 0.0
        else:
            progress_ratio = max(0.0, min(float(earned_value) / float(bac), 1.0))

        # Get OD (Original Duration) and LD (Likely Duration)
        od = original_duration
        ld = row.get("ld", row.get("likely_duration", 0.0))

        # Calculate dates based on OD, LD, and % Work Completed
        # Green segment: From Plan Start to (Plan Start + OD * % Work Completed)
        if is_valid_finite_number(od) and od > 0 and is_valid_finite_number(percent_work_completed):
            # Convert percent to decimal (e.g., 45% -> 0.45)
            work_completed_decimal = percent_work_completed / 100.0
            # Calculate progress duration in days
            od_days = pd.Timedelta(days=od * 30.44)  # Convert months to days (avg 30.44 days/month)
            progress_duration = od_days * work_completed_decimal
            progress_end = start + progress_duration
        else:
            # Fallback: use plan finish if OD or % work completed is not available
            progress_end = start

        # Ensure progress_end is within bounds
        if progress_end < start:
            progress_end = start
        if progress_end > finish:
            progress_end = finish

        # Add Progress segment (Green)
        segments.append({
            "Task": project_id,
            "Start": start,
            "Finish": progress_end,
            "Segment": "Progress",
            "project_name": project_name,
            "organization": organization,
            "bac_formatted": bac_formatted,
            "plan_start": plan_start_str,
            "plan_finish": plan_finish_str,
            "cpi": cpi,
            "spi": spi,
            "percent_budget_used": percent_budget_used,
            "percent_time_used": percent_time_used,
            "percent_work_completed": percent_work_completed
        })

        # Handle the remaining timeline based on view mode (Plan vs Predicted)
        if show_predicted and is_valid_finite_number(od) and od > 0 and is_valid_finite_number(ld) and ld > 0:
            # Predicted View: Use OD and LD to calculate segments

            # Blue segment: from (Plan Start + OD * % Work Completed) to Plan Finish
            # This represents the original planned remaining work
            if progress_end < finish:
                segments.append({
                    "Task": project_id,
                    "Start": progress_end,
                    "Finish": finish,
                    "Segment": "Planned",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })

            # Red segment: (if LD > OD) From Plan Start + OD to Plan Start + LD
            # This represents the schedule overrun
            if ld > od:
                od_end = start + pd.Timedelta(days=od * 30.44)
                ld_end = start + pd.Timedelta(days=ld * 30.44)

                segments.append({
                    "Task": project_id,
                    "Start": od_end,
                    "Finish": ld_end,
                    "Segment": "Overrun",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })
        else:
            # Plan View: Show planned completion (simple two-segment view)
            if progress_end < finish:
                segments.append({
                    "Task": project_id,
                    "Start": progress_end,
                    "Finish": finish,
                    "Segment": "Planned",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })

    return segments


def render_cash_flow_chart(filtered_df: pd.DataFrame, start_col: str = None, finish_col: str = None) -> None:
    """Render the cash flow chart with all controls and visualizations."""
    if len(filtered_df) > 0:
        # Get currency settings from session state
        currency_symbol = (
            getattr(st.session_state, 'dashboard_currency_symbol', None) or
            getattr(st.session_state, 'currency_symbol', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
        )
        currency_postfix = (
            getattr(st.session_state, 'dashboard_currency_postfix', None) or
            getattr(st.session_state, 'currency_postfix', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
        )

        # Detect possible date columns
        date_columns = []
        for col in filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['start', 'begin', 'finish', 'end', 'complete', 'date']):
                date_columns.append(col)

        # Look for specific date column patterns
        start_date_col = start_col
        plan_finish_col = finish_col
        likely_finish_col = None
        expected_finish_col = None

        for col in date_columns:
            col_lower = col.lower()
            if 'expected' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                expected_finish_col = col
            elif 'likely' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                likely_finish_col = col

        if start_date_col and (plan_finish_col or likely_finish_col or expected_finish_col):
            # Validate expected finish dates (max 4 years from plan finish)
            valid_expected_finish_col = None
            expected_date_info = ""
            if expected_finish_col and plan_finish_col:
                try:
                    # Parse dates for validation
                    temp_df = filtered_df.copy()
                    temp_df[plan_finish_col] = pd.to_datetime(temp_df[plan_finish_col], errors='coerce')
                    temp_df[expected_finish_col] = pd.to_datetime(temp_df[expected_finish_col], errors='coerce')

                    # Check if expected dates are within 4 years of plan dates
                    valid_rows = temp_df.dropna(subset=[plan_finish_col, expected_finish_col])
                    if len(valid_rows) > 0:
                        date_diff_years = (valid_rows[expected_finish_col] - valid_rows[plan_finish_col]).dt.days / 365.25
                        valid_dates = (date_diff_years <= 4) & (date_diff_years >= -1)
                        if valid_dates.all():
                            valid_expected_finish_col = expected_finish_col
                            expected_date_info = f"✅ Expected dates validated ({len(valid_rows)} projects)"
                        else:
                            invalid_count = (~valid_dates).sum()
                            expected_date_info = f"⚠️ Expected dates excluded ({invalid_count} projects exceed 4-year limit)"
                    else:
                        expected_date_info = "⚠️ No valid expected dates found"
                except Exception as e:
                    expected_date_info = f"❌ Expected date validation failed: {str(e)}"

            # Controls for cash flow chart
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                time_period = st.selectbox("Time Period",
                                           options=["Month", "Quarter", "FY"],
                                           index=0,
                                           key="cash_flow_time_period",
                                           help="Month: Monthly view, Quarter: Quarterly view, FY: Financial Year (July-June)")

            with col2:
                # Determine available finish date options (prioritize expected over likely)
                finish_options = []
                if plan_finish_col:
                    finish_options.append("Plan Finish")
                if valid_expected_finish_col:
                    finish_options.append("Expected Finish")
                elif likely_finish_col:  # Only show likely if expected is not available
                    finish_options.append("Likely Finish")

                if len(finish_options) > 1:
                    finish_date_choice = st.selectbox("Finish Date Type", finish_options, key="cash_flow_finish_type")
                else:
                    finish_date_choice = finish_options[0] if finish_options else "Plan Finish"
                    st.write(f"**Finish Date:** {finish_date_choice}")

            with col3:
                cash_flow_type = st.radio("Cash Flow Type",
                                          options=["Plan", "Predicted", "Actual", "All"],
                                          index=0,
                                          key="cash_flow_type",
                                          help="Plan: BAC + OD, Predicted: BAC + LD, Actual: AC from Start to Data Date, All: Line chart comparing all three")

            with col4:
                st.write("**Configuration:**")
                st.write(f"📊 {time_period}")
                if cash_flow_type == "Plan":
                    st.write("💰 BAC/OD")
                elif cash_flow_type == "Predicted":
                    st.write("💰 BAC/LD")
                elif cash_flow_type == "Actual":
                    st.write("💰 AC to Data Date")
                else:  # All
                    st.write("💰 Plan vs Predicted vs Actual")

            # Show expected date validation info if available
            if expected_date_info:
                st.info(expected_date_info)

            # Select finish date column based on choice
            if finish_date_choice == "Expected Finish" and valid_expected_finish_col:
                finish_col_selected = valid_expected_finish_col
            elif finish_date_choice == "Likely Finish" and likely_finish_col:
                finish_col_selected = likely_finish_col
            else:
                finish_col_selected = plan_finish_col

            if finish_col_selected:
                try:
                    # Convert date columns to datetime
                    df_cash = filtered_df.copy()

                    # Parse dates with error handling
                    df_cash[start_date_col] = pd.to_datetime(df_cash[start_date_col], errors='coerce')
                    df_cash[finish_col_selected] = pd.to_datetime(df_cash[finish_col_selected], errors='coerce')

                    # Remove rows with invalid dates
                    df_cash = df_cash.dropna(subset=[start_date_col, finish_col_selected])

                    if len(df_cash) > 0:
                        def get_financial_year(date):
                            """Get financial year string for a date (FY starts July 1st)"""
                            if date.month >= 7:  # July onwards = start of FY
                                return f"FY{date.year + 1}"  # e.g., July 2024 = FY2025
                            else:  # Jan-June = end of previous FY
                                return f"FY{date.year}"  # e.g., March 2024 = FY2024

                        def get_period_key(date, time_period):
                            """Get period key based on time period selection"""
                            if time_period == "Month":
                                return date.strftime("%Y-%b")
                            elif time_period == "Quarter":
                                quarter = f"{date.year}-Q{((date.month - 1) // 3) + 1}"
                                return quarter
                            else:  # FY
                                return get_financial_year(date)

                        def calculate_cash_flow_for_scenario(df_cash, scenario_type):
                            """Calculate cash flow for Plan (BAC/OD) or Predicted (BAC/LD) scenario"""
                            cash_flow_data = []

                            for idx, row in df_cash.iterrows():
                                start_date = row[start_date_col]

                                if pd.notna(start_date):
                                    # Ensure start_date is a proper datetime object
                                    if not isinstance(start_date, pd.Timestamp):
                                        start_date = pd.to_datetime(start_date, errors='coerce')
                                        if pd.isna(start_date):
                                            continue

                                    # Always use BAC (Budget) for both scenarios
                                    budget = row.get('bac', row.get('Budget', 0))  # Try 'bac' first, then 'Budget'

                                    if scenario_type == "Plan":
                                        # Use Original Duration (OD)
                                        if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                            duration_months = max(1, row['original_duration_months'])
                                        else:
                                            # Fallback: calculate from plan start to plan finish
                                            plan_finish = row.get('plan_finish', row.get('Plan Finish'))
                                            if pd.notna(plan_finish):
                                                plan_finish_date = pd.to_datetime(plan_finish, errors='coerce')
                                                if pd.notna(plan_finish_date):
                                                    duration_months = max(1, (plan_finish_date - start_date).days / 30.44)
                                                else:
                                                    continue
                                            else:
                                                continue
                                    else:  # Predicted
                                        # Use Likely Duration (LD) with cap check
                                        if 'ld' in row and pd.notna(row.get('ld')):
                                            ld = row['ld']
                                            # Get OD for cap calculation
                                            if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                                od = row['original_duration_months']
                                            else:
                                                # Fallback: calculate OD from plan dates
                                                plan_finish = row.get('plan_finish', row.get('Plan Finish'))
                                                if pd.notna(plan_finish):
                                                    plan_finish_date = pd.to_datetime(plan_finish, errors='coerce')
                                                    if pd.notna(plan_finish_date):
                                                        od = max(1, (plan_finish_date - start_date).days / 30.44)
                                                    else:
                                                        od = 12  # Default fallback
                                                else:
                                                    od = 12  # Default fallback

                                            # Cap LD to prevent timestamp overflow: min(LD, OD+48)
                                            duration_months = max(1, min(ld, od + 48))
                                        else:
                                            continue

                                    if budget > 0 and duration_months > 0:
                                        # Calculate monthly cash flow: BAC/Duration
                                        monthly_cash_flow = budget / duration_months

                                        # Generate monthly cash flow from plan start for duration months
                                        current_date = start_date.replace(day=1)

                                        for month in range(int(duration_months)):
                                            period_key = get_period_key(current_date, time_period)

                                            cash_flow_data.append({
                                                'Period': period_key,
                                                'Cash_Flow': monthly_cash_flow,
                                                'Project': row.get('project_name', row.get('Project Name', 'Unknown')),
                                                'Date': current_date,
                                                'Scenario': scenario_type
                                            })

                                            # Move to next month
                                            if current_date.month == 12:
                                                current_date = current_date.replace(year=current_date.year + 1, month=1)
                                            else:
                                                current_date = current_date.replace(month=current_date.month + 1)
                            return cash_flow_data

                        def calculate_actual_cash_flow(df_cash):
                            """Calculate actual cash flow using AC/AD from Plan Start for AD months"""
                            cash_flow_data = []

                            for idx, row in df_cash.iterrows():
                                start_date = row[start_date_col]

                                if pd.notna(start_date):
                                    # Ensure start_date is a proper datetime object
                                    if not isinstance(start_date, pd.Timestamp):
                                        start_date = pd.to_datetime(start_date, errors='coerce')
                                        if pd.isna(start_date):
                                            continue

                                    # Get AC (Actual Cost)
                                    ac = row.get('ac', row.get('AC', row.get('Actual Cost', 0)))

                                    # Get AD (Actual Duration)
                                    if 'actual_duration_months' in row and pd.notna(row.get('actual_duration_months')):
                                        duration_months = max(1, row['actual_duration_months'])
                                    else:
                                        # Fallback: calculate from plan start to data date
                                        data_date_col = None
                                        for col in df_cash.columns:
                                            if 'data' in col.lower() and 'date' in col.lower():
                                                data_date_col = col
                                                break

                                        if data_date_col:
                                            data_date = row.get(data_date_col)
                                            if pd.notna(data_date):
                                                data_date_parsed = pd.to_datetime(data_date, errors='coerce')
                                                if pd.notna(data_date_parsed):
                                                    duration_months = max(1, (data_date_parsed - start_date).days / 30.44)
                                                else:
                                                    continue
                                            else:
                                                continue
                                        else:
                                            continue

                                    if ac > 0 and duration_months > 0:
                                        # Calculate monthly cash flow: AC/AD
                                        monthly_cash_flow = ac / duration_months

                                        # Generate monthly cash flow from plan start for AD months
                                        current_date = start_date.replace(day=1)

                                        for month in range(int(duration_months)):
                                            period_key = get_period_key(current_date, time_period)

                                            cash_flow_data.append({
                                                'Period': period_key,
                                                'Cash_Flow': monthly_cash_flow,
                                                'Project': row.get('project_name', row.get('Project Name', 'Unknown')),
                                                'Date': current_date,
                                                'Scenario': 'Actual'
                                            })

                                            # Move to next month
                                            if current_date.month == 12:
                                                current_date = current_date.replace(year=current_date.year + 1, month=1)
                                            else:
                                                current_date = current_date.replace(month=current_date.month + 1)

                            return cash_flow_data

                        # Calculate cash flow based on selected type
                        if cash_flow_type == "Plan":
                            cash_flow_data = calculate_cash_flow_for_scenario(df_cash, "Plan")
                        elif cash_flow_type == "Predicted":
                            cash_flow_data = calculate_cash_flow_for_scenario(df_cash, "Predicted")
                        elif cash_flow_type == "Actual":
                            cash_flow_data = calculate_actual_cash_flow(df_cash)
                        else:  # All
                            plan_data = calculate_cash_flow_for_scenario(df_cash, "Plan")
                            predicted_data = calculate_cash_flow_for_scenario(df_cash, "Predicted")
                            actual_data = calculate_actual_cash_flow(df_cash)
                            # Ensure all are lists before concatenating
                            if not isinstance(plan_data, list):
                                plan_data = []
                            if not isinstance(predicted_data, list):
                                predicted_data = []
                            if not isinstance(actual_data, list):
                                actual_data = []
                            cash_flow_data = plan_data + predicted_data + actual_data

                        if cash_flow_data:
                            cash_df = pd.DataFrame(cash_flow_data)

                            # Ensure Cash_Flow column is numeric and handle any data type issues
                            try:
                                cash_df['Cash_Flow'] = pd.to_numeric(cash_df['Cash_Flow'], errors='coerce')
                                # Remove any rows with invalid cash flow values
                                cash_df = cash_df.dropna(subset=['Cash_Flow'])
                                cash_df = cash_df[cash_df['Cash_Flow'].notna() & (cash_df['Cash_Flow'] != float('inf')) & (cash_df['Cash_Flow'] != float('-inf'))]

                                # Check if we have valid data after cleaning
                                if cash_df.empty:
                                    st.warning("No valid cash flow data available after processing.")
                                    cash_df = pd.DataFrame()
                            except Exception as e:
                                st.error(f"Error processing cash flow data types: {str(e)}")
                                cash_df = pd.DataFrame()  # Empty dataframe to prevent further errors

                            def get_sort_key(period_str, time_period):
                                """Generate sort key for different time periods"""
                                if time_period == "Month":
                                    # For "2024-Jan" format
                                    try:
                                        year, month_abbr = period_str.split('-')
                                        month_num = {
                                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                        }.get(month_abbr, 1)
                                        return (int(year), month_num)
                                    except:
                                        return (2000, 1)
                                elif time_period == "Quarter":
                                    # For "2024-Q1" format
                                    try:
                                        year, quarter = period_str.split('-')
                                        quarter_num = int(quarter[1:])  # Extract number from Q1, Q2, etc.
                                        return (int(year), quarter_num)
                                    except:
                                        return (2000, 1)
                                else:  # FY
                                    # For "FY2024" format
                                    try:
                                        return (int(period_str[2:]), 1)  # Extract year from FY2024
                                    except:
                                        return (2000, 1)

                            # Only proceed if we have valid cash flow data
                            if not cash_df.empty and len(cash_df) > 0:
                                if cash_flow_type == "All":
                                    # For All option, create line chart with three series
                                    period_cash_flow = cash_df.groupby(['Period', 'Scenario'])['Cash_Flow'].sum().reset_index()
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    # For Actual in all views, remove the last point to create horizontal line effect
                                    actual_mask = period_cash_flow['Scenario'] == 'Actual'
                                    actual_data = period_cash_flow[actual_mask]
                                    if len(actual_data) > 1:
                                        # Remove the last point of Actual data
                                        last_actual_index = actual_data.index[-1]
                                        period_cash_flow = period_cash_flow.drop(last_actual_index)

                                    # Add final points for Plan and Predicted scenarios
                                    for scenario in period_cash_flow['Scenario'].unique():
                                        if scenario == 'Actual':
                                            # Skip Actual - no additional points needed
                                            continue

                                        scenario_data = period_cash_flow[period_cash_flow['Scenario'] == scenario]
                                        if not scenario_data.empty:
                                            last_period = scenario_data.iloc[-1]['Period']

                                            # For Plan and Predicted, add zero point at next period
                                            # Generate next period based on time_period type
                                            if time_period == "Month":
                                                # For Month: increment month (format: "2024-Jan")
                                                year, month_abbr = last_period.split('-')
                                                month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}.get(month_abbr, 1)
                                                if month_num == 12:
                                                    next_period = f"{int(year) + 1}-Jan"
                                                else:
                                                    # month_num is 1-12, so we need index month_num (which gives us the next month)
                                                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                                    next_month = month_names[month_num]  # month_num is the current month (1-12), so index month_num gives next month
                                                    next_period = f"{year}-{next_month}"
                                            elif time_period == "Quarter":
                                                # For Quarter: increment quarter (format: "2024-Q1")
                                                year, quarter = last_period.split('-')
                                                quarter_num = int(quarter[1:])
                                                if quarter_num == 4:
                                                    next_period = f"{int(year) + 1}-Q1"
                                                else:
                                                    next_period = f"{year}-Q{quarter_num + 1}"
                                            else:  # FY
                                                # For FY: increment year
                                                fy_year = int(last_period[2:])
                                                next_period = f"FY{fy_year + 1}"

                                            # Add the zero point for Plan and Predicted
                                            final_row = pd.DataFrame({
                                                'Period': [next_period],
                                                'Scenario': [scenario],
                                                'Cash_Flow': [0]
                                            })
                                            period_cash_flow = pd.concat([period_cash_flow, final_row], ignore_index=True)

                                    # Re-sort after adding zero points
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    chart_title = f"Portfolio Cash Flow Comparison (Plan vs Predicted vs Actual) - {time_period} View"

                                    # Define color mapping for scenarios
                                    color_map = {
                                        'Plan': 'blue',
                                        'Predicted': 'orange',
                                        'Actual': 'green'
                                    }

                                    fig_cash_flow = px.line(
                                        period_cash_flow,
                                        x='Period',
                                        y='Cash_Flow',
                                        color='Scenario',
                                        title=chart_title,
                                        labels={
                                            'Cash_Flow': f'Cash Flow ({currency_symbol})',
                                            'Period': 'Period',
                                            'Scenario': 'Scenario'
                                        },
                                        line_shape='spline',  # Makes the line smooth
                                        markers=True,
                                        color_discrete_map=color_map
                                    )

                                    # Reduce marker size to make them less distracting
                                    fig_cash_flow.update_traces(marker=dict(size=4))

                                    # Add vertical line at data date
                                    # Find the data date column and calculate average data date
                                    data_date_col = None
                                    for col in df_cash.columns:
                                        if 'data' in col.lower() and 'date' in col.lower():
                                            data_date_col = col
                                            break

                                    if data_date_col:
                                        # Get the most common data date (or average)
                                        valid_data_dates = df_cash[data_date_col].dropna()
                                        if len(valid_data_dates) > 0:
                                            # Use the most recent data date
                                            data_date_value = pd.to_datetime(valid_data_dates).max()
                                            if pd.notna(data_date_value):
                                                # Get the period key for the data date
                                                data_date_period = get_period_key(data_date_value, time_period)

                                                # Add vertical line at data date using add_shape (works with categorical x-axis)
                                                fig_cash_flow.add_shape(
                                                    type="line",
                                                    x0=data_date_period,
                                                    x1=data_date_period,
                                                    y0=0,
                                                    y1=1,
                                                    yref="paper",
                                                    line=dict(
                                                        color="red",
                                                        width=2,
                                                        dash="dash"
                                                    )
                                                )
                                                # Add annotation for the line
                                                fig_cash_flow.add_annotation(
                                                    x=data_date_period,
                                                    y=1,
                                                    yref="paper",
                                                    text="Data Date",
                                                    showarrow=False,
                                                    yshift=10,
                                                    font=dict(color="red")
                                                )
                                else:
                                    # For Plan, Predicted, or Actual, create bar chart
                                    period_cash_flow = cash_df.groupby('Period')['Cash_Flow'].sum().reset_index()
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    if cash_flow_type == 'Plan':
                                        scenario_label = "(Plan: BAC/OD)"
                                    elif cash_flow_type == 'Predicted':
                                        scenario_label = "(Predicted: BAC/LD)"
                                    else:  # Actual
                                        scenario_label = "(Actual: AC to Data Date)"
                                    chart_title = f"Portfolio Cash Flow {scenario_label} - {time_period} View"

                                    # Choose color scale based on cash flow type
                                    if cash_flow_type == 'Actual':
                                        color_scale = 'greens'
                                    else:
                                        color_scale = 'blues'

                                    fig_cash_flow = px.bar(
                                        period_cash_flow,
                                        x='Period',
                                        y='Cash_Flow',
                                        title=chart_title,
                                        labels={
                                            'Cash_Flow': f'Cash Flow ({currency_symbol})',
                                            'Period': time_period
                                        },
                                        color='Cash_Flow',
                                        color_continuous_scale=color_scale
                                    )

                                # Update layout for better visualization
                                fig_cash_flow.update_layout(
                                    height=500,
                                    showlegend=True if cash_flow_type == "All" else False,
                                    xaxis=dict(
                                        title=time_period,
                                        tickangle=45
                                    ),
                                    yaxis=dict(
                                        title=f'Cash Flow ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                        tickformat=',.0f'
                                    ),
                                    coloraxis_showscale=False if cash_flow_type != "All" else True
                                )

                                # Update traces for better appearance
                                if cash_flow_type != "All":
                                    fig_cash_flow.update_traces(
                                        texttemplate='%{y:,.0f}',
                                        textposition='outside'
                                    )

                                st.plotly_chart(fig_cash_flow, use_container_width=True)

                                # Display summary metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    if cash_flow_type != "Both":
                                        try:
                                            avg_monthly = period_cash_flow['Cash_Flow'].mean()
                                            if pd.isna(avg_monthly) or avg_monthly == float('inf') or avg_monthly == float('-inf'):
                                                st.metric("Average per Period", "N/A")
                                            else:
                                                st.metric("Average per Period", format_currency(avg_monthly, currency_symbol, currency_postfix, thousands=False))
                                        except Exception as e:
                                            st.metric("Average per Period", "Error")
                                            st.error(f"Error calculating average: {str(e)}")

                                with col2:
                                    if cash_flow_type != "Both":
                                        try:
                                            if len(period_cash_flow) > 0 and not period_cash_flow['Cash_Flow'].empty:
                                                peak_amount = period_cash_flow['Cash_Flow'].max()
                                                peak_period = period_cash_flow.loc[period_cash_flow['Cash_Flow'].idxmax(), 'Period']
                                                if pd.isna(peak_amount) or peak_amount == float('inf') or peak_amount == float('-inf'):
                                                    st.metric("Peak Period", "N/A")
                                                else:
                                                    st.metric(f"Peak Period: {peak_period}", format_currency(peak_amount, currency_symbol, currency_postfix, thousands=False))
                                            else:
                                                st.metric("Peak Period", "No Data")
                                        except Exception as e:
                                            st.metric("Peak Period", "Error")
                                            st.error(f"Error calculating peak: {str(e)}")

                                # Show detailed data table
                                with st.expander("📊 Detailed Cash Flow Data", expanded=False):
                                    if cash_flow_type == "All":
                                        # Show comparison table for all three scenarios
                                        # Create pivot table with numeric data first
                                        pivot_df = period_cash_flow.pivot_table(index='Period', columns='Scenario', values='Cash_Flow', aggfunc='sum', fill_value=0)

                                        # Reorder columns to: Plan, Actual, Predicted
                                        column_order = []
                                        if 'Plan' in pivot_df.columns:
                                            column_order.append('Plan')
                                        if 'Actual' in pivot_df.columns:
                                            column_order.append('Actual')
                                        if 'Predicted' in pivot_df.columns:
                                            column_order.append('Predicted')

                                        pivot_df = pivot_df[column_order]

                                        # Format the values for display
                                        for col in pivot_df.columns:
                                            pivot_df[col] = pivot_df[col].apply(
                                                lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False) if x != 0 else '—'
                                            )
                                        st.dataframe(pivot_df, width='stretch')
                                    elif cash_flow_type in ["Plan", "Predicted", "Actual"]:
                                        # Show single scenario table
                                        display_cash_flow = period_cash_flow[['Period', 'Cash_Flow']].copy()
                                        display_cash_flow['Cash_Flow'] = display_cash_flow['Cash_Flow'].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                                        st.dataframe(display_cash_flow, width='stretch')
                            else:
                                st.warning("No valid cash flow data could be generated from the selected projects.")

                        else:
                            st.warning("No valid cash flow data could be generated from the selected projects.")
                    else:
                        st.warning("No projects have valid start and finish dates.")

                except Exception as e:
                    st.error(f"Error processing cash flow data: {str(e)}")
                    st.info("Please check that date columns contain valid date formats.")
                    # Add detailed traceback for debugging
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("Required finish date column not found.")
        else:
            st.info("Cash flow chart requires start date and finish date columns. Available columns:")
            if date_columns:
                for col in date_columns:
                    st.write(f"• {col}")
            else:
                st.write("No date columns detected in the data.")
    else:
        st.info("No data available for cash flow analysis.")


def render_time_budget_performance(filtered_df: pd.DataFrame) -> None:
    """Render the Time/Budget Performance chart."""
    if len(filtered_df) > 0:
        st.markdown("### Time/Budget Performance Analysis")
        st.markdown("This chart shows each project's performance relative to time and budget, with reference curves for comparison.")

        # Calculate normalized values for each project
        performance_data = []

        # Get tier configuration
        tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
        default_tier_config = {
            'cutoff_points': [4000, 8000, 15000],
            'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
        }
        tier_colors = tier_config.get('colors', default_tier_config['colors'])
        tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
        cutoff_points = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])

        # Create color mapping for tiers
        tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

        # Function to determine tier based on budget
        def get_budget_tier(budget):
            """Determine tier based on budget and cutoff points."""
            if budget <= cutoff_points[0]:
                return tier_names[0]
            elif budget <= cutoff_points[1]:
                return tier_names[1]
            elif budget <= cutoff_points[2]:
                return tier_names[2]
            else:
                return tier_names[3]

        for _, project in filtered_df.iterrows():
            # Calculate % Time Used (AD/OD) - Actual Duration / Original Duration
            actual_duration = project.get('actual_duration_months', project.get('AD', project.get('Actual Duration', project.get('Actual Duration (months)', 0))))
            original_duration = project.get('original_duration_months', project.get('OD', project.get('Original Duration', project.get('Original Duration (months)', 0))))

            if pd.notna(actual_duration) and pd.notna(original_duration) and original_duration > 0:
                time_used_pct = actual_duration / original_duration
            else:
                continue  # Skip projects without duration data

            # Calculate % Budget Used (AC/BAC) - Actual Cost / Budget at Completion
            actual_cost = project.get('ac', project.get('AC', project.get('Actual Cost', 0)))
            budget = project.get('bac', project.get('BAC', project.get('Budget', 0)))

            if pd.notna(actual_cost) and pd.notna(budget) and budget > 0:
                budget_used_pct = actual_cost / budget
            else:
                continue  # Skip projects without budget/cost data

            # Determine tier based on budget (use budget_tier if available, otherwise calculate)
            if 'budget_tier' in project.index and pd.notna(project.get('budget_tier')):
                tier = project.get('budget_tier')
            else:
                tier = get_budget_tier(budget)

            color = tier_color_map.get(tier, '#888888')

            performance_data.append({
                'project_id': project.get('project_id', project.get('Project ID', 'Unknown')),
                'project_name': project.get('project_name', project.get('Project Name', 'Unknown')),
                'organization': project.get('organization', project.get('Organization', 'Unknown')),
                'time_used_pct': time_used_pct,
                'budget_used_pct': budget_used_pct,
                'tier': tier,
                'color': color,
                'bac': budget,
                'spi': project.get('spi', project.get('SPI', 0)),
                'cpi': project.get('cpi', project.get('CPI', 0))
            })

        if performance_data:
            # Calculate portfolio-level SPI and CPI for display
            spi_col = 'spi' if 'spi' in filtered_df.columns else 'SPI'
            cpi_col = 'cpi' if 'cpi' in filtered_df.columns else 'CPI'
            portfolio_spi = filtered_df[spi_col].mean() if spi_col in filtered_df.columns else 1.0
            portfolio_cpi = filtered_df[cpi_col].mean() if cpi_col in filtered_df.columns else 1.0

            # Calculate portfolio-level % time used and % budget used with weighted formulas
            # % time used = sum(AD*BAC)/sum(OD*BAC)
            # % budget used = sum(AC)/sum(BAC)
            total_ad_bac = 0
            total_od_bac = 0
            total_ac = 0
            total_bac = 0

            for _, project in filtered_df.iterrows():
                actual_duration = project.get('actual_duration_months', project.get('AD', project.get('Actual Duration', project.get('Actual Duration (months)', 0))))
                original_duration = project.get('original_duration_months', project.get('OD', project.get('Original Duration', project.get('Original Duration (months)', 0))))
                actual_cost = project.get('ac', project.get('AC', project.get('Actual Cost', 0)))
                budget = project.get('bac', project.get('BAC', project.get('Budget', 0)))

                if pd.notna(actual_duration) and pd.notna(original_duration) and pd.notna(budget) and budget > 0:
                    total_ad_bac += actual_duration * budget
                    total_od_bac += original_duration * budget

                if pd.notna(actual_cost) and pd.notna(budget):
                    total_ac += actual_cost
                    total_bac += budget

            portfolio_time_used = total_ad_bac / total_od_bac if total_od_bac > 0 else 1.0
            portfolio_budget_used = total_ac / total_bac if total_bac > 0 else 1.0

            # Create the plot using Plotly for interactivity

            fig = go.Figure()

            # Create normalized time array for reference curves
            T = np.linspace(0, 1, 101)

            # Define the performance curves (same as in project analysis)
            blue_curve = -0.794*T**3 + 0.632*T**2 + 1.162*T
            red_curve = -0.387*T**3 + 1.442*T**2 - 0.055*T

            # Plot the reference curves
            fig.add_trace(go.Scatter(
                x=T, y=blue_curve,
                mode='lines',
                name='Blue Curve (Good Performance)',
                line=dict(color='blue', width=2),
                opacity=0.7,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=T, y=red_curve,
                mode='lines',
                name='Red Curve (Poor Performance)',
                line=dict(color='red', width=2),
                opacity=0.7,
                hoverinfo='skip'
            ))

            # Plot individual projects colored by tier with tooltips
            # Get all unique tiers from the actual data
            actual_tiers = set([p['tier'] for p in performance_data])

            # Use actual tiers if they don't match configured tier names
            tiers_to_plot = tier_names if any(tier in tier_names for tier in actual_tiers) else list(actual_tiers)

            for tier in tiers_to_plot:
                tier_projects = [p for p in performance_data if p['tier'] == tier]
                if tier_projects:
                    x_vals = [p['time_used_pct'] for p in tier_projects]
                    y_vals = [p['budget_used_pct'] for p in tier_projects]
                    color = tier_color_map.get(tier, '#888888')  # Default gray if tier not in map

                    # Create custom hover text
                    hover_text = []
                    for p in tier_projects:
                        hover_text.append(
                            f"<b>{p['project_name']}</b><br>" +
                            f"Project ID: {p['project_id']}<br>" +
                            f"Organization: {p['organization']}<br>" +
                            f"Tier: {p['tier']}<br>" +
                            f"BAC: ${p['bac']:,.0f}<br>" +
                            f"SPI: {p['spi']:.2f}<br>" +
                            f"CPI: {p['cpi']:.2f}<br>" +
                            f"% Time Used: {p['time_used_pct'] * 100:.1f}%<br>" +
                            f"% Budget Used: {p['budget_used_pct'] * 100:.1f}%"
                        )

                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='markers',
                        name=tier,
                        marker=dict(
                            color=color,
                            size=10,
                            opacity=0.7,
                            line=dict(color='black', width=1)
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text
                    ))

            # Add portfolio overall performance as large yellow star
            portfolio_hover = (
                f"<b>Portfolio Overall</b><br>" +
                f"% Time Used: {portfolio_time_used * 100:.1f}%<br>" +
                f"% Budget Used: {portfolio_budget_used * 100:.1f}%<br>" +
                f"Average SPI: {portfolio_spi:.2f}<br>" +
                f"Average CPI: {portfolio_cpi:.2f}"
            )

            fig.add_trace(go.Scatter(
                x=[portfolio_time_used], y=[portfolio_budget_used],
                mode='markers',
                name=f'Portfolio Overall',
                marker=dict(
                    color='yellow',
                    size=20,
                    opacity=0.9,
                    symbol='star',
                    line=dict(color='black', width=3)
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[portfolio_hover]
            ))

            # Add reference lines
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Budget Baseline")
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Schedule Baseline")

            # Customize the layout
            fig.update_layout(
                title='Portfolio Time/Budget Performance Analysis',
                xaxis_title='% Time Used (AD/OD)',
                yaxis_title='% Budget Used (AC/BAC)',
                xaxis=dict(range=[0, 1.3], showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(range=[0, 1.3], showgrid=True, gridwidth=1, gridcolor='lightgray'),
                width=900,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(r=150)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add interpretation guide
            st.markdown("""
            **📊 Chart Interpretation:**
            - **X-axis**: % Time Used (AD/OD) - 1.0 = on schedule, <1.0 = ahead, >1.0 = delayed
            - **Y-axis**: % Budget Used (AC/BAC) - 1.0 = on budget, <1.0 = under budget, >1.0 = over budget
            - **Blue Curve**: Represents good performance trajectory
            - **Red Curve**: Represents poor performance trajectory
            - **Yellow Star**: Overall portfolio performance
            - **Project Colors**: Based on project tier classification
            """)

            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Projects Analyzed", len(performance_data))
            with col2:
                st.metric("Portfolio % Time Used", f"{portfolio_time_used * 100:.1f}%")
            with col3:
                st.metric("Portfolio % Budget Used", f"{portfolio_budget_used * 100:.1f}%")

        else:
            st.warning("Insufficient data for Time/Budget Performance analysis. Projects need both duration data (Actual Duration, Original Duration) and budget data (Actual Cost, Budget).")
    else:
        st.info("No projects available for Time/Budget Performance analysis.")


def render_portfolio_performance_curve(filtered_df: pd.DataFrame) -> None:
    """Render the Portfolio Performance Curve chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )

    # Check for required columns (allow both lowercase and uppercase variants)
    cpi_col = 'cpi' if 'cpi' in filtered_df.columns else 'CPI'
    spi_col = 'spi' if 'spi' in filtered_df.columns else 'SPI'
    budget_col = 'bac' if 'bac' in filtered_df.columns else 'BAC' if 'BAC' in filtered_df.columns else 'Budget'
    project_name_col = 'project_name' if 'project_name' in filtered_df.columns else 'Project Name'

    if len(filtered_df) > 0 and cpi_col in filtered_df.columns and spi_col in filtered_df.columns and budget_col in filtered_df.columns:
        # Get tier configuration from session state
        tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})

        # Default tier configuration if not set
        default_tier_config = {
            'cutoff_points': [4000, 8000, 15000],
            'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red
        }

        # Use saved config or defaults
        cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
        tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
        tier_colors = tier_config.get('colors', default_tier_config['colors'])

        # Create configurable BAC-based tier ranges
        budget_values = filtered_df[budget_col].dropna()
        if len(budget_values) > 0:
            def get_budget_category(budget):
                if pd.isna(budget):
                    return "Unknown"
                elif budget >= cutoffs[2]:  # Tier 4 (highest)
                    return f"{tier_names[3]}: (≥ {currency_symbol}{cutoffs[2]:,.0f})"
                elif budget >= cutoffs[1]:  # Tier 3
                    return f"{tier_names[2]}: ({currency_symbol}{cutoffs[1]:,.0f} - {currency_symbol}{cutoffs[2]:,.0f})"
                elif budget >= cutoffs[0]:  # Tier 2
                    return f"{tier_names[1]}: ({currency_symbol}{cutoffs[0]:,.0f} - {currency_symbol}{cutoffs[1]:,.0f})"
                else:  # Tier 1 (lowest)
                    return f"{tier_names[0]}: (< {currency_symbol}{cutoffs[0]:,.0f})"

            # Add budget category to filtered dataframe
            df_scatter = filtered_df.copy()
            df_scatter['Budget_Range'] = df_scatter[budget_col].apply(get_budget_category)

            # Define the tier order (Tier 4 to Tier 1 for legend display)
            tier_order = [
                f"{tier_names[3]}: (≥ {currency_symbol}{cutoffs[2]:,.0f})",      # Tier 4
                f"{tier_names[2]}: ({currency_symbol}{cutoffs[1]:,.0f} - {currency_symbol}{cutoffs[2]:,.0f})",  # Tier 3
                f"{tier_names[1]}: ({currency_symbol}{cutoffs[0]:,.0f} - {currency_symbol}{cutoffs[1]:,.0f})",  # Tier 2
                f"{tier_names[0]}: (< {currency_symbol}{cutoffs[0]:,.0f})"       # Tier 1
            ]

            # Convert Budget_Range to categorical with specific order
            df_scatter['Budget_Range'] = pd.Categorical(df_scatter['Budget_Range'], categories=tier_order, ordered=True)

            # Create scatter plot with configurable colors (Tier 4 to Tier 1 order)
            fig_performance = px.scatter(
                df_scatter,
                x=spi_col,
                y=cpi_col,
                color='Budget_Range',
                hover_data=[project_name_col, budget_col],
                title=f"Portfolio Performance Matrix ({len(filtered_df)} projects)",
                labels={
                    spi_col: 'Schedule Performance Index (SPI)',
                    cpi_col: 'Cost Performance Index (CPI)',
                    'Budget_Range': 'Budget Range'
                },
                color_discrete_sequence=[tier_colors[3], tier_colors[2], tier_colors[1], tier_colors[0]],  # Tier 4 to Tier 1
                category_orders={'Budget_Range': tier_order}
            )

            # Add quadrant lines at 1.0 for both axes
            fig_performance.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
            fig_performance.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.7)

            # Add quadrant labels
            fig_performance.add_annotation(
                x=1.3, y=1.3, text="✅ On Time<br>Under Budget",
                showarrow=False, font=dict(size=12, color="green"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=0.7, y=1.3, text="⚠️ Behind Schedule<br>Under Budget",
                showarrow=False, font=dict(size=12, color="orange"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=1.3, y=0.7, text="⚠️ On Time<br>Over Budget",
                showarrow=False, font=dict(size=12, color="orange"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=0.7, y=0.7, text="🚨 Behind Schedule<br>Over Budget",
                showarrow=False, font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.8)"
            )

            # Update layout
            fig_performance.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                xaxis=dict(title='Schedule Performance Index (SPI)<br>← Behind Schedule | Ahead of Schedule →'),
                yaxis=dict(title='Cost Performance Index (CPI)<br>← Over Budget | Under Budget →')
            )

            # Update traces for consistent dot appearance
            fig_performance.update_traces(
                marker=dict(
                    size=10,  # Consistent size for all dots
                    line=dict(width=1, color='rgba(0,0,0,0.3)')
                )
            )

            st.plotly_chart(fig_performance, use_container_width=True)

            # Filter summary for performance curve
            health_col = 'health_category' if 'health_category' in df_scatter.columns else 'Health_Category'
            if health_col in df_scatter.columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    healthy_count = len(df_scatter[df_scatter[health_col] == 'Healthy'])
                    st.metric("✅ Healthy Projects", healthy_count)
                with col2:
                    at_risk_count = len(df_scatter[df_scatter[health_col] == 'At Risk'])
                    st.metric("⚠️ At Risk Projects", at_risk_count)
                with col3:
                    critical_count = len(df_scatter[df_scatter[health_col] == 'Critical'])
                    st.metric("🚨 Critical Projects", critical_count)

            st.markdown(f"""
            **📊 How to Read This Chart:**
            - **X-axis (SPI):** Schedule Performance - Right is better (ahead of schedule)
            - **Y-axis (CPI):** Cost Performance - Up is better (under budget)
            - **Target Zone:** Upper right quadrant (SPI > 1.0, CPI > 1.0)
            - **Hover:** Click any dot to see project name and budget details
            - **Chart updates automatically** based on your filter selections above
            - **Tier Configuration:** Update tiers in File Management → Controls → Budget Tier Configuration
            """)
        else:
            st.info("No budget data available for performance curve.")
    else:
        st.info("Performance curve requires CPI, SPI, and Budget data.")


def render_portfolio_treemap(filtered_df: pd.DataFrame) -> None:
    """Render the Portfolio Treemap chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )

    if len(filtered_df) > 0:
        # Prepare data for treemap
        treemap_df = filtered_df.copy()

        # Get organization column name (check for different variants)
        org_col = None
        if 'organization' in treemap_df.columns:
            org_col = 'organization'
        elif 'Organization' in treemap_df.columns:
            org_col = 'Organization'

        # Get project name column
        project_name_col = 'project_name' if 'project_name' in treemap_df.columns else 'Project Name'

        # Get budget column
        budget_col = 'bac' if 'bac' in treemap_df.columns else 'BAC' if 'BAC' in treemap_df.columns else 'Budget'

        # Ensure we have the required columns
        if org_col and project_name_col in treemap_df.columns and budget_col in treemap_df.columns:
            # Add tier names to the dataframe if not already present
            if 'tier_names' not in treemap_df.columns:
                if 'budget_tier' in treemap_df.columns:
                    treemap_df['tier_names'] = treemap_df['budget_tier']
                elif 'Budget_Category' in treemap_df.columns:
                    treemap_df['tier_names'] = treemap_df['Budget_Category']
                else:
                    # Calculate tier based on budget
                    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])

                    def get_budget_tier(budget):
                        if pd.isna(budget):
                            return "Unknown"
                        elif budget <= cutoffs[0]:
                            return tier_names[0]
                        elif budget <= cutoffs[1]:
                            return tier_names[1]
                        elif budget <= cutoffs[2]:
                            return tier_names[2]
                        else:
                            return tier_names[3]

                    treemap_df['tier_names'] = treemap_df[budget_col].apply(get_budget_tier)

            # Create drill-down selection
            col1, col2 = st.columns([1, 3])

            with col1:
                drill_level = st.selectbox(
                    "View Level:",
                    ["Organization", "Project"],
                    help="Select Organization for high-level view, Project for detailed drill-down",
                    key="treemap_drill_level"
                )

            with col2:
                if drill_level == "Project":
                    # Show organization filter for project-level view
                    available_orgs = sorted(treemap_df[org_col].dropna().unique())
                    selected_org = st.selectbox(
                        "Filter by Organization:",
                        ["All"] + available_orgs,
                        help="Select an organization to focus on its projects",
                        key="treemap_org_filter"
                    )
                else:
                    selected_org = "All"

            # Filter data based on selection
            if drill_level == "Project" and selected_org != "All":
                plot_df = treemap_df[treemap_df[org_col] == selected_org].copy()
                if len(plot_df) == 0:
                    st.warning(f"No projects found for {selected_org}")
                    plot_df = treemap_df.copy()
            else:
                plot_df = treemap_df.copy()

            # Create treemap based on selected level
            if drill_level == "Organization":
                # Aggregate by organization
                org_summary = plot_df.groupby(org_col).agg({
                    budget_col: 'sum',
                    project_name_col: 'count'
                }).reset_index()
                org_summary = org_summary.rename(columns={project_name_col: 'Project_Count'})

                # Create organization-level treemap
                fig_treemap = px.treemap(
                    org_summary,
                    path=[org_col],
                    values=budget_col,
                    color=org_col,
                    title=f"Portfolio Treemap - Organization Level ({len(org_summary)} organizations, {len(plot_df)} projects)",
                    hover_data={'Project_Count': True},
                    labels={
                        budget_col: f'Total Budget ({currency_symbol})',
                        'Project_Count': 'Project Count'
                    }
                )

                # Update hover template for organizations
                fig_treemap.update_traces(
                    hovertemplate='<b>%{label}</b><br>' +
                                f'Budget: {currency_symbol}%{{value:,.0f}}<br>' +
                                'Projects: %{customdata[0]}<br>' +
                                '<extra></extra>'
                )

            else:  # Project level
                # Project-level treemap with tier colors
                if len(plot_df) > 0:
                    # Check for CPI and SPI columns
                    cpi_col = 'cpi' if 'cpi' in plot_df.columns else 'CPI'
                    spi_col = 'spi' if 'spi' in plot_df.columns else 'SPI'

                    fig_treemap = px.treemap(
                        plot_df,
                        path=[org_col, project_name_col],
                        values=budget_col,
                        color='tier_names',
                        title=f"Portfolio Treemap - Project Level ({selected_org if selected_org != 'All' else 'All Organizations'}, {len(plot_df)} projects)",
                        hover_data={
                            budget_col: ':,.0f',
                            cpi_col: ':.3f' if cpi_col in plot_df.columns else False,
                            spi_col: ':.3f' if spi_col in plot_df.columns else False
                        },
                        labels={
                            budget_col: f'Budget ({currency_symbol})',
                            'tier_names': 'Budget Tier'
                        }
                    )

                    # Update hover template for projects
                    hover_template = '<b>%{label}</b><br>' + f'Budget: {currency_symbol}%{{value:,.0f}}<br>'
                    if cpi_col in plot_df.columns:
                        hover_template += 'CPI: %{customdata[1]:.3f}<br>'
                    if spi_col in plot_df.columns:
                        hover_template += 'SPI: %{customdata[2] if len(customdata) > 2 else "N/A"}<br>'
                    hover_template += '<extra></extra>'

                    fig_treemap.update_traces(hovertemplate=hover_template)

            # Update layout for better appearance
            fig_treemap.update_layout(
                height=600,
                font_size=12,
                margin=dict(t=80, l=10, r=10, b=10)
            )

            # Display the treemap
            st.plotly_chart(fig_treemap, use_container_width=True)

            # Add explanatory text
            if drill_level == "Organization":
                st.markdown("""
                **📊 How to Read This Treemap:**
                - **Size**: Represents total budget for each organization
                - **Color**: Different colors for each organization
                - **Hover**: Shows organization name, total budget, and project count
                - **Click**: Switch to "Project" view above to drill down into specific organizations
                """)
            else:
                st.markdown(f"""
                **📊 How to Read This Treemap:**
                - **Size**: Represents budget for each project
                - **Color**: Shows budget tier for each project
                - **Hierarchy**: Projects are grouped by organization
                - **Hover**: Shows project details including budget, CPI, and SPI
                - **Filter**: Use the organization dropdown above to focus on specific organizations
                {f"- **Current View**: Showing {selected_org}" if selected_org != "All" else "- **Current View**: Showing all organizations"}
                """)

        else:
            st.warning("Treemap requires Organization, Project Name, and Budget columns.")
    else:
        st.info("No data available for treemap visualization.")


def render_portfolio_budget_chart(filtered_df: pd.DataFrame) -> None:
    """Render the Portfolio Budget Chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )
    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
    )

    # Get organization column name
    org_col = None
    if 'organization' in filtered_df.columns:
        org_col = 'organization'
    elif 'Organization' in filtered_df.columns:
        org_col = 'Organization'

    # Get budget column
    budget_col = 'bac' if 'bac' in filtered_df.columns else 'BAC' if 'BAC' in filtered_df.columns else 'Budget'

    if org_col and org_col in filtered_df.columns and len(filtered_df) > 0:
        try:
            # Add chart type selection
            col1, col2 = st.columns([1, 3])
            with col1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Total Budget", "Budget by Tier"],
                    help="Select chart type: Total Budget shows organization totals, Budget by Tier shows stacked bars by tier",
                    key="budget_chart_type"
                )

            if chart_type == "Total Budget":
                # Original implementation - Calculate total budget by organization
                org_budget_summary = filtered_df.groupby(org_col).agg({
                    budget_col: 'sum'
                }).reset_index()

                # Sort by budget in descending order
                org_budget_summary = org_budget_summary.sort_values(budget_col, ascending=True)  # ascending=True for horizontal bar chart

                if len(org_budget_summary) > 0:
                    # Create horizontal bar chart
                    fig_portfolio = px.bar(
                        org_budget_summary,
                        x=budget_col,
                        y=org_col,
                        orientation='h',
                        title="Total Budget by Organization",
                        labels={budget_col: f'Total Budget ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})', org_col: 'Organization'},
                        color=budget_col,
                        color_continuous_scale='viridis'
                    )

                    # Update layout for better visualization
                    fig_portfolio.update_layout(
                        height=max(400, len(org_budget_summary) * 40),  # Dynamic height based on number of organizations
                        showlegend=False,
                        xaxis=dict(tickformat=',.0f'),
                        yaxis=dict(title=org_col),
                        coloraxis_showscale=False
                    )

                    # Update traces for better appearance
                    fig_portfolio.update_traces(
                        texttemplate='%{x:,.0f}',
                        textposition='outside',
                        marker_line_width=0
                    )

                    st.plotly_chart(fig_portfolio, use_container_width=True)

                    # Add summary statistics below the chart
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Organizations", len(org_budget_summary))
                    with col2:
                        st.metric("Largest Budget", format_currency(org_budget_summary[budget_col].max(), currency_symbol, currency_postfix, thousands=False))
                    with col3:
                        st.metric("Smallest Budget", format_currency(org_budget_summary[budget_col].min(), currency_symbol, currency_postfix, thousands=False))

            else:  # Budget by Tier
                # Prepare data with tier information
                tier_df = filtered_df.copy()

                # Check for tier column
                tier_col = None
                if 'budget_tier' in tier_df.columns:
                    tier_col = 'budget_tier'
                elif 'Budget_Category' in tier_df.columns:
                    tier_col = 'Budget_Category'
                else:
                    # Calculate tier based on budget if not present
                    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names_config = tier_config.get('tier_names', default_tier_config['tier_names'])

                    def get_budget_tier(budget):
                        if pd.isna(budget):
                            return "Unknown"
                        elif budget <= cutoffs[0]:
                            return tier_names_config[0]
                        elif budget <= cutoffs[1]:
                            return tier_names_config[1]
                        elif budget <= cutoffs[2]:
                            return tier_names_config[2]
                        else:
                            return tier_names_config[3]

                    tier_df['budget_tier_calculated'] = tier_df[budget_col].apply(get_budget_tier)
                    tier_col = 'budget_tier_calculated'

                if tier_col:
                    # Create pivot table for stacked bar chart
                    pivot_df = tier_df.groupby([org_col, tier_col])[budget_col].sum().reset_index()
                    pivot_wide = pivot_df.pivot(index=org_col, columns=tier_col, values=budget_col).fillna(0)

                    # Sort organizations by total budget
                    pivot_wide['Total'] = pivot_wide.sum(axis=1)
                    pivot_wide = pivot_wide.sort_values('Total', ascending=True)
                    pivot_wide = pivot_wide.drop('Total', axis=1)

                    # Reset index to make organization a column again
                    pivot_wide = pivot_wide.reset_index()

                    # Get tier configuration for colors
                    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors = tier_config.get('colors', default_tier_config['colors'])

                    # Create color mapping for tiers
                    color_map = {}
                    available_tiers = [col for col in pivot_wide.columns if col != org_col]
                    for i, tier in enumerate(available_tiers):
                        # Try to match tier with configured names
                        tier_index = 0
                        for j, configured_tier in enumerate(tier_names):
                            if configured_tier in tier:
                                tier_index = j
                                break
                        color_map[tier] = tier_colors[tier_index % len(tier_colors)]

                    # Create stacked horizontal bar chart
                    fig_portfolio = px.bar(
                        pivot_wide,
                        x=available_tiers,
                        y=org_col,
                        orientation='h',
                        title="Budget by Organization and Tier",
                        labels={'value': f'Budget ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})', org_col: 'Organization'},
                        color_discrete_map=color_map
                    )

                    # Update layout for stacked bar chart
                    fig_portfolio.update_layout(
                        height=max(400, len(pivot_wide) * 50),  # Slightly more height for stacked bars
                        showlegend=True,
                        xaxis=dict(tickformat=',.0f'),
                        yaxis=dict(title=org_col),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            title="Budget Tiers"
                        ),
                        barmode='stack'
                    )

                    # Update traces for stacked appearance
                    fig_portfolio.update_traces(
                        textposition='inside',
                        texttemplate='%{x:,.0f}',
                        textfont_size=10
                    )

                    st.plotly_chart(fig_portfolio, use_container_width=True)

                    # Add tier summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Organizations", len(pivot_wide))
                    with col2:
                        total_budgets = pivot_wide.select_dtypes(include=[np.number]).sum(axis=1)
                        st.metric("Largest Total", format_currency(total_budgets.max(), currency_symbol, currency_postfix, thousands=False))
                    with col3:
                        st.metric("Tiers Present", len(available_tiers))
                    with col4:
                        st.metric("Total Projects", len(tier_df))

                    # Show tier distribution with proper ordering and percentages
                    st.markdown("**Tier Distribution Across Organizations:**")

                    # Calculate total budget for percentage calculations
                    total_portfolio_budget = sum(pivot_wide[tier].sum() for tier in available_tiers)

                    # Sort available tiers by their position in tier_names configuration
                    def get_tier_order(tier_name):
                        # Find the index of this tier in the configuration
                        for i, configured_tier in enumerate(tier_names):
                            if configured_tier in tier_name:
                                return i
                        return len(tier_names)  # Unknown tiers go to the end

                    ordered_tiers = sorted(available_tiers, key=get_tier_order)

                    # Display tiers in proper order with percentages
                    for tier in ordered_tiers:
                        tier_total = pivot_wide[tier].sum()
                        if tier_total > 0:
                            percentage = (tier_total / total_portfolio_budget * 100) if total_portfolio_budget > 0 else 0
                            st.write(f"• **{tier}**: {format_currency(tier_total, currency_symbol, currency_postfix, thousands=False)} [{percentage:.1f}%]")

                else:
                    st.warning("Budget tier information not available. Please ensure tier configuration is set up.")

            if len(filtered_df) == 0:
                st.info("No organization budget data available to display.")

        except Exception as e:
            st.error(f"Error creating portfolio graph: {str(e)}")
            st.info("Unable to display portfolio graph with current data.")
    else:
        if len(filtered_df) == 0:
            st.info("No projects available for portfolio graph.")
        else:
            st.info("Organization data not available for portfolio graph.")


def render_approvals_chart(filtered_df: pd.DataFrame) -> None:
    """Render the Approvals Chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )
    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
    )

    if len(filtered_df) > 0:
        # Use specific column names for approvals chart
        start_date_col = 'plan_start'
        budget_col = 'bac'

        if start_date_col in filtered_df.columns and budget_col in filtered_df.columns:
            # Controls for approvals chart
            col1, col2, col3 = st.columns(3)

            with col1:
                approval_time_period = st.selectbox("Approval Time Period",
                                                  options=["Month", "Quarter", "FY"],
                                                  index=0,
                                                  key="approval_time_period",
                                                  help="Month: Monthly approvals, Quarter: Quarterly approvals, FY: Financial Year (July-June)")

            with col2:
                show_tiers = st.toggle("Show Tiers",
                                     value=False,
                                     key="approval_show_tiers",
                                     help="Show stacked chart by budget tiers")

            with col3:
                st.write("**Configuration:**")
                st.write(f"📊 {approval_time_period}")
                if show_tiers:
                    st.write("🎯 Stacked by Tiers")
                else:
                    st.write("💰 BAC by Approval Date")

            try:
                # Prepare data for approvals chart
                df_approvals = filtered_df.copy()

                # Parse approval date (using Plan Start as proxy for approval)
                df_approvals[start_date_col] = pd.to_datetime(df_approvals[start_date_col], errors='coerce')
                df_approvals = df_approvals.dropna(subset=[start_date_col, budget_col])

                if len(df_approvals) > 0:
                    # Helper functions
                    def get_financial_year_approvals(date):
                        """Get financial year string for a date (FY starts July 1st)"""
                        if date.month >= 7:
                            return f"FY{date.year + 1}"
                        else:
                            return f"FY{date.year}"

                    def get_approval_period_key(date, time_period):
                        """Get period key based on time period selection"""
                        if time_period == "Month":
                            return date.strftime("%b-%Y")
                        elif time_period == "Quarter":
                            quarter = f"Q{((date.month - 1) // 3) + 1}-{date.year}"
                            return quarter
                        else:  # FY
                            return get_financial_year_approvals(date)

                    def get_approval_sort_key(period_str, time_period):
                        """Generate sort key for different time periods"""
                        if time_period == "Month":
                            try:
                                month_abbr, year = period_str.split('-')
                                month_num = {
                                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                }.get(month_abbr, 1)
                                return (int(year), month_num)
                            except:
                                return (2000, 1)
                        elif time_period == "Quarter":
                            try:
                                quarter, year = period_str.split('-')
                                quarter_num = int(quarter[1:])
                                return (int(year), quarter_num)
                            except:
                                return (2000, 1)
                        else:  # FY
                            try:
                                return (int(period_str[2:]), 1)
                            except:
                                return (2000, 1)

                    # Get tier configuration
                    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors = tier_config.get('colors', default_tier_config['colors'])

                    def assign_tier(budget):
                        """Assign tier based on budget"""
                        if budget >= cutoffs[2]:
                            return tier_names[3]
                        elif budget >= cutoffs[1]:
                            return tier_names[2]
                        elif budget >= cutoffs[0]:
                            return tier_names[1]
                        else:
                            return tier_names[0]

                    # Calculate approvals data
                    approvals_data = []
                    for idx, row in df_approvals.iterrows():
                        approval_date = row[start_date_col]
                        budget = row.get(budget_col, 0)

                        if pd.notna(approval_date) and budget > 0:
                            period_key = get_approval_period_key(approval_date, approval_time_period)

                            # Get financial values
                            ac = row.get('ac', row.get('Actual Cost', 0))
                            ev = row.get('ev', row.get('Earned Value', 0))
                            eac = row.get('eac', row.get('EAC', 0))
                            project_name = row.get('project_name', row.get('Project Name', 'Unknown'))

                            # Assign tier
                            tier = assign_tier(budget)

                            approvals_data.append({
                                'Period': period_key,
                                'BAC': budget,
                                'AC': ac,
                                'EV': ev,
                                'EAC': eac,
                                'Project': project_name,
                                'Date': approval_date,
                                'Tier': tier
                            })

                    if approvals_data:
                        # Create DataFrame
                        approvals_df = pd.DataFrame(approvals_data)

                        if show_tiers:
                            # Aggregate by period and tier
                            period_tier_approvals = approvals_df.groupby(['Period', 'Tier']).agg({
                                'BAC': 'sum',
                                'AC': 'sum',
                                'EV': 'sum',
                                'EAC': 'sum',
                                'Project': 'count'
                            }).rename(columns={'Project': 'Number of Projects'}).reset_index()

                            # Sort periods chronologically
                            period_tier_approvals['Sort_Key'] = period_tier_approvals['Period'].apply(
                                lambda x: get_approval_sort_key(x, approval_time_period)
                            )
                            period_tier_approvals = period_tier_approvals.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                            # Create tier color mapping
                            tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                            # Create stacked bar chart by tiers
                            chart_title = f"Project Approvals by Tier - {approval_time_period} View"

                            fig_approvals = px.bar(
                                period_tier_approvals,
                                x='Period',
                                y='BAC',
                                color='Tier',
                                title=chart_title,
                                labels={
                                    'BAC': f'Total BAC ({currency_symbol})',
                                    'Period': approval_time_period,
                                    'Tier': 'Budget Tier'
                                },
                                color_discrete_map=tier_color_map,
                                category_orders={'Tier': tier_names}
                            )

                        else:
                            # Regular aggregation by period only
                            period_approvals = approvals_df.groupby('Period').agg({
                                'BAC': 'sum',
                                'AC': 'sum',
                                'EV': 'sum',
                                'EAC': 'sum',
                                'Project': 'count'
                            }).rename(columns={'Project': 'Number of Projects'}).reset_index()

                            # Calculate percentage columns
                            period_approvals['% AC/BAC'] = (period_approvals['AC'] / period_approvals['BAC'] * 100).fillna(0)
                            period_approvals['% EV/BAC'] = (period_approvals['EV'] / period_approvals['BAC'] * 100).fillna(0)
                            period_approvals['% EAC/BAC'] = (period_approvals['EAC'] / period_approvals['BAC'] * 100).fillna(0)

                            # Sort periods chronologically
                            period_approvals['Sort_Key'] = period_approvals['Period'].apply(
                                lambda x: get_approval_sort_key(x, approval_time_period)
                            )
                            period_approvals = period_approvals.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                            # Create regular bar chart
                            chart_title = f"Project Approvals by BAC - {approval_time_period} View"

                            fig_approvals = px.bar(
                                period_approvals,
                                x='Period',
                                y='BAC',
                                title=chart_title,
                                labels={
                                    'BAC': f'Total BAC ({currency_symbol})',
                                    'Period': approval_time_period
                                },
                                color='BAC',
                                color_continuous_scale='greens'
                            )

                        # Update layout for better visualization
                        fig_approvals.update_layout(
                            height=450,
                            showlegend=False if not show_tiers else True,
                            xaxis=dict(
                                title=approval_time_period,
                                tickangle=45
                            ),
                            yaxis=dict(
                                title=f'Total BAC ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                tickformat=',.0f'
                            ),
                            coloraxis_showscale=False,
                            margin=dict(t=80, b=60, l=60, r=60)
                        )

                        # Update traces for better appearance
                        fig_approvals.update_traces(
                            texttemplate='%{y:,.0f}',
                            textposition='auto',
                            textfont=dict(size=10),
                            cliponaxis=False
                        )

                        st.plotly_chart(fig_approvals, use_container_width=True)

                        # Add tier legend when tiers are enabled
                        if show_tiers:
                            st.markdown("**🎯 Tier Legend:**")
                            tier_legend_items = []
                            for i, tier_name in enumerate(tier_names):
                                if i == 0:
                                    range_text = f"< {currency_symbol}{cutoffs[0]:,.0f}"
                                elif i == len(tier_names) - 1:
                                    range_text = f"≥ {currency_symbol}{cutoffs[2]:,.0f}"
                                else:
                                    range_text = f"{currency_symbol}{cutoffs[i-1]:,.0f} - {currency_symbol}{cutoffs[i]:,.0f}"

                                color_emoji = ["🔵", "🟢", "🟠", "🔴"][i]
                                tier_legend_items.append(f"{color_emoji} {tier_name}: {range_text}")

                            st.text(" | ".join(tier_legend_items))

                        # Show detailed data table
                        with st.expander("📊 Detailed Approvals Data", expanded=False):
                            if show_tiers:
                                display_approvals = period_tier_approvals.copy()
                                for col in ['BAC', 'AC', 'EV', 'EAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                            else:
                                display_approvals = period_approvals.copy()
                                for col in ['BAC', 'AC', 'EV', 'EAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                                for col in ['% AC/BAC', '% EV/BAC', '% EAC/BAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(lambda x: f"{x:.1f}%")

                            st.dataframe(display_approvals, width='stretch')

                    else:
                        st.warning("No valid approval data could be generated from the selected projects.")

                else:
                    st.warning("No projects have valid approval dates and budget values.")

            except Exception as e:
                st.error(f"Error processing approvals data: {str(e)}")
                st.info("Please check that date and budget columns contain valid values.")

        else:
            st.info("Approvals chart requires 'plan_start' and 'bac' columns.")
    else:
        st.info("No data available for approvals analysis.")


def render_gantt(df: pd.DataFrame, show_predicted: bool, period_choice: str, start_col: str = None) -> None:
    # Sort by start date if column is available
    if start_col and start_col in df.columns:
        df = df.sort_values(start_col).reset_index(drop=True)
    elif "plan_start" in df.columns:
        df = df.sort_values("plan_start").reset_index(drop=True)
    segments = build_segments(df, show_predicted)
    if not segments:
        st.info("No projects match the current filters.")
        return

    seg_df = pd.DataFrame(segments).sort_values(by=["Start", "Finish", "Segment"])

    if "project_id" in df.columns and start_col:
        project_order_df = df[["project_id", start_col]].dropna(subset=[start_col]).copy()
        project_order_df = project_order_df.sort_values(start_col, kind="mergesort")
        category_order = project_order_df["project_id"].astype(str).tolist()
    else:
        task_order = (
            seg_df.groupby("Task")["Start"]
            .min()
            .sort_values()
        )
        category_order = task_order.index.astype(str).tolist()

    if not category_order:
        category_order = seg_df["Task"].astype(str).unique().tolist()

    fig = px.timeline(
        seg_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Segment",
        color_discrete_map=COLOR_MAP,
        category_orders={"Task": category_order},
        custom_data=["project_name", "organization", "Segment", "bac_formatted", "plan_start", "plan_finish", "cpi", "spi", "percent_budget_used", "percent_time_used", "percent_work_completed"]
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Project ID:</b> %{y}<br>"
            "<b>Project Name:</b> %{customdata[0]}<br>"
            "<b>Organization:</b> %{customdata[1]}<br>"
            "<b>BAC:</b> %{customdata[3]}<br>"
            "<b>Plan Start:</b> %{customdata[4]}<br>"
            "<b>Plan Finish:</b> %{customdata[5]}<br>"
            "<b>CPI:</b> %{customdata[6]:.2f}<br>"
            "<b>SPI:</b> %{customdata[7]:.2f}<br>"
            "<b>% Budget Used:</b> %{customdata[8]:.1f}%<br>"
            "<b>% Time Used:</b> %{customdata[9]:.1f}%<br>"
            "<b>% Work Completed:</b> %{customdata[10]:.1f}%<br>"
            "<b>Segment:</b> %{customdata[2]}<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")

    period_meta = PERIOD_OPTIONS.get(period_choice, PERIOD_OPTIONS["Month"])

    # Calculate min/max from segment data for end range
    valid_segment_finishes = seg_df["Finish"].dropna()
    if not valid_segment_finishes.empty:
        max_finish = valid_segment_finishes.max()
        # Validate the max finish date is reasonable
        try:
            max_finish_ts = pd.Timestamp(max_finish)
            if max_finish_ts.year < 1980:  # Invalid date
                max_finish = pd.NaT
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            max_finish = pd.NaT
    else:
        max_finish = pd.NaT

    # Calculate earliest plan start from original dataframe (not segments)
    earliest_plan_start = None
    if start_col and start_col in df.columns:
        # Get valid dates only (remove NaT values)
        valid_start_dates = df[start_col].dropna()
        if not valid_start_dates.empty:
            earliest_plan_start = valid_start_dates.min()

    # Fallback to segments if no valid start dates found
    if pd.isna(earliest_plan_start) or earliest_plan_start is None:
        valid_segment_starts = seg_df["Start"].dropna()
        if not valid_segment_starts.empty:
            earliest_plan_start = valid_segment_starts.min()

    # Timeline should start 1 quarter (3 months) before earliest plan start
    if pd.notna(earliest_plan_start) and earliest_plan_start is not None:
        try:
            # Ensure we have a valid timestamp
            earliest_plan_start_ts = pd.Timestamp(earliest_plan_start)
            # Check if the timestamp is reasonable (not epoch time)
            if earliest_plan_start_ts.year > 1980:  # Reasonable check for valid dates
                axis_range_start = (earliest_plan_start_ts - pd.DateOffset(months=3)).normalize()
            else:
                # Fallback to current date if we get unreasonable dates
                axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Fallback to current date if conversion fails
            axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
    else:
        # Ultimate fallback
        axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
    # Calculate end range with validation
    if pd.notna(max_finish):
        try:
            timeline_end = pd.Timestamp(max_finish)
            axis_range_end = timeline_end + period_meta["delta"]
            year_end_candidate = timeline_end.to_period('Y').end_time
            if year_end_candidate > axis_range_end:
                axis_range_end = year_end_candidate
            axis_range_end = pd.Timestamp(axis_range_end)
            axis_range_end = axis_range_end.normalize() + pd.Timedelta(days=1)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Fallback to a reasonable end date
            axis_range_end = (pd.Timestamp.now() + pd.DateOffset(months=12)).normalize()
    else:
        # Ultimate fallback for end date
        axis_range_end = (pd.Timestamp.now() + pd.DateOffset(months=12)).normalize()
    fig.update_xaxes(
        type="date",
        dtick=period_meta["dtick"],
        tickformat="%b %Y",
        range=[axis_range_start, axis_range_end],
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )

    if pd.notna(axis_range_start) and pd.notna(axis_range_end):
        axis_start_ts = pd.Timestamp(axis_range_start)
        axis_end_ts = pd.Timestamp(axis_range_end)
        start_year = int(axis_start_ts.year)
        end_year = int(axis_end_ts.year)
        year_boundaries = []
        for year in range(start_year, end_year + 1):
            year_end = pd.Timestamp(year=year, month=12, day=31)
            if year_end < axis_start_ts or year_end > axis_end_ts:
                continue
            year_boundaries.append(year_end)
            fig.add_shape(
                type="line",
                x0=year_end,
                x1=year_end,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=YEAR_LINE_COLOR, dash="dot", width=0.8)
            )
        if period_choice in ("Quarter", "Month"):
            quarter_months = (3, 6, 9, 12)
            for year in range(start_year, end_year + 1):
                for month in quarter_months:
                    quarter_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                    if quarter_end < axis_start_ts or quarter_end > axis_end_ts:
                        continue
                    if quarter_end in year_boundaries:
                        continue
                    fig.add_shape(
                        type="line",
                        x0=quarter_end,
                        x1=quarter_end,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color=QUARTER_LINE_COLOR, dash="dot", width=0.4)
                    )

    data_dates = df["data_date"].dropna().unique()
    if data_dates.size:
        data_date = pd.to_datetime(sorted(data_dates)[-1])
        fig.add_shape(
            type="line",
            x0=data_date,
            x1=data_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#facc15", dash="dot", width=2.4)
        )
        fig.add_annotation(
            x=data_date,
            yref="paper",
            y=1.03,
            xref="x",
            text=f"<b>Data Date: {data_date.strftime('%Y-%m-%d')}</b>",
            showarrow=False,
            font=dict(color="#c53030"),
            align="center"
        )

    project_count = df.shape[0]
    if project_count <= 15:
        row_height = 36
    elif project_count <= 30:
        row_height = 24
    elif project_count <= 60:
        row_height = 16
    else:
        row_height = 10

    figure_height = 140 + project_count * row_height
    show_labels = row_height >= 16
    fig.update_layout(
        height=figure_height,
        legend_title_text="",
        margin=dict(l=120, r=40, t=60, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=category_order,
        showticklabels=show_labels,
        title_text="Project ID" if show_labels else "",
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )

    st.plotly_chart(fig, use_container_width=True)


def render_footer():
    st.markdown(
        """
        <div class="footer">
            <div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: 1rem; margin-top: 2rem;">
                <strong>Project Portfolio Intelligence Suite</strong> • Schedule Performance Overview<br>
                Generated on {date} • Confidential Executive Report
            </div>
        </div>
        """.format(date=datetime.now().strftime('%B %d, %Y at %I:%M %p')),
        unsafe_allow_html=True
    )


def main():
    st.markdown('<h1 class="main-header">📈 Portfolio Gantt</h1>', unsafe_allow_html=True)
    st.markdown("Timeline visualization for your portfolio")

    df = load_portfolio_dataframe()
    if df is None:
        st.warning("No portfolio data available. Run a batch analysis from the Portfolio Analysis page first.")
        render_footer()
        return

    # Detect available column names and coerce data types
    date_cols_to_check = ["plan_start", "plan_finish", "Plan Start", "Plan Finish", "data_date", "forecast_completion", "likely_completion"]
    for col in date_cols_to_check:
        if col in df.columns:
            df[col] = _coerce_datetime(df[col])

    numeric_cols_to_check = ["bac", "BAC", "ac", "AC", "earned_value", "original_duration_months", "actual_duration_months", "cost_performance_index", "schedule_performance_index"]
    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Find the correct start and finish date columns
    start_col = None
    if "plan_start" in df.columns:
        start_col = "plan_start"
    elif "Plan Start" in df.columns:
        start_col = "Plan Start"

    finish_col = None
    if "plan_finish" in df.columns:
        finish_col = "plan_finish"
    elif "Plan Finish" in df.columns:
        finish_col = "Plan Finish"

    if not start_col or not finish_col:
        st.warning("Project records are missing plan date columns. Please verify the source data.")
        render_footer()
        return

    # Filter out rows with missing or invalid dates
    df = df.dropna(subset=[start_col, finish_col])

    # Additional validation: remove rows with unreasonable dates (like epoch dates)
    if not df.empty:
        # Filter out dates before 1980 (likely invalid/epoch dates)
        valid_start_mask = df[start_col].apply(lambda x: pd.notna(x) and pd.Timestamp(x).year > 1980 if pd.notna(x) else False)
        valid_finish_mask = df[finish_col].apply(lambda x: pd.notna(x) and pd.Timestamp(x).year > 1980 if pd.notna(x) else False)
        df = df[valid_start_mask & valid_finish_mask]

    if df.empty:
        st.warning("Project records are missing valid plan dates. Please verify the source data.")
        render_footer()
        return

    filtered_df = apply_filters(df, start_col, finish_col)

    st.markdown(f"**Projects Displayed:** {len(filtered_df)}")

    with st.expander("📊 Portfolio Gantt Chart", expanded=False):
        # Period and View controls
        col1, col2 = st.columns([1, 1])
        with col1:
            period_choice = st.radio("Period", list(PERIOD_OPTIONS.keys()), index=2, horizontal=True, key="gantt_period_choice")  # Default to Year (index 2)
        with col2:
            show_predicted = st.toggle("Predicted View", value=False, key="gantt_predicted_view")  # Default to Plan view
            st.caption(f"Mode: {'Predicted' if show_predicted else 'Plan'}")

        render_gantt(filtered_df, show_predicted, period_choice, start_col)

    # Cash Flow Chart Expander
    with st.expander("💰 Cash Flow Chart", expanded=False):
        render_cash_flow_chart(filtered_df, start_col, finish_col)

    # Time/Budget Performance Expander
    with st.expander("📈 Time/Budget Performance", expanded=False):
        render_time_budget_performance(filtered_df)

    # Portfolio Performance Curve Expander
    with st.expander("📈 Portfolio Performance Curve", expanded=False):
        render_portfolio_performance_curve(filtered_df)

    # Portfolio Treemap Expander
    with st.expander("🗺️ Portfolio Treemap", expanded=False):
        render_portfolio_treemap(filtered_df)

    # Portfolio Budget Chart Expander
    with st.expander("📊 Portfolio Budget Chart", expanded=False):
        render_portfolio_budget_chart(filtered_df)

    # Approvals Chart Expander
    with st.expander("📈 Approvals Chart", expanded=False):
        render_approvals_chart(filtered_df)

    render_footer()


if __name__ == "__main__":
    main()

