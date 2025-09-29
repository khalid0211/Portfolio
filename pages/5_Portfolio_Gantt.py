from typing import List, Dict

from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
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


def format_currency(amount: float, symbol: str, postfix: str = "", decimals: int = 2) -> str:
    """Enhanced currency formatting with comma separators and postfix options."""
    if not is_valid_finite_number(amount):
        return "—"

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



def apply_filters(df: pd.DataFrame, start_col: str = None, finish_col: str = None) -> tuple[pd.DataFrame, bool, str]:
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

    # 1. Period and View (first row)
    col1, col2 = st.columns([1, 1])
    with col1:
        period_choice = st.radio("Period", list(PERIOD_OPTIONS.keys()), index=2, horizontal=True)  # Default to Year (index 2)
    with col2:
        show_predicted = st.toggle("Predicted View", value=False, key="gantt_predicted_view")  # Default to Plan view
        st.caption(f"Mode: {'Predicted' if show_predicted else 'Plan'}")

    # 2. Organization (single line)
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

    return filtered, show_predicted, period_choice



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
        earned_value = row.get("earned_value", 0.0)
        cpi = row.get("cost_performance_index", 0.0)
        spi = row.get("schedule_performance_index", 0.0)
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

        baseline_span = finish - start
        progress_end = start + progress_ratio * baseline_span

        if progress_end < start:
            progress_end = start
        if progress_end > finish:
            progress_end = finish

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
        forecast_finish = row.get("likely_completion", row.get("forecast_completion"))

        if show_predicted and pd.notna(forecast_finish):
            # Convert forecast_finish to datetime if it's a string
            if isinstance(forecast_finish, str):
                try:
                    # Try parsing different date formats
                    forecast_finish_dt = pd.to_datetime(forecast_finish, format='%d/%m/%Y', errors='coerce')
                    if pd.isna(forecast_finish_dt):
                        forecast_finish_dt = pd.to_datetime(forecast_finish, errors='coerce')
                    forecast_finish = forecast_finish_dt
                except:
                    forecast_finish = pd.NaT

            if pd.notna(forecast_finish) and progress_end < forecast_finish:
                # Predicted View: Show predicted completion
                if forecast_finish > finish:
                    # Predicted completion is later than planned (overrun)
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
                    segments.append({
                        "Task": project_id,
                        "Start": finish,
                        "Finish": forecast_finish,
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
                    # Predicted completion is earlier than or same as planned
                    segments.append({
                        "Task": project_id,
                        "Start": progress_end,
                        "Finish": forecast_finish,
                        "Segment": "Predicted",
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
            # Plan View: Show planned completion
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

    st.plotly_chart(fig, width='stretch')


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

    filtered_df, show_predicted, period_choice = apply_filters(df, start_col, finish_col)

    st.markdown(f"**Projects Displayed:** {len(filtered_df)}")
    render_gantt(filtered_df, show_predicted, period_choice, start_col)
    render_footer()


if __name__ == "__main__":
    main()

