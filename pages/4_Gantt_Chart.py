from typing import List, Dict

from datetime import datetime

import pandas as pd
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {
    "Progress": "#40b57b",  # lighter green
    "Planned": "#4389d1",   # lighter blue
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
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    """Render filter widgets and return the filtered DataFrame with control values."""
    organizations = sorted({str(org) for org in df.get("organization", pd.Series()).dropna() if str(org).strip()})

    numeric_bac = df["bac"].dropna().astype(float) if "bac" in df.columns else pd.Series(dtype=float)
    min_budget = float(numeric_bac.min()) if not numeric_bac.empty else 0.0
    max_budget = float(numeric_bac.max()) if not numeric_bac.empty else min_budget

    min_start = df["plan_start"].min() if "plan_start" in df.columns else pd.NaT
    max_start = df["plan_start"].max() if "plan_start" in df.columns else pd.NaT

    with st.container():
        col1, col2, col3 = st.columns([1.3, 1, 1])
        with col1:
            org_selection = st.multiselect(
                "Organization",
                options=organizations,
                default=organizations,
                placeholder="Select organization(s)" if organizations else "No organizations available"
            ) if organizations else []
        with col2:
            min_budget_toggle = st.toggle("Set Min Budget", value=False, key="gantt_min_budget_toggle")
            min_budget_value = st.number_input(
                "Min Budget",
                value=min_budget,
                step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                min_value=0.0,
                key="gantt_min_budget_value"
            ) if min_budget_toggle else min_budget

            max_budget_toggle = st.toggle("Set Max Budget", value=False, key="gantt_max_budget_toggle")
            max_budget_value = st.number_input(
                "Max Budget",
                value=max_budget,
                step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                min_value=0.0,
                key="gantt_max_budget_value"
            ) if max_budget_toggle else max_budget

        with col3:
            start_date_toggle = st.toggle("Set Start Date", value=False, key="gantt_start_date_toggle") if pd.notna(min_start) else False
            if start_date_toggle and pd.notna(min_start):
                start_date_value = st.date_input(
                    "Plan Start On/After",
                    value=min_start.date(),
                    min_value=min_start.date(),
                    max_value=max_start.date() if pd.notna(max_start) else min_start.date(),
                    key="gantt_start_date_value"
                )
            else:
                start_date_value = None

            end_date_toggle = st.toggle("Set End Date", value=False, key="gantt_end_date_toggle") if pd.notna(max_start) else False
            if end_date_toggle and pd.notna(max_start):
                end_date_value = st.date_input(
                    "Plan Start On/Before",
                    value=max_start.date(),
                    min_value=min_start.date() if pd.notna(min_start) else max_start.date(),
                    max_value=max_start.date(),
                    key="gantt_end_date_value"
                )
            else:
                end_date_value = None

    col_toggle, col_period = st.columns([1, 1])
    with col_toggle:
        show_predicted = st.toggle("Predicted View", value=True, key="gantt_predicted_view")
        st.caption(f"Mode: {'Predicted' if show_predicted else 'Plan'}")
    with col_period:
        period_choice = st.radio("Period", list(PERIOD_OPTIONS.keys()), horizontal=True)

    filtered = df.copy()
    if organizations and org_selection:
        filtered = filtered[filtered["organization"].isin(org_selection)]

    if "bac" in filtered.columns:
        if min_budget_toggle:
            filtered = filtered[filtered["bac"] >= min_budget_value]
        if max_budget_toggle:
            filtered = filtered[filtered["bac"] <= max_budget_value]

    if start_date_value is not None:
        start_dt = pd.to_datetime(start_date_value)
        filtered = filtered[filtered["plan_start"] >= start_dt]

    if end_date_value is not None:
        end_dt = pd.to_datetime(end_date_value)
        filtered = filtered[filtered["plan_start"] <= end_dt]

    return filtered, show_predicted, period_choice



def build_segments(df: pd.DataFrame, show_predicted: bool) -> List[Dict]:
    segments: List[Dict] = []
    for _, row in df.iterrows():
        start = row.get("plan_start")
        finish = row.get("plan_finish")
        if pd.isna(start) or pd.isna(finish):
            continue

        project_id = str(row.get("project_id", "")) or "Unknown"
        project_name = row.get("project_name", "")
        organization = row.get("organization", "")

        bac = row.get("bac", 0.0)
        earned_value = row.get("earned_value", 0.0)
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
            "organization": organization
        })

        if progress_end < finish:
            segments.append({
                "Task": project_id,
                "Start": progress_end,
                "Finish": finish,
                "Segment": "Planned",
                "project_name": project_name,
                "organization": organization
            })

        forecast_finish = row.get("forecast_completion")
        if show_predicted and pd.notna(forecast_finish) and forecast_finish > finish:
            segments.append({
                "Task": project_id,
                "Start": finish,
                "Finish": forecast_finish,
                "Segment": "Overrun",
                "project_name": project_name,
                "organization": organization
            })

    return segments


def render_gantt(df: pd.DataFrame, show_predicted: bool, period_choice: str) -> None:
    if "plan_start" in df.columns:
        df = df.sort_values("plan_start").reset_index(drop=True)
    segments = build_segments(df, show_predicted)
    if not segments:
        st.info("No projects match the current filters.")
        return

    seg_df = pd.DataFrame(segments).sort_values(by=["Start", "Finish", "Segment"])

    if "project_id" in df.columns:
        project_order_df = df[["project_id", "plan_start"]].dropna(subset=["plan_start"]).copy()
        project_order_df = project_order_df.sort_values("plan_start", kind="mergesort")
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
        custom_data=["project_name", "organization", "Segment"]
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Project ID:</b> %{y}<br>"
            "<b>Project:</b> %{customdata[0]}<br>"
            "<b>Organization:</b> %{customdata[1]}<br>"
            "<b>Segment:</b> %{customdata[2]}<br>"
            "<b>Start:</b> %{x|%Y-%m-%d}<br>"
            "<b>Finish:</b> %{x_end|%Y-%m-%d}<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")

    period_meta = PERIOD_OPTIONS.get(period_choice, PERIOD_OPTIONS["Month"])

    min_start = seg_df["Start"].min()
    max_finish = seg_df["Finish"].max()
    if pd.notna(min_start):
        axis_range_start = (pd.Timestamp(min_start) - pd.DateOffset(months=3)).normalize()
    else:
        axis_range_start = min_start
    axis_range_end = max_finish
    timeline_end = pd.NaT
    if pd.notna(max_finish):
        timeline_end = pd.Timestamp(max_finish)
        axis_range_end = timeline_end + period_meta["delta"]
        year_end_candidate = timeline_end.to_period('Y').end_time
        if year_end_candidate > axis_range_end:
            axis_range_end = year_end_candidate
        axis_range_end = pd.Timestamp(axis_range_end)
        axis_range_end = axis_range_end.normalize() + pd.Timedelta(days=1)
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
                <strong>Portfolio Gantt Chart</strong> • Schedule Performance Overview<br>
                Generated on {date} • Confidential Executive Report
            </div>
        </div>
        """.format(date=datetime.now().strftime('%B %d, %Y at %I:%M %p')),
        unsafe_allow_html=True
    )


def main():
    st.title("📅 Portfolio Gantt Chart")
    st.markdown(
        """
        <div style='color:#003366; font-size:16px; line-height:1.4;'>
            Portfolio Schedule Intelligence & Executive Decision Support<br>
            Developed by Dr. Khalid Ahmad Khan – <a href="https://www.linkedin.com/in/khalidahmadkhan/" target="_blank" style="color: #0066cc; text-decoration: none;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = load_portfolio_dataframe()
    if df is None:
        st.warning("No portfolio data available. Run a batch analysis from the Portfolio Analysis page first.")
        render_footer()
        return

    for col in ["plan_start", "plan_finish", "data_date", "forecast_completion"]:
        if col in df.columns:
            df[col] = _coerce_datetime(df[col])
    for col in ["bac", "earned_value"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    df = df.dropna(subset=["plan_start", "plan_finish"])
    if df.empty:
        st.warning("Project records are missing plan dates. Please verify the source data.")
        render_footer()
        return

    filtered_df, show_predicted, period_choice = apply_filters(df)

    st.markdown(f"**Projects Displayed:** {len(filtered_df)}")
    render_gantt(filtered_df, show_predicted, period_choice)
    render_footer()


if __name__ == "__main__":
    main()

