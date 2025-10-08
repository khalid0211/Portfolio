import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go

def calculate_highway_cashflow(budget, duration):
    """Calculate Highway cash flow using a calibrated formula."""
    raw_percentages = []
    for t in range(1, duration + 1):
        x = (t - 1) / (duration - 1) if duration > 1 else 0
        if x <= 0.125:
            P = 0.3 + 0.8 * (x / 0.125)**1.5
        elif x <= 0.333:
            P = 1.8 + 1.4 * ((x - 0.125) / 0.208)**0.7
        elif x <= 0.75:
            P = 2.0 + 6.1 * np.exp(-0.5 * ((x - 0.625) / 0.15)**2)
        else:
            P = 5.8 * np.exp(-2.0 * ((x - 0.75) / 0.25)) + 1.5
        raw_percentages.append(P)
    
    sum_of_percentages = sum(raw_percentages)
    scale_factor = 100 / sum_of_percentages
    monthly_cashflows = [budget * p * scale_factor / 100 for p in raw_percentages]
    return monthly_cashflows

def calculate_building_cashflow(budget, duration):
    """Calculate Building cash flow using a calibrated formula."""
    raw_percentages = []
    for t in range(1, duration + 1):
        x = (t - 1) / (duration - 1) if duration > 1 else 0
        if x <= 0.167:
            percentage = 0.5 + 2.0 * (x / 0.167)**1.2
        elif x <= 0.556:
            term = abs((x - 0.167) / 0.389 - 0.6) / 0.25
            percentage = 4.0 + 7.5 * np.exp(-0.5 * term**2)
        elif x <= 0.833:
            percentage = 8.5 + 3.0 * np.sin(np.pi * (x - 0.556) / 0.277)
        else:
            percentage = 7.0 * np.exp(-2.0 * (x - 0.833) / 0.167) + 1.0
        raw_percentages.append(percentage)

    total_percentage = sum(raw_percentages)
    scale_factor = 100 / total_percentage
    monthly_cashflows = [budget * p * scale_factor / 100 for p in raw_percentages]
    return monthly_cashflows

def calculate_scurve_cashflow(budget, duration, alpha=2, beta=2):
    """Calculate Standard S-Curve cash flow using Beta distribution."""
    from scipy.stats import beta as beta_dist

    raw_percentages = []
    for t in range(1, duration + 1):
        x = (t - 0.5) / duration  # Use midpoint of period
        # Beta distribution PDF
        percentage = beta_dist.pdf(x, alpha, beta)
        raw_percentages.append(percentage)

    # Normalize to ensure total equals budget
    total_percentage = sum(raw_percentages)
    scale_factor = 100 / total_percentage
    monthly_cashflows = [budget * p * scale_factor / 100 for p in raw_percentages]
    return monthly_cashflows

st.set_page_config(layout="wide", page_title="Project Delay Financial Impact Simulator")

# Initialize session state for baseline functionality
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = None
if 'comparison_records' not in st.session_state:
    st.session_state.comparison_records = []

# Custom CSS for professional appearance and compact layout
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }

    /* Professional Parameters Section */
    .parameters-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }

    .parameters-title {
        color: #495057;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.6rem 0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .parameter-card-header {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 0.4rem;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Compact input styling */
    .stNumberInput > div > div > input {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        font-weight: 500;
        height: 2.2rem;
        font-size: 0.9rem;
        padding: 0.3rem 0.5rem;
    }

    .stSelectbox > div > div > div {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        height: 2.2rem;
        font-size: 0.9rem;
    }

    .stNumberInput > div > div {
        width: 100%;
    }

    .stSelectbox > div {
        width: 100%;
    }

    /* Make input containers more compact */
    .stNumberInput {
        margin-bottom: 0.3rem;
    }

    .stSelectbox {
        margin-bottom: 0.3rem;
    }

    /* Compact labels */
    .stNumberInput label, .stSelectbox label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.2rem;
    }

    /* Professional results table */
    .results-table {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }

    .results-header {
        color: #495057;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        text-align: center;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 0.4rem;
    }

    .results-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-top: 0.6rem;
    }

    .result-item {
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }

    .result-label {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .result-value {
        color: #495057;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .result-positive {
        color: #dc3545;
    }

    .result-negative {
        color: #28a745;
    }

    .result-note {
        color: #6c757d;
        font-size: 0.8rem;
        font-style: italic;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .results-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Remove extra spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💸 Cash Flow Simulator</h1>', unsafe_allow_html=True)
st.markdown("Model cash flow under different scenarios")

# Professional parameters section with card-based layout
st.markdown("""
<div class="parameters-container">
</div>
""", unsafe_allow_html=True)

# Compact 3-column layout for parameters
row1_col1, row1_col2, row1_col3 = st.columns([2, 2, 2])

with row1_col1:
    st.markdown('<div class="parameter-card-header">💰 Budget & Duration</div>', unsafe_allow_html=True)
    col1a, col1b = st.columns(2)
    with col1a:
        budget = st.number_input("Budget (Millions)", min_value=1.0, max_value=999999.0, value=1000.0, step=1.0, format="%.0f",
                                help="Total project budget in millions")
    with col1b:
        duration = st.number_input("Duration (Mo)", min_value=1, max_value=999, value=12, step=1,
                                  help="Project duration in months")

with row1_col2:
    st.markdown('<div class="parameter-card-header">📊 Cash Flow Model</div>', unsafe_allow_html=True)
    col2a, col2b = st.columns(2)
    with col2a:
        cashflow_type = st.selectbox(
            "Pattern",
            ("Linear", "Highway", "Building", "S-Curve"),
            index=0,
            help="Cash flow distribution pattern"
        )
    with col2b:
        display_basis = st.selectbox("View", ("Monthly", "Quarterly", "Yearly"), help="Chart time scale")

with row1_col3:
    st.markdown('<div class="parameter-card-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
    col3a, col3b, col3c = st.columns(3)
    with col3a:
        start_delay = st.number_input("Start Delay", min_value=0, max_value=100, value=0, step=1,
                                     help="Delay before project starts (months)")
    with col3b:
        project_delay = st.number_input("Project Delay", min_value=0, max_value=100, value=0, step=1,
                                       help="Additional project duration (months)")
    with col3c:
        inflation = st.number_input("Inflation (%)", min_value=0.0, max_value=99.9, value=5.0, step=0.1, format="%.1f",
                                   help="Annual inflation rate")

# Professional Project Timeline
if cashflow_type != "Linear":
    if cashflow_type == "Highway":
        milestones = [
            (0, 'Project Start', '#6c757d'), (0.125, 'Design Phase', '#17a2b8'),
            (0.333, 'Mobilization', '#ffc107'), (0.75, 'Construction', '#dc3545'),
            (1.0, 'Project Closeout', '#28a745')
        ]
    elif cashflow_type == "Building":
        milestones = [
            (0, 'Project Start', '#6c757d'), (0.167, 'Design Phase', '#17a2b8'),
            (0.556, 'Foundation/Structure', '#ffc107'), (0.833, 'MEP/Finishing', '#dc3545'),
            (1.0, 'Project Completion', '#28a745')
        ]
    else:  # S-Curve
        milestones = [
            (0, 'Project Start', '#6c757d'), (0.25, 'Ramp-Up', '#17a2b8'),
            (0.5, 'Peak Activity', '#ffc107'), (0.75, 'Wind-Down', '#dc3545'),
            (1.0, 'Project Completion', '#28a745')
        ]

    st.markdown(f"""
    <div class="results-table" style="margin-top: 1rem;">
        <div class="results-header">Project Timeline - {cashflow_type} Pattern</div>
        <div style="background: #f8f9fa; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <div style="display: flex; width: 100%; height: 30px; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {"".join([f'<div style="width: {(milestones[i][0] - milestones[i-1][0]) * 100}%; height: 100%; background-color: {milestones[i][2]}; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 10px; text-shadow: 1px 1px 2px rgba(0,0,0,0.6); border-right: 1px solid rgba(255,255,255,0.2);">{milestones[i][1]}</div>' for i in range(1, len(milestones))])}
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8rem; color: #6c757d;">
                <span>0%</span>
                <span>25%</span>
                <span>50%</span>
                <span>75%</span>
                <span>100%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Calculation Logic
annual_inflation_rate = inflation / 100
monthly_inflation_rate = (1 + annual_inflation_rate)**(1/12) - 1

# BASELINE CASHFLOW (No delays, no inflation)
if cashflow_type == "Linear":
    baseline_monthly_cashflows = [budget / duration] * duration
elif cashflow_type == "Highway":
    baseline_monthly_cashflows = calculate_highway_cashflow(budget, duration)
elif cashflow_type == "Building":
    baseline_monthly_cashflows = calculate_building_cashflow(budget, duration)
else:  # S-Curve
    baseline_monthly_cashflows = calculate_scurve_cashflow(budget, duration)

# SIMULATED CASHFLOW (With delays and inflation)
# Total simulated duration includes project delays
simulated_duration = duration + project_delay
total_timeline = start_delay + simulated_duration

# Generate base simulated cashflow for the extended duration
if cashflow_type == "Linear":
    simulated_base_cashflows = [budget / simulated_duration] * simulated_duration
elif cashflow_type == "Highway":
    simulated_base_cashflows = calculate_highway_cashflow(budget, simulated_duration)
elif cashflow_type == "Building":
    simulated_base_cashflows = calculate_building_cashflow(budget, simulated_duration)
else:  # S-Curve
    simulated_base_cashflows = calculate_scurve_cashflow(budget, simulated_duration)

# Apply start delay (no cashflow during start delay) and inflation
simulated_monthly_cashflows = []

# Add zero cashflows for start delay period
for i in range(start_delay):
    simulated_monthly_cashflows.append(0)

# Add inflated cashflows for actual project duration
for i in range(simulated_duration):
    total_months_passed = start_delay + i
    inflation_factor = (1 + monthly_inflation_rate)**total_months_passed
    simulated_monthly_cashflows.append(simulated_base_cashflows[i] * inflation_factor)

# Prepare baseline and simulated data for display
# Baseline always starts at M1, regardless of delays
# Simulated is shifted by start delay and extended by project delay
max_timeline = len(simulated_monthly_cashflows)

# Extend baseline to match timeline: baseline starts at M1, pad end if needed
if len(baseline_monthly_cashflows) < max_timeline:
    extended_baseline = baseline_monthly_cashflows + [0] * (max_timeline - len(baseline_monthly_cashflows))
else:
    extended_baseline = baseline_monthly_cashflows[:max_timeline]

if display_basis == "Quarterly":
    num_periods = math.ceil(max_timeline / 3)
    baseline_data = [
        sum(extended_baseline[i*3:i*3+3])
        for i in range(num_periods)
    ]
    simulated_data = [
        sum(simulated_monthly_cashflows[i*3:i*3+3])
        for i in range(num_periods)
    ]
    labels = [f"Q{i+1}" for i in range(num_periods)]
    y_axis_label = "Quarterly Cash Flow (Millions)"
    x_axis_label = "Quarters"
elif display_basis == "Yearly":
    num_periods = math.ceil(max_timeline / 12)
    baseline_data = [
        sum(extended_baseline[i*12:i*12+12])
        for i in range(num_periods)
    ]
    simulated_data = [
        sum(simulated_monthly_cashflows[i*12:i*12+12])
        for i in range(num_periods)
    ]
    labels = [f"Y{i+1}" for i in range(num_periods)]
    y_axis_label = "Annual Cash Flow (Millions)"
    x_axis_label = "Years"
else:  # Monthly
    baseline_data = extended_baseline
    simulated_data = simulated_monthly_cashflows
    labels = [f"M{i+1}" for i in range(max_timeline)]
    y_axis_label = "Monthly Cash Flow (Millions)"
    x_axis_label = "Months"

# Calculate accumulated cash flows
baseline_accumulated = np.cumsum(baseline_data).tolist()
simulated_accumulated = np.cumsum(simulated_data).tolist()

# Plotly Chart
fig = go.Figure()

# Baseline Cash Flow (Green bars)
fig.add_trace(go.Bar(
    x=labels,
    y=baseline_data,
    name='Baseline Cash Flow',
    marker_color='rgba(40, 167, 69, 0.7)',
    marker_line=dict(color='rgba(40, 167, 69, 1)', width=1)
))

# Simulated Cash Flow (Blue bars)
fig.add_trace(go.Bar(
    x=labels,
    y=simulated_data,
    name='Simulated Cash Flow',
    marker_color='rgba(0, 123, 255, 0.7)',
    marker_line=dict(color='rgba(0, 123, 255, 1)', width=1)
))

# Baseline Cumulative (Green line)
fig.add_trace(go.Scatter(
    x=labels,
    y=baseline_accumulated,
    name='Baseline Cumulative',
    mode='lines+markers',
    line=dict(color='#28a745', width=3),
    marker=dict(size=5, color='#28a745'),
    yaxis='y2'
))

# Simulated Cumulative (Blue line)
fig.add_trace(go.Scatter(
    x=labels,
    y=simulated_accumulated,
    name='Simulated Cumulative',
    mode='lines+markers',
    line=dict(color='#007bff', width=3),
    marker=dict(size=5, color='#007bff'),
    yaxis='y2'
))

fig.update_layout(
    title={
        'text': f'Project Cash Flow Analysis - {cashflow_type} Pattern',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'color': '#495057', 'family': 'Arial, sans-serif'}
    },
    xaxis_title=x_axis_label,
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='#dee2e6',
        tickcolor='#6c757d'
    ),
    yaxis=dict(
        title=y_axis_label,
        side='left',
        showgrid=True,
        gridcolor='rgba(222, 226, 230, 0.5)',
        showline=True,
        linecolor='#dee2e6',
        tickcolor='#6c757d'
    ),
    yaxis2=dict(
        title='Cumulative Cash Flow (Millions)',
        side='right',
        overlaying='y',
        showgrid=False,
        showline=True,
        linecolor='#dee2e6',
        tickcolor='#6c757d'
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(248, 249, 250, 0.9)",
        bordercolor="#dee2e6",
        borderwidth=1,
        font=dict(size=11)
    ),
    plot_bgcolor='#ffffff',
    paper_bgcolor='rgba(0,0,0,0)',
    height=320,
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# Compact Results Section
baseline_budget = sum(baseline_monthly_cashflows)
simulated_budget = sum(simulated_monthly_cashflows)
budget_variance = ((simulated_budget / baseline_budget) - 1) * 100 if baseline_budget != 0 else 0

# Baseline and Export Controls
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    if st.button("📊 Set Baseline", help="Capture current scenario as baseline for comparison"):
        current_record = {
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': duration,
            'pattern': cashflow_type,
            'start_delay': start_delay,
            'project_delay': project_delay,
            'inflation': inflation,
            'simulated_budget': simulated_budget,
            'budget_variance': budget_variance
        }
        st.session_state.baseline_data = current_record
        st.session_state.comparison_records = []  # Clear existing comparisons
        st.success("✓ Baseline set successfully!")

with col2:
    if st.button("⚖️ Compare to Baseline", help="Add current scenario to comparison table"):
        if st.session_state.baseline_data is not None:
            current_record = {
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': duration,
                'pattern': cashflow_type,
                'start_delay': start_delay,
                'project_delay': project_delay,
                'inflation': inflation,
                'simulated_budget': simulated_budget,
                'budget_variance': budget_variance,
                'delta_from_baseline': simulated_budget - st.session_state.baseline_data['simulated_budget']
            }
            st.session_state.comparison_records.append(current_record)
            st.success("✓ Comparison added!")
        else:
            st.warning("⚠️ Please set a baseline first")

with col3:
    if st.button("💾 Export Comparisons", help="Download comparison table as CSV"):
        if st.session_state.baseline_data is not None and st.session_state.comparison_records:
            # Prepare comparison data for export
            export_data = []

            # Add baseline row
            baseline_row = st.session_state.baseline_data.copy()
            baseline_row['scenario_type'] = 'Baseline'
            baseline_row['delta_from_baseline'] = 0.0
            export_data.append(baseline_row)

            # Add comparison rows
            for record in st.session_state.comparison_records:
                record['scenario_type'] = 'Comparison'
                export_data.append(record)

            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)

            st.download_button(
                label="📁 Download Comparison CSV",
                data=csv,
                file_name=f"baseline_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No comparison data available")

with col4:
    if st.button("💰 Export Current Cashflow", help="Download current scenario cashflow as CSV"):
        # Prepare current cashflow data for export
        cashflow_data = []

        for i, (baseline_cf, simulated_cf) in enumerate(zip(baseline_data, simulated_data)):
            cashflow_data.append({
                'period': labels[i],
                'period_number': i + 1,
                'baseline_cashflow': baseline_cf,
                'simulated_cashflow': simulated_cf,
                'variance': simulated_cf - baseline_cf,
                'baseline_cumulative': baseline_accumulated[i],
                'simulated_cumulative': simulated_accumulated[i]
            })

        df = pd.DataFrame(cashflow_data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="📈 Download Cashflow CSV",
            data=csv,
            file_name=f"project_cashflow_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.markdown("**Financial Impact Analysis**")
st.markdown(f"""
<div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.5rem 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem;">
        <span><strong>Original Budget:</strong> {baseline_budget:,.1f}M</span>
        <span><strong>Simulated Budget:</strong> {simulated_budget:,.1f}M</span>
        <span><strong>Budget Variance:</strong> <span style="color: {'#dc3545' if budget_variance > 0 else '#28a745'}; font-weight: 600;">{"+" if budget_variance > 0 else ""}{budget_variance:,.1f}%</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display comparison table if baseline exists and comparisons have been made
if st.session_state.baseline_data is not None:
    st.markdown("**📊 Baseline & Comparison Analysis**")

    # Prepare display data
    display_data = []

    # Add baseline row
    baseline_display = {
        'Scenario': '🔵 Baseline',
        'Timestamp': st.session_state.baseline_data['timestamp'],
        'Duration': f"{st.session_state.baseline_data['duration']} mo",
        'Pattern': st.session_state.baseline_data['pattern'],
        'Start Delay': f"{st.session_state.baseline_data['start_delay']} mo",
        'Project Delay': f"{st.session_state.baseline_data['project_delay']} mo",
        'Inflation': f"{st.session_state.baseline_data['inflation']}%",
        'Simulated Budget': f"{st.session_state.baseline_data['simulated_budget']:,.1f}M",
        'Budget Variance': f"{st.session_state.baseline_data['budget_variance']:+.1f}%",
        'Delta from Baseline': "0.0M"
    }
    display_data.append(baseline_display)

    # Add comparison rows
    for i, record in enumerate(st.session_state.comparison_records):
        comparison_display = {
            'Scenario': f'⚖️ Comparison {i+1}',
            'Timestamp': record['timestamp'],
            'Duration': f"{record['duration']} mo",
            'Pattern': record['pattern'],
            'Start Delay': f"{record['start_delay']} mo",
            'Project Delay': f"{record['project_delay']} mo",
            'Inflation': f"{record['inflation']}%",
            'Simulated Budget': f"{record['simulated_budget']:,.1f}M",
            'Budget Variance': f"{record['budget_variance']:+.1f}%",
            'Delta from Baseline': f"{record['delta_from_baseline']:+.1f}M"
        }
        display_data.append(comparison_display)

    if display_data:
        df_display = pd.DataFrame(display_data)

        # Custom CSS for the comparison table
        st.markdown("""
        <style>
        .comparison-table {
            font-size: 0.85rem;
        }
        .comparison-table th {
            background-color: #f8f9fa;
            color: #495057;
            font-weight: 600;
            text-align: center;
            padding: 0.5rem 0.3rem;
            border: 1px solid #dee2e6;
        }
        .comparison-table td {
            text-align: center;
            padding: 0.4rem 0.3rem;
            border: 1px solid #dee2e6;
        }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(
            df_display,
            width='stretch',
            hide_index=True,
            column_config={
                "Scenario": st.column_config.TextColumn("Scenario", width="small"),
                "Timestamp": st.column_config.TextColumn("Time", width="medium"),
                "Duration": st.column_config.TextColumn("Duration", width="small"),
                "Pattern": st.column_config.TextColumn("Pattern", width="small"),
                "Start Delay": st.column_config.TextColumn("Start Delay", width="small"),
                "Project Delay": st.column_config.TextColumn("Proj. Delay", width="small"),
                "Inflation": st.column_config.TextColumn("Inflation", width="small"),
                "Simulated Budget": st.column_config.TextColumn("Sim. Budget", width="medium"),
                "Budget Variance": st.column_config.TextColumn("Budget Var.", width="small"),
                "Delta from Baseline": st.column_config.TextColumn("Δ from Baseline", width="medium")
            }
        )

        # Summary statistics if there are comparisons
        if st.session_state.comparison_records:
            st.markdown("**📈 Impact Summary**")

            max_delta = max([abs(r['delta_from_baseline']) for r in st.session_state.comparison_records])
            avg_delta = sum([r['delta_from_baseline'] for r in st.session_state.comparison_records]) / len(st.session_state.comparison_records)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Budget Impact", f"{max_delta:,.1f}M")
            with col2:
                st.metric("Avg Budget Impact", f"{avg_delta:+.1f}M")
            with col3:
                st.metric("Total Comparisons", len(st.session_state.comparison_records))

