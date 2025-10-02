import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- Page configuration ---
st.set_page_config(
    page_title="EVM Performance Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for professional appearance
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

    .parameters-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }

    .parameter-card-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.875rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stNumberInput > div > div > input {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
        height: 2.5rem;
        font-size: 0.875rem;
        padding: 0.5rem 0.75rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1), 0 2px 6px rgba(0, 0, 0, 0.15);
        outline: none;
        transform: translateY(-1px);
    }

    .stNumberInput > div > div > input:hover {
        border-color: #b8b9ba;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
    }

    .stSelectbox > div > div > div {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox > div > div > div:hover {
        border-color: #b8b9ba;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
    }

    .stSelectbox > div > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1), 0 2px 6px rgba(0, 0, 0, 0.15);
    }

    .stNumberInput {
        margin-bottom: 0.3rem;
    }

    .stNumberInput label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.2rem;
    }

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

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }

    .metric-label {
        font-weight: 600;
        color: #495057;
        font-size: 0.9rem;
    }

    .metric-value {
        font-weight: 700;
        font-size: 1rem;
    }

    .metric-formula {
        font-size: 0.8rem;
        color: #6c757d;
        font-style: italic;
    }

    .status-good {
        color: #28a745;
        text-shadow: 0 1px 2px rgba(40, 167, 69, 0.3);
    }
    .status-warning {
        color: #ffc107;
        text-shadow: 0 1px 2px rgba(255, 193, 7, 0.3);
    }
    .status-danger {
        color: #dc3545;
        text-shadow: 0 1px 2px rgba(220, 53, 69, 0.3);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 1rem 0 0.8rem 0;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    }

    .subsection-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a5568;
        margin: 0.8rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 EVM Calculator</h1>', unsafe_allow_html=True)
st.markdown("Earned Value Management calculations")

# Professional Project Input Section
st.markdown('<div class="section-header">Project Parameters</div>', unsafe_allow_html=True)

# Row 1: Primary Values
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Budget & Cost**")
    bac = st.number_input("Budget at Completion (BAC)", min_value=0.0, value=5000.0, step=100.0, format="%.0f",
                         help="Total project budget", key="bac")
    ac = st.number_input("Actual Cost (AC)", min_value=0.0, value=900.0, step=50.0, format="%.0f",
                        help="Actual cost incurred for work completed", key="ac")

with col2:
    st.markdown("**📋 Planned Values**")
    pv_method = st.selectbox("PV Method", ["Enter Value", "Linear Curve", "S-Curve"],
                            help="Choose how to determine Planned Value", key="pv_method")

    if pv_method == "Enter Value":
        pv = st.number_input("Planned Value (PV)", min_value=0.0, value=1000.0, step=50.0, format="%.0f",
                            help="Budgeted cost of work scheduled", key="pv")
    else:
        pv = 0  # Will be calculated from curve

    percent_complete = st.number_input("% Complete", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                                      help="Percentage of project completed", key="pct")

with col3:
    st.markdown("**⏱️ Duration**")
    original_duration = st.number_input("Original Duration (Days)", min_value=1.0, value=100.0, step=1.0, format="%.0f",
                                       help="Planned project duration", key="od")
    actual_duration = st.number_input("Used Duration (Days)", min_value=1.0, value=60.0, step=1.0, format="%.0f",
                                     help="Actual time elapsed", key="ad")

# S-curve function (using beta distribution approximation)
def calculate_s_curve_value(time_ratio, alpha=2, beta=2):
    """Calculate S-curve value using beta distribution approximation"""
    if time_ratio <= 0:
        return 0
    elif time_ratio >= 1:
        return 1
    else:
        # Simplified S-curve using polynomial approximation
        return 3 * time_ratio**2 - 2 * time_ratio**3

# Calculate derived values
ev = bac * (percent_complete / 100)
time_elapsed_pct = (actual_duration / original_duration) * 100 if original_duration > 0 else 0
budget_utilized_pct = (ac / bac) * 100 if bac > 0 else 0

# Calculate PV based on method and find Earned Schedule (ES) for SPIe
es = None
spie = None

if pv_method == "Linear Curve":
    # Linear curve: PV at current time
    time_ratio = actual_duration / original_duration if original_duration > 0 else 0
    pv = bac * min(time_ratio, 1.0)

    # Find Earned Schedule (ES) where PV = EV
    if ev > 0 and bac > 0:
        ev_ratio = ev / bac
        es = ev_ratio * original_duration
        spie = es / actual_duration if actual_duration > 0 else 0

elif pv_method == "S-Curve":
    # S-curve: PV at current time
    time_ratio = actual_duration / original_duration if original_duration > 0 else 0
    s_curve_value = calculate_s_curve_value(time_ratio)
    pv = bac * s_curve_value

    # Find Earned Schedule (ES) where S-curve PV = EV using iterative search
    if ev > 0 and bac > 0:
        ev_ratio = ev / bac
        # Search for time where s-curve equals ev_ratio
        for search_day in range(1, int(original_duration * 2)):
            search_ratio = search_day / original_duration
            if calculate_s_curve_value(search_ratio) >= ev_ratio:
                es = search_day
                break

        if es:
            spie = es / actual_duration if actual_duration > 0 else 0

# Calculated Values - Compact professional display
st.markdown('<div class="subsection-title">Calculated Values</div>', unsafe_allow_html=True)
completion_efficiency = (percent_complete / time_elapsed_pct) if time_elapsed_pct > 0 else 0

if pv_method == "Enter Value":
    # Standard 4-column layout for manual PV entry
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 0.6rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.3rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.9rem;">
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Earned Value (EV)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #007bff;">{ev:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">BAC × {percent_complete:.1f}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Time Elapsed</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #495057;">{time_elapsed_pct:.1f}%</div>
                <div style="font-size: 0.75rem; color: #6c757d;">UD/OD = {actual_duration:.0f}/{original_duration:.0f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Budget Utilized</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if budget_utilized_pct <= 100 else '#dc3545'};">{budget_utilized_pct:.1f}%</div>
                <div style="font-size: 0.75rem; color: #6c757d;">AC/BAC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Completion Efficiency</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if completion_efficiency >= 1 else '#dc3545' if completion_efficiency < 0.8 else '#ffc107'};">{completion_efficiency:.2f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">% Complete / % Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 6-column layout for curve methods with SPIe
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 0.6rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.3rem 0;">
        <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.8rem; font-size: 0.85rem;">
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Earned Value (EV)</div>
                <div style="font-size: 1rem; font-weight: 700; color: #007bff;">{ev:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6c757d;">BAC × {percent_complete:.1f}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Planned Value (PV)</div>
                <div style="font-size: 1rem; font-weight: 700; color: #28a745;">{pv:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6c757d;">{pv_method}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Time Elapsed</div>
                <div style="font-size: 1rem; font-weight: 700; color: #495057;">{time_elapsed_pct:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6c757d;">UD/OD</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Budget Utilized</div>
                <div style="font-size: 1rem; font-weight: 700; color: {'#28a745' if budget_utilized_pct <= 100 else '#dc3545'};">{budget_utilized_pct:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6c757d;">AC/BAC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Earned Schedule (ES)</div>
                <div style="font-size: 1rem; font-weight: 700; color: #6f42c1;">{'%.1f' % es if es is not None else 'N/A'}</div>
                <div style="font-size: 0.7rem; color: #6c757d;">Days @ PV=EV</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">SPIe</div>
                <div style="font-size: 1rem; font-weight: 700; color: {'#28a745' if spie and spie >= 1 else '#dc3545' if spie and spie < 1 else '#6c757d'};">{'%.3f' % spie if spie is not None else 'N/A'}</div>
                <div style="font-size: 0.7rem; color: #6c757d;">ES/AD</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Calculation logic ---
if pv >= 0 and ev >= 0 and ac >= 0 and bac > 0:
    # Calculate basic EVM metrics
    cv = ev - ac  # Cost Variance
    sv = ev - pv  # Schedule Variance
    cpi = ev / ac if ac > 0 else float('nan')  # Cost Performance Index (N/A for AC=0)
    spi = ev / pv if pv > 0 else 0  # Schedule Performance Index

    # Calculate forecasting metrics
    etc = (bac - ev) / cpi if ac > 0 and cpi > 0 else float('inf')  # Estimate to Complete
    eac = ac + etc  # Estimate at Completion
    vac = bac - eac  # Variance at Completion
    tcpi_bac = (bac - ev) / (bac - ac) if (bac - ac) != 0 else 0  # To Complete Performance Index

    # Compact EVM Analysis Results
    st.markdown('<div class="section-header">EVM Analysis Results</div>', unsafe_allow_html=True)

    # Performance Metrics - Compact single row
    st.markdown('<div class="subsection-title">Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 0.6rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.3rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.9rem;">
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Cost Variance (CV)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if cv > 0 else '#dc3545' if cv < 0 else '#ffc107'};">{cv:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">EV - AC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Schedule Variance (SV)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if sv > 0 else '#dc3545' if sv < 0 else '#ffc107'};">{sv:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">EV - PV</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Cost Performance (CPI)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if cpi > 1 else '#dc3545' if cpi < 1 else '#ffc107'};">{cpi:.3f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">EV / AC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Schedule Performance (SPI)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if spi > 1 else '#dc3545' if spi < 1 else '#ffc107'};">{spi:.3f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">EV / PV</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Forecasting - Compact single row
    st.markdown('<div class="subsection-title">Project Forecasting</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 0.6rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.3rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.9rem;">
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Estimate at Completion</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if eac <= bac else '#dc3545'};">{eac:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">AC + ETC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Estimate to Complete</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #495057;">{etc:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">(BAC-EV)/CPI</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">Variance at Completion</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if vac >= 0 else '#dc3545'};">{vac:,.0f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">BAC - EAC</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; color: #495057;">To Complete Index</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#28a745' if tcpi_bac <= 1 else '#ffc107' if tcpi_bac <= 1.2 else '#dc3545'};">{tcpi_bac:.3f}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">(BAC-EV)/(BAC-AC)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create EVM Progress Chart (only for curve methods)
    if pv_method != "Enter Value":
        st.subheader("EVM Curve Analysis")

        fig = go.Figure()

        # Timeline for full PV curve (0 to original duration)
        pv_timeline = np.linspace(0, original_duration, int(original_duration) + 1)

        # Timeline for EV and AC (0 to used duration)
        actual_timeline = np.linspace(0, actual_duration, int(actual_duration) + 1)

        # Generate PV curve data
        if pv_method == "Linear Curve":
            pv_curve = [(day / original_duration) * bac for day in pv_timeline]
        else:  # S-Curve
            pv_curve = [bac * calculate_s_curve_value(day / original_duration) for day in pv_timeline]

        # Generate EV curve based on selected curve method
        if pv_method == "Linear Curve":
            ev_curve = [(day / actual_duration) * ev for day in actual_timeline]
        else:  # S-Curve
            # EV follows S-curve pattern scaled to actual progress
            ev_curve = [ev * calculate_s_curve_value(day / actual_duration) for day in actual_timeline]

        # Generate AC curve based on selected curve method
        if pv_method == "Linear Curve":
            ac_curve = [(day / actual_duration) * ac for day in actual_timeline]
        else:  # S-Curve
            # AC follows S-curve pattern scaled to actual cost
            ac_curve = [ac * calculate_s_curve_value(day / actual_duration) for day in actual_timeline]

        # Add PV curve (full duration)
        fig.add_trace(go.Scatter(
            x=pv_timeline, y=pv_curve,
            mode='lines',
            name=f'Planned Value ({pv_method})',
            line=dict(color='#28a745', width=3)
        ))

        # Add EV curve (to used duration)
        fig.add_trace(go.Scatter(
            x=actual_timeline, y=ev_curve,
            mode='lines',
            name='Earned Value (EV)',
            line=dict(color='#007bff', width=3)
        ))

        # Add AC curve (to used duration)
        fig.add_trace(go.Scatter(
            x=actual_timeline, y=ac_curve,
            mode='lines',
            name='Actual Cost (AC)',
            line=dict(color='#dc3545', width=3)
        ))

        # Add current point markers
        fig.add_trace(go.Scatter(
            x=[actual_duration], y=[pv],
            mode='markers',
            name='Current PV',
            marker=dict(size=10, color='#28a745', symbol='circle')
        ))

        fig.add_trace(go.Scatter(
            x=[actual_duration], y=[ev],
            mode='markers',
            name='Current EV',
            marker=dict(size=10, color='#007bff', symbol='square')
        ))

        fig.add_trace(go.Scatter(
            x=[actual_duration], y=[ac],
            mode='markers',
            name='Current AC',
            marker=dict(size=10, color='#dc3545', symbol='diamond')
        ))

        # Add vertical line at used duration
        fig.add_vline(
            x=actual_duration,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Used Duration: {actual_duration:.0f} days",
            annotation_position="top"
        )

        # Add earned schedule line if available
        if es:
            fig.add_vline(
                x=es,
                line_dash="dash",
                line_color="purple",
                annotation_text=f"Earned Schedule: {es:.1f} days",
                annotation_position="bottom"
            )

        fig.update_layout(
            title={
                'text': f'EVM Curve Analysis - {pv_method}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#495057'}
            },
            xaxis_title='Project Timeline (Days)',
            yaxis_title='Cost',
            plot_bgcolor='#ffffff',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(248, 249, 250, 0.9)",
                bordercolor="#dee2e6",
                borderwidth=1
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(222, 226, 230, 0.5)',
                showline=True,
                linecolor='#dee2e6',
                range=[0, original_duration * 1.05]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(222, 226, 230, 0.5)',
                showline=True,
                linecolor='#dee2e6'
            )
        )

        st.plotly_chart(fig, width='stretch')

    # Compact Project Status Summary
    st.markdown('<div class="subsection-title">Project Status Summary</div>', unsafe_allow_html=True)

    # Determine overall status
    cost_status = "Under Budget" if cv > 0 else "Over Budget" if cv < 0 else "On Budget"
    schedule_status = "Ahead" if sv > 0 else "Behind" if sv < 0 else "On Schedule"
    cost_color = "#28a745" if cv > 0 else "#dc3545" if cv < 0 else "#ffc107"
    schedule_color = "#28a745" if sv > 0 else "#dc3545" if sv < 0 else "#ffc107"

    # Overall project health assessment
    health_score = (cpi + spi) / 2
    health_status = "Excellent" if health_score > 1.1 else "Good" if health_score > 0.95 else "Fair" if health_score > 0.85 else "Poor"
    health_color = "#28a745" if health_score > 0.95 else "#ffc107" if health_score > 0.85 else "#dc3545"

    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; border: 1px solid #dee2e6; margin: 0.3rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.85rem; text-align: center;">
            <div><strong>Cost:</strong> <span style="color: {cost_color}; font-weight: 600;">{cost_status}</span></div>
            <div><strong>Schedule:</strong> <span style="color: {schedule_color}; font-weight: 600;">{schedule_status}</span></div>
            <div><strong>Health:</strong> <span style="color: {health_color}; font-weight: 600;">{health_status}</span></div>
            <div><strong>Progress:</strong> <span style="font-weight: 600;">{percent_complete:.1f}% Complete</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please ensure all input values are valid and Budget at Completion is greater than zero.")

st.markdown("""
<div style="text-align: center; margin-top: 1rem; padding: 0.6rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 6px; border-top: 2px solid #6c757d;">
    <div style="color: #495057; font-size: 0.85rem; margin-bottom: 0.2rem;">
        <strong>EVM Performance Analyzer v1.0</strong>
    </div>
    <div style="color: #6c757d; font-size: 0.75rem;">
        Developed by <strong>Dr. Khalid Ahmad Khan</strong> • Engineering Management Solutions • September 2025
    </div>
</div>
""", unsafe_allow_html=True)