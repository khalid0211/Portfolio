import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import core utilities
from core.utils import safe_divide

# Add pages directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from Project Analysis for Executive Brief
create_portfolio_executive_summary = None
safe_llm_request = None

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "project_analysis",
        os.path.join(os.path.dirname(__file__), "3_Project_Analysis.py")
    )
    project_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_analysis)
    create_portfolio_executive_summary = project_analysis.create_portfolio_executive_summary
    safe_llm_request = project_analysis.safe_llm_request
except Exception as e:
    print(f"Warning: Could not import from Project Analysis: {e}")
    create_portfolio_executive_summary = None
    safe_llm_request = None

# Page configuration
st.set_page_config(
    page_title="CPO Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header simple style */
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Executive metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 15px 15px 0 0;
    }
    
    /* Alert banners with modern styling */
    .alert-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 82, 0.3);
        position: relative;
        overflow: hidden;
        animation: alertPulse 3s ease-in-out infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 15px 35px rgba(238, 90, 82, 0.3); }
        50% { box-shadow: 0 20px 45px rgba(238, 90, 82, 0.5); }
    }
    
    .alert-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: alertShine 2s infinite;
    }
    
    @keyframes alertShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Section cards with premium styling */
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.12);
    }
    
    /* Modern metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        min-height: 130px;
        max-height: 160px;
        overflow: visible;
        width: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="metric-container"] > div > div > div > div {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        word-wrap: break-word !important;
        white-space: normal !important;
        line-height: 1.2 !important;
        max-width: 100% !important;
        overflow-wrap: break-word !important;
        hyphens: auto !important;
        display: block !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }

    /* Additional responsive sizing for very long numbers */
    @media (max-width: 1400px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 1rem !important;
        }
    }

    @media (max-width: 1200px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 0.95rem !important;
        }
    }

    @media (max-width: 768px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 0.9rem !important;
        }
    }
    
    /* Executive section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators with professional styling */
    .status-critical {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        border-left: 5px solid #e53e3e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(229, 62, 62, 0.15);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fef5e7 0%, #feebc8 100%);
        border-left: 5px solid #d69e2e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(214, 158, 46, 0.15);
    }
    
    .status-success {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #38a169;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(56, 161, 105, 0.15);
    }
    
    /* Action items with executive styling */
    .action-item {
        background: linear-gradient(135deg, rgba(255,243,205,0.9) 0%, rgba(254,235,200,0.9) 100%);
        border-left: 4px solid #d69e2e;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(214, 158, 46, 0.1);
        transition: all 0.3s ease;
    }
    
    .action-item:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 30px rgba(214, 158, 46, 0.2);
    }
    
    /* Executive button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Selectbox and slider styling */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def format_currency(amount, currency_symbol='$', currency_postfix='', thousands=True):
    """Format currency values with proper symbol and postfix"""
    if thousands:
        formatted_amount = f"{amount/1000:.0f}K"
    else:
        formatted_amount = f"{amount:,.0f}"

    if currency_postfix:
        return f"{currency_symbol} {formatted_amount} {currency_postfix}"
    else:
        return f"{currency_symbol} {formatted_amount}"

def categorize_project_health(row):
    """Categorize project health based on CPI and SPI"""
    cpi = row.get('CPI', 0)
    spi = row.get('SPI', 0)

    if cpi >= 0.95 and spi >= 0.95:
        return 'Healthy'
    elif cpi >= 0.85 and spi >= 0.85:
        return 'At Risk'
    else:
        return 'Critical'

def map_columns_to_standard(df):
    """Map Portfolio Analysis columns to dashboard expected columns"""
    # Create a mapping from Portfolio Analysis format to Dashboard format
    column_mapping = {
        # Real batch calculation results (from Project Analysis)
        'bac': 'Budget',
        'ac': 'Actual Cost',
        'ev': 'Earned Value',
        'pv': 'Plan Value',
        'etc': 'ETC',
        'eac': 'EAC',
        'cpi': 'CPI',
        'spi': 'SPI',
        'spie': 'SPIe',
        'project_name': 'Project Name',
        'project_id': 'Project ID',
        'organization': 'Organization',
        'project_manager': 'Project Manager',
        'plan_start': 'Plan Start',
        'plan_finish': 'Plan Finish',
        'cv': 'Cost Variance',
        'sv': 'Schedule Variance',
        'vac': 'VAC',
        'percent_complete': 'Percent Complete',
        'actual_duration_months': 'Actual Duration',
        'original_duration_months': 'Original Duration',
        'forecast_duration': 'Forecast Duration',
        'present_value': 'Present Value',
        # File Management data format (uppercase) - for fallback
        'BAC': 'Budget',
        'AC': 'Actual Cost',
        'Project ID': 'Project ID',
        'Project': 'Project Name',
        'Organization': 'Organization',
        'Project Manager': 'Project Manager',
        'Plan Start': 'Plan Start',
        'Plan Finish': 'Plan Finish',
        'CPI': 'CPI',
        'SPI': 'SPI'
    }

    # Create a copy of the dataframe
    mapped_df = df.copy()

    # Rename columns if they exist in the source format
    for source_col, target_col in column_mapping.items():
        if source_col in mapped_df.columns and target_col not in mapped_df.columns:
            mapped_df[target_col] = mapped_df[source_col]

    # Ensure all expected columns exist with default values
    expected_columns = {
        'Budget': 0,
        'Actual Cost': 0,
        'Earned Value': 0,
        'Plan Value': 0,
        'CPI': 1.0,
        'SPI': 1.0,
        'SPIe': 1.0,
        'ETC': 0,
        'EAC': 0,
        'Project Name': 'Unknown',
        'Project ID': 'Unknown',
        'Organization': 'Unknown',
        'Project Manager': 'Unknown'
    }

    for col, default_value in expected_columns.items():
        if col not in mapped_df.columns:
            if col in ['Project Name', 'Project ID', 'Organization', 'Project Manager']:
                mapped_df[col] = default_value
            else:
                mapped_df[col] = default_value

    return mapped_df

def calculate_portfolio_metrics(df):
    """Calculate key portfolio-level metrics"""
    # First map columns to expected format
    df = map_columns_to_standard(df)

    metrics = {}

    # Basic counts and totals
    metrics['total_projects'] = len(df)

    # Use get() with default values for missing columns
    metrics['total_budget'] = df.get('Budget', pd.Series([0])).sum()
    metrics['total_actual_cost'] = df.get('Actual Cost', pd.Series([0])).sum()
    metrics['total_earned_value'] = df.get('Earned Value', pd.Series([0])).sum()
    metrics['total_planned_value'] = df.get('Plan Value', pd.Series([0])).sum()
    metrics['total_etc'] = df.get('ETC', pd.Series([0])).sum()
    metrics['total_eac'] = df.get('EAC', pd.Series([0])).sum()
    
    # Portfolio performance indices - use portfolio-level sums to avoid unrealistic individual values
    # CPI = SUM(EV)/SUM(AC), SPI = SUM(EV)/SUM(PV)
    metrics['portfolio_cpi'] = metrics['total_earned_value'] / metrics['total_actual_cost'] if metrics['total_actual_cost'] > 0 else 0
    metrics['portfolio_spi'] = metrics['total_earned_value'] / metrics['total_planned_value'] if metrics['total_planned_value'] > 0 else 0
    
    # Forecast metrics
    metrics['forecast_overrun'] = metrics['total_eac'] - metrics['total_budget']
    metrics['overrun_percentage'] = (metrics['forecast_overrun'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
    
    # Health distribution
    df['Health_Category'] = df.apply(categorize_project_health, axis=1)
    health_counts = df['Health_Category'].value_counts()
    metrics['critical_projects'] = health_counts.get('Critical', 0)
    metrics['at_risk_projects'] = health_counts.get('At Risk', 0)
    metrics['healthy_projects'] = health_counts.get('Healthy', 0)
    
    return metrics

def main():
    # Executive Header
    st.markdown('<h1 class="main-header">📊 Portfolio Executive Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Portfolio Analytics and graphs")
    
    # Get currency settings from Portfolio Analysis if available
    # Try multiple sources in order of preference:
    # 1. Dashboard-specific settings (from Generate Executive Dashboard button)
    # 2. Widget session state (from current Portfolio Analysis inputs)
    # 3. Saved controls (from config_dict)
    # 4. Default values

    saved_controls = getattr(st.session_state, 'config_dict', {}).get('controls', {})

    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        saved_controls.get('currency_symbol', '$')
    )

    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        saved_controls.get('currency_postfix', '')
    )



    # Check for data from Portfolio Analysis first
    if hasattr(st.session_state, 'dashboard_data') and st.session_state.dashboard_data is not None:
        df = st.session_state.dashboard_data.copy()
        st.sidebar.success("✅ Using data from Portfolio Analysis")
        st.sidebar.info(f"📊 {len(df)} projects loaded")
        if currency_symbol != '$' or currency_postfix != '':
            st.sidebar.info(f"💱 Currency: {currency_symbol} {currency_postfix}")
    elif hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
        # Fallback: use batch_results directly
        df = st.session_state.batch_results.copy()
        st.sidebar.success("✅ Using batch results from Portfolio Analysis")
        st.sidebar.info(f"📊 {len(df)} projects loaded")
    else:
        # File upload option as fallback
        st.sidebar.info("💡 **Recommended Workflow:**")
        st.sidebar.markdown("1. Go to **Portfolio Analysis**")
        st.sidebar.markdown("2. Upload data & run calculations")
        st.sidebar.markdown("3. Click **Generate Executive Dashboard**")
        st.sidebar.markdown("---")

        uploaded_file = st.sidebar.file_uploader(
            "Or Upload Portfolio Data Directly",
            type=['csv'],
            help="Upload your batch_evm_results.csv file or use Portfolio Analysis to generate data"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Clean and process the uploaded data
            df.columns = df.columns.str.strip()
            numeric_columns = ['Budget', 'Actual Cost', 'Earned Value', 'Plan Value',
                              'CPI', 'SPI', 'ETC', 'EAC', '% Budget Used', '% Time Used']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            # No data available - inform user of proper workflow
            st.sidebar.warning("⚠️ No portfolio data available")
            st.sidebar.markdown("**Please use one of these options:**")
            st.sidebar.markdown("• **File Management** → Run Batch EVM → **Portfolio Analysis**")
            st.sidebar.markdown("• **Upload CSV file** using the uploader above")
            df = None
    
    if df is None:
        # Show helpful guidance when no data is available
        st.markdown("## 🚀 Get Started")
        st.info("To view your portfolio analysis dashboard, you'll need to process your project data first.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📁 Recommended Workflow
            1. **File Management** → Upload your project data
            2. **Run Batch EVM** → Process all projects
            3. **Portfolio Analysis** → Return here for insights
            """)
            if st.button("📁 Go to File Management", type="primary"):
                st.switch_page("pages/1_File_Management.py")

        with col2:
            st.markdown("""
            ### 📊 Direct Upload
            Use the **CSV file uploader** in the sidebar to directly upload your batch EVM results.
            """)

        st.stop()

    # Map columns to expected format and calculate metrics
    df = map_columns_to_standard(df)
    metrics = calculate_portfolio_metrics(df)

    # Create budget tier categories early for use in charts
    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
    default_tier_config = {
        'cutoff_points': [4000, 8000, 15000],
        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
    }
    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])

    def categorize_budget_by_tier(budget):
        """Assign tier based on configurable budget ranges"""
        if pd.isna(budget):
            return "Unknown"
        elif budget >= cutoffs[2]:  # Tier 4 (highest)
            return tier_names[3]
        elif budget >= cutoffs[1]:  # Tier 3
            return tier_names[2]
        elif budget >= cutoffs[0]:  # Tier 2
            return tier_names[1]
        else:  # Tier 1 (lowest)
            return tier_names[0]

    df['Budget_Category'] = df['Budget'].apply(categorize_budget_by_tier)

    # Use portfolio-level calculations for CPI/SPI, keep weighted average for SPIe
    portfolio_cpi_weighted = metrics['portfolio_cpi']  # Already calculated as SUM(EV)/SUM(AC)
    portfolio_spi_weighted = metrics['portfolio_spi']  # Already calculated as SUM(EV)/SUM(PV)
    portfolio_spie_weighted = (df['SPIe'] * df['Budget']).sum() / df['Budget'].sum() if df['Budget'].sum() > 0 else 0  # Keep weighted for SPIe

    # Critical Alert Banner
    if portfolio_cpi_weighted < 0.85 or portfolio_spi_weighted < 0.85:
        st.markdown(f"""
        <div class="alert-banner">
            🚨 CRITICAL PORTFOLIO ALERT: Immediate intervention required •
            Portfolio CPI: {portfolio_cpi_weighted:.2f} •
            Portfolio SPI: {portfolio_spi_weighted:.2f}
        </div>
        """, unsafe_allow_html=True)

    # Key Performance Indicators
    st.markdown('<div class="section-header">📈 Executive Portfolio Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Projects",
            value=f"{metrics['total_projects']:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Portfolio CPI",
            value=f"{portfolio_cpi_weighted:.3f}",
            delta=f"{(portfolio_cpi_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_cpi_weighted < 1 else "normal"
        )

    with col3:
        st.metric(
            label="Portfolio SPI",
            value=f"{portfolio_spi_weighted:.3f}",
            delta=f"{(portfolio_spi_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_spi_weighted < 1 else "normal"
        )

    with col4:
        st.metric(
            label="Portfolio SPIe",
            value=f"{portfolio_spie_weighted:.3f}",
            delta=f"{(portfolio_spie_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_spie_weighted < 1 else "normal"
        )
    
    # Performance Metrics Section
    st.markdown('<div class="section-header">⚡ Strategic Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CPI Gauge - Dial with Needle
        fig_cpi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_cpi_weighted,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cost Performance Index (CPI)", 'font': {'size': 14}},
            number = {'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 2.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.8], 'color': "#ff6b6b"},      # Red - Poor
                    {'range': [0.8, 1.0], 'color': "#ffd93d"},    # Yellow - Caution
                    {'range': [1.0, 1.5], 'color': "#6bcf7f"},    # Green - Good
                    {'range': [1.5, 2.0], 'color': "#4ecdc4"}     # Teal - Excellent
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_cpi_weighted
                }
            }
        ))
        fig_cpi.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cpi, use_container_width=True)
    
    with col2:
        # SPI Gauge - Dial with Needle
        fig_spi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_spi_weighted,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Schedule Performance Index (SPI)", 'font': {'size': 14}},
            number = {'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 2.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.8], 'color': "#ff6b6b"},      # Red - Behind Schedule
                    {'range': [0.8, 1.0], 'color': "#ffd93d"},    # Yellow - Slightly Behind
                    {'range': [1.0, 1.5], 'color': "#6bcf7f"},    # Green - On/Ahead Schedule
                    {'range': [1.5, 2.0], 'color': "#4ecdc4"}     # Teal - Excellent
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_spi_weighted
                }
            }
        ))
        fig_spi.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_spi, use_container_width=True)

    with col3:
        # % Budget Used (AC/BAC) Gauge - Dial with Needle
        portfolio_budget_used = (metrics['total_actual_cost'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
        fig_budget_used = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_budget_used,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Budget Used (AC/BAC)", 'font': {'size': 14}},
            number = {'suffix': "%", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': "#6bcf7f"},       # Green - Under Budget
                    {'range': [70, 90], 'color': "#ffd93d"},      # Yellow - Approaching Budget
                    {'range': [90, 110], 'color': "#ff9500"},     # Orange - At Budget
                    {'range': [110, 150], 'color': "#ff6b6b"}     # Red - Over Budget
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_budget_used
                }
            }
        ))
        fig_budget_used.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_budget_used, use_container_width=True)

    with col4:
        # % Earned Value (EV/BAC) Gauge - Dial with Needle
        portfolio_earned_value_pct = (metrics['total_earned_value'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
        fig_earned_value = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_earned_value_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Earned Value (EV/BAC)", 'font': {'size': 14}},
            number = {'suffix': "%", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 120], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': "#ff6b6b"},       # Red - Low Performance
                    {'range': [60, 80], 'color': "#ffd93d"},      # Yellow - Below Target
                    {'range': [80, 100], 'color': "#6bcf7f"},     # Green - Good Performance
                    {'range': [100, 120], 'color': "#4ecdc4"}     # Teal - Excellent Performance
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_earned_value_pct
                }
            }
        ))
        fig_earned_value.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_earned_value, use_container_width=True)

    # Project Health Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">🏥 Portfolio Health Analytics</div>', unsafe_allow_html=True)
        
        df['Health_Category'] = df.apply(categorize_project_health, axis=1)
        health_counts = df['Health_Category'].value_counts()
        
        colors = {'Critical': '#e74c3c', 'At Risk': '#f39c12', 'Healthy': '#27ae60'}
        
        fig_health = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            color=health_counts.index,
            color_discrete_map=colors,
            title="Project Health Distribution"
        )
        fig_health.update_traces(textposition='inside', textinfo='percent+label')
        fig_health.update_layout(height=400)
        st.plotly_chart(fig_health, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">💰 Financial Performance Intelligence</div>', unsafe_allow_html=True)
        
        # Financial comparison chart
        financial_data = {
            'Metric': ['Budget', 'Actual Cost', 'Earned Value', 'Planned Value', 'EAC'],
            'Value': [
                metrics['total_budget'],
                metrics['total_actual_cost'],
                metrics['total_earned_value'],
                metrics['total_planned_value'],
                metrics['total_eac']
            ]
        }
        
        fig_financial = px.bar(
            financial_data,
            x='Metric',
            y='Value',
            title="Portfolio Financial Overview",
            color='Metric',
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#e74c3c']
        )
        fig_financial.update_layout(
            height=400,
            showlegend=False,
            yaxis=dict(
                tickformat=',.0f',
                title=f'Amount ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})'
            )
        )
        st.plotly_chart(fig_financial, use_container_width=True)

    # Portfolio by Tier Analysis
    st.markdown('<div class="section-header">🎯 Portfolio Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Portfolio Budget by Tier
        if 'Budget_Category' in df.columns and 'Budget' in df.columns:
            tier_budget = df.groupby('Budget_Category')['Budget'].sum().sort_values(ascending=False)

            # Get tier colors from config
            tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
            tier_colors = tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
            tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

            # Create color map based on tier names
            color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

            fig_budget_tier = px.pie(
                values=tier_budget.values,
                names=tier_budget.index,
                title="Portfolio Value by Budget Category",
                color=tier_budget.index,
                color_discrete_map=color_map
            )
            fig_budget_tier.update_traces(textposition='inside', textinfo='percent+label')
            fig_budget_tier.update_layout(height=400)
            st.plotly_chart(fig_budget_tier, use_container_width=True)
        else:
            st.info("Budget tier data not available")

    with col2:
        # Portfolio Projects by Tier
        if 'Budget_Category' in df.columns:
            tier_count = df['Budget_Category'].value_counts().sort_index()

            # Get tier colors from config
            tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
            tier_colors = tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
            tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

            # Create color map based on tier names
            color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

            fig_project_tier = px.pie(
                values=tier_count.values,
                names=tier_count.index,
                title="Portfolio Projects by Budget Category",
                color=tier_count.index,
                color_discrete_map=color_map
            )
            fig_project_tier.update_traces(textposition='inside', textinfo='percent+label')
            fig_project_tier.update_layout(height=400)
            st.plotly_chart(fig_project_tier, use_container_width=True)
        else:
            st.info("Project tier data not available")

    # Project Spotlight Section
    with st.expander("🎯 Project Spotlight", expanded=False):
        # Dropdown for view selection
        view_options = [
            "1. Critical Projects",
            "2. Top 10 Budget",
            "3. Top 10 Earliest",
            "4. Top 10 Latest",
            "5. Top 10 Highest SPI",
            "6. Top 10 Highest CPI",
            "7. Top 10 Lowest SPI",
            "8. Top 10 Lowest CPI",
            "9. Top 10 Longest Duration",
            "10. Top 10 Shortest Duration",
            "11. Top 10 Longest Delay",
            "12. Top 10 Highest ETC"
        ]

        selected_view = st.selectbox("Select View:", view_options, key="project_spotlight_view")

        # Initialize variables
        spotlight_projects = df.copy()
        view_description = ""

        # Filter and sort based on selected view
        if selected_view == "1. Critical Projects":
            spotlight_projects = df[df['Health_Category'] == 'Critical'].copy()
            spotlight_projects = spotlight_projects.sort_values('CPI').head(10)
            view_description = "⚠️ These projects require immediate executive intervention due to critical performance issues."

        elif selected_view == "2. Top 10 Budget":
            spotlight_projects = df.nlargest(10, 'Budget')
            view_description = "💰 Projects with the highest budgets in the portfolio."

        elif selected_view == "3. Top 10 Earliest":
            # Check for possible Plan Start column names
            plan_start_col = None
            for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                if col in df.columns:
                    plan_start_col = col
                    break

            if plan_start_col:
                # Convert to datetime and sort for earliest dates (ascending)
                df_temp = df.copy()
                df_temp[f'{plan_start_col}_datetime'] = pd.to_datetime(df_temp[plan_start_col], errors='coerce')
                valid_dates = df_temp.dropna(subset=[f'{plan_start_col}_datetime'])
                if not valid_dates.empty:
                    spotlight_projects = valid_dates.nsmallest(10, f'{plan_start_col}_datetime')
                else:
                    spotlight_projects = pd.DataFrame()
            else:
                spotlight_projects = pd.DataFrame()
            view_description = "📅 Projects with the earliest planned start dates."

        elif selected_view == "4. Top 10 Latest":
            # Check for possible Plan Start column names
            plan_start_col = None
            for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                if col in df.columns:
                    plan_start_col = col
                    break

            if plan_start_col:
                # Convert to datetime and sort for latest dates (descending)
                df_temp = df.copy()
                df_temp[f'{plan_start_col}_datetime'] = pd.to_datetime(df_temp[plan_start_col], errors='coerce')
                valid_dates = df_temp.dropna(subset=[f'{plan_start_col}_datetime'])
                if not valid_dates.empty:
                    spotlight_projects = valid_dates.nlargest(10, f'{plan_start_col}_datetime')
                else:
                    spotlight_projects = pd.DataFrame()
            else:
                spotlight_projects = pd.DataFrame()
            view_description = "📅 Projects with the latest planned start dates."

        elif selected_view == "5. Top 10 Highest SPI":
            spotlight_projects = df.nlargest(10, 'SPI')
            view_description = "🚀 Projects with the best schedule performance (ahead of schedule)."

        elif selected_view == "6. Top 10 Highest CPI":
            spotlight_projects = df.nlargest(10, 'CPI')
            view_description = "💎 Projects with the best cost performance (under budget)."

        elif selected_view == "7. Top 10 Lowest SPI":
            spotlight_projects = df.nsmallest(10, 'SPI')
            view_description = "⏰ Projects with the worst schedule performance (behind schedule)."

        elif selected_view == "8. Top 10 Lowest CPI":
            spotlight_projects = df.nsmallest(10, 'CPI')
            view_description = "💸 Projects with the worst cost performance (over budget)."

        elif selected_view == "9. Top 10 Longest Duration":
            # Look for duration columns - try multiple possible names
            duration_col = None
            duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
            for col in duration_types:
                if col in df.columns:
                    duration_col = col
                    break

            if duration_col:
                spotlight_projects = df.nlargest(10, duration_col)
                view_description = f"⏳ Projects with the longest duration ({duration_col})."
            else:
                spotlight_projects = pd.DataFrame()
                view_description = "⏳ Duration data not available."

        elif selected_view == "10. Top 10 Shortest Duration":
            # Look for duration columns
            duration_col = None
            duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
            for col in duration_types:
                if col in df.columns:
                    duration_col = col
                    break

            if duration_col:
                spotlight_projects = df.nsmallest(10, duration_col)
                view_description = f"⚡ Projects with the shortest duration ({duration_col})."
            else:
                spotlight_projects = pd.DataFrame()
                view_description = "⚡ Duration data not available."

        elif selected_view == "11. Top 10 Longest Delay":
            # Calculate delay as LD-OD (Likely Duration - Original Duration)
            df_temp = df.copy()

            # Find LD (Likely Duration) column
            ld_col = None
            for col in ['Likely Dur', 'forecast_duration', 'LD']:
                if col in df_temp.columns:
                    ld_col = col
                    break

            # Find OD (Original Duration) column
            od_col = None
            for col in ['Original Dur', 'original_duration_months', 'OD']:
                if col in df_temp.columns:
                    od_col = col
                    break

            if ld_col and od_col:
                # Calculate delay = LD - OD
                df_temp['Delay'] = df_temp[ld_col] - df_temp[od_col]
                # Filter projects: positive delay AND delay ≤ 3 × OD
                delayed_projects = df_temp[
                    (df_temp['Delay'] > 0) &
                    (df_temp['Delay'] <= 3 * df_temp[od_col])
                ]
                if not delayed_projects.empty:
                    spotlight_projects = delayed_projects.nlargest(10, 'Delay')
                    view_description = f"🐌 Projects with the longest delays (LD - OD ≤ 3×OD, in months)."
                else:
                    spotlight_projects = pd.DataFrame()
                    view_description = "🐌 No projects with valid delays found (within 3×OD limit)."
            else:
                # Fallback to lowest SPI if duration columns not available
                spotlight_projects = df.nsmallest(10, 'SPI')
                view_description = "🐌 Projects with the most significant schedule delays (using SPI)."

        elif selected_view == "12. Top 10 Highest ETC":
            # Calculate ETC if not available (ETC = EAC - AC)
            df_temp = df.copy()
            if 'ETC' not in df_temp.columns:
                if 'EAC' in df_temp.columns and 'Actual Cost' in df_temp.columns:
                    df_temp['ETC'] = df_temp['EAC'] - df_temp['Actual Cost']
                elif 'EAC' in df_temp.columns and 'AC' in df_temp.columns:
                    df_temp['ETC'] = df_temp['EAC'] - df_temp['AC']
                else:
                    df_temp['ETC'] = 0  # Default if we can't calculate

            spotlight_projects = df_temp.nlargest(10, 'ETC')
            view_description = "🔮 Projects with the highest remaining costs to complete (ETC = EAC - AC)."

        # Ensure ETC column is available if needed (calculate if missing)
        if 'ETC' not in spotlight_projects.columns and selected_view == "12. Top 10 Highest ETC":
            if 'EAC' in spotlight_projects.columns and 'Actual Cost' in spotlight_projects.columns:
                spotlight_projects['ETC'] = spotlight_projects['EAC'] - spotlight_projects['Actual Cost']
            elif 'EAC' in spotlight_projects.columns and 'AC' in spotlight_projects.columns:
                spotlight_projects['ETC'] = spotlight_projects['EAC'] - spotlight_projects['AC']

        # Ensure Delay column is available if needed (calculate if missing)
        if 'Delay' not in spotlight_projects.columns and selected_view == "11. Top 10 Longest Delay":
            # Find LD (Likely Duration) column
            ld_col = None
            for col in ['Likely Dur', 'forecast_duration', 'LD']:
                if col in spotlight_projects.columns:
                    ld_col = col
                    break

            # Find OD (Original Duration) column
            od_col = None
            for col in ['Original Dur', 'original_duration_months', 'OD']:
                if col in spotlight_projects.columns:
                    od_col = col
                    break

            if ld_col and od_col:
                # Calculate delay and apply 3×OD restriction
                spotlight_projects['Delay'] = spotlight_projects[ld_col] - spotlight_projects[od_col]
                # Cap delay at 3 times the original duration
                spotlight_projects['Delay'] = spotlight_projects['Delay'].where(
                    spotlight_projects['Delay'] <= 3 * spotlight_projects[od_col],
                    3 * spotlight_projects[od_col]
                )

        # Display the results
        if not spotlight_projects.empty:
            # Prepare data for table display with appropriate columns for each view
            # Base columns that should appear in all views
            base_columns = ['Project Name']

            # Financial columns (using different possible names)
            financial_columns = []
            # BAC (Budget at Completion)
            for col in ['Budget', 'BAC', 'bac']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # AC (Actual Cost)
            for col in ['Actual Cost', 'AC', 'ac']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # EV (Earned Value)
            for col in ['Earned Value', 'EV', 'ev']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # Performance columns
            performance_columns = ['CPI', 'SPI', 'SPIe']

            # View-specific columns
            if selected_view in ["2. Top 10 Budget", "12. Top 10 Highest ETC"]:
                specific_columns = ['ETC', 'EAC']
                display_columns = base_columns + financial_columns + performance_columns + specific_columns

            elif selected_view in ["3. Top 10 Earliest", "4. Top 10 Latest"]:
                # Find the plan start column
                plan_start_col = None
                for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                    if col in spotlight_projects.columns:
                        plan_start_col = col
                        break

                specific_columns = [plan_start_col] if plan_start_col else []
                display_columns = base_columns + specific_columns + financial_columns + performance_columns

            elif selected_view in ["9. Top 10 Longest Duration", "10. Top 10 Shortest Duration"]:
                # Dynamically determine which duration column to show
                duration_display_col = None
                duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
                for col in duration_types:
                    if col in spotlight_projects.columns:
                        duration_display_col = col
                        break

                specific_columns = [duration_display_col] if duration_display_col else []
                display_columns = base_columns + specific_columns + financial_columns + performance_columns

            elif selected_view == "11. Top 10 Longest Delay":
                # For delay view, show Delay column plus original and likely durations
                delay_columns = []

                # Add Delay column if it exists
                if 'Delay' in spotlight_projects.columns:
                    delay_columns.append('Delay')

                # Add original duration column
                for col in ['Original Dur', 'original_duration_months', 'OD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                # Add actual duration column
                for col in ['Actual Dur', 'actual_duration_months', 'AD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                # Add likely duration column
                for col in ['Likely Dur', 'forecast_duration', 'LD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                display_columns = base_columns + delay_columns + financial_columns + performance_columns

            else:
                # Default view - include EAC
                specific_columns = ['EAC']
                display_columns = base_columns + financial_columns + performance_columns + specific_columns

            # Remove None values and duplicates while preserving order
            display_columns = [col for col in display_columns if col is not None]
            display_columns = list(dict.fromkeys(display_columns))  # Remove duplicates while preserving order

            # Check which columns are available
            available_columns = [col for col in display_columns if col in spotlight_projects.columns]

            if available_columns:
                spotlight_table = spotlight_projects[available_columns].copy()

                # Update column headers to include currency information instead of formatting values
                currency_columns = ['Budget', 'BAC', 'bac', 'Actual Cost', 'AC', 'ac', 'Earned Value', 'EV', 'ev', 'Planned Value', 'PV', 'pv', 'EAC', 'ETC']
                spotlight_column_renames = {}
                for col in currency_columns:
                    if col in spotlight_table.columns:
                        if col in ['Budget', 'BAC', 'bac']:
                            spotlight_column_renames[col] = f'BAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'BAC ({currency_symbol})'
                        elif col in ['Actual Cost', 'AC', 'ac']:
                            spotlight_column_renames[col] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        elif col in ['Earned Value', 'EV', 'ev']:
                            spotlight_column_renames[col] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        elif col in ['Planned Value', 'PV', 'pv']:
                            spotlight_column_renames[col] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        elif col == 'EAC':
                            spotlight_column_renames[col] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'
                        elif col == 'ETC':
                            spotlight_column_renames[col] = f'ETC ({currency_symbol}{currency_postfix})' if currency_postfix else f'ETC ({currency_symbol})'

                # Apply column renames for spotlight table
                spotlight_table = spotlight_table.rename(columns=spotlight_column_renames)

                # Create column configuration for spotlight table
                spotlight_column_config = {}

                # Configure currency columns for spotlight table
                spotlight_currency_original_columns = ['Budget', 'BAC', 'bac', 'Actual Cost', 'AC', 'ac', 'Earned Value', 'EV', 'ev', 'Planned Value', 'PV', 'pv', 'EAC', 'ETC']
                for col in spotlight_currency_original_columns:
                    if col in spotlight_projects.columns:  # Check original column names before renaming
                        renamed_col = spotlight_column_renames.get(col, col)  # Get the renamed column name
                        if renamed_col in spotlight_table.columns:
                            spotlight_column_config[renamed_col] = st.column_config.NumberColumn(
                                renamed_col,
                                format="%.2f",
                                help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                            )

                # Configure performance indices columns for spotlight table
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in spotlight_table.columns:
                        spotlight_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.3f",
                            help=f"{col} performance index"
                        )

                # Configure duration columns
                duration_format_cols = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
                for col in duration_format_cols:
                    if col in spotlight_table.columns:
                        spotlight_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.1f",
                            help="Duration in months"
                        )

                # Configure Delay column
                if 'Delay' in spotlight_table.columns:
                    spotlight_column_config['Delay'] = st.column_config.NumberColumn(
                        'Delay',
                        format="%.1f",
                        help="Schedule delay in months"
                    )

                # Format dates if present - handle multiple possible date column names
                date_columns = ['Plan Start', 'plan_start', 'Start Date', 'start_date']
                for col in date_columns:
                    if col in spotlight_table.columns:
                        spotlight_table[col] = pd.to_datetime(spotlight_table[col], errors='coerce').dt.strftime('%Y-%m-%d')

                # Duration and delay columns are now handled by column configuration
                # (Remove string formatting to preserve numeric sorting)

                # Apply conditional styling based on view type
                if selected_view == "1. Critical Projects":
                    # Style critical projects with red background
                    def highlight_critical_projects(val):
                        return 'background-color: #ffebee; color: #d32f2f; font-weight: bold;'
                    try:
                        styled_table = spotlight_table.style.applymap(highlight_critical_projects)
                        st.dataframe(styled_table, width='stretch', height=300, column_config=spotlight_column_config)
                    except:
                        st.dataframe(spotlight_table, width='stretch', height=300, column_config=spotlight_column_config)
                else:
                    # Standard display for other views
                    st.dataframe(spotlight_table, width='stretch', height=300, column_config=spotlight_column_config)

                st.markdown(f"**{view_description}**")
            else:
                st.info("Required data columns not available for this view.")
        else:
            st.info(f"No data available for {selected_view.split('.')[1].strip()}.")


    # Interactive Data Explorer
    with st.expander("Advanced Portfolio Analytics", expanded=False):
        # Budget_Category already created earlier in the code

        # ═══════════════════════════════════════════════════════════════════
        # 🎯 FILTER CONTROLS SECTION
        # ═══════════════════════════════════════════════════════════════════

        st.markdown("### 🔍 Filter Controls")

        # ─────────────────────────────────────────────────────────────────
        # BASIC FILTERS
        # ─────────────────────────────────────────────────────────────────
        st.markdown("**📊 Basic Filters**")
        col1, col2 = st.columns(2)

        with col1:
            # Organization filter
            org_columns = [col for col in df.columns if 'org' in col.lower() or 'department' in col.lower() or 'division' in col.lower()]
            if org_columns:
                org_options = df[org_columns[0]].dropna().unique().tolist()
                organization_filter = st.multiselect(
                    "🏢 Organization",
                    options=org_options,
                    default=org_options,
                    help="Filter projects by organization or department"
                )
            else:
                # Create dummy organizations for demonstration
                np.random.seed(42)
                orgs = ['Engineering', 'Infrastructure', 'IT', 'Construction', 'Energy', 'Healthcare']
                df['Organization'] = np.random.choice(orgs, len(df))
                organization_filter = st.multiselect(
                    "🏢 Organization",
                    options=orgs,
                    default=orgs,
                    help="Filter projects by organization or department"
                )

        with col2:
            health_filter = st.multiselect(
                "🏥 Health Status",
                options=['Critical', 'At Risk', 'Healthy'],
                default=['Critical', 'At Risk', 'Healthy'],
                help="Filter projects by their health status based on CPI and SPI"
            )

        st.markdown("")  # Add spacing

        # ─────────────────────────────────────────────────────────────────
        # PERFORMANCE FILTERS
        # ─────────────────────────────────────────────────────────────────
        st.markdown("**📈 Performance Filters**")
        col1, col2, col3 = st.columns(3)

        with col1:
            enable_cpi_filter = st.checkbox("Enable CPI Filter", value=False, key="cpi_filter_toggle")
            if enable_cpi_filter:
                cpi_range = st.slider(
                    "💰 CPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(0.0, 3.0),
                    step=0.01,
                    help="Cost Performance Index: >1.0 = under budget, <1.0 = over budget"
                )
            else:
                cpi_range = (0.0, 3.0)  # Default to full range when disabled

        with col2:
            enable_spi_filter = st.checkbox("Enable SPI Filter", value=False, key="spi_filter_toggle")
            if enable_spi_filter:
                spi_range = st.slider(
                    "⏱️ SPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(0.0, 3.0),
                    step=0.01,
                    help="Schedule Performance Index: >1.0 = ahead of schedule, <1.0 = behind schedule"
                )
            else:
                spi_range = (0.0, 3.0)  # Default to full range when disabled

        with col3:
            enable_spie_filter = st.checkbox("Enable SPIe Filter", value=False, key="spie_filter_toggle")
            if enable_spie_filter:
                spie_range = st.slider(
                    "📊 SPIe Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(0.0, 3.0),
                    step=0.01,
                    help="Schedule Performance Index Estimate"
                )
            else:
                spie_range = (0.0, 3.0)  # Default to full range when disabled

        st.markdown("")  # Add spacing

        # ─────────────────────────────────────────────────────────────────
        # BUDGET FILTERS
        # ─────────────────────────────────────────────────────────────────
        with st.expander("💵 Budget Range Filters", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                enable_lower_budget = st.checkbox("Enable Minimum Budget", value=False, key="lower_budget_toggle")
                if enable_lower_budget:
                    min_budget = st.number_input(
                        f"Minimum Budget ({currency_symbol})",
                        min_value=0,
                        value=0,
                        step=1000,
                        key="min_budget_value",
                        help="Set the minimum budget threshold"
                    )
                else:
                    min_budget = 0

            with col2:
                enable_upper_budget = st.checkbox("Enable Maximum Budget", value=False, key="upper_budget_toggle")
                if enable_upper_budget:
                    max_budget = st.number_input(
                        f"Maximum Budget ({currency_symbol})",
                        min_value=0,
                        value=1000000,
                        step=1000,
                        key="max_budget_value",
                        help="Set the maximum budget threshold"
                    )
                else:
                    max_budget = float('inf')

        # ─────────────────────────────────────────────────────────────────
        # DATE FILTERS (Based on Plan Start Date)
        # ─────────────────────────────────────────────────────────────────
        with st.expander("📅 Plan Start Date Filters", expanded=False):
            # Check if plan_start column exists
            plan_start_col = 'plan_start'
            if plan_start_col in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    enable_start_date = st.checkbox("Enable Start Date Filter", value=False, key="start_date_toggle")
                    if enable_start_date:
                        try:
                            temp_dates = pd.to_datetime(df[plan_start_col], errors='coerce').dropna()
                            if len(temp_dates) > 0:
                                min_date = temp_dates.min().date()
                                max_date = temp_dates.max().date()
                                start_date_filter = st.date_input(
                                    "From Date",
                                    value=min_date,
                                    min_value=min_date,
                                    max_value=max_date,
                                    key="start_date_value",
                                    help="Filter projects with plan start date from this date onwards"
                                )
                            else:
                                start_date_filter = None
                                st.warning("⚠️ No valid plan start dates found")
                        except:
                            start_date_filter = None
                            st.error("❌ Error processing plan start dates")
                    else:
                        start_date_filter = None

                with col2:
                    enable_end_date = st.checkbox("Enable End Date Filter", value=False, key="end_date_toggle")
                    if enable_end_date:
                        try:
                            temp_dates = pd.to_datetime(df[plan_start_col], errors='coerce').dropna()
                            if len(temp_dates) > 0:
                                min_date = temp_dates.min().date()
                                max_date = temp_dates.max().date()
                                end_date_filter = st.date_input(
                                    "To Date",
                                    value=max_date,
                                    min_value=min_date,
                                    max_value=max_date,
                                    key="end_date_value",
                                    help="Filter projects with plan start date up to this date"
                                )
                            else:
                                end_date_filter = None
                                st.warning("⚠️ No valid plan start dates found")
                        except:
                            end_date_filter = None
                            st.error("❌ Error processing plan start dates")
                    else:
                        end_date_filter = None

                selected_date_column = plan_start_col
            else:
                selected_date_column = None
                start_date_filter = None
                end_date_filter = None
                st.info("ℹ️ Plan start date column not found in the dataset")
        
        # Apply filters
        filtered_df = df.copy()

        # Apply organization filter (include projects with missing org data when all orgs are selected)
        if org_columns:
            org_col = org_columns[0]
            all_orgs = df[org_col].dropna().unique().tolist()
            if set(organization_filter) == set(all_orgs):
                # All organizations selected - include projects with missing org data
                filtered_df = filtered_df[
                    filtered_df[org_col].isin(organization_filter) |
                    filtered_df[org_col].isnull()
                ]
            else:
                # Specific organizations selected - exclude missing org data
                filtered_df = filtered_df[filtered_df[org_col].isin(organization_filter)]
        else:
            all_orgs = ['Engineering', 'Infrastructure', 'IT', 'Construction', 'Energy', 'Healthcare']
            if set(organization_filter) == set(all_orgs):
                # All organizations selected - include projects with missing org data
                filtered_df = filtered_df[
                    filtered_df['Organization'].isin(organization_filter) |
                    filtered_df['Organization'].isnull()
                ]
            else:
                # Specific organizations selected - exclude missing org data
                filtered_df = filtered_df[filtered_df['Organization'].isin(organization_filter)]

        # Apply date range filter
        if selected_date_column and (start_date_filter or end_date_filter):
            try:
                # Convert the selected date column to datetime
                filtered_df[selected_date_column] = pd.to_datetime(filtered_df[selected_date_column], errors='coerce')

                # Apply start date filter
                if start_date_filter:
                    start_datetime = pd.to_datetime(start_date_filter)
                    filtered_df = filtered_df[
                        (filtered_df[selected_date_column].isna()) |
                        (filtered_df[selected_date_column] >= start_datetime)
                    ]

                # Apply end date filter
                if end_date_filter:
                    end_datetime = pd.to_datetime(end_date_filter)
                    filtered_df = filtered_df[
                        (filtered_df[selected_date_column].isna()) |
                        (filtered_df[selected_date_column] <= end_datetime)
                    ]
            except Exception as e:
                st.warning(f"Error applying date filter: {str(e)}")

        # Apply other filters
        filtered_df = filtered_df[
            (filtered_df['Health_Category'].isin(health_filter)) &
            (filtered_df['Budget'] >= min_budget) &
            (filtered_df['Budget'] <= max_budget) &
            (filtered_df['CPI'].between(cpi_range[0], cpi_range[1])) &
            (filtered_df['SPI'].between(spi_range[0], spi_range[1])) &
            (filtered_df['SPIe'].between(spie_range[0], spie_range[1]))
        ]
        
        # Check if any filters are actually applied
        filters_active = False

        # Check organization filter
        total_orgs = len(df[org_columns[0]].dropna().unique()) if org_columns else len(['Engineering', 'Infrastructure', 'IT', 'Construction', 'Energy', 'Healthcare'])
        if len(organization_filter) < total_orgs:
            filters_active = True

        # Check health filter
        if len(health_filter) < 3:  # Less than all 3 health statuses
            filters_active = True

        # Check budget filter
        budget_min = df['Budget'].min()
        budget_max = df['Budget'].max()
        if min_budget > budget_min or max_budget < budget_max:
            filters_active = True

        # Check performance filters (CPI, SPI, SPIe)
        if cpi_range != [df['CPI'].min(), df['CPI'].max()]:
            filters_active = True
        if spi_range != [df['SPI'].min(), df['SPI'].max()]:
            filters_active = True
        if spie_range != [df['SPIe'].min(), df['SPIe'].max()]:
            filters_active = True

        # Check date filters
        if (selected_date_column and (start_date_filter or end_date_filter)):
            filters_active = True

        # Display appropriate message
        if filters_active:
            st.write(f"Showing {len(filtered_df)} projects (filtered from {len(df)} total)")
        else:
            st.write(f"Showing all {len(filtered_df)} projects")


        # ═══════════════════════════════════════════════════════════════════
        # FILTERED PORTFOLIO PERFORMANCE CURVE
        # ═══════════════════════════════════════════════════════════════════

        # Portfolio Time/Budget Performance Curve (Filtered)
        # Portfolio Performance Curve has been moved to 5_Portfolio_Charts.py

        # Projects Expander
        with st.expander("📋 Projects", expanded=False):
            # Display filtered data with enhanced columns
            if org_columns:
                display_columns = ['Project Name', org_columns[0], 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
            else:
                display_columns = ['Project Name', 'Organization', 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']

            available_columns = [col for col in display_columns if col in filtered_df.columns]

            if available_columns:
                # Format the dataframe for better display
                display_df = filtered_df[available_columns].copy()

                # Update column names to include currency information
                column_renames = {}
                if 'Budget' in display_df.columns:
                    column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                if 'Actual Cost' in display_df.columns:
                    column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                if 'Plan Value' in display_df.columns:
                    column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                if 'Earned Value' in display_df.columns:
                    column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                if 'EAC' in display_df.columns:
                    column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'

                # Apply column renames
                display_df = display_df.rename(columns=column_renames)

                # Create column configuration for proper numeric formatting and alignment
                column_config = {}

                # Configure currency columns
                currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                for col in currency_value_columns:
                    if col in filtered_df.columns:  # Check original column names
                        renamed_col = column_renames.get(col, col)  # Get the renamed column name
                        if renamed_col in display_df.columns:
                            column_config[renamed_col] = st.column_config.NumberColumn(
                                renamed_col,
                                format="%.2f",
                                help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                            )

                # Configure performance indices columns
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in display_df.columns:
                        column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.3f",
                            help=f"{col} performance index"
                        )

                # Configure other numeric columns
                if 'Project Count' in display_df.columns:
                    column_config['Project Count'] = st.column_config.NumberColumn(
                        'Project Count',
                        format="%d"
                    )

                # Color-code the health status
                def highlight_health(val):
                    if val == 'Critical':
                        return 'background-color: #ffebee'
                    elif val == 'At Risk':
                        return 'background-color: #fff3e0'
                    elif val == 'Healthy':
                        return 'background-color: #e8f5e8'
                    return ''

                if 'Health_Category' in display_df.columns:
                    styled_df = display_df.style.applymap(highlight_health, subset=['Health_Category'])
                    st.dataframe(styled_df, width='stretch', height=400, column_config=column_config)
                else:
                    st.dataframe(display_df, width='stretch', height=400, column_config=column_config)

        # Organizations Expander
        with st.expander("🏢 Organizations", expanded=False):
            # Get organization column name
            org_col = org_columns[0] if org_columns else 'Organization'

            if org_col in filtered_df.columns and len(filtered_df) > 0:
                # Group by organization and calculate consolidated metrics
                try:
                    # Define aggregation columns dynamically
                    agg_dict = {
                        'Project Name': 'count',  # Project count
                        'Budget': 'sum',          # Total budget
                        'Actual Cost': 'sum',     # Total actual cost
                        'EAC': 'sum',            # Total EAC
                        'Health_Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # Most common health status
                    }

                    # Add Plan Value and Earned Value if they exist
                    if 'Plan Value' in filtered_df.columns:
                        agg_dict['Plan Value'] = 'sum'
                    if 'Earned Value' in filtered_df.columns:
                        agg_dict['Earned Value'] = 'sum'

                    org_summary = filtered_df.groupby(org_col).agg(agg_dict).rename(columns={'Project Name': 'Project Count'})

                    # Calculate weighted averages for CPI, SPI, SPIe
                    org_weighted_metrics = filtered_df.groupby(org_col).apply(
                        lambda group: pd.Series({
                            'CPI': (group['CPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPI': (group['SPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPIe': (group['SPIe'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0
                        })
                    )

                    # Combine the metrics
                    org_display = pd.concat([org_summary, org_weighted_metrics], axis=1)

                    # Check if we have valid data
                    if len(org_display) > 0:
                        # Reorder columns to match project view (without Budget_Category)
                        base_org_columns = ['Project Count', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost']
                        optional_columns = []
                        if 'Plan Value' in org_display.columns:
                            optional_columns.append('Plan Value')
                        if 'Earned Value' in org_display.columns:
                            optional_columns.append('Earned Value')
                        if 'EAC' in org_display.columns:
                            optional_columns.append('EAC')

                        org_display_columns = base_org_columns + optional_columns
                        available_org_columns = [col for col in org_display_columns if col in org_display.columns]
                        org_display = org_display[available_org_columns]

                        # Reset index to ensure unique indices for styling
                        org_display = org_display.reset_index()

                        # Format the organizational data for display
                        org_display_formatted = org_display.copy()

                        # Update column names to include currency information for organizations
                        org_column_renames = {}
                        if 'Budget' in org_display_formatted.columns:
                            org_column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                        if 'Actual Cost' in org_display_formatted.columns:
                            org_column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        if 'Plan Value' in org_display_formatted.columns:
                            org_column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        if 'Earned Value' in org_display_formatted.columns:
                            org_column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        if 'EAC' in org_display_formatted.columns:
                            org_column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'

                        # Apply column renames for organizations
                        org_display_formatted = org_display_formatted.rename(columns=org_column_renames)

                        # Create column configuration for organizations table
                        org_column_config = {}

                        # Configure currency columns for organizations
                        org_currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                        for col in org_currency_value_columns:
                            if col in org_display.columns:  # Check original column names before renaming
                                renamed_col = org_column_renames.get(col, col)  # Get the renamed column name
                                if renamed_col in org_display_formatted.columns:
                                    org_column_config[renamed_col] = st.column_config.NumberColumn(
                                        renamed_col,
                                        format="%.2f",
                                        help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                                    )

                        # Configure performance indices columns for organizations
                        for col in ['CPI', 'SPI', 'SPIe']:
                            if col in org_display_formatted.columns:
                                org_column_config[col] = st.column_config.NumberColumn(
                                    col,
                                    format="%.3f",
                                    help=f"{col} performance index"
                                )

                        # Configure Project Count column
                        if 'Project Count' in org_display_formatted.columns:
                            org_column_config['Project Count'] = st.column_config.NumberColumn(
                                'Project Count',
                                format="%d"
                            )

                        # Apply health status styling with error handling
                        try:
                            if 'Health_Category' in org_display_formatted.columns and len(org_display_formatted) > 0:
                                styled_org_df = org_display_formatted.style.applymap(highlight_health, subset=['Health_Category'])
                                st.dataframe(styled_org_df, width='stretch', height=300, column_config=org_column_config)
                            else:
                                st.dataframe(org_display_formatted, width='stretch', height=300, column_config=org_column_config)
                        except (KeyError, ValueError) as e:
                            # Fallback to unstyled dataframe if styling fails
                            st.dataframe(org_display_formatted, width='stretch', height=300)
                    else:
                        st.info("No organization data available with current filters.")

                except Exception as e:
                    st.error(f"Error processing organization data: {str(e)}")
                    st.info("Unable to display organizational consolidation with current data.")
            else:
                if len(filtered_df) == 0:
                    st.info("No projects available for organizational consolidation.")
                else:
                    st.info("Organization data not available for consolidation.")

        # Portfolio Budget Chart Expander
        # Portfolio Budget Chart has been moved to 5_Portfolio_Charts.py

        # Cash Flow Chart has been moved to 5_Portfolio_Charts.py

        # Approvals Chart has been moved to 5_Portfolio_Charts.py

        # Financial Summary Expander
        with st.expander("💰 Financial Summary", expanded=False):
            if len(filtered_df) > 0:
                # Enhanced styling for Financial Summary
                st.markdown("""
                <style>
                .financial-metric {
                    font-size: 1.1rem;
                    line-height: 1.4;
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }
                .financial-value {
                    font-size: 1.3rem;
                    font-weight: bold;
                    color: #1f77b4;
                    margin-top: 8px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Calculate portfolio totals with safe column access
                total_budget = filtered_df.get('Budget', pd.Series([0])).sum()
                total_actual_cost = filtered_df.get('Actual Cost', pd.Series([0])).sum()
                total_earned_value = filtered_df.get('Earned Value', pd.Series([0])).sum()
                total_planned_value = filtered_df.get('Plan Value', pd.Series([0])).sum()
                total_eac = filtered_df.get('EAC', pd.Series([0])).sum()

                # Calculate ETC as EAC - AC
                total_etc = total_eac - total_actual_cost

                # Row 1: Budget (BAC) and Actual Cost (AC)
                col1, col2 = st.columns(2)
                with col1:
                    if 'Budget' in filtered_df.columns and total_budget > 0:
                        bac_formatted = format_currency(total_budget, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💰 **Portfolio Budget (BAC)**<br><span class="financial-value">{bac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💰 **Portfolio Budget (BAC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col2:
                    if 'Actual Cost' in filtered_df.columns:
                        ac_formatted = format_currency(total_actual_cost, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💸 **Portfolio Actual Cost (AC)**<br><span class="financial-value">{ac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💸 **Portfolio Actual Cost (AC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 2: Planned Value (PV) and Earned Value (EV)
                col3, col4 = st.columns(2)
                with col3:
                    if 'Plan Value' in filtered_df.columns:
                        pv_formatted = format_currency(total_planned_value, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">📊 **Portfolio Planned Value (PV)**<br><span class="financial-value">{pv_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">📊 **Portfolio Planned Value (PV)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col4:
                    if 'Earned Value' in filtered_df.columns:
                        ev_formatted = format_currency(total_earned_value, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💎 **Portfolio Earned Value (EV)**<br><span class="financial-value">{ev_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💎 **Portfolio Earned Value (EV)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 3: ETC and EAC
                col5, col6 = st.columns(2)
                with col5:
                    if 'EAC' in filtered_df.columns and 'Actual Cost' in filtered_df.columns:
                        etc_formatted = format_currency(total_etc, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">🔧 **Portfolio Estimate to Complete (ETC)**<br><span class="financial-value">{etc_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">🔧 **Portfolio Estimate to Complete (ETC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col6:
                    if 'EAC' in filtered_df.columns:
                        eac_formatted = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">🎯 **Portfolio Estimate at Completion (EAC)**<br><span class="financial-value">{eac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">🎯 **Portfolio Estimate at Completion (EAC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Portfolio EAC Status Alert
                forecast_overrun = total_eac - total_budget
                if forecast_overrun > 0:
                    eac_fmt = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                    overrun_fmt = format_currency(forecast_overrun, currency_symbol, currency_postfix, thousands=False)
                    st.markdown(
                        f"""<div style="background-color: #fee; border-left: 4px solid #c33; padding: 1rem; border-radius: 4px; margin: 1rem 0; font-family: sans-serif; letter-spacing: normal; word-spacing: normal;">
                        📢 <strong>Portfolio EAC Status:</strong> {eac_fmt} (+{overrun_fmt} over budget)
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    eac_fmt = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                    st.markdown(
                        f"""<div style="background-color: #efe; border-left: 4px solid #3c3; padding: 1rem; border-radius: 4px; margin: 1rem 0; font-family: sans-serif; letter-spacing: normal; word-spacing: normal;">
                        ✅ <strong>Portfolio EAC Status:</strong> {eac_fmt} (Under budget)
                        </div>""",
                        unsafe_allow_html=True
                    )

            else:
                st.info("No data available for financial summary.")

        # Durations Expander
        with st.expander("⏱️ Durations", expanded=False):
            if len(filtered_df) > 0:
                # Enhanced styling for Duration metrics (same as Financial Summary)
                st.markdown("""
                <style>
                .duration-metric {
                    font-size: 1.1rem;
                    line-height: 1.4;
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #28a745;
                }
                .duration-value {
                    font-size: 1.3rem;
                    font-weight: bold;
                    color: #28a745;
                    margin-top: 8px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Check for duration columns and calculate metrics
                od_col = 'original_duration_months'
                ad_col = 'actual_duration_months'
                ld_col = 'ld'
                bac_col = 'bac' if 'bac' in filtered_df.columns else 'Budget'

                # Initialize variables
                avg_od = wt_avg_od = avg_ad = wt_avg_ad = avg_ld = wt_avg_ld = None
                avg_delay = wt_avg_delay = None

                # Calculate OD metrics
                if od_col in filtered_df.columns:
                    od_data = filtered_df[filtered_df[od_col].notna() & (filtered_df[od_col] > 0)]
                    if len(od_data) > 0:
                        avg_od = od_data[od_col].mean()
                        if bac_col in od_data.columns:
                            total_bac = od_data[bac_col].sum()
                            if total_bac > 0:
                                wt_avg_od = (od_data[od_col] * od_data[bac_col]).sum() / total_bac

                # Calculate AD metrics
                if ad_col in filtered_df.columns:
                    ad_data = filtered_df[filtered_df[ad_col].notna() & (filtered_df[ad_col] > 0)]
                    if len(ad_data) > 0:
                        avg_ad = ad_data[ad_col].mean()
                        if bac_col in ad_data.columns:
                            total_bac = ad_data[bac_col].sum()
                            if total_bac > 0:
                                wt_avg_ad = (ad_data[ad_col] * ad_data[bac_col]).sum() / total_bac

                # Calculate LD metrics with upper limit check
                if ld_col in filtered_df.columns:
                    ld_data = filtered_df[filtered_df[ld_col].notna() & (filtered_df[ld_col] > 0)].copy()
                    if len(ld_data) > 0:
                        # Apply upper limit cap: min(LD, OD + 48)
                        if od_col in ld_data.columns:
                            # For projects with OD data, cap LD to OD + 48
                            ld_data_with_od = ld_data[ld_data[od_col].notna() & (ld_data[od_col] > 0)].copy()
                            if len(ld_data_with_od) > 0:
                                ld_data_with_od[ld_col + '_capped'] = ld_data_with_od.apply(
                                    lambda row: min(row[ld_col], row[od_col] + 48), axis=1
                                )
                                # Use capped values for calculation
                                avg_ld = ld_data_with_od[ld_col + '_capped'].mean()
                                if bac_col in ld_data_with_od.columns:
                                    total_bac = ld_data_with_od[bac_col].sum()
                                    if total_bac > 0:
                                        wt_avg_ld = (ld_data_with_od[ld_col + '_capped'] * ld_data_with_od[bac_col]).sum() / total_bac
                            else:
                                # No OD data available, use original LD values
                                avg_ld = ld_data[ld_col].mean()
                                if bac_col in ld_data.columns:
                                    total_bac = ld_data[bac_col].sum()
                                    if total_bac > 0:
                                        wt_avg_ld = (ld_data[ld_col] * ld_data[bac_col]).sum() / total_bac
                        else:
                            # No OD column available, use original LD values
                            avg_ld = ld_data[ld_col].mean()
                            if bac_col in ld_data.columns:
                                total_bac = ld_data[bac_col].sum()
                                if total_bac > 0:
                                    wt_avg_ld = (ld_data[ld_col] * ld_data[bac_col]).sum() / total_bac

                # Calculate Delay metrics
                if avg_ld is not None and avg_od is not None:
                    avg_delay = avg_ld - avg_od
                if wt_avg_ld is not None and wt_avg_od is not None:
                    wt_avg_delay = wt_avg_ld - wt_avg_od

                # Row 1: Plan Duration (OD)
                col1, col2 = st.columns(2)
                with col1:
                    if avg_od is not None:
                        st.markdown(f'<div class="duration-metric">📅 **Avg Plan Duration (OD)**<br><span class="duration-value">{round(avg_od)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">📅 **Avg Plan Duration (OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col2:
                    if wt_avg_od is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Plan Duration (wt OD)**<br><span class="duration-value">{round(wt_avg_od)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Plan Duration (wt OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 2: Actual Duration (AD)
                col3, col4 = st.columns(2)
                with col3:
                    if avg_ad is not None:
                        st.markdown(f'<div class="duration-metric">📊 **Avg Actual Duration (AD)**<br><span class="duration-value">{round(avg_ad)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">📊 **Avg Actual Duration (AD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col4:
                    if wt_avg_ad is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Actual Duration (Wt AD)**<br><span class="duration-value">{round(wt_avg_ad)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Actual Duration (Wt AD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 3: Likely Duration (LD)
                col5, col6 = st.columns(2)
                with col5:
                    if avg_ld is not None:
                        st.markdown(f'<div class="duration-metric">🔮 **Avg Likely Duration (LD)**<br><span class="duration-value">{round(avg_ld)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">🔮 **Avg Likely Duration (LD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col6:
                    if wt_avg_ld is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Likely Duration (wt LD)**<br><span class="duration-value">{round(wt_avg_ld)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Likely Duration (wt LD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 4: Delay
                col7, col8 = st.columns(2)
                with col7:
                    if avg_delay is not None:
                        delay_color = "#dc3545" if avg_delay > 0 else "#28a745"
                        delay_sign = "+" if avg_delay > 0 else ""
                        st.markdown(f'<div class="duration-metric">⏰ **Avg Delay (Avg LD - Avg OD)**<br><span class="duration-value" style="color: {delay_color};">{delay_sign}{round(avg_delay)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⏰ **Avg Delay (Avg LD - Avg OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col8:
                    if wt_avg_delay is not None:
                        delay_color = "#dc3545" if wt_avg_delay > 0 else "#28a745"
                        delay_sign = "+" if wt_avg_delay > 0 else ""
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Delay (wt Avg LD - wt Avg OD)**<br><span class="duration-value" style="color: {delay_color};">{delay_sign}{round(wt_avg_delay)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Delay (wt Avg LD - wt Avg OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

            else:
                st.info("No data available for duration analysis.")

        # Time/Budget Performance Chart
        # Time/Budget Performance has been moved to 5_Portfolio_Charts.py

        # Portfolio Treemap
        # Portfolio Treemap has been moved to 5_Portfolio_Charts.py

        # Summary statistics for filtered data
        if len(filtered_df) > 0:
            st.markdown('<div class="section-header">📊 Filtered Portfolio Analytics</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            # Calculate portfolio-level indices for filtered data (CPI/SPI) and weighted average for SPIe
            total_budget = filtered_df['Budget'].sum()
            total_earned_value = filtered_df.get('Earned Value', pd.Series([0])).sum()
            total_actual_cost = filtered_df.get('Actual Cost', pd.Series([0])).sum()
            total_planned_value = filtered_df.get('Plan Value', pd.Series([0])).sum()

            weighted_cpi = total_earned_value / total_actual_cost if total_actual_cost > 0 else 0
            weighted_spi = total_earned_value / total_planned_value if total_planned_value > 0 else 0
            weighted_spie = (filtered_df['SPIe'] * filtered_df['Budget']).sum() / total_budget if total_budget > 0 else 0

            with col1:
                st.metric("📈 Portfolio CPI", f"{weighted_cpi:.3f}", help="Portfolio Cost Performance Index: SUM(Earned Value) / SUM(Actual Cost)")
            with col2:
                st.metric("⏱️ Portfolio SPI", f"{weighted_spi:.3f}", help="Portfolio Schedule Performance Index: SUM(Earned Value) / SUM(Plan Value)")
            with col3:
                st.metric("📊 Avg SPIe", f"{weighted_spie:.3f}", help="Weighted Average Schedule Performance Index Estimate (weighted by budget)")


    # Executive Portfolio Brief Section (Enhanced with Project Details)
    with st.expander("📊 Executive Portfolio Brief (AI-Generated)", expanded=False):
        st.markdown("""
        ### 🤖 AI-Powered Portfolio Executive Report

        Generate a comprehensive executive briefing of your portfolio health using AI with deep analysis of hidden trends.

        **Advanced Analysis includes:**
        - Portfolio dashboard with key metrics
        - Executive summary with overall health rating
        - Performance analysis (cost and schedule)
        - **Trend Analysis**: Patterns by organization, project size, tier, and timeline
        - **Root Cause Analysis**: Identify systemic issues affecting performance
        - Risk assessment across all dimensions
        - Strategic recommendations and corrective actions
        - Decision points requiring executive approval
        """)

        # Check if LLM is configured
        llm_config = st.session_state.config_dict.get('llm_config', {})

        if not llm_config or not llm_config.get('has_api_key'):
            st.warning("⚠️ LLM Provider not configured. Please configure it in File Management to enable AI-generated reports.")
        elif not safe_llm_request:
            st.error("❌ Required functions not available. Please check imports.")
        else:
            if st.button("🚀 Generate Executive Portfolio Brief", type="primary", key="gen_portfolio_brief"):
                with st.spinner("Analyzing portfolio data and generating comprehensive brief..."):
                    try:
                        # Calculate quadrants with correct key names
                        on_budget_on_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        over_budget_on_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        on_budget_behind_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        over_budget_behind_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0

                        # Get controls for currency formatting
                        controls = st.session_state.config_dict.get('controls', {
                            'currency_symbol': '$',
                            'currency_postfix': ''
                        })

                        currency = controls.get('currency_symbol', '$')
                        postfix = controls.get('currency_postfix', '')

                        # Prepare detailed project data for trend analysis
                        project_details = []
                        for _, row in df.iterrows():
                            project_info = {
                                'name': row.get('Project', row.get('Project Name', 'Unknown')),
                                'budget': row.get('Budget', row.get('BAC', 0)),
                                'cpi': row.get('CPI', 0),
                                'spi': row.get('SPI', 0),
                                'progress': row.get('Completion %', 0),
                                'organization': row.get('Organization', row.get('Org', 'Unknown')),
                                'tier': row.get('Budget_Category', 'Unknown'),
                                'health': row.get('Health_Category', 'Unknown'),
                            }
                            # Add dates if available
                            if 'Plan Start' in row:
                                project_info['start_date'] = str(row['Plan Start'])[:10] if pd.notna(row['Plan Start']) else 'N/A'
                            if 'Plan Finish' in row:
                                project_info['finish_date'] = str(row['Plan Finish'])[:10] if pd.notna(row['Plan Finish']) else 'N/A'

                            project_details.append(project_info)

                        # Analyze trends by organization
                        org_analysis = df.groupby('Organization').agg({
                            'CPI': 'mean',
                            'SPI': 'mean',
                            'Budget': 'sum'
                        }).round(2).to_dict('index') if 'Organization' in df.columns else {}

                        # Analyze trends by tier
                        tier_analysis = df.groupby('Budget_Category').agg({
                            'CPI': 'mean',
                            'SPI': 'mean',
                            'Budget': 'sum'
                        }).round(2).to_dict('index') if 'Budget_Category' in df.columns else {}

                        # Create enhanced prompt with detailed project data
                        prompt = f"""
You are a seasoned Chief Portfolio Officer preparing an executive briefing for the C-Suite. Create a comprehensive, narrative-style report that tells the story of this portfolio's performance with actionable insights.

Write in a professional yet conversational tone that executives can quickly scan but also dive deep into. Use clear headings, bullet points, and specific numbers to support your analysis.

---

# PORTFOLIO EXECUTIVE BRIEFING
**Report Date:** {datetime.now().strftime('%B %d, %Y')}

## DATA SNAPSHOT

**Portfolio Scale:**
- {metrics['total_projects']} active projects under management
- Total Budget at Completion (BAC): {currency}{metrics['total_budget']:,.0f}{' ' + postfix if postfix else ''}
- Current Actual Cost (AC): {currency}{metrics['total_actual_cost']:,.0f}{' ' + postfix if postfix else ''}
- Earned Value (EV): {currency}{metrics['total_earned_value']:,.0f}{' ' + postfix if postfix else ''}
- Average Portfolio Progress: {df['Completion %'].mean() if 'Completion %' in df.columns else 0:.1f}%

**Key Performance Indicators:**
- Portfolio CPI: {metrics['portfolio_cpi']:.2f} (Every dollar spent delivers {currency}{metrics['portfolio_cpi']:.2f} of value)
- Portfolio SPI: {metrics['portfolio_spi']:.2f} (Portfolio is {'ahead of' if metrics['portfolio_spi'] > 1 else 'behind'} schedule)
- Forecast at Completion (EAC): {currency}{metrics['total_eac']:,.0f}{' ' + postfix if postfix else ''}
- Projected Overrun: {currency}{metrics['forecast_overrun']:,.0f}{' ' + postfix if postfix else ''} ({'over' if metrics['forecast_overrun'] > 0 else 'under'} budget)

**Portfolio Health Status:**
- 🟢 Healthy Projects: {metrics['healthy_projects']} ({(metrics['healthy_projects']/metrics['total_projects'])*100:.0f}%)
- 🟡 At Risk Projects: {metrics['at_risk_projects']} ({(metrics['at_risk_projects']/metrics['total_projects'])*100:.0f}%)
- 🔴 Critical Projects: {metrics['critical_projects']} ({(metrics['critical_projects']/metrics['total_projects'])*100:.0f}%)

**Performance Quadrant Distribution:**
- ✅ On Budget & On Schedule: {on_budget_on_schedule} projects ({(on_budget_on_schedule/metrics['total_projects'])*100:.0f}%)
- ⚠️ Over Budget but On Schedule: {over_budget_on_schedule} projects ({(over_budget_on_schedule/metrics['total_projects'])*100:.0f}%)
- ⏰ On Budget but Behind Schedule: {on_budget_behind_schedule} projects ({(on_budget_behind_schedule/metrics['total_projects'])*100:.0f}%)
- 🚨 Over Budget & Behind Schedule: {over_budget_behind_schedule} projects ({(over_budget_behind_schedule/metrics['total_projects'])*100:.0f}%)

**Performance by Organization:**
{chr(10).join([f"- **{org}**: CPI {data['CPI']:.2f}, SPI {data['SPI']:.2f}, Managing {currency}{data['Budget']:,.0f}{' ' + postfix if postfix else ''} ({(data['Budget']/metrics['total_budget'])*100:.0f}% of portfolio)" for org, data in sorted(org_analysis.items(), key=lambda x: x[1]['Budget'], reverse=True)]) if org_analysis else "Organization data not available"}

**Performance by Project Size (Tier):**
{chr(10).join([f"- **{tier}**: CPI {data['CPI']:.2f}, SPI {data['SPI']:.2f}, Total Value {currency}{data['Budget']:,.0f}{' ' + postfix if postfix else ''}" for tier, data in sorted(tier_analysis.items(), key=lambda x: x[1]['Budget'], reverse=True)]) if tier_analysis else "Tier data not available"}

**🚨 Projects Requiring Immediate Attention (Worst CPI):**
{chr(10).join([f"{i+1}. **{p['name']}** ({p['organization']}, {p['tier']})\n   - Cost Performance: CPI {p['cpi']:.2f} (Spending {currency}{(1/p['cpi'] if p['cpi'] > 0 else 0):.2f} for every {currency}1.00 of value)\n   - Schedule: SPI {p['spi']:.2f}\n   - Physical Progress: {p['progress']:.0f}%\n   - Status: {p['health']}" for i, p in enumerate(sorted(project_details, key=lambda x: x['cpi'])[:5])])}

**⏰ Projects with Significant Schedule Delays (Worst SPI):**
{chr(10).join([f"{i+1}. **{p['name']}** ({p['organization']}, {p['tier']})\n   - Schedule Performance: SPI {p['spi']:.2f} ({'Ahead' if p['spi'] > 1 else f'{((1-p['spi'])*100):.0f}% behind'} schedule)\n   - Cost: CPI {p['cpi']:.2f}\n   - Progress: {p['progress']:.0f}%\n   - Status: {p['health']}" for i, p in enumerate(sorted(project_details, key=lambda x: x['spi'])[:5])])}

---

Now, based on this comprehensive data, provide your executive analysis:

## 1. EXECUTIVE SUMMARY
Write 3-5 concise bullets that capture the most important insights an executive needs to know. Each bullet should:
- State the finding clearly with specific numbers
- Explain what it means in business terms
- Indicate if action is required

Example: "Portfolio is tracking 8% over budget with EAC of $X.X billion, driven primarily by Organization Y's projects which represent 35% of overruns despite managing only 20% of portfolio value."

## 2. FINANCIAL PERFORMANCE NARRATIVE
Tell the story of the portfolio's financial health:
- How efficiently are we converting budget into value?
- Which organizations or project types are performing well vs. struggling?
- What does the CPI trend tell us about our project management capabilities?
- Are cost overruns concentrated or widespread?
- What's driving the forecast overrun/underrun?

Use specific examples and numbers. Compare performance across organizations and tiers.

## 3. SCHEDULE PERFORMANCE NARRATIVE
Tell the story of timeline adherence:
- Are we meeting our delivery commitments?
- Where are the delays concentrated? (organization, tier, project type?)
- Is poor schedule performance correlated with cost issues?
- What does this mean for our strategic timeline goals?

## 4. DEEP DIVE: PATTERNS & ROOT CAUSES
This is the most critical section. Analyze the data for hidden patterns:

**Organizational Performance:**
- Are certain organizations consistently outperforming or underperforming?
- Is this a capability issue, resource issue, or complexity issue?
- What can high-performing organizations teach struggling ones?

**Project Size/Complexity Patterns:**
- Do larger projects (higher tiers) perform differently than smaller ones?
- Is our PMO better equipped for certain project scales?
- Should we adjust governance based on tier?

**Systemic vs. Isolated Issues:**
- Are problems concentrated in a few projects or spread across the portfolio?
- Do we have systemic process issues or just a few troubled projects?
- What percentage of the budget is in healthy vs. troubled projects?

**Strategic Implications:**
- What does this performance pattern mean for our strategic objectives?
- Are we at risk of missing key business milestones?

## 5. PORTFOLIO HEALTH RATING & TRAJECTORY

Give an overall rating: **EXCELLENT** | **GOOD** | **FAIR** | **POOR** | **CRITICAL**

Explain your rating with specific rationale. Then discuss:
- Is performance improving, stable, or declining?
- What's the trajectory if current trends continue?
- How does this compare to industry benchmarks or past performance?

## 6. RISK ASSESSMENT & IMPACT QUANTIFICATION

Identify 3-5 specific risks with:
- **Risk Description:** What could go wrong?
- **Likelihood:** High/Medium/Low
- **Impact:** Quantify in dollars and timeline
- **Affected Projects/Organizations:** Be specific
- **Mitigation Status:** What's being done?

Example: "**Risk:** Organization X's consistent underperformance (CPI 0.72) may lead to $15M additional overrun by Q4. **Impact:** Would consume 40% of portfolio contingency reserve. **Mitigation:** Requires immediate intervention."

## 7. STRATEGIC RECOMMENDATIONS (Prioritized)

Provide 5-8 actionable recommendations ranked by impact. For each:
- **Recommendation:** What should be done?
- **Rationale:** Why this matters (tie to data)
- **Expected Impact:** Quantify the benefit
- **Owner:** Who should lead this?
- **Timeline:** When should this happen?
- **Success Metric:** How will we measure success?

Focus on strategic moves, not tactical project management. Think:
- Organizational restructuring
- Resource reallocation
- Process improvements
- Governance changes
- Portfolio rebalancing

## 8. IMMEDIATE ACTION PLAN (Next 60 Days)

Create a prioritized action list with:
- **Action Item:** Specific, clear directive
- **Owner:** Organization or role responsible
- **Deadline:** Specific date or timeframe
- **Success Criteria:** How we'll know it's done
- **Dependencies:** What's needed to execute

Focus on actions that will have the highest impact on portfolio health.

## 9. EXECUTIVE DECISION POINTS

List 2-4 decisions that require C-suite approval:
- Budget reallocation or contingency release
- Project cancellation/pause considerations
- Organizational restructuring
- Strategic priority changes
- Major resource shifts

For each, provide:
- The decision needed
- Options available
- Recommendation with rationale
- Consequences of inaction

---

**WRITING GUIDELINES:**
- Use clear, confident language appropriate for senior executives
- Avoid jargon; explain technical terms when needed
- Use storytelling: connect the dots between data points
- Be specific: use actual numbers, names, and examples
- Be actionable: every insight should lead to a decision or action
- Be balanced: acknowledge both risks and opportunities
- Use formatting (bold, bullets, sections) to make it scannable
- Think like a trusted advisor: be honest about problems and realistic about solutions

Write this report as if you're presenting it to the CEO and Board. They trust your judgment and need you to tell them what's really happening and what they need to do about it.
"""

                        # Make LLM request
                        brief_response = safe_llm_request(
                            llm_config.get('provider', ''),
                            llm_config.get('model', ''),
                            llm_config.get('api_key', ''),
                            llm_config.get('temperature', 0.3),  # Slightly higher for more creative insights
                            llm_config.get('timeout', 180),  # Longer timeout for complex analysis
                            prompt
                        )

                        st.session_state.portfolio_executive_brief = brief_response

                    except Exception as e:
                        st.error(f"Failed to generate portfolio brief: {e}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")

            # Display brief if generated
            if "portfolio_executive_brief" in st.session_state:
                brief = st.session_state.portfolio_executive_brief
                if brief.startswith("Error:") or brief == "No API key available":
                    if brief == "No API key available":
                        st.warning("⚠️ No API key available. Please upload an API key file in File Management.")
                    else:
                        st.error(brief)
                else:
                    st.markdown("#### 📄 Executive Portfolio Report")
                    # Clean up LaTeX/math formatting issues from LLM response
                    # Replace inline math delimiters that shouldn't be interpreted as LaTeX
                    cleaned_brief = brief.replace('$', r'\$')  # Escape dollar signs to prevent LaTeX rendering
                    st.markdown(cleaned_brief)

                    # Download button
                    st.download_button(
                        "📥 Download Portfolio Brief",
                        brief,
                        file_name=f"portfolio_executive_brief_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        key="download_portfolio_brief"
                    )


    # Sidebar with executive styling
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Executive Controls</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📈 Portfolio KPIs</div>', unsafe_allow_html=True)

        # Use the same weighted calculations as the main dashboard for consistency
        st.metric("⭐ Portfolio CPI", f"{portfolio_cpi_weighted:.3f}", help="Weighted Portfolio Cost Performance Index (weighted by budget)")
        st.metric("🎯 Portfolio SPI", f"{portfolio_spi_weighted:.3f}", help="Weighted Portfolio Schedule Performance Index (weighted by budget)")
        st.metric("📊 Portfolio SPIe", f"{portfolio_spie_weighted:.3f}", help="Weighted Schedule Performance Index Estimate (weighted by budget)")
        st.metric("🏢 Total Projects", len(df), help="Active projects in portfolio")
        
        # Add executive summary
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        portfolio_health = "Critical" if portfolio_cpi_weighted < 0.8 else "At Risk" if portfolio_cpi_weighted < 0.95 else "Healthy"
        health_color = "🔴" if portfolio_health == "Critical" else "🟡" if portfolio_health == "At Risk" else "🟢"
        
        st.markdown(f"""
        **Portfolio Status:** {health_color} {portfolio_health}
        
        **Key Insights:**
        - {metrics['critical_projects']} projects need immediate attention
        - {format_currency(metrics['forecast_overrun'], currency_symbol, currency_postfix, thousands=False)} projected overrun
        - {metrics['overrun_percentage']:.1f}% budget variance
        """)

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: 1rem; margin-top: 2rem;">
            <strong>Portfolio Executive Dashboard</strong> • Real-time Intelligence for Strategic Decision Making<br>
            Generated on {date} • Confidential Executive Report
        </div>
    </div>
    """.format(date=datetime.now().strftime('%B %d, %Y at %I:%M %p')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()