import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header with glassmorphism effect */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        opacity: 0.1;
        animation: gradientShift 8s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 2;
    }

    .main-header h3 {
        font-size: 1.2rem;
        font-weight: 400;
        color: #4a5568;
        position: relative;
        z-index: 2;
        margin-bottom: 0.3rem;
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

@st.cache_data
def load_data():
    """Load and process the portfolio data"""
    try:
        # You can replace this with your actual file path
        df = pd.read_csv('batch_evm_results.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle missing values
        numeric_columns = ['Budget', 'Actual Cost', 'Earned Value', 'Plan Value', 
                          'CPI', 'SPI', 'ETC', 'EAC', '% Budget Used', '% Time Used']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except FileNotFoundError:
        st.error("Please upload the batch_evm_results.csv file to proceed.")
        return None

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
        # Portfolio Analysis -> Dashboard expected
        'bac': 'Budget',
        'ac': 'Actual Cost',
        'earned_value': 'Earned Value',
        'planned_value': 'Plan Value',
        'etc': 'ETC',
        'estimate_at_completion': 'EAC',
        'cost_performance_index': 'CPI',
        'schedule_performance_index': 'SPI',
        'spie': 'SPIe',
        'project_name': 'Project Name',
        'project_id': 'Project ID'
    }

    # Create a copy of the dataframe
    mapped_df = df.copy()

    # Rename columns if they exist in the source format
    for source_col, target_col in column_mapping.items():
        if source_col in mapped_df.columns and target_col not in mapped_df.columns:
            mapped_df[target_col] = mapped_df[source_col]

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
    st.markdown("""
    <div class="main-header">
        <h1>📊 Portfolio Executive Command Center</h1>
        <h3>Chief Projects Officer • Strategic Portfolio Intelligence</h3>
        <div style="margin-top: 0.8rem; font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">
            Portfolio Health Monitoring & Executive Decision Support
        </div>
        <p style="margin-top: 0.8rem; font-size: 0.9em; color: #666; font-style: italic; margin-bottom: 0;">
            Developed by Dr. Khalid Ahmad Khan – <a href="https://www.linkedin.com/in/khalidahmadkhan/" target="_blank" style="color: #0066cc; text-decoration: none;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
            df = load_data()
    
    if df is None:
        st.stop()

    # Map columns to expected format and calculate metrics
    df = map_columns_to_standard(df)
    metrics = calculate_portfolio_metrics(df)
    
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
        
        # Health metrics with enhanced styling
        st.markdown(f'<div class="status-success">✅ Healthy Projects: {metrics["healthy_projects"]} ({metrics["healthy_projects"]/metrics["total_projects"]*100:.0f}%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-warning">⚠️ At Risk Projects: {metrics["at_risk_projects"]} ({metrics["at_risk_projects"]/metrics["total_projects"]*100:.0f}%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-critical">🚨 Critical Projects: {metrics["critical_projects"]} ({metrics["critical_projects"]/metrics["total_projects"]*100:.0f}%)</div>', unsafe_allow_html=True)
    
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
        
        # Financial alert
        if metrics['forecast_overrun'] > 0:
            st.error(f"📢 Portfolio EAC: {format_currency(metrics['total_eac'], currency_symbol, currency_postfix, thousands=False)} (+{format_currency(metrics['forecast_overrun'], currency_symbol, currency_postfix, thousands=False)} over budget)")
        else:
            st.success(f"✅ Portfolio EAC: {format_currency(metrics['total_eac'], currency_symbol, currency_postfix, thousands=False)} (Under budget)")


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
                        st.dataframe(styled_table, use_container_width=True, height=300, column_config=spotlight_column_config)
                    except:
                        st.dataframe(spotlight_table, use_container_width=True, height=300, column_config=spotlight_column_config)
                else:
                    # Standard display for other views
                    st.dataframe(spotlight_table, use_container_width=True, height=300, column_config=spotlight_column_config)

                st.markdown(f"**{view_description}**")
            else:
                st.info("Required data columns not available for this view.")
        else:
            st.info(f"No data available for {selected_view.split('.')[1].strip()}.")
    
    
    # Interactive Data Explorer
    with st.expander("📋 Executive Project Intelligence Center", expanded=False):
        st.markdown('<div class="section-header">🔍 Advanced Portfolio Analytics & Filtering</div>', unsafe_allow_html=True)
        
        # Create budget categories for filtering
        def categorize_budget(budget, symbol=currency_symbol):
            if budget < 1000:
                return f"Under {symbol} 1K"
            elif budget < 10000:
                return f"{symbol} 1K - {symbol} 10K"
            elif budget < 50000:
                return f"{symbol} 10K - {symbol} 50K"
            elif budget < 100000:
                return f"{symbol} 50K - {symbol} 100K"
            elif budget < 500000:
                return f"{symbol} 100K - {symbol} 500K"
            else:
                return f"Over {symbol} 500K"
        
        df['Budget_Category'] = df['Budget'].apply(categorize_budget)

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

        # Apply organization filter
        if org_columns:
            filtered_df = filtered_df[filtered_df[org_columns[0]].isin(organization_filter)]
        else:
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
        
        st.write(f"Showing {len(filtered_df)} projects (filtered from {len(df)} total)")

        # ═══════════════════════════════════════════════════════════════════
        # FILTERED PORTFOLIO PERFORMANCE CURVE
        # ═══════════════════════════════════════════════════════════════════

        # Portfolio Time/Budget Performance Curve (Filtered)
        with st.expander("📈 Portfolio Performance Curve (Filtered)", expanded=True):
            if len(filtered_df) > 0 and 'CPI' in filtered_df.columns and 'SPI' in filtered_df.columns and 'Budget' in filtered_df.columns:
                # Create fixed BAC-based tier ranges
                budget_values = filtered_df['Budget'].dropna()
                if len(budget_values) > 0:
                    def get_budget_category(budget):
                        if pd.isna(budget):
                            return "Unknown"
                        elif budget >= 15000:
                            return f"Tier 5: Strategic (≥ {currency_symbol}15,000)"
                        elif budget >= 8500:
                            return f"Tier 4: Major ({currency_symbol}8,500 - {currency_symbol}15,000)"
                        elif budget >= 6000:
                            return f"Tier 3: Large-Scale ({currency_symbol}6,000 - {currency_symbol}8,500)"
                        elif budget >= 4000:
                            return f"Tier 2: Mid-Range ({currency_symbol}4,000 - {currency_symbol}6,000)"
                        else:
                            return f"Tier 1: Small-Scale (< {currency_symbol}4,000)"

                    # Add budget category to filtered dataframe
                    df_scatter = filtered_df.copy()
                    df_scatter['Budget_Range'] = df_scatter['Budget'].apply(get_budget_category)

                    # Define the desired order for the legend (Tier 5 to Tier 1)
                    tier_order = [
                        f"Tier 5: Strategic (≥ {currency_symbol}15,000)",
                        f"Tier 4: Major ({currency_symbol}8,500 - {currency_symbol}15,000)",
                        f"Tier 3: Large-Scale ({currency_symbol}6,000 - {currency_symbol}8,500)",
                        f"Tier 2: Mid-Range ({currency_symbol}4,000 - {currency_symbol}6,000)",
                        f"Tier 1: Small-Scale (< {currency_symbol}4,000)"
                    ]

                    # Convert Budget_Range to categorical with specific order
                    df_scatter['Budget_Range'] = pd.Categorical(df_scatter['Budget_Range'], categories=tier_order, ordered=True)

                    # Create scatter plot with consistent dot sizes and explicit ordering
                    fig_performance = px.scatter(
                        df_scatter,
                        x='SPI',
                        y='CPI',
                        color='Budget_Range',
                        hover_data=['Project Name', 'Budget'],
                        title=f"Filtered Portfolio Performance Matrix ({len(filtered_df)} projects)",
                        labels={
                            'SPI': 'Schedule Performance Index (SPI)',
                            'CPI': 'Cost Performance Index (CPI)',
                            'Budget_Range': 'Budget Range'
                        },
                        color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#3498db'],
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        healthy_count = len(df_scatter[df_scatter['Health_Category'] == 'Healthy']) if 'Health_Category' in df_scatter.columns else 0
                        st.metric("✅ Healthy Projects", healthy_count)
                    with col2:
                        at_risk_count = len(df_scatter[df_scatter['Health_Category'] == 'At Risk']) if 'Health_Category' in df_scatter.columns else 0
                        st.metric("⚠️ At Risk Projects", at_risk_count)
                    with col3:
                        critical_count = len(df_scatter[df_scatter['Health_Category'] == 'Critical']) if 'Health_Category' in df_scatter.columns else 0
                        st.metric("🚨 Critical Projects", critical_count)

                    # Add interpretation guide
                    st.markdown(f"""
                    **📊 How to Read This Chart:**
                    - **X-axis (SPI):** Schedule Performance - Right is better (ahead of schedule)
                    - **Y-axis (CPI):** Cost Performance - Up is better (under budget)
                    - **Dot Colors:** Project tiers by budget:
                      - 🔴 **Tier 5:** Strategic (≥ {currency_symbol}15K{currency_postfix})
                      - 🟠 **Tier 4:** Major ({currency_symbol}8.5K{currency_postfix} - {currency_symbol}15K{currency_postfix})
                      - 🟡 **Tier 3:** Large-Scale ({currency_symbol}6K{currency_postfix} - {currency_symbol}8.5K{currency_postfix})
                      - 🟢 **Tier 2:** Mid-Range ({currency_symbol}4K{currency_postfix} - {currency_symbol}6K{currency_postfix})
                      - 🔵 **Tier 1:** Small-Scale (< {currency_symbol}4K{currency_postfix})
                    - **Target Zone:** Upper right quadrant (SPI > 1.0, CPI > 1.0)
                    - **Hover:** Click any dot to see project name and budget details
                    - **Chart updates automatically** based on your filter selections above
                    """)
                else:
                    st.info("No budget data available for performance curve.")
            else:
                st.info("Performance curve requires CPI, SPI, and Budget data.")

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
                    st.dataframe(styled_df, use_container_width=True, height=400, column_config=column_config)
                else:
                    st.dataframe(display_df, use_container_width=True, height=400, column_config=column_config)

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
                                st.dataframe(styled_org_df, use_container_width=True, height=300, column_config=org_column_config)
                            else:
                                st.dataframe(org_display_formatted, use_container_width=True, height=300, column_config=org_column_config)
                        except (KeyError, ValueError) as e:
                            # Fallback to unstyled dataframe if styling fails
                            st.dataframe(org_display_formatted, use_container_width=True, height=300)
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

        # Portfolio Graph Expander
        with st.expander("📊 Portfolio Graph", expanded=False):
            # Get organization column name
            org_col = org_columns[0] if org_columns else 'Organization'

            if org_col in filtered_df.columns and len(filtered_df) > 0:
                try:
                    # Calculate total budget by organization
                    org_budget_summary = filtered_df.groupby(org_col).agg({
                        'Budget': 'sum'
                    }).reset_index()

                    # Sort by budget in descending order
                    org_budget_summary = org_budget_summary.sort_values('Budget', ascending=True)  # ascending=True for horizontal bar chart

                    if len(org_budget_summary) > 0:
                        # Create horizontal bar chart
                        fig_portfolio = px.bar(
                            org_budget_summary,
                            x='Budget',
                            y=org_col,
                            orientation='h',
                            title="Total Budget by Organization",
                            labels={'Budget': f'Total Budget ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})', org_col: 'Organization'},
                            color='Budget',
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
                            st.metric("Largest Budget", format_currency(org_budget_summary['Budget'].max(), currency_symbol, currency_postfix, thousands=False))
                        with col3:
                            st.metric("Smallest Budget", format_currency(org_budget_summary['Budget'].min(), currency_symbol, currency_postfix, thousands=False))

                    else:
                        st.info("No organization budget data available to display.")

                except Exception as e:
                    st.error(f"Error creating portfolio graph: {str(e)}")
                    st.info("Unable to display portfolio graph with current data.")
            else:
                if len(filtered_df) == 0:
                    st.info("No projects available for portfolio graph.")
                else:
                    st.info("Organization data not available for portfolio graph.")

        # Cash Flow Chart Expander
        with st.expander("💰 Cash Flow Chart", expanded=False):
            if len(filtered_df) > 0:
                # Detect possible date columns
                date_columns = []
                for col in filtered_df.columns:
                    if any(keyword in col.lower() for keyword in ['start', 'begin', 'finish', 'end', 'complete', 'date']):
                        date_columns.append(col)

                # Look for specific date column patterns
                start_date_col = None
                plan_finish_col = None
                likely_finish_col = None
                expected_finish_col = None

                for col in date_columns:
                    col_lower = col.lower()
                    if 'start' in col_lower or 'begin' in col_lower:
                        start_date_col = col
                    elif 'expected' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                        expected_finish_col = col
                    elif 'likely' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                        likely_finish_col = col
                    elif 'plan' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                        plan_finish_col = col
                    elif 'finish' in col_lower or 'end' in col_lower or 'complete' in col_lower:
                        if plan_finish_col is None:  # Use as plan finish if not already found
                            plan_finish_col = col

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
                                                  options=["Plan", "Predicted", "Both"],
                                                  index=0,
                                                  key="cash_flow_type",
                                                  help="Plan: BAC + OD, Predicted: EAC + LD, Both: Line chart comparing both")

                    with col4:
                        st.write("**Configuration:**")
                        st.write(f"📊 {time_period}")
                        if cash_flow_type == "Plan":
                            st.write("💰 BAC/OD")
                        elif cash_flow_type == "Predicted":
                            st.write("💰 BAC/LD")
                        else:  # Both
                            st.write("💰 BAC/OD vs BAC/LD")

                    # Show expected date validation info if available
                    if expected_date_info:
                        st.info(expected_date_info)

                    # Select finish date column based on choice
                    if finish_date_choice == "Expected Finish" and valid_expected_finish_col:
                        finish_col = valid_expected_finish_col
                    elif finish_date_choice == "Likely Finish" and likely_finish_col:
                        finish_col = likely_finish_col
                    else:
                        finish_col = plan_finish_col

                    if finish_col:
                        try:
                            # Convert date columns to datetime
                            df_cash = filtered_df.copy()

                            # Parse dates with error handling
                            df_cash[start_date_col] = pd.to_datetime(df_cash[start_date_col], errors='coerce')
                            df_cash[finish_col] = pd.to_datetime(df_cash[finish_col], errors='coerce')

                            # Remove rows with invalid dates
                            df_cash = df_cash.dropna(subset=[start_date_col, finish_col])

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
                                        return date.strftime("%b-%Y")
                                    elif time_period == "Quarter":
                                        quarter = f"Q{((date.month - 1) // 3) + 1}-{date.year}"
                                        return quarter
                                    else:  # FY
                                        return get_financial_year(date)

                                def calculate_cash_flow_for_scenario(df_cash, scenario_type):
                                    """Calculate cash flow for Plan (BAC/OD) or Predicted (BAC/LD) scenario"""
                                    cash_flow_data = []

                                    for idx, row in df_cash.iterrows():
                                        start_date = row[start_date_col]

                                        if pd.notna(start_date):
                                            # Always use BAC (Budget) for both scenarios
                                            budget = row.get('Budget', 0)  # BAC

                                            if scenario_type == "Plan":
                                                # Use Original Duration (OD)
                                                if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                                    duration_months = max(1, row['original_duration_months'])
                                                else:
                                                    # Fallback: calculate from plan start to plan finish
                                                    plan_finish = row.get('Plan Finish')
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
                                                if 'forecast_duration' in row and pd.notna(row.get('forecast_duration')):
                                                    ld = row['forecast_duration']
                                                    # Get OD for cap calculation
                                                    if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                                        od = row['original_duration_months']
                                                    else:
                                                        # Fallback: calculate OD from plan dates
                                                        plan_finish = row.get('Plan Finish')
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
                                                        'Project': row.get('Project Name', 'Unknown'),
                                                        'Date': current_date,
                                                        'Scenario': scenario_type
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
                                else:  # Both
                                    plan_data = calculate_cash_flow_for_scenario(df_cash, "Plan")
                                    predicted_data = calculate_cash_flow_for_scenario(df_cash, "Predicted")
                                    cash_flow_data = plan_data + predicted_data

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
                                            # For "Jan-2024" format
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
                                            # For "Q1-2024" format
                                            try:
                                                quarter, year = period_str.split('-')
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
                                        if cash_flow_type == "Both":
                                            # For Both option, create line chart with two series
                                            period_cash_flow = cash_df.groupby(['Period', 'Scenario'])['Cash_Flow'].sum().reset_index()
                                            period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                                lambda x: get_sort_key(x, time_period)
                                            )
                                            period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                            chart_title = f"Portfolio Cash Flow Comparison (Plan: BAC/OD vs Predicted: BAC/LD) - {time_period} View"

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
                                                markers=True
                                            )
                                        else:
                                            # For Plan or Predicted, create bar chart
                                            period_cash_flow = cash_df.groupby('Period')['Cash_Flow'].sum().reset_index()
                                            period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                                lambda x: get_sort_key(x, time_period)
                                            )
                                            period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                            scenario_label = f"({'Plan: BAC/OD' if cash_flow_type == 'Plan' else 'Predicted: BAC/LD'})"
                                            chart_title = f"Portfolio Cash Flow {scenario_label} - {time_period} View"

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
                                                color_continuous_scale='blues'
                                            )

                                        # Update layout for better visualization
                                        fig_cash_flow.update_layout(
                                            height=500,
                                            showlegend=True if cash_flow_type == "Both" else False,
                                            xaxis=dict(
                                                title=time_period,
                                                tickangle=45
                                            ),
                                            yaxis=dict(
                                                title=f'Cash Flow ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                                tickformat=',.0f'
                                            ),
                                            coloraxis_showscale=False if cash_flow_type != "Both" else True
                                        )

                                        # Update traces for better appearance
                                        if cash_flow_type != "Both":
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
                                            if cash_flow_type == "Both":
                                                # Show comparison table for both scenarios
                                                # Create pivot table with numeric data first
                                                pivot_df = period_cash_flow.pivot_table(index='Period', columns='Scenario', values='Cash_Flow', aggfunc='sum', fill_value=0)
                                                # Then format the values for display
                                                for col in pivot_df.columns:
                                                    pivot_df[col] = pivot_df[col].apply(
                                                        lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                                    )
                                                st.dataframe(pivot_df, use_container_width=True)
                                            else:
                                                # Show single scenario table
                                                display_cash_flow = period_cash_flow.copy()
                                                display_cash_flow['Cash_Flow'] = display_cash_flow['Cash_Flow'].apply(
                                                    lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                                )
                                                st.dataframe(display_cash_flow, use_container_width=True)
                                    else:
                                        st.warning("No valid cash flow data could be generated from the selected projects.")

                                else:
                                    st.warning("No valid cash flow data could be generated from the selected projects.")
                            else:
                                st.warning("No projects have valid start and finish dates.")

                        except Exception as e:
                            st.error(f"Error processing cash flow data: {str(e)}")
                            st.info("Please check that date columns contain valid date formats.")
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

        # Approvals Chart Expander
        with st.expander("📈 Approvals Chart", expanded=False):
            if len(filtered_df) > 0:
                # Use specific column names for approvals chart (matching your data)
                start_date_col = 'plan_start'
                budget_col = 'bac'

                if start_date_col in filtered_df.columns and budget_col in filtered_df.columns:
                    # Controls for approvals chart
                    col1, col2 = st.columns(2)

                    with col1:
                        approval_time_period = st.selectbox("Approval Time Period",
                                                          options=["Month", "Quarter", "FY"],
                                                          index=0,
                                                          key="approval_time_period",
                                                          help="Month: Monthly approvals, Quarter: Quarterly approvals, FY: Financial Year (July-June)")

                    with col2:
                        st.write("**Configuration:**")
                        st.write(f"📊 {approval_time_period}")
                        st.write("💰 BAC by Approval Date")

                    try:
                        # Prepare data for approvals chart
                        df_approvals = filtered_df.copy()

                        # Parse approval date (using Plan Start as proxy for approval)
                        df_approvals[start_date_col] = pd.to_datetime(df_approvals[start_date_col], errors='coerce')
                        df_approvals = df_approvals.dropna(subset=[start_date_col, budget_col])

                        if len(df_approvals) > 0:
                            # Helper functions (reuse from cash flow)
                            def get_financial_year_approvals(date):
                                """Get financial year string for a date (FY starts July 1st)"""
                                if date.month >= 7:  # July onwards = start of FY
                                    return f"FY{date.year + 1}"
                                else:  # Jan-June = end of previous FY
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

                            # Calculate approvals data
                            approvals_data = []
                            for idx, row in df_approvals.iterrows():
                                approval_date = row[start_date_col]
                                budget = row.get(budget_col, 0)

                                if pd.notna(approval_date) and budget > 0:
                                    period_key = get_approval_period_key(approval_date, approval_time_period)

                                    # Get financial values
                                    ac = row.get('Actual Cost', 0)
                                    ev = row.get('Earned Value', 0)
                                    eac = row.get('EAC', 0)

                                    approvals_data.append({
                                        'Period': period_key,
                                        'BAC': budget,
                                        'AC': ac,
                                        'EV': ev,
                                        'EAC': eac,
                                        'Project': row.get('Project Name', 'Unknown'),
                                        'Date': approval_date
                                    })

                            if approvals_data:
                                # Create DataFrame and aggregate by period
                                approvals_df = pd.DataFrame(approvals_data)
                                period_approvals = approvals_df.groupby('Period').agg({
                                    'BAC': 'sum',
                                    'AC': 'sum',
                                    'EV': 'sum',
                                    'EAC': 'sum',
                                    'Project': 'count'  # Count number of projects
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

                                # Create the approvals chart
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
                                    height=450,  # Increased height to accommodate labels
                                    showlegend=False,
                                    xaxis=dict(
                                        title=approval_time_period,
                                        tickangle=45
                                    ),
                                    yaxis=dict(
                                        title=f'Total BAC ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                        tickformat=',.0f'
                                    ),
                                    coloraxis_showscale=False,
                                    margin=dict(t=80, b=60, l=60, r=60)  # Add margins for text labels
                                )

                                # Update traces for better appearance
                                fig_approvals.update_traces(
                                    texttemplate='%{y:,.0f}',
                                    textposition='auto',  # Changed from 'outside' to 'auto' for better positioning
                                    textfont=dict(size=10),  # Smaller text to fit better
                                    cliponaxis=False  # Prevent clipping of text labels
                                )

                                st.plotly_chart(fig_approvals, use_container_width=True)

                                # Show detailed data table
                                with st.expander("📊 Detailed Approvals Data", expanded=False):
                                    display_approvals = period_approvals.copy()

                                    # Format currency columns
                                    for col in ['BAC', 'AC', 'EV', 'EAC']:
                                        if col in display_approvals.columns:
                                            display_approvals[col] = display_approvals[col].apply(
                                                lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                            )

                                    # Format percentage columns
                                    for col in ['% AC/BAC', '% EV/BAC', '% EAC/BAC']:
                                        if col in display_approvals.columns:
                                            display_approvals[col] = display_approvals[col].apply(lambda x: f"{x:.1f}%")

                                    st.dataframe(display_approvals, use_container_width=True)

                            else:
                                st.warning("No valid approval data could be generated from the selected projects.")

                        else:
                            st.warning("No projects have valid approval dates and budget values.")

                    except Exception as e:
                        st.error(f"Error processing approvals data: {str(e)}")
                        st.info("Please check that date and budget columns contain valid values.")

                else:
                    st.info("Approvals chart requires 'plan_start' and 'bac' columns.")
                    st.info("Available columns:")
                    available_cols = list(filtered_df.columns)
                    for col in available_cols:
                        st.write(f"• {col}")
            else:
                st.info("No data available for approvals analysis.")

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
                ld_col = 'forecast_duration'
                bac_col = 'Budget'

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