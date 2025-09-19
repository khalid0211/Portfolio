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
        padding: 3rem 2rem;
        border-radius: 20px;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
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
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 2;
    }
    
    .main-header h3 {
        font-size: 1.3rem;
        font-weight: 400;
        color: #4a5568;
        position: relative;
        z-index: 2;
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
    
    # Portfolio performance indices
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
        <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
            Real-time Portfolio Health Monitoring & Executive Decision Support
        </div>
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
    
    # Critical Alert Banner
    if metrics['portfolio_cpi'] < 0.85 or metrics['portfolio_spi'] < 0.85:
        st.markdown(f"""
        <div class="alert-banner">
            🚨 CRITICAL PORTFOLIO ALERT: Immediate intervention required • 
            Portfolio CPI: {metrics['portfolio_cpi']:.2f} • 
            Portfolio SPI: {metrics['portfolio_spi']:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Key Performance Indicators
    st.markdown('<div class="section-header">📈 Executive Portfolio Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Calculate weighted Portfolio SPI and SPIe
    portfolio_spi_weighted = (df['SPI'] * df['Budget']).sum() / df['Budget'].sum() if df['Budget'].sum() > 0 else 0
    portfolio_spie_weighted = (df['SPIe'] * df['Budget']).sum() / df['Budget'].sum() if df['Budget'].sum() > 0 else 0

    with col1:
        st.metric(
            label="Total Projects",
            value=f"{metrics['total_projects']:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Portfolio CPI",
            value=f"{metrics['portfolio_cpi']:.3f}",
            delta=f"{(metrics['portfolio_cpi'] - 1) * 100:.1f}%",
            delta_color="inverse" if metrics['portfolio_cpi'] < 1 else "normal"
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPI Gauge
        fig_cpi = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics['portfolio_cpi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cost Performance Index (CPI)"},
            delta = {'reference': 1.0},
            gauge = {
                'axis': {'range': [None, 1.5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.8], 'color': "lightgray"},
                    {'range': [0.8, 1.0], 'color': "yellow"},
                    {'range': [1.0, 1.5], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig_cpi.update_layout(height=300)
        st.plotly_chart(fig_cpi, use_container_width=True)
    
    with col2:
        # SPI Gauge
        fig_spi = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics['portfolio_spi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Schedule Performance Index (SPI)"},
            delta = {'reference': 1.0},
            gauge = {
                'axis': {'range': [None, 1.5]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 0.8], 'color': "lightgray"},
                    {'range': [0.8, 1.0], 'color': "yellow"},
                    {'range': [1.0, 1.5], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig_spi.update_layout(height=300)
        st.plotly_chart(fig_spi, use_container_width=True)
    
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
            yaxis=dict(tickformat='$,.0f')
        )
        st.plotly_chart(fig_financial, use_container_width=True)
        
        # Financial alert
        if metrics['forecast_overrun'] > 0:
            st.error(f"📢 Portfolio EAC: {format_currency(metrics['total_eac'], currency_symbol, currency_postfix, thousands=False)} (+{format_currency(metrics['forecast_overrun'], currency_symbol, currency_postfix, thousands=False)} over budget)")
        else:
            st.success(f"✅ Portfolio EAC: {format_currency(metrics['total_eac'], currency_symbol, currency_postfix, thousands=False)} (Under budget)")
    
    # Critical Projects Section
    with st.expander("🔥 Critical Projects - Executive Intervention Required", expanded=True):
        critical_projects = df[df['Health_Category'] == 'Critical'].copy()
        critical_projects = critical_projects.sort_values('CPI').head(10)

        if not critical_projects.empty:
            # Prepare data for table display
            critical_display_columns = ['Project Name', 'Budget', 'CPI', 'SPI', 'SPIe', 'Actual Cost', 'EAC']

            # Check which columns are available
            available_critical_columns = [col for col in critical_display_columns if col in critical_projects.columns]

            if available_critical_columns:
                critical_table = critical_projects[available_critical_columns].copy()

                # Format the data for display
                if 'Budget' in critical_table.columns:
                    critical_table['Budget'] = critical_table['Budget'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                if 'Actual Cost' in critical_table.columns:
                    critical_table['Actual Cost'] = critical_table['Actual Cost'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                if 'EAC' in critical_table.columns:
                    critical_table['EAC'] = critical_table['EAC'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))

                # Format performance indices
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in critical_table.columns:
                        critical_table[col] = critical_table[col].apply(lambda x: f"{x:.3f}")

                # Style the table with critical project highlighting
                def highlight_critical_projects(val):
                    return 'background-color: #ffebee; color: #d32f2f; font-weight: bold;'

                try:
                    styled_critical_table = critical_table.style.applymap(highlight_critical_projects)
                    st.dataframe(styled_critical_table, use_container_width=True, height=300)
                except:
                    # Fallback without styling
                    st.dataframe(critical_table, use_container_width=True, height=300)

                st.markdown("**⚠️ These projects require immediate executive intervention due to critical performance issues.**")
            else:
                st.info("Critical project data not available for display.")
        else:
            st.info("No critical projects found.")
    
    
    # Interactive Data Explorer
    with st.expander("📋 Executive Project Intelligence Center"):
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
        
        # Filters - Row 1
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Organization filter (check if organization columns exist)
            org_columns = [col for col in df.columns if 'org' in col.lower() or 'department' in col.lower() or 'division' in col.lower()]
            if org_columns:
                org_options = df[org_columns[0]].dropna().unique().tolist()
                organization_filter = st.multiselect(
                    "Filter by Organization",
                    options=org_options,
                    default=org_options
                )
            else:
                # Create dummy organizations for demonstration
                np.random.seed(42)
                orgs = ['Engineering', 'Infrastructure', 'IT', 'Construction', 'Energy', 'Healthcare']
                df['Organization'] = np.random.choice(orgs, len(df))
                organization_filter = st.multiselect(
                    "Filter by Organization",
                    options=orgs,
                    default=orgs
                )
        
        with col2:
            health_filter = st.multiselect(
                "Filter by Health Status",
                options=['Critical', 'At Risk', 'Healthy'],
                default=['Critical', 'At Risk', 'Healthy']
            )
        
        with col3:
            # Budget range toggle controls
            st.write("**Budget Range Controls**")

            # Lower budget range toggle
            enable_lower_budget = st.checkbox("Enable Lower Budget Limit", value=False, key="lower_budget_toggle")
            if enable_lower_budget:
                min_budget = st.number_input(
                    f"Minimum Budget ({currency_symbol})",
                    min_value=0,
                    value=0,
                    step=1000,
                    key="min_budget_value"
                )
            else:
                min_budget = 0

            # Upper budget range toggle
            enable_upper_budget = st.checkbox("Enable Upper Budget Limit", value=False, key="upper_budget_toggle")
            if enable_upper_budget:
                max_budget = st.number_input(
                    f"Maximum Budget ({currency_symbol})",
                    min_value=0,
                    value=1000000,
                    step=1000,
                    key="max_budget_value"
                )
            else:
                max_budget = float('inf')
        
        # Filters - Row 2
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpi_range = st.slider(
                "CPI Range (All)",
                min_value=0.0,
                max_value=2.0,
                value=(0.0, 2.0),
                step=0.01,
                help="Cost Performance Index: >1.0 is good, <1.0 indicates cost overrun. Default: All (0.0-2.0)"
            )
        
        with col2:
            spi_range = st.slider(
                "SPI Range (All)",
                min_value=0.0,
                max_value=2.0,
                value=(0.0, 2.0),
                step=0.01,
                help="Schedule Performance Index: >1.0 is ahead, <1.0 is behind schedule. Default: All (0.0-2.0)"
            )
        
        with col3:
            spie_range = st.slider(
                "SPIe Range (All)",
                min_value=0.0,
                max_value=2.0,
                value=(0.0, 2.0),
                step=0.01,
                help="Schedule Performance Index Estimate. Default: All (0.0-2.0)"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply organization filter
        if org_columns:
            filtered_df = filtered_df[filtered_df[org_columns[0]].isin(organization_filter)]
        else:
            filtered_df = filtered_df[filtered_df['Organization'].isin(organization_filter)]
        
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

        # Projects Expander
        with st.expander("📋 Projects", expanded=True):
            # Display filtered data with enhanced columns
            if org_columns:
                display_columns = ['Project Name', org_columns[0], 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'EAC']
            else:
                display_columns = ['Project Name', 'Organization', 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'EAC']

            available_columns = [col for col in display_columns if col in filtered_df.columns]

            if available_columns:
                # Format the dataframe for better display
                display_df = filtered_df[available_columns].copy()

                # Format budget columns
                if 'Budget' in display_df.columns:
                    display_df['Budget'] = display_df['Budget'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                if 'Actual Cost' in display_df.columns:
                    display_df['Actual Cost'] = display_df['Actual Cost'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                if 'EAC' in display_df.columns:
                    display_df['EAC'] = display_df['EAC'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))

                # Format performance indices
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

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
                    st.dataframe(styled_df, use_container_width=True, height=400)
                else:
                    st.dataframe(display_df, use_container_width=True, height=400)

        # Organizations Expander
        with st.expander("🏢 Organizations", expanded=False):
            # Get organization column name
            org_col = org_columns[0] if org_columns else 'Organization'

            if org_col in filtered_df.columns and len(filtered_df) > 0:
                # Group by organization and calculate consolidated metrics
                try:
                    org_summary = filtered_df.groupby(org_col).agg({
                        'Project Name': 'count',  # Project count
                        'Budget': 'sum',          # Total budget
                        'Actual Cost': 'sum',     # Total actual cost
                        'EAC': 'sum',            # Total EAC
                        'Health_Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # Most common health status
                    }).rename(columns={'Project Name': 'Project Count'})

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
                        org_display_columns = ['Project Count', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'EAC']
                        org_display = org_display[org_display_columns]

                        # Reset index to ensure unique indices for styling
                        org_display = org_display.reset_index()

                        # Format the organizational data for display
                        org_display_formatted = org_display.copy()

                        # Format budget columns
                        if 'Budget' in org_display_formatted.columns:
                            org_display_formatted['Budget'] = org_display_formatted['Budget'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                        if 'Actual Cost' in org_display_formatted.columns:
                            org_display_formatted['Actual Cost'] = org_display_formatted['Actual Cost'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))
                        if 'EAC' in org_display_formatted.columns:
                            org_display_formatted['EAC'] = org_display_formatted['EAC'].apply(lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False))

                        # Format performance indices
                        for col in ['CPI', 'SPI', 'SPIe']:
                            if col in org_display_formatted.columns:
                                org_display_formatted[col] = org_display_formatted[col].apply(lambda x: f"{x:.3f}")

                        # Apply health status styling with error handling
                        try:
                            if 'Health_Category' in org_display_formatted.columns and len(org_display_formatted) > 0:
                                styled_org_df = org_display_formatted.style.applymap(highlight_health, subset=['Health_Category'])
                                st.dataframe(styled_org_df, use_container_width=True, height=300)
                            else:
                                st.dataframe(org_display_formatted, use_container_width=True, height=300)
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
                        use_quarterly = st.checkbox("Use Quarterly View", value=False, key="cash_flow_quarterly")

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
                        use_eac = st.checkbox("Use EAC instead of BAC", value=False, key="cash_flow_use_eac")

                    with col4:
                        st.write("**Configuration:**")
                        st.write("📊 Monthly" if not use_quarterly else "📊 Quarterly")
                        st.write("💰 EAC" if use_eac else "💰 BAC")

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
                                # Calculate cash flow for each project
                                cash_flow_data = []

                                for idx, row in df_cash.iterrows():
                                    start_date = row[start_date_col]
                                    finish_date = row[finish_col]

                                    # Use EAC or BAC based on toggle
                                    if use_eac and 'EAC' in row and pd.notna(row['EAC']) and row['EAC'] > 0:
                                        budget = row['EAC']
                                        budget_type = "EAC"
                                    else:
                                        budget = row.get('Budget', 0)
                                        budget_type = "BAC"

                                    if pd.notna(start_date) and pd.notna(finish_date) and budget > 0:
                                        # Calculate project duration in months
                                        duration_months = max(1, (finish_date - start_date).days / 30.44)  # Average days per month
                                        monthly_cash_flow = budget / duration_months

                                        # Generate monthly cash flow from start to finish
                                        current_date = start_date.replace(day=1)  # Start of month
                                        finish_month = finish_date.replace(day=1)

                                        while current_date <= finish_month:
                                            if use_quarterly:
                                                # Group by quarter
                                                quarter = f"Q{((current_date.month - 1) // 3) + 1}-{current_date.year}"
                                                period_key = quarter
                                            else:
                                                # Monthly view
                                                period_key = current_date.strftime("%b-%Y")

                                            cash_flow_data.append({
                                                'Period': period_key,
                                                'Cash_Flow': monthly_cash_flow,
                                                'Project': row.get('Project Name', 'Unknown'),
                                                'Date': current_date
                                            })

                                            # Move to next month
                                            if current_date.month == 12:
                                                current_date = current_date.replace(year=current_date.year + 1, month=1)
                                            else:
                                                current_date = current_date.replace(month=current_date.month + 1)

                                if cash_flow_data:
                                    # Create DataFrame and aggregate by period
                                    cash_df = pd.DataFrame(cash_flow_data)

                                    if use_quarterly:
                                        # For quarterly, aggregate by quarter and sum cash flows
                                        period_cash_flow = cash_df.groupby('Period')['Cash_Flow'].sum().reset_index()
                                        # Sort by year and quarter
                                        period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                            lambda x: (int(x.split('-')[1]), int(x.split('-')[0][1:]))
                                        )
                                        period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)
                                    else:
                                        # For monthly, aggregate and sort chronologically
                                        period_cash_flow = cash_df.groupby(['Period', 'Date'])['Cash_Flow'].sum().reset_index()
                                        period_cash_flow = period_cash_flow.sort_values('Date')
                                        period_cash_flow = period_cash_flow[['Period', 'Cash_Flow']]

                                    # Create the cash flow chart
                                    budget_type_used = "EAC" if use_eac else "BAC"
                                    chart_title = f"Portfolio Cash Flow ({'EAC' if use_eac else 'BAC'}) - {'Quarterly' if use_quarterly else 'Monthly'} View"

                                    fig_cash_flow = px.bar(
                                        period_cash_flow,
                                        x='Period',
                                        y='Cash_Flow',
                                        title=chart_title,
                                        labels={
                                            'Cash_Flow': f'Cash Flow ({currency_symbol})',
                                            'Period': 'Quarter' if use_quarterly else 'Month'
                                        },
                                        color='Cash_Flow',
                                        color_continuous_scale='blues'
                                    )

                                    # Update layout for better visualization
                                    fig_cash_flow.update_layout(
                                        height=500,
                                        showlegend=False,
                                        xaxis=dict(
                                            title='Quarter' if use_quarterly else 'Month',
                                            tickangle=45
                                        ),
                                        yaxis=dict(
                                            title=f'Cash Flow ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                            tickformat=',.0f'
                                        ),
                                        coloraxis_showscale=False
                                    )

                                    # Update traces for better appearance
                                    fig_cash_flow.update_traces(
                                        texttemplate='%{y:,.0f}',
                                        textposition='outside',
                                        marker_line_width=0
                                    )

                                    st.plotly_chart(fig_cash_flow, use_container_width=True)

                                    # Summary statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Projects", len(df_cash))
                                    with col2:
                                        total_cash_flow = period_cash_flow['Cash_Flow'].sum()
                                        st.metric("Total Cash Flow", format_currency(total_cash_flow, currency_symbol, currency_postfix, thousands=False))
                                    with col3:
                                        avg_monthly = period_cash_flow['Cash_Flow'].mean()
                                        st.metric(f"Avg {'Quarterly' if use_quarterly else 'Monthly'}", format_currency(avg_monthly, currency_symbol, currency_postfix, thousands=False))
                                    with col4:
                                        peak_period = period_cash_flow.loc[period_cash_flow['Cash_Flow'].idxmax(), 'Period']
                                        peak_amount = period_cash_flow['Cash_Flow'].max()
                                        st.metric("Peak Period", f"{peak_period}")
                                        st.caption(f"Amount: {format_currency(peak_amount, currency_symbol, currency_postfix, thousands=False)}")

                                    # Show period breakdown table
                                    with st.expander("📋 Period Breakdown"):
                                        # Format cash flow for display
                                        display_cash_flow = period_cash_flow.copy()
                                        display_cash_flow['Cash_Flow'] = display_cash_flow['Cash_Flow'].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                                        st.dataframe(display_cash_flow, use_container_width=True)

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

        # Summary statistics for filtered data
        if len(filtered_df) > 0:
            st.markdown('<div class="section-header">📊 Filtered Portfolio Analytics</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            # Calculate weighted averages
            total_budget = filtered_df['Budget'].sum()
            weighted_cpi = (filtered_df['CPI'] * filtered_df['Budget']).sum() / total_budget if total_budget > 0 else 0
            weighted_spi = (filtered_df['SPI'] * filtered_df['Budget']).sum() / total_budget if total_budget > 0 else 0
            weighted_spie = (filtered_df['SPIe'] * filtered_df['Budget']).sum() / total_budget if total_budget > 0 else 0

            with col1:
                st.metric("📈 Avg CPI", f"{weighted_cpi:.3f}", help="Weighted Average Cost Performance Index for filtered projects (weighted by budget)")
            with col2:
                st.metric("⏱️ Avg SPI", f"{weighted_spi:.3f}", help="Weighted Average Schedule Performance Index for filtered projects (weighted by budget)")
            with col3:
                st.metric("📊 Avg SPIe", f"{weighted_spie:.3f}", help="Weighted Average Schedule Performance Index Estimate for filtered projects (weighted by budget)")
    
    # Sidebar with executive styling
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Executive Controls</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📤 Portfolio Reports</div>', unsafe_allow_html=True)
        if st.button("📋 Export Critical Projects", key="export_critical"):
            critical_data = df[df['Health_Category'] == 'Critical']
            csv = critical_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Critical Projects Report",
                data=csv,
                file_name=f"critical_projects_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        if st.button("📊 Export Complete Portfolio", key="export_all"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Full Portfolio Report",
                data=csv,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown('<div class="section-header">📈 Portfolio KPIs</div>', unsafe_allow_html=True)

        # Calculate weighted averages for sidebar KPIs
        total_budget_sidebar = df['Budget'].sum()
        weighted_cpi_sidebar = (df['CPI'] * df['Budget']).sum() / total_budget_sidebar if total_budget_sidebar > 0 else 0
        weighted_spi_sidebar = (df['SPI'] * df['Budget']).sum() / total_budget_sidebar if total_budget_sidebar > 0 else 0
        weighted_spie_sidebar = (df['SPIe'] * df['Budget']).sum() / total_budget_sidebar if total_budget_sidebar > 0 else 0

        st.metric("⭐ Average CPI", f"{weighted_cpi_sidebar:.3f}", help="Weighted Portfolio Cost Performance Index (weighted by budget)")
        st.metric("🎯 Average SPI", f"{weighted_spi_sidebar:.3f}", help="Weighted Portfolio Schedule Performance Index (weighted by budget)")
        st.metric("📊 Average SPIe", f"{weighted_spie_sidebar:.3f}", help="Weighted Schedule Performance Index Estimate (weighted by budget)")
        st.metric("🏢 Total Projects", len(df), help="Active projects in portfolio")
        
        # Add executive summary
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        portfolio_health = "Critical" if metrics['portfolio_cpi'] < 0.8 else "At Risk" if metrics['portfolio_cpi'] < 0.95 else "Healthy"
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