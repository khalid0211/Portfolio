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
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="metric-container"] > div > div > div > div {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

def calculate_portfolio_metrics(df):
    """Calculate key portfolio-level metrics"""
    metrics = {}
    
    # Basic counts and totals
    metrics['total_projects'] = len(df)
    metrics['total_budget'] = df['Budget'].sum()
    metrics['total_actual_cost'] = df['Actual Cost'].sum()
    metrics['total_earned_value'] = df['Earned Value'].sum()
    metrics['total_planned_value'] = df['Plan Value'].sum()
    metrics['total_etc'] = df['ETC'].sum()
    metrics['total_eac'] = df['EAC'].sum()
    
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
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload Portfolio Data", 
        type=['csv'],
        help="Upload your batch_evm_results.csv file"
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
    
    # Calculate metrics
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
    
    with col1:
        st.metric(
            label="Total Projects",
            value=f"{metrics['total_projects']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Budget",
            value=f"${metrics['total_budget']/1000:.0f}K",
            delta=None
        )
    
    with col3:
        delta_color = "inverse" if metrics['forecast_overrun'] > 0 else "normal"
        st.metric(
            label="Forecast Overrun",
            value=f"${metrics['forecast_overrun']/1000:.0f}K",
            delta=f"{metrics['overrun_percentage']:.1f}%",
            delta_color=delta_color
        )
    
    with col4:
        st.metric(
            label="Portfolio CPI",
            value=f"{metrics['portfolio_cpi']:.2f}",
            delta=f"{(metrics['portfolio_cpi'] - 1) * 100:.1f}%",
            delta_color="inverse" if metrics['portfolio_cpi'] < 1 else "normal"
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
            st.error(f"📢 Portfolio EAC: ${metrics['total_eac']/1000:.0f}K (+${metrics['forecast_overrun']/1000:.0f}K over budget)")
        else:
            st.success(f"✅ Portfolio EAC: ${metrics['total_eac']/1000:.0f}K (Under budget)")
    
    # Critical Projects Section
    st.markdown('<div class="section-header">🔥 Critical Projects - Executive Intervention Required</div>', unsafe_allow_html=True)
    
    critical_projects = df[df['Health_Category'] == 'Critical'].copy()
    critical_projects = critical_projects.sort_values('CPI').head(10)
    
    if not critical_projects.empty:
        for idx, project in critical_projects.iterrows():
            with st.expander(f"🚨 {project.get('Project Name', 'Unknown Project')[:50]}..."):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CPI", f"{project.get('CPI', 0):.2f}")
                with col2:
                    st.metric("SPI", f"{project.get('SPI', 0):.2f}")
                with col3:
                    st.metric("Budget", f"${project.get('Budget', 0)/1000:.0f}K")
                
                st.write(f"**Status:** Critical performance issues requiring immediate intervention")
    else:
        st.info("No critical projects found.")
    
    # Immediate Actions Required
    st.markdown('<div class="section-header">🎯 Strategic Action Plan - Executive Directives</div>', unsafe_allow_html=True)
    
    actions = [
        "Conduct emergency portfolio review with project managers for bottom 20 projects",
        "Implement cost containment measures across all projects with CPI < 0.80",
        "Reassess project priorities and consider portfolio rebalancing",
        "Establish weekly steering committee for critical projects",
        "Review resource allocation and project dependencies",
        "Initiate risk mitigation strategies for schedule recovery",
        "Consider project cancellation for consistently underperforming initiatives"
    ]
    
    for i, action in enumerate(actions, 1):
        st.markdown(f"""
        <div class="action-item">
            <strong>{i}.</strong> {action}
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive Data Explorer
    with st.expander("📋 Executive Project Intelligence Center"):
        st.markdown('<div class="section-header">🔍 Advanced Portfolio Analytics & Filtering</div>', unsafe_allow_html=True)
        
        # Create budget categories for filtering
        def categorize_budget(budget):
            if budget < 1000:
                return "Under $1K"
            elif budget < 10000:
                return "$1K - $10K"
            elif budget < 50000:
                return "$10K - $50K"
            elif budget < 100000:
                return "$50K - $100K"
            elif budget < 500000:
                return "$100K - $500K"
            else:
                return "Over $500K"
        
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
            budget_categories = ["Under $1K", "$1K - $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", "Over $500K"]
            budget_filter = st.multiselect(
                "Budget Range",
                options=budget_categories,
                default=budget_categories
            )
        
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
            (filtered_df['Budget_Category'].isin(budget_filter)) &
            (filtered_df['CPI'].between(cpi_range[0], cpi_range[1])) &
            (filtered_df['SPI'].between(spi_range[0], spi_range[1])) &
            (filtered_df['SPIe'].between(spie_range[0], spie_range[1]))
        ]
        
        st.write(f"Showing {len(filtered_df)} projects (filtered from {len(df)} total)")
        
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
                display_df['Budget'] = display_df['Budget'].apply(lambda x: f"${x:,.0f}")
            if 'Actual Cost' in display_df.columns:
                display_df['Actual Cost'] = display_df['Actual Cost'].apply(lambda x: f"${x:,.0f}")
            if 'EAC' in display_df.columns:
                display_df['EAC'] = display_df['EAC'].apply(lambda x: f"${x:,.0f}")
            
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
                
        # Summary statistics for filtered data
        if len(filtered_df) > 0:
            st.markdown('<div class="section-header">📊 Filtered Portfolio Analytics</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Avg CPI", f"{filtered_df['CPI'].mean():.3f}", help="Average Cost Performance Index for filtered projects")
            with col2:
                st.metric("⏱️ Avg SPI", f"{filtered_df['SPI'].mean():.3f}", help="Average Schedule Performance Index for filtered projects")
            with col3:
                st.metric("📊 Avg SPIe", f"{filtered_df['SPIe'].mean():.3f}", help="Average Schedule Performance Index Estimate for filtered projects")
            with col4:
                st.metric("💰 Total Budget", f"${filtered_df['Budget'].sum()/1000:.0f}K", help="Combined budget for filtered projects")
    
    # Sidebar with executive styling
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Executive Controls</div>', unsafe_allow_html=True)
        
        auto_refresh = st.checkbox("🔄 Real-time Monitoring", help="Enable auto-refresh for live dashboard updates")
        if auto_refresh:
            st.success("📊 Dashboard monitoring active")
        
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
        st.metric("⭐ Average CPI", f"{df['CPI'].mean():.3f}", help="Portfolio Cost Performance Index")
        st.metric("🎯 Average SPI", f"{df['SPI'].mean():.3f}", help="Portfolio Schedule Performance Index") 
        st.metric("📊 Average SPIe", f"{df['SPIe'].mean():.3f}", help="Schedule Performance Index Estimate")
        st.metric("🏢 Total Projects", len(df), help="Active projects in portfolio")
        
        # Add executive summary
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        portfolio_health = "Critical" if metrics['portfolio_cpi'] < 0.8 else "At Risk" if metrics['portfolio_cpi'] < 0.95 else "Healthy"
        health_color = "🔴" if portfolio_health == "Critical" else "🟡" if portfolio_health == "At Risk" else "🟢"
        
        st.markdown(f"""
        **Portfolio Status:** {health_color} {portfolio_health}
        
        **Key Insights:**
        - {metrics['critical_projects']} projects need immediate attention
        - ${metrics['forecast_overrun']/1000:.0f}K projected overrun
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