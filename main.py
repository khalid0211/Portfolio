"""
Portfolio Management Suite - Main Navigation
A comprehensive project portfolio analysis and executive dashboard system.
"""

import streamlit as st
from utils.auth import check_authentication, show_user_info_sidebar
from utils.auth_utils import check_page_access, is_admin
from utils.firebase_db import db

# Page configuration
st.set_page_config(
    page_title="Portfolio Management Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication first
if not check_authentication():
    st.stop()  # Stop here if not authenticated

# Custom CSS for navigation
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        margin: 0.5rem;
        display: inline-block;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio Analysis Suite</h1>
    <h3>Project Portfolio Analysis & Executive Dashboard</h3>
    <p style="margin-top: 1rem; font-size: 1.1em; color: #666; font-style: italic;">
        Smarter Projects and Portfolios with Earned Value Analysis and AI-Powered Executive Reporting<br>
        <strong>Beta Version 0.9 • Released September 30, 2025</strong><br>
        Developed by Dr. Khalid Ahmad Khan – <a href="https://www.linkedin.com/in/khalidahmadkhan/" target="_blank" style="color: #0066cc; text-decoration: none;">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
st.markdown("## Choose Your Tool")

# Get user email for access checks
user_email = st.session_state.get('user_email')

# Determine which pages to show
show_file_mgmt = check_page_access(db, user_email, 'file_management')
show_manual_entry = check_page_access(db, user_email, 'manual_data_entry')
show_project_analysis = check_page_access(db, user_email, 'project_analysis')
show_portfolio_analysis = check_page_access(db, user_email, 'portfolio_analysis')
show_portfolio_charts = check_page_access(db, user_email, 'portfolio_charts')
show_cash_flow = check_page_access(db, user_email, 'cash_flow_simulator')
show_evm_simulator = check_page_access(db, user_email, 'evm_simulator')

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: File Management, Manual Data Entry, Project Analysis
# ═══════════════════════════════════════════════════════════════════
section1_pages = [show_file_mgmt, show_manual_entry, show_project_analysis]
if any(section1_pages):
    col1, col2, col3 = st.columns(3)

    with col1:
        if show_file_mgmt:
            st.markdown("""
            #### 📁 File Management
            - Data import (CSV/JSON)
            - Configuration settings
            - Batch calculations
            - Export & download options
            """)
            if st.button("📁 Open File Management", key="file_mgmt_btn", use_container_width=True):
                st.switch_page("pages/1_File_Management.py")

    with col2:
        if show_manual_entry:
            st.markdown("""
            #### 📝 Manual Data Entry
            - Quick project data input
            - Direct data entry interface
            - Alternative to file upload
            - Instant data validation
            """)
            if st.button("✏️ Open Manual Data Entry", key="manual_btn", use_container_width=True):
                st.switch_page("pages/2_Manual_Data_Entry.py")

    with col3:
        if show_project_analysis:
            st.markdown("""
            #### 🔍 Project Analysis
            - Single project EVM analysis
            - Individual project insights
            - Detailed calculations
            - Project-level charts
            """)
            if st.button("🚀 Open Project Analysis", key="project_btn", use_container_width=True):
                st.switch_page("pages/3_Project_Analysis.py")

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Portfolio Analysis, Portfolio Charts
# ═══════════════════════════════════════════════════════════════════
section2_pages = [show_portfolio_analysis, show_portfolio_charts]
if any(section2_pages):
    col1, col2, col3 = st.columns(3)

    with col1:
        if show_portfolio_analysis:
            st.markdown("""
            #### 📈 Portfolio Analysis
            - Portfolio health metrics
            - Multi-project comparisons
            - Strategic performance indicators
            - Executive summary reports
            """)
            if st.button("📊 Open Portfolio Analysis", key="portfolio_btn", use_container_width=True):
                st.switch_page("pages/4_Portfolio_Analysis.py")

    with col2:
        if show_portfolio_charts:
            st.markdown("""
            #### 📊 Portfolio Charts
            - Interactive baseline vs forecast timeline
            - Organization, budget, and date filtering
            - EV progress shading with forecast alerts
            - Hover insights for project detail
            """)
            if st.button("📊 Open Portfolio Charts", key="gantt_btn", use_container_width=True):
                st.switch_page("pages/5_Portfolio_Charts.py")

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Cash Flow Simulator, EVM Simulator
# ═══════════════════════════════════════════════════════════════════
section3_pages = [show_cash_flow, show_evm_simulator]
if any(section3_pages):
    col1, col2, col3 = st.columns(3)

    with col1:
        if show_cash_flow:
            st.markdown("""
            #### 💸 Cash Flow Simulator
            - Project delay impact analysis
            - Multiple cash flow patterns (Linear, S-Curve, Highway, Building)
            - Inflation and delay modeling
            - Baseline comparison & export capabilities
            """)
            if st.button("📈 Open Cash Flow Simulator", key="cashflow_btn", use_container_width=True):
                st.switch_page("pages/6_Cash_Flow_Simulator.py")

    with col2:
        if show_evm_simulator:
            st.markdown("""
            #### 🎯 EVM Simulator
            - Interactive EVM scenario modeling
            - Performance index simulations
            - Schedule and cost impact analysis
            - Advanced forecasting tools
            """)
            if st.button("🎯 Open EVM Simulator", key="evm_btn", use_container_width=True):
                st.switch_page("pages/7_EVM_Simulator.py")

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: User Management (Admin Only)
# ═══════════════════════════════════════════════════════════════════
if is_admin(user_email):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 👥 User Management
        - Manage user access permissions
        - Grant/revoke page access
        - View user statistics
        - Admin-only access control
        """)
        if st.button("👥 Open User Management", key="user_mgmt_btn", use_container_width=True):
            st.switch_page("pages/8_User_Management.py")

    st.markdown("---")

# Quick stats if data exists
if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
    st.markdown("---")
    st.markdown("### 📋 Current Session Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Projects in Session", len(st.session_state.batch_results))
    with col2:
        if 'CPI' in st.session_state.batch_results.columns:
            avg_cpi = st.session_state.batch_results['CPI'].mean()
            st.metric("Average CPI", f"{avg_cpi:.2f}")
    with col3:
        st.success("✅ Data Ready for Dashboard")

# Help section
with st.expander("ℹ️ How to Use This System"):
    st.markdown("""
    **Portfolio Management Workflow:**
    1. **Start with Portfolio Analysis** to upload your project data
    2. **Run batch calculations** to process all projects
    3. **Click "Generate Executive Dashboard"** to view executive summary with filtering
    4. **Or navigate manually** using the buttons above

    **Individual Project Tools:**
    - **Manual Data Entry:** Quick single project analysis
    - **Cash Flow Simulator:** Model project delays and financial impacts
    - **EVM Simulator:** Advanced earned value modeling and forecasting

    **Data Flow:**
    - Portfolio tools share session data automatically
    - Individual tools work independently
    - All tools provide export capabilities for further analysis
    """)

# Show user info in sidebar
show_user_info_sidebar()