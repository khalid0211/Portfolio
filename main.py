"""
Portfolio Management Suite - Main Navigation
A comprehensive project portfolio analysis and executive dashboard system.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Portfolio Management Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    <h1>📊 Portfolio Management Suite</h1>
    <h3>Project Portfolio Analysis & Executive Dashboard System</h3>
</div>
""", unsafe_allow_html=True)

# Navigation
st.markdown("## Choose Your Application")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔍 Portfolio Analysis
    - Upload project data (CSV/JSON)
    - Perform EVM calculations
    - Batch processing for multiple projects
    - Download detailed results
    """)

    if st.button("🚀 Open Portfolio Analysis", key="portfolio_btn", use_container_width=True):
        st.switch_page("pages/1_Portfolio_Analysis.py")

with col2:
    st.markdown("""
    ### 📈 Executive Dashboard
    - View portfolio health metrics
    - Critical project alerts
    - Strategic performance indicators
    - Executive summary reports
    """)

    if st.button("📊 Open Executive Dashboard", key="dashboard_btn", use_container_width=True):
        st.switch_page("pages/2_Executive_Dashboard.py")

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
    **Typical Workflow:**
    1. **Start with Portfolio Analysis** to upload your project data
    2. **Run batch calculations** to process all projects
    3. **Click "Generate Executive Dashboard"** to view executive summary
    4. **Or navigate manually** using the buttons above

    **Data Flow:**
    - Data processed in Portfolio Analysis is automatically available in Executive Dashboard
    - No need to download/upload files between applications
    - Both applications share the same session data
    """)