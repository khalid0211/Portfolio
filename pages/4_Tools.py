"""
Standalone Tools - Simulation Utilities
Independent analysis and simulation tools for financial and project management.
"""

import streamlit as st
import subprocess
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Tools - Portfolio Management Suite",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .tool-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    .tool-button {
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
        width: 100%;
    }
    .tool-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="tool-card">
    <h1>🛠️ Standalone Tools</h1>
    <h3>Independent Simulation and Analysis Utilities</h3>
    <p style="margin-top: 1rem; font-size: 1.1em; color: #666; font-style: italic;">
        Specialized tools for financial modeling and project analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Tools section
st.markdown("## Available Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tool-card">
        <h3>💰 Cash Flow Simulator</h3>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Monte Carlo cash flow simulations</li>
            <li>Risk analysis and forecasting</li>
            <li>Sensitivity analysis</li>
            <li>Interactive visualizations</li>
            <li>Export simulation results</li>
        </ul>
        <p><strong>Use Case:</strong> Financial planning, investment analysis, and cash flow forecasting with uncertainty modeling.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Launch Cash Flow Simulator", key="cashflow_btn", use_container_width=True):
        st.switch_page("pages/5_Cash_Flow_Simulator.py")

with col2:
    st.markdown("""
    <div class="tool-card">
        <h3>📊 EVM Simulator</h3>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Earned Value Management simulations</li>
            <li>Project performance modeling</li>
            <li>Schedule and cost variance analysis</li>
            <li>Forecasting and trend analysis</li>
            <li>Performance indicator calculations</li>
        </ul>
        <p><strong>Use Case:</strong> Project management analysis, performance forecasting, and EVM scenario modeling.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Launch EVM Simulator", key="evm_btn", use_container_width=True):
        st.switch_page("pages/6_EVM_Simulator.py")

# Information section
st.markdown("---")
st.markdown("## 📋 Tool Information")

with st.expander("ℹ️ How to Use These Tools"):
    st.markdown("""
    **Getting Started:**
    1. **Click the launch button** for the tool you want to use
    2. **The tool will open in the same app** (use sidebar to navigate back)
    3. **Each tool runs independently** - no data sharing with main application
    4. **Use the sidebar** to navigate between tools and main app

    **Important Notes:**
    - These tools are **integrated pages** in the same app
    - They do **not share data** with the main Portfolio Management Suite
    - Use the **sidebar navigation** to switch between tools
    - You can run **multiple tools simultaneously**
    - Tools will run on **different ports** automatically

    **System Requirements:**
    - Python with Streamlit installed
    - Required dependencies for each simulation tool
    - Modern web browser for visualization
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Cash Flow Simulator:**
    - Monte Carlo simulation engine
    - Statistical analysis capabilities
    - Interactive plotting with Plotly
    - Excel/CSV export functionality

    **EVM Simulator:**
    - Earned Value Management calculations
    - Performance index computations
    - Forecasting algorithms
    - Variance analysis tools

    **Architecture:**
    - Independent Streamlit applications
    - Modular design for easy maintenance
    - No dependencies on main application
    """)

# Navigation back to Main
st.markdown("---")
if st.button("🏠 Back to Main Menu", use_container_width=True):
    st.switch_page("Main.py")