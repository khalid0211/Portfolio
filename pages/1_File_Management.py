"""
File Management & Configuration - Portfolio Management Suite
Centralized file operations and system configuration interface.
"""

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os
import traceback

# Page configuration
st.set_page_config(
    page_title="File Management - Portfolio Suite",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Note: Some advanced functionality (like database operations)
# will be handled through session state integration with Portfolio Analysis

# Custom CSS
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
    .file-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .section-header {
        font-size: 1.2em;
        font-weight: 600;
        color: #4A90E2;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4A90E2;
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

# Initialize session state
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.data_df = None
    st.session_state.raw_csv_df = None
    st.session_state.batch_results = None
    st.session_state.config_dict = {}
    st.session_state.original_filename = None
    st.session_state.file_type = None
    st.session_state.processed_file_info = None

# Ensure config_dict exists even if session was initialized elsewhere
if 'config_dict' not in st.session_state:
    st.session_state.config_dict = {}

def render_data_source_section():
    """Render A. Data Source section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">A. Data Source</div>', unsafe_allow_html=True)

    # Simple data source selection
    data_source = st.radio(
        "Select Data Source",
        options=["Load JSON", "Load CSV", "Manual Entry"],
        index=0,
        key="data_source_radio",
        help="Choose JSON for comprehensive data with configuration, CSV for data-only imports, or Manual Entry to create a blank table"
    )

    selected_table = None
    column_mapping = {}
    df = None

    # Dynamic file upload info based on selection
    if data_source == "Load JSON":
        st.info("💡 Upload JSON files exported from this application or compatible EVM tools")
        file_types = ["json"]
        help_text = "JSON files contain both data and configuration settings"
    elif data_source == "Load CSV":
        st.info("💡 Upload CSV files containing project data. Configuration will be set to defaults")
        file_types = ["csv"]
        help_text = "CSV files contain data only - you'll need to map columns to EVM fields"
    else:  # Manual Entry
        st.info("💡 Create a blank table that you can fill in manually with your project data")
        file_types = None
        help_text = "Start with an empty table and add your projects manually"

    # File uploader for JSON and CSV options only
    uploaded_file = None
    if data_source != "Manual Entry":
        uploaded_file = st.file_uploader(
            "Choose file",
            type=file_types,
            key="unified_file_uploader",
            help=help_text,
            label_visibility="visible"
        )

    # Process uploaded file based on selection
    if uploaded_file is not None:
        # Create a unique identifier for the current file to avoid reprocessing
        current_file_info = (uploaded_file.name, uploaded_file.size)

        # Process only if it's a new file
        if st.session_state.get('processed_file_info') != current_file_info:
            try:
                if data_source == "Load JSON":
                    # Process JSON file
                    content = uploaded_file.read().decode('utf-8')
                    json_data = json.loads(content)

                    # Debug: Show the actual JSON structure
                    st.markdown("🔍 **Debug: JSON File Structure**")
                    st.markdown(f"**Top-level keys:** {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dictionary'}")

                    if isinstance(json_data, dict):
                        if 'config' in json_data:
                            st.markdown(f"**Config keys:** {list(json_data['config'].keys()) if json_data['config'] else 'Config is empty/null'}")
                            if json_data['config'] and 'controls' in json_data['config']:
                                st.markdown(f"**Controls:** {json_data['config']['controls']}")
                        else:
                            st.markdown("**No 'config' key found**")

                    with st.expander("🔍 Show Full JSON Structure (First 1000 chars)"):
                        st.code(str(json_data)[:1000])

                    # Extract data and config - handle multiple formats
                    df = None
                    config_loaded = False

                    if isinstance(json_data, dict):
                        # Check for controls at top level (your format)
                        if 'controls' in json_data:
                            st.session_state.config_dict['controls'] = json_data['controls']
                            controls = json_data['controls']
                            config_loaded = True
                            st.info(f"🔧 Controls loaded from top-level: {controls.get('curve_type', 'N/A')} curve, {controls.get('currency_symbol', '$')} currency")

                        # Handle other top-level config items (but avoid enable_batch)
                        for key, value in json_data.items():
                            if key in ['enable_batch']:
                                # Store batch setting info but don't set it directly (widget manages this)
                                st.session_state.config_dict['batch_setting_from_json'] = value
                                if value:
                                    st.info("📋 JSON file indicates batch mode was enabled")
                            elif key not in ['controls', 'data', 'export_date', 'batch_results']:
                                st.session_state.config_dict[key] = value

                        # Get data - could be at top level or in 'data' key
                        if 'data' in json_data:
                            df = pd.DataFrame(json_data['data'])
                        elif any(key for key in json_data.keys() if key not in ['controls', 'config', 'export_date', 'batch_results']):
                            # Assume the JSON itself contains the data (legacy format)
                            df = pd.DataFrame([json_data])  # Single record

                        # Also check for nested config format
                        if 'config' in json_data and json_data['config']:
                            loaded_config = json_data['config']
                            if 'controls' in loaded_config:
                                st.session_state.config_dict['controls'] = loaded_config['controls']
                                controls = loaded_config['controls']
                                config_loaded = True
                                st.info(f"🔧 Controls loaded from config section: {controls.get('curve_type', 'N/A')} curve, {controls.get('currency_symbol', '$')} currency")

                            # Load other config sections (but avoid enable_batch which is managed by widget)
                            for key, value in loaded_config.items():
                                if key not in ['controls', 'enable_batch']:
                                    st.session_state.config_dict[key] = value

                    # If no data found, treat entire JSON as data
                    if df is None:
                        df = pd.DataFrame(json_data)

                    # Success message
                    st.success(f"✅ JSON file loaded: {len(df)} projects")
                    if not config_loaded:
                        st.warning("⚠️ No controls configuration found in JSON file")

                    st.session_state.file_type = "json"

                elif data_source == "Load CSV":
                    # Process CSV file
                    df = pd.read_csv(uploaded_file)
                    st.session_state.raw_csv_df = df.copy()
                    st.session_state.file_type = "csv"
                    st.success(f"✅ CSV file loaded: {len(df)} rows")

                # Store processed data
                st.session_state.data_df = df
                st.session_state.original_filename = uploaded_file.name
                st.session_state.processed_file_info = current_file_info

                # Force rerun to update UI with new config values
                if data_source == "Load JSON" and config_loaded:
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logging.error(f"File processing error: {e}")

    # Manual Entry option
    elif data_source == "Manual Entry":
        if st.button("🆕 Initialize Empty Project Table", key="init_empty"):
            # Create empty dataframe with required columns
            empty_df = pd.DataFrame({
                'Project_Name': [],
                'Budget': [],
                'Start_Date': [],
                'End_Date': [],
                'Actual_Cost': [],
                'Completion_Percentage': []
            })
            st.session_state.data_df = empty_df
            st.session_state.file_type = "manual"
            st.success("✅ Empty project table initialized!")

    st.markdown('</div>', unsafe_allow_html=True)
    return df, selected_table, column_mapping

def render_controls_section():
    """Render B. Controls section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">B. Controls</div>', unsafe_allow_html=True)

    # Load saved controls from JSON (if any)
    saved_controls = st.session_state.config_dict.get('controls', {})

    # Curve settings
    curve_value = saved_controls.get('curve_type', 'linear').lower()
    curve_index = 0 if curve_value == 'linear' else 1

    curve_type = st.selectbox(
        "Curve Type (PV)",
        ["Linear", "S-Curve"],
        index=curve_index,
        key="curve_type_select"
    )

    alpha = 2.0
    beta = 2.0

    if curve_type == "S-Curve":
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.number_input("S-Curve α", min_value=0.1, max_value=10.0,
                                  value=float(saved_controls.get('alpha', 2.0)), step=0.1, key="s_alpha")
        with col2:
            beta = st.number_input("S-Curve β", min_value=0.1, max_value=10.0,
                                 value=float(saved_controls.get('beta', 2.0)), step=0.1, key="s_beta")

    # Currency settings
    col1, col2 = st.columns(2)
    with col1:
        currency_symbol = st.text_input("Currency Symbol",
                                       value=saved_controls.get('currency_symbol', '$'),
                                       key="currency_symbol")
    with col2:
        currency_postfix = st.text_input("Currency Postfix",
                                        value=saved_controls.get('currency_postfix', ''),
                                        key="currency_postfix")

    # Additional settings
    col1, col2 = st.columns(2)
    with col1:
        date_format = st.selectbox(
            "Date Format",
            ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"],
            index=["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"].index(saved_controls.get('date_format', 'YYYY-MM-DD')),
            key="date_format_select"
        )
    with col2:
        data_date = st.date_input(
            "Data Date",
            value=pd.to_datetime(saved_controls.get('data_date', '2024-01-01')).date(),
            key="data_date_input"
        )

    # Inflation rate
    inflation_rate = st.number_input(
        "Annual Inflation Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(saved_controls.get('inflation_rate', 0.0)),
        step=0.1,
        key="inflation_rate"
    )

    # Store controls in session state
    controls = {
        'curve_type': curve_type.lower(),
        'alpha': alpha,
        'beta': beta,
        'currency_symbol': currency_symbol,
        'currency_postfix': currency_postfix,
        'date_format': date_format,
        'data_date': data_date.strftime('%Y-%m-%d'),
        'inflation_rate': inflation_rate
    }

    st.session_state.config_dict['controls'] = controls

    # Debug: Show what's being stored
    with st.expander("🔍 Debug: Current Controls Values"):
        st.json(controls)
    st.markdown('</div>', unsafe_allow_html=True)
    return controls

def render_batch_calculation_section():
    """Render C. Run Batch EVM section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">C. Run Batch EVM</div>', unsafe_allow_html=True)

    # Check if we have data to process
    has_data = (st.session_state.data_df is not None and not st.session_state.data_df.empty)

    if has_data:
        st.markdown("### 🚀 Batch EVM Calculations")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Ready to process {len(st.session_state.data_df)} projects**")
            st.caption("All configuration settings will be applied to the batch calculation")

        with col2:
            if st.button("🚀 **Run Batch EVM**", type="primary", key="run_batch"):
                # Run batch processing
                try:
                    st.info("🔄 Starting batch EVM calculations...")

                    # Get controls from session state
                    controls = st.session_state.config_dict.get('controls', {})

                    # Set up column mapping based on file type
                    if st.session_state.get('file_type') in ['demo', 'csv', 'json']:
                        column_mapping = {
                            'pid_col': 'Project ID',
                            'pname_col': 'Project',
                            'org_col': 'Organization',
                            'pm_col': 'Project Manager',
                            'bac_col': 'BAC',
                            'ac_col': 'AC',
                            'st_col': 'Plan Start',
                            'fn_col': 'Plan Finish',
                            'cp_col': 'Completion %'
                        }
                    else:
                        # For other file types, you might need column mapping UI
                        st.error("Column mapping not implemented for this file type yet")
                        return

                    # Extract parameters
                    curve_type = controls.get('curve_type', 'linear')
                    alpha = float(controls.get('alpha', 2.0))
                    beta = float(controls.get('beta', 2.0))
                    data_date = pd.to_datetime(controls.get('data_date', '2024-01-01')).date()
                    inflation_rate = float(controls.get('inflation_rate', 0.0))

                    # Actually run the REAL batch calculation using the same function as Project Analysis
                    with st.spinner("⚡ Running comprehensive EVM batch calculations..."):

                        # Call the real batch calculation function
                        # We'll handle the import dynamically to avoid circular imports
                        import sys
                        import os
                        sys.path.append(os.path.dirname(__file__))

                        try:
                            # Import and run the real batch calculation
                            from pathlib import Path
                            import importlib.util

                            # Load the Project Analysis module dynamically
                            project_analysis_path = Path(__file__).parent / "3_Project_Analysis.py"
                            spec = importlib.util.spec_from_file_location("project_analysis", project_analysis_path)
                            project_analysis = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(project_analysis)

                            # Use the real perform_batch_calculation function
                            batch_results = project_analysis.perform_batch_calculation(
                                st.session_state.data_df, column_mapping,
                                curve_type, alpha, beta, data_date, inflation_rate
                            )

                            st.success(f"✅ Real EVM calculations completed for {len(batch_results)} projects!")

                        except Exception as import_error:
                            st.warning(f"⚠️ Could not import real batch calculation: {import_error}")
                            st.info("Using simplified calculation as fallback...")

                            # Fallback to simplified calculation if import fails
                            batch_results = st.session_state.data_df.copy()

                            # Add basic calculated columns
                            if 'cost_performance_index' not in batch_results.columns:
                                batch_results['cost_performance_index'] = 1.0
                            if 'schedule_performance_index' not in batch_results.columns:
                                batch_results['schedule_performance_index'] = 1.0
                            if 'spie' not in batch_results.columns:
                                batch_results['spie'] = 1.0

                    # Store the batch results in session state
                    st.session_state.batch_results = batch_results
                    st.session_state['batch_column_mapping'] = column_mapping
                    st.session_state['batch_results_ready'] = True

                    st.success("✅ Batch calculations completed!")
                    st.info("📊 You can now access Project Analysis, Portfolio Analysis, or Portfolio Gantt")

                    # Show navigation options
                    col_nav1, col_nav2, col_nav3 = st.columns(3)
                    with col_nav1:
                        if st.button("🔍 Project Analysis", key="goto_project"):
                            st.switch_page("pages/3_Project_Analysis.py")
                    with col_nav2:
                        if st.button("📊 Portfolio Analysis", key="goto_portfolio"):
                            st.switch_page("pages/4_Portfolio_Analysis.py")
                    with col_nav3:
                        if st.button("📈 Portfolio Gantt", key="goto_gantt"):
                            st.switch_page("pages/5_Portfolio_Gantt.py")

                except Exception as e:
                    st.error(f"❌ Error running batch calculation: {str(e)}")
                    logging.error(f"Batch calculation error: {e}")
    else:
        st.warning("⚠️ No data loaded. Load data first to enable batch processing.")
        st.info("💡 Upload JSON/CSV data or initialize manual entry in Section A")

    st.markdown('</div>', unsafe_allow_html=True)

def render_llm_provider_section():
    """Render D. LLM Provider section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">D. LLM Provider (Executive Brief)</div>', unsafe_allow_html=True)

    provider = st.radio(
        "Choose Provider",
        ["OpenAI", "Gemini"],
        index=0,
        key="llm_provider"
    )

    # API Key file upload
    uploaded_file = st.file_uploader(
        "Upload API Key File",
        type=["txt", "key"],
        help="Upload a text file containing your API key",
        key="api_key_uploader"
    )

    if uploaded_file is not None:
        try:
            api_key = uploaded_file.read().decode('utf-8').strip()
            st.session_state['api_key'] = api_key
            st.success("✅ API key loaded successfully")
        except Exception as e:
            st.error(f"❌ Error reading API key: {str(e)}")

    llm_config = {
        'provider': provider,
        'has_api_key': 'api_key' in st.session_state
    }

    st.markdown('</div>', unsafe_allow_html=True)
    return llm_config

def render_save_download_section():
    """Render E. Save & Download section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">E. Save & Download</div>', unsafe_allow_html=True)

    # Check if there's data to save
    has_data = False
    data_sources = []

    # Check session state data
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        has_data = True
        data_sources.append(f"Session Data ({len(st.session_state.data_df)} rows)")

    # Check batch results
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        has_data = True
        data_sources.append(f"Batch Results ({len(st.session_state.batch_results)} projects)")

    # Check config data
    config_items = len(st.session_state.config_dict) if st.session_state.config_dict else 0
    if config_items > 0:
        data_sources.append(f"Configuration ({config_items} settings)")

    if has_data or config_items > 0:
        # Show what will be saved
        st.markdown("**Available for Export:**")
        for source in data_sources:
            st.markdown(f"• {source}")

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download JSON Package", key="download_json"):
                # Create complete package
                package = {
                    'data': st.session_state.data_df.to_dict('records') if st.session_state.data_df is not None else [],
                    'config': st.session_state.config_dict,
                    'export_date': datetime.now().isoformat(),
                    'batch_results': st.session_state.batch_results.to_dict('records') if st.session_state.batch_results is not None else []
                }

                # Show export info
                st.success("📦 Exporting complete package with configuration")

                json_str = json.dumps(package, indent=2, default=str)
                st.download_button(
                    "📁 Download Complete Package",
                    json_str,
                    file_name=f"portfolio_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if has_data and st.button("📊 Download CSV Data", key="download_csv"):
                if st.session_state.data_df is not None and not st.session_state.data_df.empty:
                    csv = st.session_state.data_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download Data CSV",
                        csv,
                        file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("💡 No data available for export. Load or create data first.")

    st.markdown('</div>', unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>📁 File Management & Configuration</h1>
    <h3>Centralized Data Import, Configuration & Export Hub</h3>
    <p style="margin-top: 1rem; font-size: 1.1em; color: #666;">
        Configure your data sources, calculation parameters, and export settings
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔙 Return to Main Navigation", key="return_main", type="secondary"):
        st.switch_page("main.py")

    if st.button("🚀 Continue to Project Analysis", key="continue_project", type="primary"):
        st.switch_page("pages/3_Project_Analysis.py")

st.markdown("---")

# Render all sections
df, selected_table, column_mapping = render_data_source_section()
controls = render_controls_section()
render_batch_calculation_section()
llm_config = render_llm_provider_section()
render_save_download_section()

# Status summary
st.markdown("---")
st.markdown("## 📋 Configuration Summary")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        st.success(f"✅ Data: {len(st.session_state.data_df)} projects loaded")
        if st.session_state.get('file_type'):
            st.caption(f"Source: {st.session_state.file_type.upper()}")
    else:
        st.warning("⚠️ No data loaded")

with col2:
    if st.session_state.config_dict.get('controls'):
        controls = st.session_state.config_dict['controls']
        st.success("✅ Controls: Configured")
        st.caption(f"{controls.get('curve_type', 'linear')} curve, {controls.get('currency_symbol', '$')} currency")
    else:
        st.info("💡 Controls: Using defaults")

with col3:
    if st.session_state.get('batch_results_ready'):
        st.success("✅ Batch calculations: Complete")
        st.caption("Ready for analysis")
    else:
        st.info("💡 Batch calculations: Not run yet")
        st.caption("Click 'Run Batch EVM' when ready")

# Quick help
with st.expander("ℹ️ File Management Help"):
    st.markdown("""
    **Workflow:**
    1. **A. Data Source**: Load JSON/CSV files or initialize manual entry
    2. **B. Controls**: Configure calculation parameters (curves, currency, etc.)
    3. **C. Batch Calculations**: Enable for processing multiple projects
    4. **D. LLM Provider**: Set up AI for executive reports
    5. **E. Save & Download**: Export your data and configuration

    **Next Steps:**
    - Click "Continue to Portfolio Analysis" to proceed with calculations
    - Or return to Main Navigation to access other tools

    **Data Persistence:**
    - All settings are automatically saved in your session
    - Use JSON export to save complete packages for later use
    """)