# Manual Data Entry — Enhanced Project Data Management Interface
# Professional interface for adding, editing, and managing project data

from __future__ import annotations
import io
import json
import logging
import math
import os
import re
import textwrap
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd
import numpy as np
import streamlit as st
from dateutil import parser as date_parser

# =============================================================================
# CONSTANTS
# =============================================================================

# Application constants
APP_TITLE = "Manual Data Entry 📝"
DEFAULT_DATASET_TABLE = "dataset"

# =============================================================================
# DATA PERSISTENCE FUNCTIONS (copied from main app)
# =============================================================================

def load_table(table_name: str) -> pd.DataFrame:
    """Load table from session state."""
    try:
        # Load main dataset table
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is not None:
                return st.session_state.data_df.copy()
            else:
                return pd.DataFrame()

        # Load from config tables
        if ("tables" in st.session_state.config_dict and
            table_name in st.session_state.config_dict["tables"]):
            table_records = st.session_state.config_dict["tables"][table_name]
            return pd.DataFrame(table_records)

        # Table not found
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Failed to load table {table_name}: {e}")
        return pd.DataFrame()

def save_table_replace(df: pd.DataFrame, table_name: str):
    """Save table to session state."""
    try:
        # Save to appropriate session location
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df.copy()
            st.session_state.data_loaded = True
        else:
            # Save to config tables
            if "tables" not in st.session_state.config_dict:
                st.session_state.config_dict["tables"] = {}

            st.session_state.config_dict["tables"][table_name] = df.to_dict('records')

    except Exception as e:
        st.error(f"Failed to save table {table_name}: {e}")
        raise

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "config_dict" not in st.session_state:
        st.session_state.config_dict = {}
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = None
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_project_data(data: Dict[str, Any]) -> List[str]:
    """Validate project data and return list of errors."""
    errors = []

    # Required fields
    if not data.get("Project ID", "").strip():
        errors.append("Project ID is required")
    if not data.get("Project", "").strip():
        errors.append("Project name is required")

    # Numeric validations
    try:
        bac = float(data.get("BAC", 0))
        if bac <= 0:
            errors.append("BAC must be greater than 0")
    except (ValueError, TypeError):
        errors.append("BAC must be a valid number")

    try:
        ac = float(data.get("AC", 0))
        if ac < 0:
            errors.append("AC cannot be negative")
    except (ValueError, TypeError):
        errors.append("AC must be a valid number")

    # Date validations
    try:
        start_date = date_parser.parse(data.get("Plan Start", ""))
        finish_date = date_parser.parse(data.get("Plan Finish", ""))
        if start_date >= finish_date:
            errors.append("Plan Start must be before Plan Finish")
    except:
        errors.append("Invalid date format. Use DD/MM/YYYY")

    return errors

def format_currency(value: float) -> str:
    """Format currency value with proper formatting."""
    if value is None:
        return ""
    return f"{value:,.2f}"

# =============================================================================
# DATA MANAGEMENT FUNCTIONS
# =============================================================================

def add_new_project(project_data: Dict[str, Any]) -> bool:
    """Add a new project to the dataframe."""
    try:
        # Load existing data
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty:
            # Create new dataframe with proper schema
            current_df = pd.DataFrame(columns=[
                "Project ID", "Project", "Organization", "Project Manager",
                "BAC", "AC", "Plan Start", "Plan Finish",
                "Use_Manual_PV", "Manual_PV", "Use_Manual_EV", "Manual_EV"
            ])

        # Check for duplicate Project ID
        if not current_df.empty and project_data["Project ID"] in current_df["Project ID"].values:
            st.error(f"Project ID '{project_data['Project ID']}' already exists!")
            return False

        # Add new row
        new_row = pd.DataFrame([project_data])
        updated_df = pd.concat([current_df, new_row], ignore_index=True)

        # Save using the proper persistence mechanism
        save_table_replace(updated_df, DEFAULT_DATASET_TABLE)

        # Set flag to switch to view tab
        st.session_state.active_tab = 0
        st.session_state.project_just_added = True

        st.success(f"✅ Project '{project_data['Project ID']}' added successfully! Switching to View & Edit Data tab...")
        return True

    except Exception as e:
        st.error(f"Error adding project: {e}")
        return False

def update_project(index: int, project_data: Dict[str, Any]) -> bool:
    """Update an existing project."""
    try:
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty or index >= len(current_df):
            st.error("Project not found!")
            return False

        # Check for duplicate Project ID (excluding current record)
        mask = current_df.index != index
        if project_data["Project ID"] in current_df.loc[mask, "Project ID"].values:
            st.error(f"Project ID '{project_data['Project ID']}' already exists!")
            return False

        # Update row
        for key, value in project_data.items():
            current_df.at[index, key] = value

        # Save back to persistence
        save_table_replace(current_df, DEFAULT_DATASET_TABLE)

        st.success(f"✅ Project '{project_data['Project ID']}' updated successfully!")
        return True

    except Exception as e:
        st.error(f"Error updating project: {e}")
        return False

def delete_project(index: int) -> bool:
    """Delete a project."""
    try:
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty or index >= len(current_df):
            st.error("Project not found!")
            return False

        project_id = current_df.at[index, "Project ID"]
        updated_df = current_df.drop(index).reset_index(drop=True)

        # Save back to persistence
        save_table_replace(updated_df, DEFAULT_DATASET_TABLE)

        st.success(f"✅ Project '{project_id}' deleted successfully!")
        return True

    except Exception as e:
        st.error(f"Error deleting project: {e}")
        return False

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_project_form(project_data: Dict[str, Any] = None, form_key: str = "project_form") -> Dict[str, Any]:
    """Render project input form."""

    # Default values
    if project_data is None:
        project_data = {
            "Project ID": "",
            "Project": "",
            "Organization": "",
            "Project Manager": "",
            "BAC": 1000.0,
            "AC": 0.0,
            "Plan Start": "01/01/2025",
            "Plan Finish": "31/12/2025",
            "Use_Manual_PV": False,
            "Manual_PV": 0.0,
            "Use_Manual_EV": False,
            "Manual_EV": 0.0
        }

    with st.form(form_key):
        # Row 1: Project ID and Project Name
        col1, col2 = st.columns(2)
        with col1:
            project_id = st.text_input("Project ID *", value=project_data.get("Project ID", ""))
        with col2:
            project_name = st.text_input("Project Name *", value=project_data.get("Project", ""))

        # Row 2: Organization and Project Manager
        col3, col4 = st.columns(2)
        with col3:
            organization = st.text_input("Organization", value=project_data.get("Organization", ""))
        with col4:
            project_manager = st.text_input("Project Manager", value=project_data.get("Project Manager", ""))

        # Row 3: Financial Data Header
        st.subheader("Financial Data")

        # Row 4: BAC and AC
        col5, col6 = st.columns(2)
        with col5:
            bac = st.number_input("BAC (Budget) *", min_value=0.0, value=float(project_data.get("BAC", 1000.0)), step=100.0)
        with col6:
            ac = st.number_input("AC (Actual Cost)", min_value=0.0, value=float(project_data.get("AC", 0.0)), step=50.0)

        # Row 5: Manual PV and Manual EV
        col7, col8 = st.columns(2)
        with col7:
            manual_pv = st.number_input("Manual PV (Optional)", min_value=0.0, value=float(project_data.get("Manual_PV") or 0.0), step=50.0, help="Leave at 0 for automatic calculation")
        with col8:
            manual_ev = st.number_input("Manual EV (Optional)", min_value=0.0, value=float(project_data.get("Manual_EV") or 0.0), step=50.0, help="Leave at 0 for automatic calculation")

        # Row 6: Schedule Header
        st.subheader("Schedule")

        # Row 7: Plan Start and Plan Finish
        col9, col10 = st.columns(2)
        with col9:
            plan_start = st.text_input("Plan Start (DD/MM/YYYY) *", value=project_data.get("Plan Start", "01/01/2025"))
        with col10:
            plan_finish = st.text_input("Plan Finish (DD/MM/YYYY) *", value=project_data.get("Plan Finish", "31/12/2025"))

        submitted = st.form_submit_button("💾 Save Project", type="primary")

        if submitted:
            # Auto-detect manual values based on non-zero input
            use_manual_pv = manual_pv > 0
            use_manual_ev = manual_ev > 0

            form_data = {
                "Project ID": project_id.strip(),
                "Project": project_name.strip(),
                "Organization": organization.strip(),
                "Project Manager": project_manager.strip(),
                "BAC": bac,
                "AC": ac,
                "Plan Start": plan_start.strip(),
                "Plan Finish": plan_finish.strip(),
                "Use_Manual_PV": use_manual_pv,
                "Manual_PV": manual_pv,
                "Use_Manual_EV": use_manual_ev,
                "Manual_EV": manual_ev
            }

            # Validate data
            errors = validate_project_data(form_data)
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                return None

            return form_data

    return None

def render_data_overview():
    """Render data overview metrics."""
    df = load_table(DEFAULT_DATASET_TABLE)

    if df.empty:
        st.info("📊 No project data available. Add your first project below!")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Projects", len(df))

    with col2:
        total_bac = df["BAC"].sum()
        st.metric("Total BAC", f"{format_currency(total_bac)}")

    with col3:
        total_ac = df["AC"].sum()
        st.metric("Total AC", f"{format_currency(total_ac)}")

    with col4:
        if total_bac > 0:
            progress = (total_ac / total_bac) * 100
            st.metric("Overall Progress", f"{progress:.1f}%")
        else:
            st.metric("Overall Progress", "0%")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.title("📝 Manual Data Entry")
    st.markdown("Professional interface for managing your project portfolio data")
    st.markdown("""
    **Smarter Projects and Portfolios with Earned Value Analysis and AI-Powered Executive Reporting**
    Developed by Dr. Khalid Ahmad Khan – [LinkedIn](https://www.linkedin.com/in/khalidahmadkhan/)
    """, unsafe_allow_html=True)

    # Check if data exists
    current_df = load_table(DEFAULT_DATASET_TABLE)
    has_data = (current_df is not None and not current_df.empty)

    if not has_data:
        st.warning("⚠️ No project data found. Please create a demo project or load data from the main page first.")

        if st.button("🚀 Create Demo Project", type="primary"):
            # Create demo project
            demo_data = {
                "Project ID": "Demo",
                "Project": "Demo Project",
                "Organization": "Demo Org",
                "Project Manager": "Demo Manager",
                "BAC": 1000.0,
                "AC": 500.0,
                "Plan Start": "01/01/2025",
                "Plan Finish": "31/10/2025",
                "Use_Manual_PV": False,
                "Manual_PV": 0.0,
                "Use_Manual_EV": True,
                "Manual_EV": 500.0
            }

            # Use proper persistence mechanism
            demo_df = pd.DataFrame([demo_data])
            save_table_replace(demo_df, DEFAULT_DATASET_TABLE)

            # Update config
            st.session_state.config_dict.update({
                "curve_type": "S-curve",
                "data_date": "30/05/2025",
                "currency_symbol": "PKR",
                "currency_postfix": "Million"
            })

            st.success("✅ Demo project created successfully!")
            st.rerun()

        return

    # Data overview
    render_data_overview()
    st.divider()

    # Main interface tabs with session state control
    tab_options = ["📋 View & Edit Data", "➕ Add New Project", "🔧 Bulk Operations"]

    # Use radio buttons for better control over active tab
    selected_tab = st.radio(
        "Choose Action:",
        options=tab_options,
        index=st.session_state.get("active_tab", 0),
        horizontal=True,
        key="tab_selector"
    )

    # Update active tab index
    st.session_state.active_tab = tab_options.index(selected_tab)

    st.divider()

    if selected_tab == "📋 View & Edit Data":
        st.subheader("Project Data")

        # Show success message if project was just added
        if st.session_state.get("project_just_added", False):
            st.success("🎉 Project added successfully! Here's your updated data:")
            # Reset the flag
            st.session_state.project_just_added = False

        # Load and display dataframe with selection
        current_df = load_table(DEFAULT_DATASET_TABLE)
        if not current_df.empty:
            # Try to get SPI and CPI from batch results first
            current_df = current_df.copy()

            # Check if batch results are available with EVM calculations
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                batch_df = st.session_state.batch_results.copy()

                # Merge EVM metrics from batch results
                # The batch results use 'project_id' (lowercase) as the column name
                project_id_col = None
                if 'project_id' in batch_df.columns:
                    project_id_col = 'project_id'
                elif 'Project ID' in batch_df.columns:
                    project_id_col = 'Project ID'

                if project_id_col:
                    # Select relevant EVM columns from batch results
                    evm_columns = [project_id_col]

                    # Rename batch results columns to match our display names
                    column_renames = {}
                    if 'schedule_performance_index' in batch_df.columns:
                        evm_columns.append('schedule_performance_index')
                        column_renames['schedule_performance_index'] = 'SPI'
                    if 'cost_performance_index' in batch_df.columns:
                        evm_columns.append('cost_performance_index')
                        column_renames['cost_performance_index'] = 'CPI'
                    if 'earned_value' in batch_df.columns:
                        evm_columns.append('earned_value')
                        column_renames['earned_value'] = 'EV'
                    if 'planned_value' in batch_df.columns:
                        evm_columns.append('planned_value')
                        column_renames['planned_value'] = 'PV'

                    if len(evm_columns) > 1:  # More than just project ID
                        # Select only the columns we need
                        batch_evm_df = batch_df[evm_columns].copy()

                        # Rename columns
                        if column_renames:
                            batch_evm_df = batch_evm_df.rename(columns=column_renames)

                        # Rename project_id column to match current_df
                        if project_id_col == 'project_id':
                            batch_evm_df = batch_evm_df.rename(columns={'project_id': 'Project ID'})

                        # Ensure both Project ID columns are strings to avoid merge type errors
                        current_df['Project ID'] = current_df['Project ID'].astype(str)
                        batch_evm_df['Project ID'] = batch_evm_df['Project ID'].astype(str)

                        # Merge the EVM data
                        current_df = current_df.merge(
                            batch_evm_df,
                            on='Project ID',
                            how='left'
                        )
                        st.info("📊 Using EVM calculations from Portfolio Analysis batch results")
                    else:
                        st.warning("⚠️ No EVM metrics found in batch results. Please run 'Batch EVM Calculation' in Portfolio Analysis first.")
                        # Add empty columns for consistency
                        current_df['SPI'] = 0.0
                        current_df['CPI'] = 0.0
                else:
                    st.warning("⚠️ Batch results don't contain project identifier column. Please run 'Batch EVM Calculation' in Portfolio Analysis first.")
                    # Add empty columns for consistency
                    current_df['SPI'] = 0.0
                    current_df['CPI'] = 0.0
            else:
                st.warning("⚠️ No batch results found. Please run 'Batch EVM Calculation' in Portfolio Analysis to get SPI and CPI values.")
                # Add empty columns for consistency
                current_df['SPI'] = 0.0
                current_df['CPI'] = 0.0

            # Add comprehensive filters
            filter_header_col1, filter_header_col2 = st.columns([3, 1])
            with filter_header_col1:
                st.subheader("🔧 Filters")
            with filter_header_col2:
                if st.button("🔄 Clear All Filters", key="clear_filters"):
                    st.rerun()

            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                # Project ID search filter
                search_project_id = st.text_input("🔍 Project ID", placeholder="Search by Project ID...")

            with filter_col2:
                # Organization filter
                org_options = ["All"] + sorted(current_df["Organization"].dropna().unique().tolist())
                selected_org = st.selectbox("🏢 Organization", options=org_options)

            with filter_col3:
                # Plan Start date range filter
                if 'Plan Start' in current_df.columns:
                    try:
                        # Convert Plan Start to datetime if it's not already
                        if current_df['Plan Start'].dtype == 'object':
                            current_df['Plan Start'] = pd.to_datetime(current_df['Plan Start'], format='%d/%m/%Y', errors='coerce')

                        # Get min and max dates, ignoring NaT values
                        valid_dates = current_df['Plan Start'].dropna()
                        if len(valid_dates) > 0:
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()

                            # Ensure we have valid datetime objects
                            if pd.notna(min_date) and pd.notna(max_date) and hasattr(min_date, 'date'):
                                date_range = st.date_input(
                                    "📅 Plan Start Range",
                                    value=(min_date.date(), max_date.date()),
                                    min_value=min_date.date(),
                                    max_value=max_date.date()
                                )
                            else:
                                st.text("📅 Plan Start Range")
                                st.caption("No valid dates found")
                                date_range = None
                        else:
                            st.text("📅 Plan Start Range")
                            st.caption("No valid dates found")
                            date_range = None
                    except Exception as e:
                        st.text("📅 Plan Start Range")
                        st.caption(f"Date parsing error: {str(e)}")
                        date_range = None
                else:
                    date_range = None

            with filter_col4:
                # SPI/CPI range filters
                if 'SPI' in current_df.columns and 'CPI' in current_df.columns:
                    metric_filter = st.selectbox("📊 Performance Filter",
                        options=["All", "SPI < 1.0 (Behind Schedule)", "SPI >= 1.0 (On/Ahead Schedule)",
                                "CPI < 1.0 (Over Budget)", "CPI >= 1.0 (On/Under Budget)",
                                "SPI < 1.0 AND CPI < 1.0 (Critical)", "SPI >= 1.0 AND CPI >= 1.0 (Healthy)"])
                else:
                    metric_filter = "All"

            # Store original dataframe and indices for mapping
            original_df = current_df.copy()

            # Apply filters
            filtered_df = current_df.copy()

            # Project ID filter
            if search_project_id:
                filtered_df = filtered_df[filtered_df["Project ID"].astype(str).str.contains(search_project_id, case=False, na=False)]

            # Organization filter
            if selected_org != "All":
                filtered_df = filtered_df[filtered_df["Organization"] == selected_org]

            # Date range filter
            if date_range and len(date_range) == 2:
                try:
                    start_date, end_date = date_range
                    # Ensure Plan Start is datetime before filtering
                    if 'Plan Start' in filtered_df.columns:
                        if filtered_df['Plan Start'].dtype == 'object':
                            filtered_df['Plan Start'] = pd.to_datetime(filtered_df['Plan Start'], format='%d/%m/%Y', errors='coerce')

                        # Filter only rows with valid dates
                        valid_date_mask = filtered_df['Plan Start'].notna()
                        if valid_date_mask.any():
                            filtered_df = filtered_df[
                                valid_date_mask &
                                (filtered_df['Plan Start'].dt.date >= start_date) &
                                (filtered_df['Plan Start'].dt.date <= end_date)
                            ]
                except Exception as e:
                    st.error(f"Error filtering by date range: {str(e)}")

            # Performance metric filter
            if metric_filter != "All" and 'SPI' in filtered_df.columns and 'CPI' in filtered_df.columns:
                if metric_filter == "SPI < 1.0 (Behind Schedule)":
                    filtered_df = filtered_df[filtered_df['SPI'] < 1.0]
                elif metric_filter == "SPI >= 1.0 (On/Ahead Schedule)":
                    filtered_df = filtered_df[filtered_df['SPI'] >= 1.0]
                elif metric_filter == "CPI < 1.0 (Over Budget)":
                    filtered_df = filtered_df[filtered_df['CPI'] < 1.0]
                elif metric_filter == "CPI >= 1.0 (On/Under Budget)":
                    filtered_df = filtered_df[filtered_df['CPI'] >= 1.0]
                elif metric_filter == "SPI < 1.0 AND CPI < 1.0 (Critical)":
                    filtered_df = filtered_df[(filtered_df['SPI'] < 1.0) & (filtered_df['CPI'] < 1.0)]
                elif metric_filter == "SPI >= 1.0 AND CPI >= 1.0 (Healthy)":
                    filtered_df = filtered_df[(filtered_df['SPI'] >= 1.0) & (filtered_df['CPI'] >= 1.0)]

            # Update current_df to filtered results
            current_df = filtered_df

            # Show filter results
            if len(current_df) != len(original_df):
                st.info(f"📊 Showing {len(current_df)} of {len(original_df)} projects after filtering")

            if current_df.empty:
                st.warning("No projects match the selected filters. Please adjust your filter criteria.")
                return

            # Create a display version with formatted numbers
            display_df = current_df.copy()

            # Format currency columns
            display_df["BAC"] = display_df["BAC"].apply(format_currency)
            display_df["AC"] = display_df["AC"].apply(format_currency)
            if "Manual_PV" in display_df.columns:
                display_df["Manual_PV"] = display_df["Manual_PV"].apply(format_currency)
            if "Manual_EV" in display_df.columns:
                display_df["Manual_EV"] = display_df["Manual_EV"].apply(format_currency)
            if "PV" in display_df.columns:
                display_df["PV"] = display_df["PV"].apply(format_currency)
            if "EV" in display_df.columns:
                display_df["EV"] = display_df["EV"].apply(format_currency)

            # Format SPI and CPI as ratios with 3 decimal places
            if "SPI" in display_df.columns:
                display_df["SPI"] = display_df["SPI"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            if "CPI" in display_df.columns:
                display_df["CPI"] = display_df["CPI"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

            # Format dates back to string for display
            if 'Plan Start' in display_df.columns:
                try:
                    # Only use dt accessor if column is actually datetime
                    if pd.api.types.is_datetime64_any_dtype(display_df['Plan Start']):
                        display_df['Plan Start'] = display_df['Plan Start'].dt.strftime('%d/%m/%Y')
                    else:
                        # If it's already string or other type, keep as is
                        display_df['Plan Start'] = display_df['Plan Start'].astype(str)
                except:
                    # Fallback to string conversion
                    display_df['Plan Start'] = display_df['Plan Start'].astype(str)

            if 'Plan Finish' in display_df.columns:
                try:
                    # Only use dt accessor if column is actually datetime
                    if pd.api.types.is_datetime64_any_dtype(display_df['Plan Finish']):
                        display_df['Plan Finish'] = display_df['Plan Finish'].dt.strftime('%d/%m/%Y')
                    else:
                        # If it's already string or other type, keep as is
                        display_df['Plan Finish'] = display_df['Plan Finish'].astype(str)
                except:
                    # Fallback to string conversion
                    display_df['Plan Finish'] = display_df['Plan Finish'].astype(str)

            # Select key columns for display (avoid showing too many columns)
            display_columns = ['Project ID', 'Project', 'Organization', 'BAC', 'AC', 'Plan Start', 'Plan Finish']
            if 'SPI' in display_df.columns:
                display_columns.append('SPI')
            if 'CPI' in display_df.columns:
                display_columns.append('CPI')

            # Only show columns that exist in the dataframe
            display_columns = [col for col in display_columns if col in display_df.columns]
            display_df = display_df[display_columns]

            # Display table
            event = st.dataframe(
                display_df,
                width='stretch',
                on_select="rerun",
                selection_mode="single-row"
            )

            # Handle row selection
            if event.selection.rows:
                selected_display_idx = event.selection.rows[0]
                # Map the display index back to the original dataframe index
                actual_idx = current_df.index[selected_display_idx]
                st.session_state.selected_row_index = actual_idx

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Selected Project", type="primary"):
                        st.session_state.edit_mode = True
                        st.rerun()

                with col2:
                    if st.button("🗑️ Delete Selected Project", type="secondary"):
                        if delete_project(actual_idx):
                            st.session_state.selected_row_index = None
                            st.rerun()

                # Show edit form if in edit mode
                if st.session_state.edit_mode and st.session_state.selected_row_index is not None:
                    st.divider()
                    st.subheader("Edit Project")

                    current_data = original_df.iloc[st.session_state.selected_row_index].to_dict()
                    updated_data = render_project_form(current_data, "edit_form")

                    if updated_data:
                        if update_project(st.session_state.selected_row_index, updated_data):
                            st.session_state.edit_mode = False
                            st.session_state.selected_row_index = None
                            st.rerun()

                    if st.button("❌ Cancel Edit"):
                        st.session_state.edit_mode = False
                        st.rerun()

    elif selected_tab == "➕ Add New Project":
        st.subheader("Add New Project")
        new_project_data = render_project_form(form_key="add_form")

        if new_project_data:
            if add_new_project(new_project_data):
                st.rerun()

    elif selected_tab == "🔧 Bulk Operations":
        st.subheader("Bulk Operations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export to CSV"):
                export_df = load_table(DEFAULT_DATASET_TABLE)
                if not export_df.empty:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"project_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        with col2:
            if st.button("🗑️ Clear All Data"):
                current_df = load_table(DEFAULT_DATASET_TABLE)
                if not current_df.empty:
                    if st.checkbox("⚠️ I understand this will delete all project data"):
                        # Save empty dataframe to clear data
                        empty_df = pd.DataFrame()
                        save_table_replace(empty_df, DEFAULT_DATASET_TABLE)
                        st.success("All data cleared!")
                        st.rerun()

if __name__ == "__main__":
    main()