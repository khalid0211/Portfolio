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
            manual_pv = st.number_input("Manual PV (Optional)", min_value=0.0, value=float(project_data.get("Manual_PV", 0.0)), step=50.0, help="Leave at 0 for automatic calculation")
        with col8:
            manual_ev = st.number_input("Manual EV (Optional)", min_value=0.0, value=float(project_data.get("Manual_EV", 0.0)), step=50.0, help="Leave at 0 for automatic calculation")

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
            # Create a display version with formatted numbers
            display_df = current_df.copy()
            display_df["BAC"] = display_df["BAC"].apply(format_currency)
            display_df["AC"] = display_df["AC"].apply(format_currency)
            if "Manual_PV" in display_df.columns:
                display_df["Manual_PV"] = display_df["Manual_PV"].apply(format_currency)
            if "Manual_EV" in display_df.columns:
                display_df["Manual_EV"] = display_df["Manual_EV"].apply(format_currency)

            # Display table
            event = st.dataframe(
                display_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            # Handle row selection
            if event.selection.rows:
                selected_idx = event.selection.rows[0]
                st.session_state.selected_row_index = selected_idx

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Selected Project", type="primary"):
                        st.session_state.edit_mode = True
                        st.rerun()

                with col2:
                    if st.button("🗑️ Delete Selected Project", type="secondary"):
                        if delete_project(selected_idx):
                            st.session_state.selected_row_index = None
                            st.rerun()

                # Show edit form if in edit mode
                if st.session_state.edit_mode and st.session_state.selected_row_index is not None:
                    st.divider()
                    st.subheader("Edit Project")

                    current_data = current_df.iloc[selected_idx].to_dict()
                    updated_data = render_project_form(current_data, "edit_form")

                    if updated_data:
                        if update_project(selected_idx, updated_data):
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