"""
User Management - Portfolio Management Suite
Admin interface for managing user access permissions
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.auth import check_authentication, require_page_access
from utils.auth_utils import get_all_users, update_user_permissions, is_admin
from utils.firebase_db import db, is_db_configured

# Page configuration
st.set_page_config(
    page_title="User Management - Portfolio Suite",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('user_management', 'User Management')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .user-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .permission-toggle {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">👥 User Management</h1>', unsafe_allow_html=True)
st.markdown("Manage user access permissions for the Portfolio Management Suite")

# Check if Firebase is configured
if not is_db_configured():
    st.error("❌ Firebase database is not configured")
    st.info("""
    **Firebase Configuration Required:**
    User management requires Firebase Firestore to be configured.
    Please add Firebase credentials to `.streamlit/secrets.toml`
    """)
    st.stop()

# Get current user
current_user_email = st.session_state.get('user_email')

if not is_admin(current_user_email):
    st.error("❌ Access Denied")
    st.info("Only administrators can access the User Management page.")
    st.stop()

# Page permissions configuration
PAGE_PERMISSIONS = {
    'file_management': '📁 File Management',
    'manual_data_entry': '📝 Manual Data Entry',
    'project_analysis': '🔍 Project Analysis',
    'portfolio_analysis': '📈 Portfolio Analysis',
    'portfolio_charts': '📊 Portfolio Charts',
    'cash_flow_simulator': '💸 Cash Flow Simulator',
    'evm_simulator': '🎯 EVM Simulator',
    'user_management': '👥 User Management'
}

# Fetch all users
users = get_all_users(db)

if not users:
    st.warning("⚠️ No users found in the database")
    st.info("Users will be automatically added when they sign in with Google OAuth")
else:
    st.success(f"✅ Managing {len(users)} user(s)")

    # Create summary table
    st.markdown("## 📋 Users Overview")

    summary_data = []
    for user in users:
        def format_date_short(date_obj):
            if date_obj:
                try:
                    if hasattr(date_obj, 'strftime'):
                        return date_obj.strftime('%Y-%m-%d %H:%M')
                    else:
                        return str(date_obj)[:16]
                except:
                    return "Unknown"
            return "Never"

        summary_data.append({
            'Name': user.get('name', 'User'),
            'Email': user.get('email', 'Unknown'),
            'Created': format_date_short(user.get('created_at')),
            'Last Access': format_date_short(user.get('last_access_date')),
            'Access Count': user.get('access_count', 0),
            'Role': '🔑 Admin' if is_admin(user.get('email')) else '👤 User'
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("## 👥 Detailed User Management")

    # Display users in expandable cards
    for user_idx, user in enumerate(users):
        user_email = user.get('email', 'Unknown')
        user_name = user.get('name', 'User')
        permissions = user.get('permissions', {})
        created_at = user.get('created_at')
        last_access_date = user.get('last_access_date')
        access_count = user.get('access_count', 0)

        # Format dates
        def format_date(date_obj):
            if date_obj:
                try:
                    if hasattr(date_obj, 'strftime'):
                        return date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        return str(date_obj)
                except:
                    return "Unknown"
            return "Never"

        created_at_str = format_date(created_at)
        last_access_str = format_date(last_access_date)

        # User card
        with st.expander(f"👤 {user_name} ({user_email})", expanded=(user_email == current_user_email)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Email:** {user_email}")
                st.markdown(f"**Name:** {user_name}")
                st.markdown(f"**Created Date:** {created_at_str}")
                st.markdown(f"**Last Access Date:** {last_access_str}")
                st.markdown(f"**Access Count:** {access_count} time(s)")

                # Show admin badge
                if is_admin(user_email):
                    st.markdown("🔑 **Administrator** (Full Access)")

            with col2:
                if user_email == current_user_email:
                    st.info("📌 This is you")

                # Show activity metrics
                st.metric("Total Accesses", access_count)

            st.markdown("---")
            st.markdown("### 🔐 Access Permissions")

            # Create permission toggles
            updated_permissions = {}
            cols = st.columns(2)

            for idx, (perm_key, perm_label) in enumerate(PAGE_PERMISSIONS.items()):
                col = cols[idx % 2]

                with col:
                    # Disable toggle for admins (they always have full access)
                    disabled = is_admin(user_email)

                    current_value = permissions.get(perm_key, False)

                    # Create unique key using user index to avoid duplicates
                    unique_key = f"user_{user_idx}_{perm_key}"

                    # If admin, always show as enabled
                    if disabled:
                        st.checkbox(
                            perm_label,
                            value=True,
                            key=unique_key,
                            disabled=True,
                            help="Administrators always have full access"
                        )
                        updated_permissions[perm_key] = True
                    else:
                        new_value = st.checkbox(
                            perm_label,
                            value=current_value,
                            key=unique_key
                        )
                        updated_permissions[perm_key] = new_value

            # Save button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("💾 Save Permissions", key=f"save_{user_idx}", type="primary"):
                    if not is_admin(user_email):  # Only update if not admin
                        success = update_user_permissions(db, user_email, updated_permissions)

                        if success:
                            st.success("✅ Permissions updated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to update permissions")
                    else:
                        st.info("ℹ️ Cannot modify administrator permissions")

            with col2:
                if st.button("🔄 Reset to Default", key=f"reset_{user_idx}"):
                    if not is_admin(user_email):
                        default_permissions = {key: False for key in PAGE_PERMISSIONS.keys()}
                        success = update_user_permissions(db, user_email, default_permissions)

                        if success:
                            st.success("✅ Permissions reset to default (all disabled)")
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset permissions")
                    else:
                        st.info("ℹ️ Cannot reset administrator permissions")

            with col3:
                # Permission summary
                enabled_count = sum(1 for v in updated_permissions.values() if v)
                st.caption(f"Access: {enabled_count}/{len(PAGE_PERMISSIONS)} pages enabled")

# Summary section
st.markdown("---")
st.markdown("## 📊 Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Users", len(users))

with col2:
    admin_count = sum(1 for u in users if is_admin(u.get('email')))
    st.metric("Administrators", admin_count)

with col3:
    active_users = sum(1 for u in users if u.get('last_login'))
    st.metric("Users with Login", active_users)

# Help section
with st.expander("ℹ️ User Management Help"):
    st.markdown("""
    **How User Management Works:**

    1. **Automatic User Creation**: When a user signs in with Google OAuth, they are automatically added to the database

    2. **Default Permissions**:
       - khalid0211@gmail.com gets full access automatically (Administrator)
       - Other users get no access by default (all toggles off)

    3. **Managing Permissions**:
       - Toggle permissions on/off for each page
       - Click "Save Permissions" to apply changes
       - Use "Reset to Default" to disable all permissions

    4. **Administrator Role**:
       - Administrators always have full access (cannot be changed)
       - Only administrators can access this User Management page
       - Primary admin: khalid0211@gmail.com

    5. **Page Access Control**:
       - Users can only access pages they have permission for
       - If denied, they'll see an "Access Denied" message
       - User Management is only accessible to administrators

    **Note:** Changes take effect immediately. Users may need to refresh their page to see updated access.
    """)

# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()
