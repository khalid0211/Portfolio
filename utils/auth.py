"""
Authentication module for Portfolio Management Suite
Handles Google OAuth authentication and session management
"""

import streamlit as st
import os
import json
from urllib.parse import urlencode
import requests

# OAuth Configuration - Try Streamlit secrets first, then environment variables
try:
    GOOGLE_CLIENT_ID = st.secrets.get("google_oauth", {}).get("client_id", "")
    GOOGLE_CLIENT_SECRET = st.secrets.get("google_oauth", {}).get("client_secret", "")
except:
    # Fallback to environment variables
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# This will be set dynamically in check_authentication()
REDIRECT_URI = None

# Google OAuth endpoints
AUTHORIZATION_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def check_authentication():
    """
    Check if user is authenticated using Google OAuth
    Returns True if authenticated, False otherwise
    """
    global REDIRECT_URI

    # Dynamically detect redirect URI based on current URL
    if REDIRECT_URI is None:
        # Try to get the current URL from Streamlit's context
        try:
            # Check environment variable first
            env_uri = os.environ.get("REDIRECT_URI", "")
            if env_uri:
                REDIRECT_URI = env_uri
            # Check if hostname indicates Streamlit Cloud
            elif os.environ.get("HOSTNAME") == "streamlit":
                REDIRECT_URI = "https://portfolio-suite.streamlit.app/"
            else:
                # Default to localhost
                REDIRECT_URI = "http://localhost:8501"
        except:
            REDIRECT_URI = "http://localhost:8501"

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

    # If already authenticated, return True
    if st.session_state.authenticated:
        return True

    # Show login page
    st.markdown("# 🔐 Loading Portfolio Management Suite")
    st.markdown("Authentication complete")

    # Check if OAuth credentials are configured
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        st.error("❌ OAuth credentials not configured")
        st.info("""
        **Setup Instructions:**
        1. Create a Google Cloud Project
        2. Enable Google OAuth 2.0
        3. Add credentials to `.streamlit/secrets.toml`
        """)
        return False

    # Check if we have an authorization code in the URL
    query_params = st.query_params

    if 'code' in query_params:
        # Exchange authorization code for access token
        auth_code = query_params['code']

        try:
            # Exchange code for token
            token_data = {
                'code': auth_code,
                'client_id': GOOGLE_CLIENT_ID,
                'client_secret': GOOGLE_CLIENT_SECRET,
                'redirect_uri': REDIRECT_URI,
                'grant_type': 'authorization_code'
            }

            token_response = requests.post(TOKEN_URL, data=token_data)

            if token_response.status_code == 200:
                token_json = token_response.json()
                access_token = token_json.get('access_token')

                # Get user info using access token
                headers = {'Authorization': f'Bearer {access_token}'}
                userinfo_response = requests.get(USERINFO_URL, headers=headers)

                if userinfo_response.status_code == 200:
                    user_info = userinfo_response.json()

                    # Store user information in session state
                    st.session_state.authenticated = True
                    st.session_state.user_email = user_info.get('email')
                    st.session_state.user_info = user_info

                    # Create/update user record in Firebase
                    from utils.firebase_db import db
                    from utils.auth_utils import ensure_user_exists
                    if db is not None:
                        ensure_user_exists(db, user_info.get('email'), user_info.get('name', 'User'))

                    # Clear the code from URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error(f"Failed to get user info: {userinfo_response.status_code}")
            else:
                st.error(f"Failed to exchange code for token: {token_response.status_code}")
                st.code(token_response.text)

        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.info("Please try signing in again or contact support if the issue persists.")

        return False

    # Show Google Sign-In button
    st.markdown("### Sign in with Google")

    # Debug info - show which redirect URI is being used
    with st.expander("🔧 Debug Info"):
        st.code(f"Redirect URI: {REDIRECT_URI}")
        st.code(f"Environment check:\nSTREAMLIT_SHARING_MODE: {os.environ.get('STREAMLIT_SHARING_MODE', 'Not set')}\nHOSTNAME: {os.environ.get('HOSTNAME', 'Not set')}\nSTREAMLIT_SERVER_HEADLESS: {os.environ.get('STREAMLIT_SERVER_HEADLESS', 'Not set')}")

    # Generate OAuth URL
    oauth_params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'online',
        'prompt': 'select_account'
    }

    auth_url = f"{AUTHORIZATION_BASE_URL}?{urlencode(oauth_params)}"

    # Display login button as a link
    st.markdown(f"""
    <a href="{auth_url}" target="_self">
        <button style="
            background-color: #4285f4;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <svg width="18" height="18" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Sign in with Google
        </button>
    </a>
    """, unsafe_allow_html=True)

    return False


def require_page_access(page_name, page_title):
    """
    Check if user has access to a specific page
    Stops execution if access is denied

    Args:
        page_name: Internal page identifier
        page_title: Display name of the page
    """
    from utils.auth_utils import check_page_access
    from utils.firebase_db import db

    user_email = st.session_state.get('user_email')

    if not check_page_access(db, user_email, page_name):
        st.error(f"❌ Access Denied to {page_title}")
        st.info("You do not have permission to access this page. Please contact your administrator.")
        st.stop()


def show_user_info_sidebar():
    """
    Display user information and logout button in sidebar
    """
    if st.session_state.get('authenticated') and st.session_state.get('user_info'):
        user_info = st.session_state.user_info

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Info")

            # Show user picture if available
            if user_info.get('picture'):
                st.image(user_info['picture'], width=60)

            st.markdown(f"**{user_info.get('name', 'User')}**")
            st.markdown(f"_{user_info.get('email', 'N/A')}_")

            # Logout button
            if st.button("🚪 Logout", width="stretch"):
                # Clear session state
                st.session_state.authenticated = False
                st.session_state.user_info = None
                st.session_state.user_email = None
                st.rerun()
