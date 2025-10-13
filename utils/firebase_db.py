"""
Firebase Database Configuration
Handles connection to Firebase Firestore for user management
"""

import os
import json

# Initialize database variable
db = None

try:
    # Import Firebase modules
    import firebase_admin
    from firebase_admin import credentials, firestore
    import streamlit as st

    # Try to get Firebase credentials from Streamlit secrets
    if hasattr(st, 'secrets') and 'firebase' in st.secrets:
        # Use credentials from secrets.toml
        firebase_config = dict(st.secrets['firebase'])

        # Initialize Firebase app if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)

        # Get Firestore client
        db = firestore.client()

    else:
        # Fallback to file-based credentials
        firebase_creds_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")

        if firebase_creds_path and os.path.exists(firebase_creds_path):
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(firebase_creds_path)
                firebase_admin.initialize_app(cred)

            # Get Firestore client
            db = firestore.client()

except ImportError:
    pass  # Firebase not installed
except Exception as e:
    pass  # Firebase initialization failed


def get_db():
    """
    Get the Firebase database instance

    Returns:
        Firestore client or None if not configured
    """
    return db


def is_db_configured() -> bool:
    """
    Check if Firebase database is configured and available

    Returns:
        bool: True if database is configured, False otherwise
    """
    return db is not None
