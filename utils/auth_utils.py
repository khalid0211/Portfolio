"""
Authentication utility functions
Helper functions for access control and user management
"""

from typing import Optional

# Try to import firestore for SERVER_TIMESTAMP
try:
    from google.cloud import firestore
except ImportError:
    firestore = None


def check_page_access(db, user_email: Optional[str], page_name: str) -> bool:
    """
    Check if a user has access to a specific page

    Args:
        db: Firebase database instance (or None if not configured)
        user_email: User's email address
        page_name: Internal page identifier

    Returns:
        bool: True if user has access, False otherwise
    """
    # If no database configured, allow access (development mode)
    if db is None:
        return True

    # If no user email, deny access
    if not user_email:
        return False

    # Admin users have access to everything
    if is_admin(user_email):
        return True

    try:
        # Try to fetch user permissions from database
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        if user_doc.exists:
            user_data = user_doc.to_dict()
            permissions = user_data.get('permissions', {})

            # Check if page access is granted
            return permissions.get(page_name, False)
        else:
            # User not in database - grant basic access by default
            # You can change this to False for stricter access control
            return True

    except Exception as e:
        # If database access fails, log error and allow access
        # (to avoid locking users out due to connection issues)
        print(f"Database access error: {e}")
        return True


def is_admin(user_email: Optional[str]) -> bool:
    """
    Check if a user is an administrator

    Args:
        user_email: User's email address

    Returns:
        bool: True if user is admin, False otherwise
    """
    # Define admin emails
    ADMIN_EMAILS = [
        "khalid0211@gmail.com",  # Primary admin
        "dev@localhost",         # Development mode
    ]

    return user_email in ADMIN_EMAILS


def get_user_permissions(db, user_email: str) -> dict:
    """
    Get all permissions for a user

    Args:
        db: Firebase database instance
        user_email: User's email address

    Returns:
        dict: Dictionary of page permissions
    """
    if db is None:
        # Return default permissions for development mode
        return {
            'file_management': True,
            'manual_data_entry': True,
            'project_analysis': True,
            'portfolio_analysis': True,
            'portfolio_charts': True,
            'cash_flow_simulator': True,
            'evm_simulator': True,
            'user_management': is_admin(user_email)
        }

    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        if user_doc.exists:
            user_data = user_doc.to_dict()
            return user_data.get('permissions', {})
        else:
            # Return default permissions for new users
            return {
                'file_management': True,
                'manual_data_entry': True,
                'project_analysis': True,
                'portfolio_analysis': True,
                'portfolio_charts': True,
                'cash_flow_simulator': True,
                'evm_simulator': True,
                'user_management': False
            }
    except Exception as e:
        print(f"Error fetching user permissions: {e}")
        return {}


def update_user_permissions(db, user_email: str, permissions: dict) -> bool:
    """
    Update user permissions in the database

    Args:
        db: Firebase database instance
        user_email: User's email address
        permissions: Dictionary of page permissions

    Returns:
        bool: True if successful, False otherwise
    """
    if db is None:
        return False

    try:
        user_ref = db.collection('users').document(user_email)
        user_ref.set({
            'email': user_email,
            'permissions': permissions
        }, merge=True)
        return True
    except Exception as e:
        print(f"Error updating user permissions: {e}")
        return False


def ensure_user_exists(db, user_email: str, user_name: str = "User") -> None:
    """
    Create or update user record in Firebase on login
    Grants full access to khalid0211@gmail.com by default
    Tracks: Create Date, Last Access Date, Number of Times Accessed

    Args:
        db: Firebase database instance
        user_email: User's email address
        user_name: User's display name
    """
    if db is None:
        return

    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        # Check if user is admin
        admin_user = is_admin(user_email)

        if not user_doc.exists:
            # Create new user with default permissions
            default_permissions = {
                'file_management': admin_user,
                'manual_data_entry': admin_user,
                'project_analysis': admin_user,
                'portfolio_analysis': admin_user,
                'portfolio_charts': admin_user,
                'cash_flow_simulator': admin_user,
                'evm_simulator': admin_user,
                'user_management': admin_user
            }

            user_ref.set({
                'email': user_email,
                'name': user_name,
                'permissions': default_permissions,
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_access_date': firestore.SERVER_TIMESTAMP,
                'access_count': 1
            })

            pass  # User created successfully
        else:
            # Get current access count
            user_data = user_doc.to_dict()
            current_count = user_data.get('access_count', 0)

            # Update last access time and increment access count
            user_ref.update({
                'last_access_date': firestore.SERVER_TIMESTAMP,
                'name': user_name,  # Update name in case it changed
                'access_count': firestore.Increment(1)
            })

    except Exception as e:
        pass  # Silently fail if there's an error


def get_all_users(db) -> list:
    """
    Get all users from the database

    Args:
        db: Firebase database instance

    Returns:
        list: List of user dictionaries
    """
    if db is None:
        return []

    try:
        users_ref = db.collection('users')
        users = users_ref.stream()

        user_list = []
        for user in users:
            user_data = user.to_dict()
            user_data['id'] = user.id
            user_list.append(user_data)

        return user_list
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
