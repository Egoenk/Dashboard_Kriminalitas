# auth.py
import hashlib
import uuid
from datetime import datetime, timedelta
from server import db
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.sessions = {}  # In-memory session storage
        self.session_timeout = timedelta(hours=24)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, role="user"):
        """Create a new user in Firestore"""
        try:
            # Check if user already exists
            users_ref = db.collection('users')
            existing_user = users_ref.where('username', '==', username).get()
            
            if len(existing_user) > 0:
                return False, "Username already exists"
            
            # Check if email already exists
            existing_email = users_ref.where('email', '==', email).get()
            if len(existing_email) > 0:
                return False, "Email already exists"
            
            # Create new user
            user_data = {
                'username': username,
                'email': email,
                'password': self.hash_password(password),
                'role': role,
                'created_at': datetime.now(),
                'last_login': None,
                'is_active': True
            }
            
            doc_ref = users_ref.document()
            doc_ref.set(user_data)
            
            logger.info(f"User {username} created successfully")
            return True, "User created successfully"
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, f"Error creating user: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials"""
        try:
            users_ref = db.collection('users')
            user_query = users_ref.where('username', '==', username).get()
            
            if len(user_query) == 0:
                return False, None, "Invalid username or password"
            
            user_doc = user_query[0]
            user_data = user_doc.to_dict()
            
            # Check if user is active
            if not user_data.get('is_active', True):
                return False, None, "Account is deactivated"
            
            # Verify password
            if user_data['password'] == self.hash_password(password):
                # Update last login
                user_doc.reference.update({'last_login': datetime.now()})
                
                # Create session
                session_id = str(uuid.uuid4())
                session_data = {
                    'user_id': user_doc.id,
                    'username': username,
                    'email': user_data['email'],
                    'role': user_data['role'],
                    'created_at': datetime.now(),
                    'expires_at': datetime.now() + self.session_timeout
                }
                
                self.sessions[session_id] = session_data
                logger.info(f"User {username} authenticated successfully")
                return True, session_id, "Login successful"
            else:
                return False, None, "Invalid username or password"
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return False, None, f"Authentication error: {str(e)}"
    
    def validate_session(self, session_id):
        """Validate if session is still active"""
        if not session_id or session_id not in self.sessions:
            return False, None
        
        session_data = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() > session_data['expires_at']:
            del self.sessions[session_id]
            return False, None
        
        return True, session_data
    
    def logout_user(self, session_id):
        """Logout user by removing session"""
        if session_id in self.sessions:
            username = self.sessions[session_id]['username']
            del self.sessions[session_id]
            logger.info(f"User {username} logged out")
            return True
        return False
    
    def get_user_role(self, session_id):
        """Get user role from session"""
        is_valid, session_data = self.validate_session(session_id)
        if is_valid:
            return session_data['role']
        return None
    
    def require_auth(self, session_id):
        """Decorator function to require authentication"""
        is_valid, session_data = self.validate_session(session_id)
        return is_valid, session_data

# Initialize the auth manager
auth_manager = AuthManager()

# Create default admin user if it doesn't exist
def create_default_users():
    """Create default users for the system"""
    try:
        # Check if admin user exists
        users_ref = db.collection('users')
        admin_query = users_ref.where('username', '==', 'admin').get()
        
        if len(admin_query) == 0:
            # Create admin user
            auth_manager.create_user('admin', 'admin@crimereport.com', 'admin123', 'admin')
            print("Default admin user created: admin/admin123")
        
        # Check if analyst user exists
        analyst_query = users_ref.where('username', '==', 'analyst').get()
        
        if len(analyst_query) == 0:
            # Create analyst user
            auth_manager.create_user('analyst', 'analyst@crimereport.com', 'analyst123', 'analyst')
            print("Default analyst user created: analyst/analyst123")
            
    except Exception as e:
        logger.error(f"Error creating default users: {e}")

# Run this when the module is imported
create_default_users()