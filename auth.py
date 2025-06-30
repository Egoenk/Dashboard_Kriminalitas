# auth.py - Authentication module
import firebase_admin
from firebase_admin import credentials, firestore
from dash import html, dcc, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import hashlib
import logging

logger = logging.getLogger(__name__)

# Initialize Firebase (add your config)
# cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)
db = firestore.client()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """Verify user credentials against Firestore"""
    try:
        # Query users collection
        users_ref = db.collection('users')
        query = users_ref.where('username', '==', username).limit(1)
        docs = query.stream()
        
        for doc in docs:
            user_data = doc.to_dict()
            hashed_input = hash_password(password)
            
            # Check if password matches (assuming passwords are hashed in DB)
            if user_data.get('password') == hashed_input:
                return {
                    'authenticated': True,
                    'username': username,
                    'role': user_data.get('role', 'user'),
                    'user_id': doc.id
                }
        
        return {'authenticated': False}
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return {'authenticated': False}

def get_firestore_collections():
    """Retrieve all collections in the Firestore database"""
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        return collection_names
    except Exception as e:
        logger.error(f"Error retrieving collections: {e}")
        return []

def create_login_page():
    """Create login page layout"""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2("Login Dashboard Kriminalitas", className="text-center mb-4"),
                            html.Div(id="login-alert"),
                            dbc.Form([
                                dbc.Row([
                                    dbc.Label("Username", html_for="username-input"),
                                    dbc.Input(
                                        id="username-input",
                                        type="text",
                                        placeholder="Enter username",
                                        className="mb-3"
                                    ),
                                ]),
                                dbc.Row([
                                    dbc.Label("Password", html_for="password-input"),
                                    dbc.Input(
                                        id="password-input",
                                        type="password",
                                        placeholder="Enter password",
                                        className="mb-3"
                                    ),
                                ]),
                                dbc.Button(
                                    "Login",
                                    id="login-button",
                                    color="primary",
                                    className="w-100",
                                    n_clicks=0
                                ),
                            ])
                        ])
                    ], className="shadow")
                ], width=6, className="mx-auto")
            ], className="justify-content-center", style={"margin-top": "5rem"})
        ], fluid=True)
    ])

# Session management using dcc.Store
def create_session_stores():
    """Create session storage components"""
    return html.Div([
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='auth-state', storage_type='session', data={'authenticated': False}),
        dcc.Location(id='login-redirect', refresh=True)
    ])

def check_authentication(session_data):
    """Check if user is authenticated"""
    if not session_data:
        return False
    return session_data.get('authenticated', False)

def get_user_role(session_data):
    """Get user role from session"""
    if not session_data or not session_data.get('authenticated'):
        return None
    return session_data.get('role', 'user')

def has_permission(session_data, required_permission):
    """Check if user has required permission"""
    role = get_user_role(session_data)
    if not role:
        return False
    
    permissions = {
        'admin': ['view', 'upload', 'edit', 'delete', 'manage_users'],
        'researcher': ['view', 'upload'],
        'user': ['view']
    }
    
    return required_permission in permissions.get(role, [])