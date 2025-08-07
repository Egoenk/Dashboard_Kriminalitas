# auth.py - Improved authentication module
import firebase_admin
from firebase_admin import credentials, firestore
from dash import html, dcc, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import hashlib
import logging
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Get Firestore client from server.py to avoid re-initialization
try:
    from server import db
except ImportError:
    # Fallback if imported separately
    db = firestore.client()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

@lru_cache(maxsize=100)  # Cache user lookups for better performance
def get_user_from_db(username):
    """Get user data from Firestore with caching"""
    try:
        users_ref = db.collection('users')
        query = users_ref.where('username', '==', username).limit(1)
        docs = list(query.stream())  # Convert to list for better handling
        
        if docs:
            doc = docs[0]
            user_data = doc.to_dict()
            user_data['user_id'] = doc.id
            return user_data
        return None
    except Exception as e:
        logger.error(f"Error fetching user data: {e}")
        return None

def verify_user(username, password):
    """Verify user credentials against Firestore"""
    try:
        start_time = time.time()
        
        # Get user data
        user_data = get_user_from_db(username)
        
        if user_data:
            hashed_input = hash_password(password)
            
            # Check if password matches
            if user_data.get('password') == hashed_input:
                logger.info(f"Authentication successful for {username} in {time.time() - start_time:.2f}s")
                return {
                    'authenticated': True,
                    'username': username,
                    'role': user_data.get('role', 'user'),
                    'user_id': user_data.get('user_id')
                }
        
        logger.warning(f"Authentication failed for {username}")
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
    """Create login page layout with better styling"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H2("Sistem Informasi Kriminalitas & Analisis Prediktif Banyumas (SIKAPMAS)", 
                                   className="text-center mb-2 text-primary"),
                            html.H5("Sistem Prediksi Kriminalitas", 
                                   className="text-center mb-4 text-muted"),
                        ]),
                        html.Div(id="login-alert"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Label("Username", html_for="username-input", 
                                         className="form-label"),
                                dbc.Input(
                                    id="username-input",
                                    type="text",
                                    placeholder="Masukan username",
                                    className="mb-3",
                                    size="lg"
                                ),
                            ]),
                            dbc.Row([
                                dbc.Label("Password", html_for="password-input",
                                         className="form-label"),
                                dbc.Input(
                                    id="password-input",
                                    type="password",
                                    placeholder="Masukan Password",
                                    className="mb-4",
                                    size="lg"
                                ),
                            ]),
                            dbc.Button(
                                [
                                    dbc.Spinner(size="sm", color="light", id="login-spinner", 
                                              spinner_style={"display": "none"}),
                                    html.Span("Login", id="login-text")
                                ],
                                id="login-button",
                                color="primary",
                                className="w-100",
                                size="lg",
                                n_clicks=0
                            ),
                        ])
                    ])
                ], className="shadow-lg border-0")
            ], width=6, lg=4, className="mx-auto")
        ], className="justify-content-center min-vh-100 align-items-center")
    ], fluid=True, className="bg-light")

# Session management using dcc.Store
def create_session_stores():
    """Create session storage components"""
    return html.Div([
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='auth-state', storage_type='session', 
                 data={'authenticated': False}),
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
        'admin': ['view', 'upload', 'edit', 'delete', 'manage_users','augmentation','prediction'],
        'researcher': ['view', 'upload','augmentation','prediction'],
        'user': ['view','']
    }
    
    return required_permission in permissions.get(role, [])

# Add callback for login button loading state
@callback(
    [Output('login-spinner', 'spinner_style'),
     Output('login-text', 'children')],
    [Input('login-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value')],
    prevent_initial_call=True
)
def update_login_button_state(n_clicks, username, password):
    """Update login button state during authentication"""
    if n_clicks and n_clicks > 0 and username and password:
        return {"display": "inline-block"}, "Logging in..."
    return {"display": "none"}, "Login"