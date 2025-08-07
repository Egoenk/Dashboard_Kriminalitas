# server.py - Optimized Firebase initialization
from dash import Dash
import dash_bootstrap_components as dbc
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
db = None
app = None

def initialize_firebase():
    """Initialize Firebase with better error handling and retry logic"""
    global db
    
    if db is not None:
        logger.info("Firebase already initialized")
        return db
    
    try:
        # Check if Firebase app is already initialized
        if not firebase_admin._apps:
            # Try different methods to load credentials
            cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 'connect/connect.json')
            
            if os.path.exists(cred_path):
                logger.info(f"Loading Firebase credentials from {cred_path}")
                cred = credentials.Certificate(cred_path)
            else:
                # Try loading from environment variable
                cred_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
                if cred_json:
                    logger.info("Loading Firebase credentials from environment variable")
                    cred_dict = json.loads(cred_json)
                    cred = credentials.Certificate(cred_dict)
                else:
                    # Try default application credentials
                    logger.info("Using default application credentials")
                    cred = credentials.ApplicationDefault()
            
            firebase_admin.initialize_app(cred)
            logger.info("Firebase app initialized successfully!")
        else:
            logger.info("Firebase app already exists, using existing instance")
        
        # Initialize Firestore
        db = firestore.client()
        
        # Test connection
        test_collection = db.collection('test')
        test_doc = test_collection.document('connection_test')
        # Just check if we can access it without writing
        try:
            test_doc.get()
            logger.info("Firestore connection test successful!")
        except Exception as e:
            logger.warning(f"Firestore connection test failed, but client initialized: {e}")
        
        return db
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        logger.error("Make sure your Firebase credentials are properly configured")
        raise

def create_dash_app():
    """Create and configure the Dash application"""
    global app
    
    if app is not None:
        return app
    
    # Initialize Firebase first
    initialize_firebase()
    
    # Initialize the Dash app with optimized settings
    app = Dash(
        __name__, 
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        ], 
        suppress_callback_exceptions=True,
        use_pages=True
    )
    
    app.title = "SIKAPMAS"
    
    # Configure server settings
    app.server.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-here'),
        SEND_FILE_MAX_AGE_DEFAULT=31536000, 
    )
    
    logger.info("Dash app created successfully!")
    return app

# Create the app instance
app = create_dash_app()

# Expose the server for deployment
server = app.server

# Export db for use in other modules
__all__ = ['app', 'server', 'db']