# server.py
from dash import Dash
import dash_bootstrap_components as dbc
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Attempt to load Firebase credentials
    cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 'connect/firebase-service-account-key.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing Firebase: {e}")
    raise

# Initialize Firestore
try:
    db = firestore.client()
    logger.info("Firestore initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing Firestore: {e}")
    raise

# Initialize the Dash app
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP], 
           suppress_callback_exceptions=True,
           use_pages=True)
app.title = "Crime Rate Prediction Dashboard"

# Expose the server for deployment
server = app.server