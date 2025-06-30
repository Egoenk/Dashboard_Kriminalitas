# login.py - Fixed login page component
import dash
from dash import html

# Register the login page
dash.register_page(__name__, path="/login", title="Login")

# Simple layout - the actual login page will be handled by the main app
layout = html.Div([
    html.H1("Redirecting to login...", className="text-center")
])