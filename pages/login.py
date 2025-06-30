# login.py - Login page component
import dash
from dash import html, dcc
from auth import create_login_page

# Register the login page
dash.register_page(__name__, path="/login", title="Login")

# Layout for the login page
layout = html.Div([
    create_login_page()
])