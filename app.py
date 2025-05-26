# app.py - Updated with authentication
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
from server import app
from auth import auth_manager

# Sidebar layout (only shown when logged in)
def create_sidebar(username=None, role=None):
    user_info = html.Div([
        html.P(f"Welcome, {username}!" if username else "Welcome!", 
               className="text-primary fw-bold mb-1"),
        html.P(f"Role: {role.title()}" if role else "", 
               className="text-muted small mb-3"),
    ]) if username else html.Div()
    
    return html.Div([
        html.H2("Crime Dashboard", className="display-6 text-primary"),
        html.Hr(),
        user_info,
        dbc.Nav(
            [
                dbc.NavLink("Data", href="/data", active="exact"),
                dbc.NavLink("Upload CSV", href="/upload", active="exact"),
                dbc.NavLink("Visualisasi", href="/visualisasi", active="exact"),
                dbc.NavLink("Prediksi", href="/prediksi", active="exact"),
                html.Hr(),
                dbc.NavLink("Logout", href="/logout", active="exact", 
                           style={"color": "#dc3545"}),
            ],
            vertical=True,
            pills=True,
        ),
    ], style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "border-right": "1px solid #dee2e6"
    })

# Main app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="login-status-store", storage_type="session"),
    html.Div(id="app-content")
])

# Main callback to handle authentication and page routing
@callback(
    Output("app-content", "children"),
    [Input("url", "pathname"),
     Input("login-status-store", "data")]
)
def display_page(pathname, login_data):
    # Check if user is logged in
    if login_data and login_data.get("logged_in"):
        session_id = login_data.get("session_id")
        is_valid, session_data = auth_manager.validate_session(session_id)
        
        if is_valid:
            # User is authenticated, show main dashboard
            username = session_data.get("username")
            role = session_data.get("role")
            
            sidebar = create_sidebar(username, role)
            
            # Handle different routes
            if pathname == "/" or pathname is None:
                # Redirect to data page by default
                return html.Div([
                    sidebar,
                    html.Div(
                        [dcc.Location(id="redirect", pathname="/data")],
                        style={"margin-left": "18rem", "padding": "2rem 1rem"}
                    )
                ])
            elif pathname == "/login":
                # Already logged in, redirect to data
                return html.Div([
                    dcc.Location(id="redirect", pathname="/data")
                ])
            else:
                # Show the requested page with sidebar
                return html.Div([
                    sidebar,
                    html.Div(
                        dash.page_container,
                        style={"margin-left": "18rem", "padding": "2rem 1rem"}
                    )
                ])
        else:
            # Session expired, redirect to login
            return html.Div([
                dcc.Location(id="redirect", pathname="/login")
            ])
    else:
        # User not logged in
        if pathname == "/login" or pathname is None or pathname == "/":
            # Show login page
            return dash.page_container
        else:
            # Redirect to login for any other page
            return html.Div([
                dcc.Location(id="redirect", pathname="/login")
            ])

# Callback to handle automatic session validation
@callback(
    Output("login-status-store", "data", allow_duplicate=True),
    Input("url", "pathname"),
    State("login-status-store", "data"),
    prevent_initial_call=True
)
def validate_session_on_navigation(pathname, login_data):
    if login_data and login_data.get("logged_in"):
        session_id = login_data.get("session_id")
        is_valid, session_data = auth_manager.validate_session(session_id)
        
        if not is_valid:
            # Session expired, clear login data
            return {}
    
    return dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True)