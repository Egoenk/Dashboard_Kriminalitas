# app.py - Updated with authentication
from dash import html, dcc, Input, Output, State, callback, ctx, clientside_callback
import dash_bootstrap_components as dbc
import dash
from server import app
from auth import (
    create_login_page, 
    create_session_stores, 
    verify_user, 
    check_authentication,
    get_user_role,
    has_permission
)

# Sidebar layout with role-based navigation
def create_sidebar(user_role=None):
    nav_items = [
        dbc.NavLink("2. Lihat Data", href="/data", active="exact"),
        dbc.NavLink("3. Visualisasi Data", href="/visualisasi", active="exact"),
        dbc.NavLink("4. Augmentasi Data", href="/augmentation", active="exact"),
        dbc.NavLink("5. Prediksi Data", href="/prediction", active="exact"),
    ]
    
    # Add upload link based on role
    if user_role in ['admin', 'researcher']:
        nav_items.insert(0, dbc.NavLink("1. Upload CSV", href="/upload", active="exact"))
    
    return html.Div([
        html.H2("Dashboard Kriminalitas", className="display-6 text-primary"),
        html.Hr(),
        dbc.Nav(nav_items, vertical=True, pills=True),
        html.Hr(),
        html.Div([
            html.P(f"Role: {user_role or 'Guest'}", className="text-muted small"),
            dbc.Button(
                "Logout",
                id="logout-button",
                color="outline-danger",
                size="sm",
                className="w-100"
            )
        ])
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

# Main app layout with session management
app.layout = html.Div([
    dcc.Location(id="url"),
    create_session_stores(),
    html.Div(id="app-content")
])

# Login callback
@callback(
    [Output('session-store', 'data'),
     Output('auth-state', 'data'),
     Output('login-alert', 'children'),
     Output('login-redirect', 'pathname')],
    [Input('login-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value')],
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    if n_clicks > 0 and username and password:
        auth_result = verify_user(username, password)
        
        if auth_result['authenticated']:
            session_data = {
                'authenticated': True,
                'username': auth_result['username'],
                'role': auth_result['role'],
                'user_id': auth_result.get('user_id')
            }
            
            return (
                session_data,
                {'authenticated': True},
                None,
                '/data'  # Redirect to data page after successful login
            )
        else:
            alert = dbc.Alert(
                "Invalid username or password",
                color="danger",
                dismissable=True
            )
            return dash.no_update, dash.no_update, alert, dash.no_update
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Logout callback
@callback(
    [Output('session-store', 'data', allow_duplicate=True),
     Output('auth-state', 'data', allow_duplicate=True),
     Output('login-redirect', 'pathname', allow_duplicate=True)],
    [Input('logout-button', 'n_clicks')],
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    if n_clicks > 0:
        return {}, {'authenticated': False}, '/login'
    return dash.no_update, dash.no_update, dash.no_update

# Main callback to handle page routing with authentication
@callback(
    Output("app-content", "children"),
    [Input("url", "pathname"),
     Input("auth-state", "data")],
    [State("session-store", "data")],
)
def display_page(pathname, auth_state, session_data):
    # Handle login page
    if pathname == "/login" or not auth_state.get('authenticated', False):
        return create_login_page()
    
    # If authenticated, get user role
    user_role = get_user_role(session_data)
    sidebar = create_sidebar(user_role)
    
    # Handle routing for authenticated users
    if pathname == "/" or pathname is None:
        # Default page - redirect to data
        return html.Div([
            sidebar,
            html.Div([
                dcc.Location(id="redirect", pathname="/data")
            ], style={"margin-left": "18rem", "padding": "2rem 1rem"})
        ])
    
    # Check permissions for restricted pages
    if pathname in ["/upload"] and not has_permission(session_data, 'upload'):
        return html.Div([
            sidebar,
            html.Div([
                dbc.Alert(
                    f"Access denied. Your role ({user_role}) does not have permission to access this page.",
                    color="danger"
                )
            ], style={"margin-left": "18rem", "padding": "2rem 1rem"})
        ])
    
    # Show the requested page with sidebar
    return html.Div([
        sidebar,
        html.Div(
            dash.page_container,
            style={"margin-left": "18rem", "padding": "2rem 1rem"}
        )
    ])

# Client-side callback to handle initial authentication state
clientside_callback(
    """
    function(pathname) {
        // Check if we have session data and redirect accordingly
        const sessionData = sessionStorage.getItem('session-store');
        const authState = sessionStorage.getItem('auth-state');
        
        if (!authState || !JSON.parse(authState).authenticated) {
            if (pathname !== '/login') {
                return '/login';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('login-redirect', 'pathname', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call=True
)

if __name__ == "__main__":
    app.run_server(debug=True)