# pages/login.py
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from auth import auth_manager

dash.register_page(__name__, path="/login", name="Login")

# Login form layout
layout = html.Div([
    dcc.Store(id="login-status-store"),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H2("Crime Analysis Dashboard", 
                                   className="text-center mb-2",
                                   style={"color": "#2c3e50"}),
                            html.P("Please login to access the dashboard", 
                                  className="text-center text-muted mb-4"),
                        ]),
                        
                        html.Form([
                            dbc.InputGroup([
                                dbc.InputGroupText(html.I(className="fas fa-user")),
                                dbc.Input(
                                    id="login-username",
                                    placeholder="Username",
                                    type="text",
                                    required=True
                                )
                            ], className="mb-3"),
                            
                            dbc.InputGroup([
                                dbc.InputGroupText(html.I(className="fas fa-lock")),
                                dbc.Input(
                                    id="login-password",
                                    placeholder="Password",
                                    type="password",
                                    required=True
                                )
                            ], className="mb-3"),
                            
                            dbc.Button(
                                "Login",
                                id="login-submit-btn",
                                color="primary",
                                className="w-100 mb-3",
                                size="lg"
                            ),
                        ]),
                        
                        html.Div(id="login-message", className="mt-3"),
                        
                        html.Hr(),
                        
                        html.Div([
                            html.P("Default accounts:", className="text-muted small mb-1"),
                            html.P("Admin: admin / admin123", className="text-muted small mb-1"),
                            html.P("Analyst: analyst / analyst123", className="text-muted small"),
                        ], className="text-center")
                    ])
                ], style={
                    "max-width": "400px",
                    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                    "border": "none"
                })
            ], width=12, className="d-flex justify-content-center")
        ], className="min-vh-100 d-flex align-items-center")
    ], fluid=True, style={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "min-height": "100vh"
    })
])

# Login callback
@callback(
    [Output("login-message", "children"),
     Output("login-status-store", "data"),
     Output("url", "pathname", allow_duplicate=True)],
    Input("login-submit-btn", "n_clicks"),
    [State("login-username", "value"),
     State("login-password", "value")],
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    if n_clicks and username and password:
        success, session_id, message = auth_manager.authenticate_user(username, password)
        
        if success:
            # Store session in browser
            login_data = {
                "logged_in": True,
                "session_id": session_id,
                "username": username
            }
            
            success_message = dbc.Alert(
                "Login successful! Redirecting...",
                color="success",
                dismissable=False
            )
            
            return success_message, login_data, "/data"
        else:
            error_message = dbc.Alert(
                f"Login failed: {message}",
                color="danger",
                dismissable=True
            )
            return error_message, {}, dash.no_update
    
    return "", {}, dash.no_update

# Handle Enter key press for login
@callback(
    Output("login-submit-btn", "n_clicks"),
    [Input("login-username", "n_submit"),
     Input("login-password", "n_submit")],
    State("login-submit-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_enter_key(username_submit, password_submit, current_clicks):
    if username_submit or password_submit:
        return (current_clicks or 0) + 1
    return current_clicks or 0