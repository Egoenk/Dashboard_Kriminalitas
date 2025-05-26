# pages/logout.py
import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from auth import auth_manager

dash.register_page(__name__, path="/logout", name="Logout")

layout = html.Div([
    dcc.Store(id="logout-trigger", data={"logout": True}),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-sign-out-alt fa-3x text-primary mb-3"),
                            html.H3("Logging out...", className="text-center"),
                            html.P("Please wait while we log you out.", className="text-center text-muted"),
                        ], className="text-center py-4")
                    ])
                ], style={"max-width": "400px"})
            ], width=12, className="d-flex justify-content-center")
        ], className="min-vh-100 d-flex align-items-center")
    ])
])

@callback(
    [Output("login-status-store", "data", allow_duplicate=True),
     Output("url", "pathname", allow_duplicate=True)],
    Input("logout-trigger", "data"),
    prevent_initial_call=False
)
def handle_logout(logout_data):
    if logout_data and logout_data.get("logout"):
        # Clear session data
        return {}, "/login"
    return dash.no_update, dash.no_update