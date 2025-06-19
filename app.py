# app.py - Simplified without authentication
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import dash
from server import app

# Sidebar layout
def create_sidebar():
    return html.Div([
        html.H2("Dashboard Kriminalitas", className="display-6 text-primary"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Data", href="/data", active="exact"),
                dbc.NavLink("Upload CSV", href="/upload", active="exact"),
                dbc.NavLink("Visualisasi", href="/visualisasi", active="exact"),
                dbc.NavLink("Prediksi", href="/prediction", active="exact"),
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
    html.Div(id="app-content")
])

# Main callback to handle page routing
@callback(
    Output("app-content", "children"),
    Input("url", "pathname"),
    allow_dupolicate=True
)
def display_page(pathname):
    sidebar = create_sidebar()
    
    # Handle different routes
    if pathname == "/" or pathname is None:
        # Default page - redirect to data
        return html.Div([
            sidebar,
            html.Div([
                dcc.Location(id="redirect", pathname="/data")
            ], style={"margin-left": "18rem", "padding": "2rem 1rem"})
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

if __name__ == "__main__":
    app.run_server(debug=True)