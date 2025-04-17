import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize app with Dash Pages & Bootstrap
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server  # for deployment

# Sidebar layout
sidebar = html.Div([
    html.H2("Menu", className="display-6"),
    html.Hr(),
    dbc.Nav(
        [
            dbc.NavLink("Data", href="/data", active="exact"),
            dbc.NavLink("Visualisasi", href="/visualisasi", active="exact"),
            dbc.NavLink("Prediksi", href="/prediksi", active="exact"),
            dbc.NavLink("Logout", href="/Logout", active="exact"),

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
})

# Main app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    html.Div(
        dash.page_container,
        style={"margin-left": "18rem", "padding": "2rem 1rem"},
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
