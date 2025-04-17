from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from server import app
from pages import dashboard
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dbc.NavbarSimple(
        brand="Crime Rate Prediction Dashboard",
        brand_href="/dashboard",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard")),
        ],
    ),
    html.Div(id="page-content", className="container-fluid p-4")
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    try:
        if pathname == "/dashboard":
            return dashboard.layout
        return dcc.Location(pathname="/dashboard", id="redirect-dashboard")
    except Exception as e:
        logger.error(f"Error rendering page: {str(e)}")
        return dbc.Alert("An error occurred while loading the page.", color="danger")

if __name__ == "__main__":
    app.title = "Crime Rate Prediction Dashboard"
    app.run_server(debug=True)