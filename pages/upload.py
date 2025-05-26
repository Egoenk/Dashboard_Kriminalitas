import dash
from dash import html

dash.register_page(__name__, path="/upload", name="Upload")

layout = html.Div([
    html.H2("Halaman Upload"),
    html.P("Ini adalah upload.")
])
