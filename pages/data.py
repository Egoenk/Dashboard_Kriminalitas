import dash
from dash import html

dash.register_page(__name__, path="/data", name="Data")

layout = html.Div([
    html.H2("Halaman Data"),
    html.P("Ini adalah halaman data.")
])
