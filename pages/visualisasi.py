import dash
from dash import html

dash.register_page(__name__, path="/visualisasi", name="Visualisasi")

layout = html.Div([
    html.H2("Halaman Visualisasi"),
    html.P("Ini adalah halaman visualisasi.")
])
