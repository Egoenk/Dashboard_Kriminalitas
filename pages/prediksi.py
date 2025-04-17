import dash
from dash import html

dash.register_page(__name__, path="/prediksi", name="Prediksi")

layout = html.Div([
    html.H2("Halaman Prediksi"),
    html.P("Ini adalah halaman prediksi.")
])
