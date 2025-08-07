# pages/landing.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import callback, Input, Output, State

dash.register_page(__name__, path="/dashboard", name="Dashboard")

def create_feature_card(icon, title, description, color="primary"):
    """Create a feature card component"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas fa-{icon} fa-2x text-{color} mb-3"),
                html.H5(title, className="card-title"),
                html.P(description, className="card-text text-muted small")
            ], className="text-center")
        ])
    ], className="h-100 shadow-sm")

def create_step_card(step_number, title, description):
    """Create a step card for usage instructions"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Span(step_number, className="display-6 fw-bold text-primary")
                ], className="text-center mb-3"),
                html.H6(title, className="card-title text-center"),
                html.P(description, className="card-text text-muted small text-center")
            ])
        ])
    ], className="h-100 shadow-sm")

# Main layout
layout = html.Div([
    # Header Section
    html.Div([
        dbc.Container([
            html.Div([
                html.H1("SIKAPMAS", className="display-4 fw-bold text-primary mb-3"),
                html.H4("Sistem Informasi Kriminalitas & Analisis Prediktif Banyumas", 
                       className="text-secondary mb-4"),
                html.P([
                    "Selamat datang di platform analisis prediktif tingkat kriminalitas untuk wilayah Banyumas. ",
                    "SIKAPMAS menggunakan teknologi ", 
                    html.Strong("Random Forest Regression"), 
                    " untuk memberikan insight berdasarkan data historis kriminalitas."
                ], className="lead mb-4"),
                html.Hr(className="my-4"),
                html.P([
                    "Platform ini dikembangkan untuk membantu analisis dan prediksi tingkat kriminalitas ",
                    "dengan menggunakan metode machine learning yang dan teknik augmentasi data."
                ], className="text-muted")
            ], className="text-center")
        ], className="py-5")
    ], className="bg-light"),
    
    # Features Section
    html.Div([
        dbc.Container([
            html.H2("Fitur Utama", className="text-center mb-5"),
            dbc.Row([
                dbc.Col([
                    create_feature_card(
                        "upload", 
                        "Upload Data", 
                        "Upload file CSV data kriminalitas untuk dianalisis dan diproses",
                        "success"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_feature_card(
                        "table", 
                        "Lihat Data", 
                        "Tampilkan data kriminalitas dalam format tabel yang mudah dibaca",
                        "info"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_feature_card(
                        "chart-bar", 
                        "Visualisasi Data", 
                        "Buat grafik dan chart interaktif untuk analisis visual data",
                        "warning"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_feature_card(
                        "magic", 
                        "Augmentasi Data", 
                        "Gaussian, Interpolasi, dan Bootstrapping untuk memperkaya dataset",
                        "purple"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_feature_card(
                        "brain", 
                        "Prediksi Data", 
                        "Random Forest Regression dengan GridSearch, Standard Scaler, dan Time-Series Split",
                        "danger"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_feature_card(
                        "analytics", 
                        "Analisis Prediktif", 
                        "Evaluasi model dan visualisasi hasil prediksi tingkat kriminalitas",
                        "dark"
                    )
                ], width=12, md=6, lg=4, className="mb-4"),
            ])
        ], className="py-5")
    ]),
    
    # Usage Instructions Section
    html.Div([
        dbc.Container([
            html.H2("Cara Penggunaan", className="text-center mb-5"),
            dbc.Row([
                dbc.Col([
                    create_step_card(
                        "1", 
                        "Upload Data", 
                        "Mulai dengan mengupload file CSV yang berisi data kriminalitas historis"
                    )
                ], width=12, md=6, lg=3, className="mb-4"),
                
                dbc.Col([
                    create_step_card(
                        "2", 
                        "Eksplorasi Data", 
                        "Lihat dan eksplorasi data melalui tabel dan visualisasi interaktif"
                    )
                ], width=12, md=6, lg=3, className="mb-4"),
                
                dbc.Col([
                    create_step_card(
                        "3", 
                        "Augmentasi Data", 
                        "Perkaya dataset dengan teknik Gaussian, Interpolasi, dan Bootstrapping"
                    )
                ], width=12, md=6, lg=3, className="mb-4"),
                
                dbc.Col([
                    create_step_card(
                        "4", 
                        "Prediksi & Analisis", 
                        "Jalankan model Random Forest untuk prediksi dan analisis hasil"
                    )
                ], width=12, md=6, lg=3, className="mb-4"),
            ])
        ], className="py-5")
    ], className="bg-light"),
    
    # Technical Details Section
    html.Div([
        dbc.Container([
            html.H2("Teknologi yang Digunakan", className="text-center mb-5"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Machine Learning", className="card-title text-primary"),
                            html.Ul([
                                html.Li("Random Forest Regression"),
                                html.Li("GridSearch CV untuk hyperparameter tuning"),
                                html.Li("Standard Scaler untuk normalisasi data"),
                                html.Li("Time-Series Split untuk validasi temporal")
                            ], className="mb-0")
                        ])
                    ], className="h-100 shadow-sm")
                ], width=12, md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Augmentasi Data", className="card-title text-success"),
                            html.Ul([
                                html.Li("Interpolasi"),
                                html.Li("Bootstrapping"),
                                html.Li("Gaussian Noise")
                            ], className="mb-0")
                        ])
                    ], className="h-100 shadow-sm")
                ], width=12, md=6, className="mb-4"),
            ])
        ], className="py-5")
    ]),
    
    # Quick Start Section
    html.Div([
        dbc.Container([
            html.Div([
                html.H2("Mulai Sekarang", className="text-center mb-4"),
                html.P([
                    "Pilih menu di sidebar untuk memulai analisis data kriminalitas Anda. ",
                    "Pastikan data CSV sudah siap untuk diupload dan dianalisis."
                ], className="text-center text-muted mb-4"),
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-upload me-2"),
                        "Upload Data"
                    ], href="/upload", color="primary", size="lg", className="me-3"),
                    dbc.Button([
                        html.I(className="fas fa-table me-2"),
                        "Lihat Data"
                    ], href="/data", color="outline-primary", size="lg"),
                ], className="text-center")
            ])
        ], className="py-5")
    ], className="bg-light"),
])

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]