import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import plotly.graph_objects as go
import plotly.express as px
from server import db
import logging
import json

dash.register_page(__name__, path="/prediction", name="Crime Rate Prediction")

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "model_1": {
        "name": "Model 1: Usia 15 Tahun ke Atas",
        "target": "jumlah_kriminalitas_sebenarnya_diatas_15_tahun",
        "features": [
            "jumlah_penduduk_u15",
            "jumlah_berpendidikan_diatas_15_tahun", 
            "jumlah_tidak_berpendidikan_diatas_15_tahun",
            "jumlah_miskin_diatas_15_tahun",
            "jumlah_tidak_miskin_diatas_15_tahun",
        ],
        "description": "Model khusus untuk prediksi kriminalitas pada populasi usia 15 tahun ke atas"
    },
    "model_2": {
        "name": "Model 2: Seluruh Populasi",
        "target": "jumlah_kriminalitas_sebenarnya",
        "features": [
            "jumlah_penduduk_seluruh",
            "jumlah_bekerja",
            "jumlah_tidak_bekerja",
            "jumlah_berpendidikan",
            "jumlah_tidak_berpendidikan",
            "jumlah_miskin",
            "jumlah_tidak_miskin",
        ],
        "description": "Model untuk prediksi kriminalitas pada seluruh populasi"
    }
}

def get_firestore_collections():
    """Get all collection names from Firestore"""
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        return collection_names
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []

def get_collection_data(collection_name):
    """Get data from a specific Firestore collection"""
    try:
        docs = db.collection(collection_name).stream()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            data.append(doc_data)
        return data
    except Exception as e:
        logger.error(f"Error getting data from collection {collection_name}: {e}")
        return []

def get_all_data(collection_name='crime_data'):
    """Get data from specified collection"""
    try:
        data = get_collection_data(collection_name)
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def validate_model_data(df, model_key):
    """Validate if data contains required columns for the selected model"""
    model_config = MODEL_CONFIGS[model_key]
    required_cols = model_config["features"] + [model_config["target"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing columns for {model_config['name']}: {missing_cols}")
        return False, missing_cols
    
    return True, []

def preprocess_data(df, model_key):
    """Preprocess the data based on selected model"""
    if df.empty:
        logger.error("No data available for preprocessing.")
        return None, None, None, []

    model_config = MODEL_CONFIGS[model_key]
    
    # Validate data
    is_valid, missing_cols = validate_model_data(df, model_key)
    if not is_valid:
        return None, None, None, missing_cols

    # Remove outliers using Z-score
    numeric_cols = df.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) > 0:
        z_scores = zscore(numeric_cols)
        outliers = (np.abs(z_scores) > 3).any(axis=1)
        if outliers.any():
            logger.warning(f"Outliers detected and removed: {outliers.sum()} rows")
            df = df[~outliers]

    # Separate features and target based on model
    feature_cols = model_config["features"]
    target_col = model_config["target"]
    
    # Filter available features (in case some are optional)
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df[target_col].values

    logger.info(f"Using {len(available_features)} features for {model_config['name']}")
    return X, y, df, []

def split_data(X, y, n_splits=5):
    """Split data into training and testing sets using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    train_index, test_index = splits[-1]  # Use the last split
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the Random Forest model."""
    model = RandomForestRegressor(random_state=48)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=TimeSeriesSplit(n_splits=3), 
        scoring="r2",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    logger.info(f"Best model parameters: {grid_search.best_params_}")

    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = r2_score(y_test, y_pred) * 100
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    logger.info(f"Model Evaluation - R²: {accuracy:.2f}%, MAPE: {mape:.2f}%, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    return best_model, accuracy, mape, mse, rmse, y_pred, y_test

def predict_next_year(model, df, scaler, model_key):
    """Predict the next year's crime rate using the trained model."""
    if df.empty:
        return None
    
    model_config = MODEL_CONFIGS[model_key]
    available_features = [col for col in model_config["features"] if col in df.columns]
    
    X = df[available_features]
    next_year_X = X.iloc[-1:].copy()
    next_year_X_scaled = scaler.transform(next_year_X)
    y_next = model.predict(next_year_X_scaled)[0]
    
    logger.info(f"Predicted crime rate for next year: {y_next:.2f}")
    return y_next

def run_model_pipeline(df, model_key):
    """Run the entire model pipeline: preprocessing, training, evaluation, and prediction."""
    if df.empty:
        logger.error("No data provided for the model pipeline.")
        return None, None, None, None, None, None, None, None, []

    # Preprocess data
    X, y, processed_df, missing_cols = preprocess_data(df, model_key)
    if X is None:
        return None, None, None, None, None, None, None, None, missing_cols

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # Train and evaluate model
    model, accuracy, mape, mse, rmse, y_pred, y_test = train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # Predict next year's value
    y_next = predict_next_year(model, processed_df, scaler, model_key)

    return model, accuracy, mape, mse, rmse, y_next, y_pred, y_test, []

# Layout
layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Crime Rate Prediction Dashboard", className="text-center mb-4"),
            html.P("Sistem Prediksi Tingkat Kriminalitas dengan Dual Model", 
                   className="text-center text-muted mb-4"),
            html.Hr(),
        ], width=12)
    ]),

    # Section 1: Model & Data Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pilih Model & Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Pilih Model Prediksi:"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[
                                    {"label": config["name"], "value": key}
                                    for key, config in MODEL_CONFIGS.items()
                                ],
                                value="model_1",
                                className="mb-3"
                            ),
                            html.Div(id="model-description", className="mb-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Pilih Collection Data:"),
                            dcc.Dropdown(
                                id="collection-dropdown-prediction",
                                placeholder="Select a collection...",
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Data Status:"),
                            html.Div(id="data-validation-status", className="mb-3"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Model Status:"),
                            html.Div(id="model-status", className="mb-3"),
                        ], width=6),
                    ]),
                    
                    dbc.Button("Latih Model", id="train-button", color="primary", className="w-100", disabled=True),
                    
                    # Store components
                    dcc.Store(id="model-store"),
                    dcc.Store(id="data-store"),
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 2: Model Performance Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Matriks Performa Model", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="metrics-content", className="mb-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 3: Feature Importance
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Analisis Feature Importance", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id="feature-importance-graph")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 4: Prediction Results and Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Visualisasi Hasil Prediksi", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id="prediction-graph")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 5: Future Prediction
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Prediksi Kriminalitas Di Masa Depan", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="future-prediction", className="mb-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 6: Status & Logs
    dbc.Row([
        dbc.Col([
            html.Div(id="prediction-output-status", className="mb-4")
        ])
    ])
])

# Callbacks
@callback(
    [Output("collection-dropdown-prediction", "options"),
     Output("collection-dropdown-prediction", "value")],
    [Input("collection-dropdown-prediction", "id")]
)
def update_collections(_):
    collections = get_firestore_collections()
    options = [{"label": col, "value": col} for col in collections]
    default_value = "crime_data" if "crime_data" in collections else (collections[0] if collections else None)
    return options, default_value

@callback(
    Output("model-description", "children"),
    Input("model-dropdown", "value")
)
def update_model_description(model_key):
    if model_key and model_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_key]
        return dbc.Alert([
            html.H6("Model Description:", className="mb-2"),
            html.P(config["description"], className="mb-2"),
            html.P(f"Target Variable: {config['target']}", className="mb-2"),
            html.P(f"Features: {len(config['features'])} variables", className="mb-0")
        ], color="info")
    return html.Div()

@callback(
    [Output("data-validation-status", "children"),
     Output("train-button", "disabled"),
     Output("model-status", "children")],
    [Input("collection-dropdown-prediction", "value"),
     Input("model-dropdown", "value")]
)
def validate_data_and_model(collection_name, model_key):
    if not collection_name or not model_key:
        return (
            dbc.Badge("Select collection and model", color="secondary"),
            True,
            dbc.Badge("Not ready", color="secondary")
        )
    
    data = get_all_data(collection_name)
    if data.empty:
        return (
            dbc.Badge("No data available", color="danger"),
            True,
            dbc.Badge("No data", color="danger")
        )
    
    # Validate model requirements
    is_valid, missing_cols = validate_model_data(data, model_key)
    
    if is_valid:
        status_msg = dbc.Badge(f"Data valid ({len(data)} records)", color="success")
        model_status = dbc.Badge("Ready to train", color="success")
        return status_msg, False, model_status
    else:
        missing_msg = f"Missing: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}"
        status_msg = dbc.Badge(missing_msg, color="warning")
        model_status = dbc.Badge("Data incomplete", color="warning")
        return status_msg, True, model_status

@callback(
    [Output("model-store", "data"),
     Output("data-store", "data"),
     Output("metrics-content", "children"),
     Output("feature-importance-graph", "figure"),
     Output("prediction-graph", "figure"),
     Output("future-prediction", "children"),
     Output("prediction-output-status", "children")],
    [Input("train-button", "n_clicks")],
    [State("collection-dropdown-prediction", "value"),
     State("model-dropdown", "value")]
)
def train_model_and_predict(n_clicks, collection_name, model_key):
    if not n_clicks or not collection_name or not model_key:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Train the model to see results", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return None, None, html.P("Click 'Train Model' to start."), empty_fig, empty_fig, html.P(""), html.Div()

    try:
        # Get data and run model pipeline
        df = get_all_data(collection_name)
        
        if df.empty:
            empty_fig = go.Figure()
            return None, None, dbc.Alert("No data available for training.", color="danger"), empty_fig, empty_fig, html.P(""), html.Div()

        # Run the model pipeline
        model, accuracy, mape, mse, rmse, y_next, y_pred, y_test, missing_cols = run_model_pipeline(df, model_key)
        
        if model is None:
            error_msg = "Model training failed."
            if missing_cols:
                error_msg += f" Missing columns: {', '.join(missing_cols)}"
            empty_fig = go.Figure()
            return None, None, dbc.Alert(error_msg, color="danger"), empty_fig, empty_fig, html.P(""), html.Div()

        # Create metrics display
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{accuracy:.2f}%", className="text-primary"),
                        html.P("R² Score", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{mape:.2f}%", className="text-warning"),
                        html.P("MAPE", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{rmse:.2f}", className="text-info"),
                        html.P("RMSE", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{mse:.2f}", className="text-success"),
                        html.P("MSE", className="text-muted")
                    ])
                ])
            ], width=3),
        ])

        # Feature importance graph
        feature_names = MODEL_CONFIGS[model_key]["features"]
        available_features = [f for f in feature_names if f in df.columns]
        
        if hasattr(model, 'feature_importances_'):
            importance_fig = go.Figure(go.Bar(
                x=model.feature_importances_[:len(available_features)],
                y=available_features,
                orientation='h'
            ))
            importance_fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=400
            )
        else:
            importance_fig = go.Figure()
            importance_fig.add_annotation(text="Feature importance not available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        # Create prediction visualization
        pred_fig = go.Figure()
        
        # Add actual vs predicted scatter plot
        pred_fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Sebenarnya vs Diprediksi',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        pred_fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        pred_fig.update_layout(
            title=f"Sebenarnya vs Diprediksi - {MODEL_CONFIGS[model_key]['name']}",
            xaxis_title="Kriminalitas Sebernanya",
            yaxis_title="Kriminalitas Diprediksi",
            hovermode='closest'
        )

        # Future prediction display
        future_pred_card = dbc.Card([
            dbc.CardBody([
                html.H3(f"{y_next:.0f}", className="text-center text-primary"),
                html.P("Predicted Crime Cases for Next Period", className="text-center text-muted"),
                html.Hr(),
                html.P(f"Model: {MODEL_CONFIGS[model_key]['name']}", 
                      className="text-center small text-muted"),
                html.P(f"Accuracy: {accuracy:.1f}% R² Score", 
                      className="text-center small text-muted")
            ])
        ], color="light", outline=True)

        # Status message
        status_alert = dbc.Alert([
            html.H5("Pelatihan Model Berhasil!", className="alert-heading"),
            html.P(f"Model: {MODEL_CONFIGS[model_key]['name']}"),
            html.P(f"Dilatih dari  {len(df)} records dari '{collection_name}' collection."),
            html.P(f"Akurasi model {accuracy:.2f}% R² score dengan {mape:.2f}% MAPE.", className="mb-0")
        ], color="success")

        return (
            {"trained": True, "collection": collection_name, "model": model_key}, 
            df.to_dict('records'),
            metrics_cards, 
            importance_fig,
            pred_fig, 
            future_pred_card, 
            status_alert
        )

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        error_alert = dbc.Alert(f"Error during model training: {str(e)}", color="danger")
        empty_fig = go.Figure()
        return None, None, error_alert, empty_fig, empty_fig, html.P(""), html.Div()