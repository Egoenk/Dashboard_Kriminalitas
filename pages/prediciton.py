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

def preprocess_data(df):
    """Preprocess the data: handle outliers, check correlation, and normalize features."""
    if df.empty:
        logger.error("No data available for preprocessing.")
        return None, None, None

    # Remove outliers using Z-score
    numeric_cols = df.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) > 0:
        z_scores = zscore(numeric_cols)
        outliers = (np.abs(z_scores) > 3).any(axis=1)
        if outliers.any():
            logger.warning(f"Outliers detected and removed: {outliers.sum()} rows")
            df = df[~outliers]

    # Separate features and target
    if "Reported crimeRate" not in df.columns:
        logger.error("Target column 'Reported crimeRate' not found in data")
        return None, None, None
    
    X = df.drop(columns=["id", "Reported crimeRate"])
    y = df["Reported crimeRate"].values

    return X, y, df

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
        "n_estimators": [200],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring="r2")
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

def predict_next_year(model, df, scaler):
    """Predict the next year's crime rate using the trained model."""
    if df.empty:
        return None
    X = df.drop(columns=["id", "Reported crimeRate"])
    next_year_X = X.iloc[-1:].copy()
    next_year_X_scaled = scaler.transform(next_year_X)
    y_next = model.predict(next_year_X_scaled)[0]
    logger.info(f"Predicted crime rate for next year: {y_next:.2f}")
    return y_next

def run_model_pipeline(df):
    """Run the entire model pipeline: preprocessing, training, evaluation, and prediction."""
    if df.empty:
        logger.error("No data provided for the model pipeline.")
        return None, None, None, None, None, None, None, None

    # Preprocess data
    X, y, processed_df = preprocess_data(df)
    if X is None:
        return None, None, None, None, None, None, None, None

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # Train and evaluate model
    model, accuracy, mape, mse, rmse, y_pred, y_test = train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # Predict next year's value
    y_next = predict_next_year(model, processed_df, scaler)

    return model, accuracy, mape, mse, rmse, y_next, y_pred, y_test

# Layout
layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Crime Rate Prediction Dashboard", className="text-center mb-4"),
            html.Hr(),
        ], width=12)
    ]),

    # Section 1: Data Selection and Model Training
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Data Selection & Model Training", className="mb-0")),
                dbc.CardBody([
                    html.P("Select your crime data collection and train the Random Forest model for prediction.", 
                          className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Collection:"),
                            dcc.Dropdown(
                                id="collection-dropdown-predicition",
                                placeholder="Select a collection...",
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Model Status:"),
                            html.Div(id="model-status", className="mb-3"),
                        ], width=6),
                    ]),
                    
                    dbc.Button("Train Model", id="train-button", color="primary", className="w-100", disabled=True),
                    
                    # Store components for model data
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
                dbc.CardHeader(html.H4("Model Performance Metrics", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="metrics-content", className="mb-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 3: Prediction Results and Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Prediction Results & Visualization", className="mb-0")),
                dbc.CardBody([
                    html.P("Visualize actual vs predicted values and future crime rate predictions.", 
                          className="text-muted mb-3"),
                    dcc.Graph(id="prediction-graph")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 4: Next Year Prediction
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Future Crime Rate Prediction", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="future-prediction", className="mb-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 5: Logs / Status / Results
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
    [Input("collection-dropdown-prediction", "value"),
     Input("some-other-component", "n_clicks")],  # Use a button or other trigger
    allow_duplicate=True
)
def update_collections(_):
    collections = get_firestore_collections()
    options = [{"label": col, "value": col} for col in collections]
    default_value = "crime_data" if "crime_data" in collections else (collections[0] if collections else None)
    return options, default_value

@callback(
    Output("train-button", "disabled"),
    Output("model-status", "children"),
    Input("collection-dropdown-prediction", "value")
)
def update_train_button(collection_name):
    if collection_name:
        data = get_all_data(collection_name)
        if not data.empty:
            status = dbc.Badge(f"Ready to train ({len(data)} records)", color="success")
            return False, status
        else:
            status = dbc.Badge("No data available", color="warning")
            return True, status
    else:
        status = dbc.Badge("Select a collection", color="secondary")
        return True, status

@callback(
    [Output("model-store", "data"),
     Output("data-store", "data"),
     Output("metrics-content", "children"),
     Output("prediction-graph", "figure"),
     Output("future-prediction", "children"),
     Output("prediction-output-status", "children")],
    [Input("train-button", "n_clicks")],
    [State("collection-dropdown-prediction", "value")]
)
def train_model_and_predict(n_clicks, collection_name):
    if not n_clicks or not collection_name:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Train the model to see predictions", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return None, None, html.P("Click 'Train Model' to start."), empty_fig, html.P(""), html.Div()

    try:
        # Get data and run model pipeline
        df = get_all_data(collection_name)
        
        if df.empty:
            return None, None, dbc.Alert("No data available for training.", color="danger"), go.Figure(), html.P(""), html.Div()

        # Run the model pipeline
        model, accuracy, mape, mse, rmse, y_next, y_pred, y_test = run_model_pipeline(df)
        
        if model is None:
            return None, None, dbc.Alert("Model training failed. Check your data format.", color="danger"), go.Figure(), html.P(""), html.Div()

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

        # Create prediction visualization
        fig = go.Figure()
        
        # Add actual vs predicted scatter plot
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Actual vs Predicted',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Crime Rates",
            xaxis_title="Actual Crime Rate",
            yaxis_title="Predicted Crime Rate",
            hovermode='closest'
        )

        # Future prediction display
        future_pred_card = dbc.Card([
            dbc.CardBody([
                html.H3(f"{y_next:.2f}", className="text-center text-primary"),
                html.P("Predicted Crime Rate for Next Year", className="text-center text-muted"),
                html.Hr(),
                html.P(f"Based on Random Forest model with {accuracy:.1f}% accuracy", 
                      className="text-center small text-muted")
            ])
        ], color="light", outline=True)

        # Status message
        status_alert = dbc.Alert([
            html.H5("Model Training Completed Successfully!", className="alert-heading"),
            html.P(f"Trained on {len(df)} records from '{collection_name}' collection."),
            html.P(f"Model achieved {accuracy:.2f}% R² score with {mape:.2f}% MAPE.", className="mb-0")
        ], color="success")

        return (
            {"trained": True, "collection": collection_name}, 
            df.to_dict('records'),
            metrics_cards, 
            fig, 
            future_pred_card, 
            status_alert
        )

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        error_alert = dbc.Alert(f"Error during model training: {str(e)}", color="danger")
        return None, None, error_alert, go.Figure(), html.P(""), html.Div()