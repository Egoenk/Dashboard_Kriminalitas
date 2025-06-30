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
    """Get data from specified collection and sort chronologically by tahun"""
    try:
        data = get_collection_data(collection_name)
        if not data:
            logger.warning(f"No data found in collection: {collection_name}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # CRITICAL FIX: Sort by year to ensure chronological order for time series
        if 'tahun' in df.columns:
            # Handle potential data type issues
            df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
            df = df.dropna(subset=['tahun'])  # Remove rows with invalid years
            df = df.sort_values('tahun', ascending=True).reset_index(drop=True)
            
            years = df['tahun'].unique()
            logger.info(f"Data sorted chronologically by tahun: {sorted(years)}")
            logger.info(f"Total records: {len(df)} spanning {len(years)} years ({min(years)}-{max(years)})")
        else:
            logger.warning("Column 'tahun' not found - chronological sorting not applied!")
            logger.warning("This may cause issues with time series predictions!")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching and sorting data: {e}")
        return pd.DataFrame()

def validate_temporal_data(df):
    """Validate temporal aspects of the data"""
    if df.empty:
        return False, "No data available"
    
    if 'tahun' not in df.columns:
        return False, "Missing 'tahun' column - cannot perform time series analysis"
    
    years = df['tahun'].dropna().sort_values()
    if len(years) == 0:
        return False, "No valid year data found"
    
    year_counts = df['tahun'].value_counts().sort_index()
    
    logger.info(f"Temporal validation:")
    logger.info(f"  Year range: {years.min()} to {years.max()}")
    logger.info(f"  Years with data: {list(year_counts.index)}")
    logger.info(f"  Records per year: {dict(year_counts)}")
    
    # Check for reasonable time series (at least 3 years for proper cross-validation)
    if len(year_counts) < 3:
        return False, f"Insufficient temporal data: only {len(year_counts)} years (minimum 3 required)"
    
    return True, f"Temporal data validated: {len(year_counts)} years of data"

def validate_model_data(df, model_key):
    """Validate if data contains required columns for the selected model"""
    model_config = MODEL_CONFIGS[model_key]
    required_cols = model_config["features"] + [model_config["target"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing columns for {model_config['name']}: {missing_cols}")
        return False, missing_cols
    
    # Check for sufficient non-null data
    null_counts = df[required_cols].isnull().sum()
    problematic_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()  # >50% null
    
    if problematic_cols:
        logger.warning(f"Columns with excessive null values: {problematic_cols}")
        return False, problematic_cols
    
    logger.info(f"Data validation passed for {model_config['name']}")
    return True, []

def preprocess_data(df, model_key):
    """Preprocess the data based on selected model"""
    if df.empty:
        logger.error("No data available for preprocessing.")
        return None, None, None, []

    # Validate temporal data first
    is_temporal_valid, temp_msg = validate_temporal_data(df)
    if not is_temporal_valid:
        logger.error(f"Temporal validation failed: {temp_msg}")
        return None, None, None, [temp_msg]

    model_config = MODEL_CONFIGS[model_key]
    
    # Validate model data requirements
    is_valid, missing_cols = validate_model_data(df, model_key)
    if not is_valid:
        return None, None, None, missing_cols

    # Ensure data is sorted chronologically (double-check)
    if 'tahun' in df.columns:
        df = df.sort_values('tahun', ascending=True).reset_index(drop=True)
        logger.info(f"Data confirmed chronologically sorted: {df['tahun'].min()} to {df['tahun'].max()}")

    # Get required columns
    feature_cols = model_config["features"]
    target_col = model_config["target"]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target_col])
    
    # Remove outliers using Z-score on numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
    
    for col in numeric_cols:
        if col in available_features + [target_col]:
            z_scores = np.abs(zscore(df_clean[col].dropna()))
            col_outliers = z_scores > 3
            outlier_mask = outlier_mask | col_outliers
    
    if outlier_mask.any():
        logger.warning(f"Outliers detected and removed: {outlier_mask.sum()} rows out of {len(df_clean)}")
        df_clean = df_clean[~outlier_mask]

    # Final feature and target extraction
    X = df_clean[available_features]
    y = df_clean[target_col].values

    # Check for sufficient data after cleaning
    if len(X) < 10:
        logger.error(f"Insufficient data after cleaning: only {len(X)} records")
        return None, None, None, ["Insufficient data after preprocessing"]

    logger.info(f"Preprocessing complete:")
    logger.info(f"  Model: {model_config['name']}")
    logger.info(f"  Features used: {len(available_features)} out of {len(feature_cols)}")
    logger.info(f"  Final dataset size: {len(X)} records")
    logger.info(f"  Year range: {df_clean['tahun'].min()} - {df_clean['tahun'].max()}")
    
    return X, y, df_clean, []

def split_data(X, y, df, n_splits=5):
    """Split data into training and testing sets using TimeSeriesSplit with proper logging"""
    
    # Validate minimum data requirements
    if len(X) < n_splits + 1:
        logger.warning(f"Insufficient data for {n_splits} splits. Reducing to {len(X)-1} splits")
        n_splits = max(2, len(X) - 1)
    
    # Log the chronological order before splitting
    if 'tahun' in df.columns:
        years = df['tahun'].values
        logger.info(f"Time series split preparation:")
        logger.info(f"  Data chronological order: {list(years)}")
        logger.info(f"  Unique years: {sorted(set(years))}")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    
    # Use the last split for final train/test
    train_index, test_index = splits[-1]
    
    # Log the train/test distribution
    if 'tahun' in df.columns:
        train_years = df.iloc[train_index]['tahun'].values
        test_years = df.iloc[test_index]['tahun'].values
        
        logger.info(f"Time series split results:")
        logger.info(f"  Training data: {len(train_index)} records from years {sorted(set(train_years))}")
        logger.info(f"  Testing data: {len(test_index)} records from years {sorted(set(test_years))}")
        
        # Validate that test years are after train years (as expected in time series)
        max_train_year = max(train_years)
        min_test_year = min(test_years)
        
        if min_test_year <= max_train_year:
            logger.warning(f"Time series split validation: Test year {min_test_year} overlaps with training years!")
        else:
            logger.info(f"Time series split validated: Training ends at {max_train_year}, testing starts at {min_test_year}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    
    # Log original feature statistics
    logger.info("Feature normalization:")
    logger.info(f"  Training features shape: {X_train.shape}")
    logger.info(f"  Testing features shape: {X_test.shape}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log scaling statistics
    feature_names = X_train.columns.tolist()
    for i, feature in enumerate(feature_names[:3]):  # Log first 3 features
        logger.info(f"  {feature}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names):
    """Train and evaluate the Random Forest model with comprehensive logging."""
    
    logger.info("Starting model training with hyperparameter optimization...")
    
    # Base model
    model = RandomForestRegressor(random_state=48)
    
    # Parameter grid for optimization
    param_grid = {
        "n_estimators": [300],
        "max_depth": [15],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": [0.8],
        "bootstrap": [True]
    }
    
    # Grid search with time series cross-validation
    cv_splits = 5
    if len(X_train) < 500:
        cv_splits = 3  
    
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=cv_splits),
    scoring='r2',
    n_jobs=-1,  
    verbose=2,
    refit=True,
    return_train_score=True,
    error_score='raise'
)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    logger.info(f"Best model parameters found:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")

    # Evaluate model on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate comprehensive metrics
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Additional metrics
    mae = np.mean(np.abs(y_test - y_pred))
    accuracy_percentage = r2 * 100
    
    logger.info(f"Model evaluation results:")
    logger.info(f"  R² Score: {r2:.4f} ({accuracy_percentage:.2f}%)")
    logger.info(f"  MAPE: {mape:.4f} ({mape*100:.2f}%)")
    logger.info(f"  MSE: {mse:.2f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_') and len(feature_names) > 0:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 5 most important features:")
        for idx, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return best_model, accuracy_percentage, mape*100, mse, rmse, y_pred, y_test

def predict_next_year(model, df, scaler, model_key):
    """Predict the next year's crime rate using the trained model."""
    if df.empty:
        logger.error("No data available for future prediction")
        return None
    
    model_config = MODEL_CONFIGS[model_key]
    available_features = [col for col in model_config["features"] if col in df.columns]
    
    # Use the most recent year's data for prediction
    latest_data = df.iloc[-1:].copy()
    latest_year = latest_data['tahun'].iloc[0] if 'tahun' in df.columns else "Unknown"
    
    logger.info(f"Making prediction for period after {latest_year}")
    logger.info(f"Using data: {latest_data[available_features].iloc[0].to_dict()}")
    
    X = latest_data[available_features]
    X_scaled = scaler.transform(X)
    y_next = model.predict(X_scaled)[0]
    
    logger.info(f"Predicted crime count for next period: {y_next:.2f}")
    return y_next

def run_model_pipeline(df, model_key):
    """Run the entire model pipeline: preprocessing, training, evaluation, and prediction."""
    logger.info(f"Starting model pipeline for {MODEL_CONFIGS[model_key]['name']}")
    
    if df.empty:
        logger.error("No data provided for the model pipeline.")
        return None, None, None, None, None, None, None, None, ["No data available"]

    # Preprocess data
    X, y, processed_df, errors = preprocess_data(df, model_key)
    if X is None:
        logger.error("Preprocessing failed")
        return None, None, None, None, None, None, None, None, errors

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, processed_df)

    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # Train and evaluate model
    model, accuracy, mape, mse, rmse, y_pred, y_test = train_and_evaluate_model(
        X_train_scaled, y_train, X_test_scaled, y_test, X.columns.tolist()
    )

    # Predict next year's value
    y_next = predict_next_year(model, processed_df, scaler, model_key)

    logger.info("Model pipeline completed successfully")
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
    
    # Validate temporal data
    is_temporal_valid, temp_msg = validate_temporal_data(data)
    if not is_temporal_valid:
        return (
            dbc.Badge(f"Temporal issue: {temp_msg}", color="warning"),
            True,
            dbc.Badge("Temporal data invalid", color="warning")
        )
    
    # Validate model requirements
    is_valid, missing_cols = validate_model_data(data, model_key)
    
    if is_valid:
        years = sorted(data['tahun'].unique())
        status_msg = dbc.Badge(f"Data valid ({len(data)} records, {years[0]}-{years[-1]})", color="success")
        model_status = dbc.Badge("Ready to train", color="success")
        return status_msg, False, model_status
    else:
        missing_msg = f"Missing: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}"
        status_msg = dbc.Badge(missing_msg, color="warning")
        model_status = dbc.Badge("Data incomplete", color="warning")
        return status_msg, True, model_status

# Complete the train_model_and_predict callback and remaining code

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
    """Main callback for training model and generating predictions with proper chronological sorting"""
    
    if not n_clicks or not collection_name or not model_key:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Train the model to see results", 
            xref="paper", yref="paper", x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return None, None, html.P("Click 'Train Model' to start."), empty_fig, empty_fig, html.P(""), html.Div()

    try:
        # Get data with proper chronological sorting
        df = get_all_data(collection_name)
        
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No data available", 
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color="red")
            )
            error_status = dbc.Alert([
                html.H5("❌ Training Gagal", className="alert-heading"),
                html.P("Tidak ada data dalam selection."),
                html.P("Mohon untuk mengecek data kembali.")
            ], color="danger")
            return None, None, error_status, empty_fig, empty_fig, html.P(""), html.Div()

        # CRITICAL FIX: Ensure chronological sorting before time series analysis
        if 'tahun' in df.columns:
            original_order = df['tahun'].tolist()
            df = df.sort_values('tahun', ascending=True).reset_index(drop=True)
            sorted_order = df['tahun'].tolist()
            
            logger.info(f"URUTAN TAHUN SUDAH DIUURTKAN:")
            logger.info(f"  Urutan Asli: {original_order}")
            logger.info(f"  Yang Sudah Diurut: {sorted_order}")
            logger.info(f"  Rentang Tahun: {min(sorted_order)} - {max(sorted_order)}")
        else:
            logger.warning("⚠️ Tidak ada kolom 'tahun' - Analisis Time Series Tidak Dapat Dilakukan")

        # Run the model pipeline with sorted data
        model, accuracy, mape, mse, rmse, y_next, y_pred, y_test, errors = run_model_pipeline(df, model_key)
        
        if model is None:
            error_msg = "Model training failed."
            if errors:
                error_msg += f" Issues: {'; '.join(errors[:3])}"
            
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Training Failed", 
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color="red")
            )
            
            error_status = dbc.Alert([
                html.H5("❌ Model Training Gagal", className="alert-heading"),
                html.P(error_msg),
                html.Ul([html.Li(error) for error in errors[:5]]) if errors else html.P("")
            ], color="danger")
            
            return None, None, error_status, empty_fig, empty_fig, html.P(""), html.Div()

        # SUCCESS: Create comprehensive results display
        
        # 1. Metrics Cards with enhanced styling
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{accuracy:.1f}%", className="text-primary mb-1", style={"font-weight": "bold"}),
                        html.P("Akurasi (R² Score)", className="text-muted mb-0"),
                        html.Small("Tingkat ketepatan prediksi", className="text-secondary")
                    ], className="text-center")
                ], className="h-100 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{mape:.1f}%", className="text-warning mb-1", style={"font-weight": "bold"}),
                        html.P("MAPE", className="text-muted mb-0"),
                        html.Small("Mean Absolute Percentage Error", className="text-secondary")
                    ], className="text-center")
                ], className="h-100 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{rmse:.0f}", className="text-info mb-1", style={"font-weight": "bold"}),
                        html.P("RMSE", className="text-muted mb-0"),
                        html.Small("Root Mean Square Error", className="text-secondary")
                    ], className="text-center")
                ], className="h-100 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{mse:.0f}", className="text-success mb-1", style={"font-weight": "bold"}),
                        html.P("MSE", className="text-muted mb-0"),
                        html.Small("Mean Square Error", className="text-secondary")
                    ], className="text-center")
                ], className="h-100 shadow-sm")
            ], width=3),
        ], className="g-3")

        # 2. Feature Importance Graph
        model_config = MODEL_CONFIGS[model_key]
        feature_names = model_config["features"]
        available_features = [f for f in feature_names if f in df.columns]
        
        if hasattr(model, 'feature_importances_') and len(available_features) > 0:
            # Create feature importance dataframe for better visualization
            importance_df = pd.DataFrame({
                'feature': available_features,
                'importance': model.feature_importances_[:len(available_features)]
            }).sort_values('importance', ascending=True)
            
            importance_fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{imp:.3f}" for imp in importance_df['importance']],
                textposition='auto'
            ))
            
            importance_fig.update_layout(
                title=dict(
                    text=f"Feature Importance - {model_config['name']}",
                    font=dict(size=16, family="Arial Black")
                ),
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(available_features) * 40),
                template="plotly_white",
                margin=dict(l=200, r=50, t=80, b=50)
            )
        else:
            importance_fig = go.Figure()
            importance_fig.add_annotation(
                text="Feature importance not available for this model", 
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=14, color="gray")
            )

        # 3. Enhanced Prediction Visualization
        pred_fig = go.Figure()
        
        # Add actual vs predicted scatter plot with enhanced styling
        pred_fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Actual vs Predicted',
            marker=dict(
                color='rgba(31, 119, 180, 0.7)',
                size=10,
                line=dict(width=1, color='rgba(31, 119, 180, 1)')
            ),
            hovertemplate='<b>Actual:</b> %{x:.0f}<br><b>Predicted:</b> %{y:.0f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        pred_fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction Line',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='Perfect Prediction<extra></extra>'
        ))
        
        # Add R² annotation
        pred_fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'R² = {accuracy/100:.3f}',
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        pred_fig.update_layout(
            title=dict(
                text=f"Actual vs Predicted Crime Rates - {model_config['name']}",
                font=dict(size=16, family="Arial Black")
            ),
            xaxis_title="Actual Crime Count",
            yaxis_title="Predicted Crime Count",
            template="plotly_white",
            height=500,
            hovermode='closest'
        )

        # 4. Future Prediction Display
        latest_year = df['tahun'].max() if 'tahun' in df.columns else "Unknown"
        next_year = latest_year + 1 if isinstance(latest_year, (int, float)) else "Next Period"
        
        if y_next is not None:
            # Calculate prediction confidence based on model accuracy
            confidence_level = "High" if accuracy > 80 else "Medium" if accuracy > 60 else "Low"
            confidence_color = "success" if accuracy > 80 else "warning" if accuracy > 60 else "danger"
            
            future_prediction_content = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{y_next:.0f}", className="text-primary mb-2", style={"font-size": "2.5rem", "font-weight": "bold"}),
                            html.H5(f"Prediksi Kriminalitas {next_year}", className="text-muted mb-3"),
                            dbc.Badge(f"Confidence: {confidence_level}", color=confidence_color, className="mb-2"),
                            html.P(f"Based on {latest_year} data patterns", className="text-secondary mb-0")
                        ], className="text-center")
                    ], className="shadow-sm")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Prediction Details:", className="mb-3"),
                            html.P([
                                html.Strong("Model digunakan: "), 
                                model_config['name']
                            ], className="mb-2"),
                            html.P([
                                html.Strong("Data training: "), 
                                f"{df['tahun'].min()}-{df['tahun'].max()}" if 'tahun' in df.columns else "Multiple periods"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("Feature: "), 
                                f"{len(available_features)} variables"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("Akurasi Model: "), 
                                f"{accuracy:.1f}%"
                            ], className="mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], width=6),
            ], className="g-3")
        else:
            future_prediction_content = dbc.Alert(
                "Unable to generate future prediction. Please check your data and model configuration.",
                color="warning"
            )

        # 5. Success Status with Training Summary
        success_status = dbc.Alert([
            html.H5("✅ Model Training Berhasil Diselesaikan!", className="alert-heading"),
            html.P([
                f"Model: {model_config['name']} | ",
                f"Data: {len(df)} records ({df['tahun'].min()}-{df['tahun'].max()}) | " if 'tahun' in df.columns else f"Data: {len(df)} records | ",
                f"Accuracy: {accuracy:.1f}%"
            ]),
            html.Hr(),
            html.P([
                "Model ini telah dilatih dengan data time-series yang diurutkan secara kronologis.",
                f"Confidence sebesar {confidence_level.lower()} berdasar pada skor R² {accuracy:.1f}%."
            ], className="mb-0")
        ], color="success")

        # Store model data for potential future use
        model_data = {
            "model_type": model_key,
            "accuracy": accuracy,
            "mape": mape,
            "collection": collection_name,
            "features_used": available_features,
            "training_years": f"{df['tahun'].min()}-{df['tahun'].max()}" if 'tahun' in df.columns else "Unknown"
        }

        return (
            model_data,  # model-store
            df.to_dict('records'),  # data-store
            metrics_cards,  # metrics-content
            importance_fig,  # feature-importance-graph
            pred_fig,  # prediction-graph
            future_prediction_content,  # future-prediction
            success_status  # prediction-output-status
        )

    except Exception as e:
        logger.error(f"Critical error in model training: {str(e)}")
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Training Error Occurred", 
            xref="paper", yref="paper", x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        error_status = dbc.Alert([
            html.H5("❌ Error saat training model", className="alert-heading"),
            html.P(f"Terjadi error: {str(e)}"),
            html.P("Cek format data dan coba lagi.")
        ], color="danger")
        
        return None, None, error_status, empty_fig, empty_fig, html.P(""), error_status


# Additional callback for real-time data preview
@callback(
    Output("data-preview", "children"),
    [Input("collection-dropdown-prediction", "value")],
    prevent_initial_call=True
)
def show_data_preview(collection_name):
    """Show a preview of the selected data collection"""
    if not collection_name:
        return html.Div()
    
    try:
        df = get_all_data(collection_name)
        if df.empty:
            return dbc.Alert("Tidak ada data ditemukan.", color="warning")
        
        # Sort by year for preview
        if 'tahun' in df.columns:
            df_preview = df.sort_values('tahun', ascending=True).head(5)
            
            preview_table = dash_table.DataTable(
                data=df_preview.to_dict('records'),
                columns=[{"name": col, "id": col} for col in df_preview.columns],
                style_cell={'textAlign': 'left', 'fontSize': 12},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'backgroundColor': 'rgb(248, 249, 250)'},
                page_size=5
            )
            
            return dbc.Card([
                dbc.CardHeader(html.H6(f"Preview Data - {collection_name}", className="mb-0")),
                dbc.CardBody([
                    html.P(f"Jumlah Records: {len(df)} | Tahun: {df['tahun'].min()}-{df['tahun'].max()}", 
                           className="text-muted mb-3"),
                    preview_table
                ])
            ])
        else:
            return dbc.Alert("Data tidak memuat 'tahun' sehingga tidak bisa dilakukan time-series analysis ", color="warning")
            
    except Exception as e:
        return dbc.Alert(f"Error dalam memuat data: {str(e)}", color="danger")


# Enhanced layout with data preview section (add this to your layout if needed)
data_preview_section = dbc.Row([
    dbc.Col([
        html.Div(id="data-preview", className="mb-4")
    ], width=12)
])


# Utility functions for enhanced error handling and logging

def log_model_performance(model_key, accuracy, mape, mse, rmse, data_shape):
    """Log comprehensive model performance metrics"""
    model_name = MODEL_CONFIGS[model_key]["name"]
    
    logger.info("="*80)
    logger.info(f"Rangkuman Performa Model - {model_name}")
    logger.info("="*80)
    logger.info(f"Dataset: {data_shape}")
    logger.info(f"Skor R² (Akurasi): {accuracy:.2f}%")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"MSE: {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    
    # Performance evaluation
    if accuracy > 60:
        logger.info("✅ PERMFORMA MAKSIMAL")
    elif accuracy > 30:
        logger.info("⚠️ PERFORMA BAIK")
    else:
        logger.warning("❌ PERFORMA KURANG BAIK")
    
    logger.info("="*80)


def validate_time_series_integrity(df):
    """Comprehensive validation of time series data integrity"""
    if df.empty or 'tahun' not in df.columns:
        return False, "No time series data available"
    
    years = df['tahun'].dropna().sort_values()
    
    # Check for consecutive years
    year_gaps = years.diff().dropna()
    large_gaps = year_gaps[year_gaps > 1]
    
    validation_results = {
        "total_years": len(years.unique()),
        "year_range": f"{years.min()}-{years.max()}",
        "has_gaps": len(large_gaps) > 0,
        "gap_years": large_gaps.index.tolist() if len(large_gaps) > 0 else [],
        "consecutive": year_gaps.eq(1).all() if len(year_gaps) > 0 else False
    }
    
    logger.info(f"Time Series Integrity Check: {validation_results}")
    
    if validation_results["has_gaps"]:
        logger.warning(f"⚠️ Time series has gaps at years: {validation_results['gap_years']}")
    
    return True, validation_results


# Error handling wrapper for callbacks
def safe_callback_execution(func):
    """Decorator for safe callback execution with comprehensive error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Return safe default values based on callback outputs
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="An error occurred. Please try again.", 
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False
            )
            
            error_alert = dbc.Alert(
                f"System error: {str(e)}. Please refresh and try again.",
                color="danger"
            )
            
            # Return appropriate number of None values based on expected outputs
            return [None] * 7 if func.__name__ == 'train_model_and_predict' else None
    
    return wrapper


# Apply safe execution to critical callbacks
train_model_and_predict = safe_callback_execution(train_model_and_predict)


# Additional configuration and constants
APP_CONFIG = {
    "MIN_DATA_POINTS": 10,
    "MIN_YEARS_FOR_PREDICTION": 3,
    "MAX_FEATURE_IMPORTANCE_DISPLAY": 10,
    "DEFAULT_CV_SPLITS": 5,
    "OUTLIER_Z_THRESHOLD": 3,
    "LOG_LEVEL": "INFO"
}

# Set up logging configuration
logging.basicConfig(
    level=getattr(logging, APP_CONFIG["LOG_LEVEL"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crime_prediction.log')
    ]
)

logger.info("Crime Prediction Dashboard initialized successfully")
logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")
logger.info(f"App configuration: {APP_CONFIG}")