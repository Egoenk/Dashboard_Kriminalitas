from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import logging
from server import app, db
from .model import CrimeRatePredictor
import numpy as np
from traceback import format_exc

logger = logging.getLogger(__name__)

# Column configuration
COLUMNS = {
    "id": "Tahun",
    "Reported_crimeRate": "Tingkat Kriminalitas",
    "educationAPK_SD": "APK SD",
    "educationAPK_SMP": "APK SMP",
    "educationAPK_SMA": "APK SMA",
    "educationAPM_SD": "APM SD",
    "educationAPM_SMP": "APM SMP",
    "educationAPM_SMA": "APM SMA",
    "IPM_SCORE": "IPM",
    "kepadatan_penduduk": "Kepadatan Penduduk",
    "unemploymentRate": "Pengangguran"
}

def fetch_data(db):
    """Fetch and validate data from Firestore with enhanced debugging"""
    try:
        # Log the collection being queried
        logger.info(f"Attempting to fetch data from collection: table_dummy")
        
        # Check if collection exists
        collection_ref = db.collection("table_dummy")
        docs = list(collection_ref.stream())
        
        # Log number of documents found
        logger.info(f"Number of documents found: {len(docs)}")
        
        if not docs:
            logger.warning("No documents found in the collection.")
            return pd.DataFrame()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            logger.debug(f"Raw document data: {doc_data}")  # Print raw document data
            
            # Convert column names and validate
            converted = {"id": doc.id}
            for col in COLUMNS.keys():
                if col == "id": continue
                
                # Try multiple variations of column names
                variations = [
                    col.replace("_", " "),  # e.g., "Reported crimeRate"
                    col,  # original column name
                    col.replace("_", "")  # e.g., "ReportedcrimeRate"
                ]
                
                for variation in variations:
                    value = doc_data.get(variation)
                    if value is not None:
                        converted[col] = value
                        break
                else:
                    logger.warning(f"Could not find value for column {col}")
                    converted[col] = np.nan
            
            data.append(converted)
        
        df = pd.DataFrame(data)
        logger.debug("DataFrame before dropna:")
        logger.debug(df)
        
        df_cleaned = df.dropna()
        logger.debug("Cleaned DataFrame:")
        logger.debug(df_cleaned)
        
        return df_cleaned if not df_cleaned.empty else df
        
    except Exception as e:
        logger.error(f"Detailed Error in fetch_data: {str(e)}")
        logger.error(f"Traceback: {format_exc()}")
        return pd.DataFrame()

def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Dashboard Prediksi Kriminalitas Banyumas", className="text-center mb-4"),
                dcc.Tabs([
                    dcc.Tab(label='Data', children=[
                        html.Div([
                            dbc.Button("Refresh Data", id="refresh-button", className="mb-3"),
                            DataTable(
                                id="data-table",
                                columns=[{"name": COLUMNS[col], "id": col} for col in COLUMNS],
                                data=[],
                                editable=True,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "whiteSpace": "normal",
                                    "height": "auto"
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "lightgray"
                                },
                                filter_action="native",
                                sort_action="native"
                            ),
                            dbc.Button("Simpan Perubahan", id="save-button", color="primary", className="mt-3"),
                            html.Div(id="save-status")
                        ])
                    ]),
                    dcc.Tab(label='Visualisasi', children=[
                        dcc.Graph(id="crime-trend"),
                        dcc.Graph(id="feature-correlation")
                    ]),
                    dcc.Tab(label='Prediksi', children=[
                        dcc.Loading(
                            id="loading-prediction",
                            children=html.Div(id="prediction-output"),
                            type="circle"
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="prediction-graph"), width=8),
                            dbc.Col(html.Div(id="model-metrics"), width=4)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)

# Create the layout
layout = create_layout()

def fetch_data(db):
    """Fetch and validate data from Firestore with enhanced debugging"""
    try:
        # Log the collection being queried
        print(f"Attempting to fetch data from collection: table_dummy")
        
        # Check if collection exists
        collection_ref = db.collection("table_dummy")
        docs = list(collection_ref.stream())
        
        # Log number of documents found
        print(f"Number of documents found: {len(docs)}")
        
        if not docs:
            print("No documents found in the collection.")
            return pd.DataFrame()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            print(f"Raw document data: {doc_data}")  # Print raw document data
            
            # Convert column names and validate
            converted = {"id": doc.id}
            for col in COLUMNS.keys():
                if col == "id": continue
                
                # Try multiple variations of column names
                variations = [
                    col.replace("_", " "),  # e.g., "Reported crimeRate"
                    col,  # original column name
                    col.replace("_", "")  # e.g., "ReportedcrimeRate"
                ]
                
                for variation in variations:
                    value = doc_data.get(variation)
                    if value is not None:
                        converted[col] = value
                        break
                else:
                    print(f"Warning: Could not find value for column {col}")
                    converted[col] = np.nan
            
            data.append(converted)
        
        df = pd.DataFrame(data)
        print("DataFrame before dropna:")
        print(df)
        
        df_cleaned = df.dropna()
        print("Cleaned DataFrame:")
        print(df_cleaned)
        
        return df_cleaned if not df_cleaned.empty else df
        
    except Exception as e:
        print(f"Detailed Error in fetch_data: {str(e)}")
        print(f"Traceback: {format_exc()}")
        return pd.DataFrame()

def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Dashboard Prediksi Kriminalitas Banyumas", className="text-center mb-4"),
                dcc.Tabs([
                    dcc.Tab(label='Data', children=[
                        html.Div([
                            dbc.Button("Refresh Data", id="refresh-button", className="mb-3"),
                            DataTable(
                                id="data-table",
                                columns=[{"name": COLUMNS[col], "id": col} for col in COLUMNS],
                                data=[],
                                editable=True,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "whiteSpace": "normal",
                                    "height": "auto"
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "lightgray"
                                },
                                filter_action="native",
                                sort_action="native"
                            ),
                            dbc.Button("Simpan Perubahan", id="save-button", color="primary", className="mt-3"),
                            html.Div(id="save-status")
                        ])
                    ]),
                    dcc.Tab(label='Visualisasi', children=[
                        dcc.Graph(id="crime-trend"),
                        dcc.Graph(id="feature-correlation")
                    ]),
                    dcc.Tab(label='Prediksi', children=[
                        dcc.Loading(
                            id="loading-prediction",
                            children=html.Div(id="prediction-output"),
                            type="circle"
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="prediction-graph"), width=8),
                            dbc.Col(html.Div(id="model-metrics"), width=4)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)

layout = create_layout()

@callback(
    [Output("save-status", "children"), 
     Output("data-table", "data"),
     Output("crime-trend", "figure")],
    [Input("save-button", "n_clicks"),
     Input("refresh-button", "n_clicks")],
    [State("data-table", "data"), 
     State("data-table", "columns")],
    prevent_initial_call=True
)
def handle_table_updates(save_clicks, refresh_clicks, table_data, table_columns):
    ctx = callback_context
    
    if not ctx.triggered:
        return "", [], px.scatter(title="Silakan refresh data")
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "refresh-button":
        df = fetch_data(db)
        if df.empty:
            return dbc.Alert("Tidak ada data yang tersedia", color="warning"), [], px.scatter(title="Tidak ada data")
        
        fig = px.line(
            df, 
            x="id", 
            y="Reported_crimeRate",
            title="Trend Kriminalitas",
            labels=COLUMNS
        )
        return "", df.to_dict("records"), fig
    
    elif trigger_id == "save-button":
        try:
            df = pd.DataFrame(table_data)
            if df.empty or "id" not in df.columns:
                return dbc.Alert("Data tidak valid untuk disimpan", color="danger"), table_data, px.scatter(title="Data tidak valid")
            
            collection_ref = db.collection("table_dummy")
            batch = db.batch()
            
            for _, row in df.iterrows():
                doc_ref = collection_ref.document(str(row["id"]))
                data = {
                    "Reported crimeRate": int(row["Reported_crimeRate"]),
                    "educationAPK SD": float(row["educationAPK_SD"]),
                    "educationAPK SMP": float(row["educationAPK_SMP"]),
                    "educationAPK SMA": float(row["educationAPK_SMA"]),
                    "educationAPM SD": float(row["educationAPM_SD"]),
                    "educationAPM SMP": float(row["educationAPM_SMP"]),
                    "educationAPM SMA": float(row["educationAPM_SMA"]),
                    "IPM SCORE": float(row["IPM_SCORE"]),
                    "kepadatan penduduk": int(row["kepadatan_penduduk"]),
                    "unemploymentRate": float(row["unemploymentRate"])
                }
                batch.set(doc_ref, data)
            
            batch.commit()
            return dbc.Alert("Data berhasil disimpan!", color="success"), table_data, px.scatter(title="Data tersimpan")
        
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}\n{format_exc()}")
            return dbc.Alert(f"Gagal menyimpan data: {str(e)}", color="danger"), table_data, px.scatter(title="Error")

@callback(
    Output("feature-correlation", "figure"),
    Input("data-table", "data")
)
def update_correlation(data):
    if not data:
        return px.scatter(title="Tidak ada data yang tersedia")
    
    try:
        df = pd.DataFrame(data)
        
        # Select only numeric columns and exclude 'id'
        numeric_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col]) and col != 'id']
        
        if len(numeric_cols) < 2:
            return px.scatter(title="Minimal 2 variabel numerik diperlukan")
        
        # Convert to numeric and drop NA
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_numeric) < 1:
            return px.scatter(title="Tidak ada data valid setelah pembersihan")
        
        # Calculate correlation
        corr_matrix = df_numeric.corr().round(2)
        
        # Create heatmap with proper labels
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Korelasi Antar Variabel",
            labels=dict(
                x="Variabel",
                y="Variabel",
                color="Korelasi"
            ),
            x=[COLUMNS.get(col, col) for col in corr_matrix.columns],
            y=[COLUMNS.get(col, col) for col in corr_matrix.index]
        )
        
        # Improve layout
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0),
            coloraxis_colorbar=dict(
                title="Korelasi",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1 (Negatif)", "-0.5", "0", "0.5", "1 (Positif)"]
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation: {str(e)}\n{format_exc()}")
        return px.scatter(title="Terjadi error saat menghitung korelasi")

@callback(
    [Output("prediction-output", "children"),
     Output("prediction-graph", "figure"),
     Output("model-metrics", "children")],
    Input("data-table", "data")
)
def update_prediction(data):
    if not data:
        return (
            dbc.Alert("Tidak ada data untuk diprediksi", color="warning"),
            px.scatter(title="Tidak ada data"),
            ""
        )
    
    try:
        df = pd.DataFrame(data)
        predictor = CrimeRatePredictor()
        results = predictor.run_pipeline(df)
        
        if not results:
            raise ValueError("Model gagal menghasilkan prediksi")
        
        # Prediction Card
        prediction_card = dbc.Card([
            dbc.CardHeader("Prediksi Tahun Depan"),
            dbc.CardBody([
                html.H4(f"{results['next_year_prediction']:.2f}", className="card-title"),
                html.P("Tingkat Kriminalitas", className="card-text"),
                html.Hr(),
                html.P(f"Interval Kepercayaan: {results['confidence_interval'][0][0]:.2f} - {results['confidence_interval'][1][0]:.2f}")
            ])
        ])
        
        # Prediction Graph
        history_df = df[['id', 'Reported_crimeRate']].copy()
        history_df['type'] = 'Historical'
        
        prediction_df = pd.DataFrame({
            'id': [str(int(df['id'].iloc[-1])) + '+1'],
            'Reported_crimeRate': [results['next_year_prediction']],
            'type': 'Predicted'
        })
        
        combined_df = pd.concat([history_df, prediction_df])
        
        fig = px.line(
            combined_df,
            x='id',
            y='Reported_crimeRate',
            color='type',
            title='Trend Historis dan Prediksi',
            markers=True,
            labels=COLUMNS
        )
        
        # Metrics Card
        metrics_card = dbc.Card([
            dbc.CardHeader("Evaluasi Model"),
            dbc.CardBody([
                dbc.ListGroup([
                    dbc.ListGroupItem(f"R²: {results['metrics']['r2']:.3f}"),
                    dbc.ListGroupItem(f"MAPE: {results['metrics']['mape']:.2f}%"),
                    dbc.ListGroupItem(f"MSE: {results['metrics']['mse']:.2f}"),
                    dbc.ListGroupItem(f"RMSE: {results['metrics']['rmse']:.2f}"),
                    dbc.ListGroupItem(f"MAE: {results['metrics']['mae']:.2f}"),
                ]),
                html.Hr(),
                html.H5("Faktor Paling Berpengaruh"),
                html.P(", ".join(results['feature_importances'].index[:3].tolist()))
            ])
        ])
        
        return prediction_card, fig, metrics_card
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{format_exc()}")
        return (
            dbc.Alert(f"Error dalam prediksi: {str(e)}", color="danger"),
            px.scatter(title="Error dalam prediksi"),
            dbc.Alert("Lihat log server untuk detail error", color="danger")
        )