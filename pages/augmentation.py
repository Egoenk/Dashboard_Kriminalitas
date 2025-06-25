import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import json
import base64
import io
from server import db
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import uuid
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/augmentation", name="Data Augmentation")

def get_firestore_collections():
    """Get all collection names from Firestore"""
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        logger.info(f"Found collections: {collection_names}")
        return collection_names
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []

def get_all_collection_data():
    """Get data from ALL Firestore collections and combine them"""
    try:
        collections = get_firestore_collections()
        all_data = []
        
        for collection_name in collections:
            docs = db.collection(collection_name).stream()
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data['id'] = doc.id
                doc_data['source_collection'] = collection_name  # Track source
                doc_data['created_at'] = pd.Timestamp.now()
                all_data.append(doc_data)
        
        logger.info(f"Retrieved {len(all_data)} total documents from {len(collections)} collections")
        return all_data
    except Exception as e:
        logger.error(f"Error getting all collection data: {e}")
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
        logger.info(f"Retrieved {len(data)} documents from {collection_name}")
        return data
    except Exception as e:
        logger.error(f"Error getting data from collection {collection_name}: {e}")
        return []

def upload_to_firestore(df, collection_name):
    """Upload DataFrame to Firestore collection"""
    try:
        # Convert DataFrame to records (list of dictionaries)
        records = df.to_dict('records')
        
        batch = db.batch()
        collection_ref = db.collection(collection_name)
        
        # Add documents in batches (Firestore limit is 500 per batch)
        batch_size = 500
        total_uploaded = 0
        
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            
            for record in batch_records:
                # Remove any NaN values and convert to native types
                clean_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        clean_record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        clean_record[key] = float(value) if isinstance(value, np.floating) else int(value)
                    else:
                        clean_record[key] = value
                
                # Create new document reference
                doc_ref = collection_ref.document()
                batch.set(doc_ref, clean_record)
            
            # Commit the batch
            batch.commit()
            total_uploaded += len(batch_records)
            logger.info(f"Uploaded batch: {total_uploaded}/{len(records)} documents")
            
            # Create new batch for next iteration
            if i + batch_size < len(records):
                batch = db.batch()
        
        logger.info(f"Successfully uploaded {total_uploaded} documents to {collection_name}")
        return True, total_uploaded
        
    except Exception as e:
        logger.error(f"Error uploading to Firestore: {e}")
        return False, 0

def augment_crime_data(df, num_entries, noise_level=0.1):
    """
    Improved augmentation that preserves tahun as numeric and adds proper tracking
    Keeps 'tahun' as base numeric value and adds comprehensive tracking
    """
    if df.empty or num_entries <= 0:
        return df
    
    # Exclude columns that shouldn't be augmented
    exclude_columns = ['source_collection', 'id', 'created_at', 'record_id', 'synthetic_id']
    augmentable_columns = [col for col in df.columns if col not in exclude_columns]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    augmentable_numeric = [col for col in numeric_columns if col not in exclude_columns]
    
    augmented_rows = []
    
    # Keep track of synthetic entries per year and method
    year_synthetic_counts = {}
    
    # Generate the specified number of synthetic entries
    for i in range(num_entries):
        # Choose augmentation method based on iteration
        if i % 3 == 0:
            # Gaussian noise augmentation
            base_row = df.sample(1).copy().iloc[0].to_dict()
            row = base_row.copy()
            
            # Keep original tahun as numeric
            base_tahun = row.get('tahun', 2020)  # Default fallback
            row['tahun'] = base_tahun
            
            # Create tracking for synthetic data
            if base_tahun not in year_synthetic_counts:
                year_synthetic_counts[base_tahun] = {'gaussian': 0, 'interpolation': 0, 'bootstrap': 0}
            year_synthetic_counts[base_tahun]['gaussian'] += 1
            
            # Add unique identifiers
            row['record_id'] = str(uuid.uuid4())[:8]
            row['synthetic_id'] = f"{base_tahun}_gauss_{year_synthetic_counts[base_tahun]['gaussian']}"
            row['is_synthetic'] = True
            row['base_year'] = base_tahun
            
            # Apply Gaussian noise to numeric columns (exclude tahun)
            for col in augmentable_numeric:
                if col in row and pd.notna(row[col]) and col != 'tahun':
                    original_value = float(row[col])
                    noise = np.random.normal(0, abs(original_value) * noise_level)
                    row[col] = max(0, original_value + noise)  # Ensure non-negative
            
            row['source'] = 'gaussian_noise_augmentation'
            row['augmentation_timestamp'] = datetime.now().isoformat()
        
        elif i % 3 == 1:
            # Interpolation-based augmentation
            indices = np.random.choice(df.index, 2, replace=False)
            row1, row2 = df.loc[indices[0]].to_dict(), df.loc[indices[1]].to_dict()
            
            row = {}
            weight = np.random.uniform(0.3, 0.7)
            
            # Use the first row's tahun as base (keep numeric)
            base_tahun = row1.get('tahun', 2020)
            row['tahun'] = base_tahun
            
            # Create tracking
            if base_tahun not in year_synthetic_counts:
                year_synthetic_counts[base_tahun] = {'gaussian': 0, 'interpolation': 0, 'bootstrap': 0}
            year_synthetic_counts[base_tahun]['interpolation'] += 1
            
            # Add unique identifiers
            row['record_id'] = str(uuid.uuid4())[:8]
            row['synthetic_id'] = f"{base_tahun}_interp_{year_synthetic_counts[base_tahun]['interpolation']}"
            row['is_synthetic'] = True
            row['base_year'] = base_tahun
            
            # Interpolate values for numeric columns (exclude tahun)
            for col in augmentable_numeric:
                if col == 'tahun':
                    continue  # Already handled above
                elif col in row1 and col in row2:
                    if pd.notna(row1[col]) and pd.notna(row2[col]):
                        row[col] = float(row1[col]) * weight + float(row2[col]) * (1 - weight)
                    else:
                        row[col] = row1[col] if pd.notna(row1[col]) else row2[col]
            
            # Handle non-numeric columns
            for col in augmentable_columns:
                if col not in augmentable_numeric and col not in ['tahun', 'record_id', 'synthetic_id', 'is_synthetic', 'base_year']:
                    if col in row1 and col in row2:
                        row[col] = row1[col] if np.random.random() > 0.5 else row2[col]
            
            # Copy excluded columns from first row
            for col in exclude_columns:
                if col in row1 and col not in ['record_id', 'synthetic_id']:
                    row[col] = row1[col]
            
            row['source'] = 'interpolation_augmentation'
            row['augmentation_timestamp'] = datetime.now().isoformat()
        
        else:
            # Bootstrap with slight variation
            base_row = df.sample(1).iloc[0].to_dict()
            row = base_row.copy()
            
            # Keep original tahun as numeric
            base_tahun = row.get('tahun', 2020)
            row['tahun'] = base_tahun
            
            # Create tracking
            if base_tahun not in year_synthetic_counts:
                year_synthetic_counts[base_tahun] = {'gaussian': 0, 'interpolation': 0, 'bootstrap': 0}
            year_synthetic_counts[base_tahun]['bootstrap'] += 1
            
            # Add unique identifiers
            row['record_id'] = str(uuid.uuid4())[:8]
            row['synthetic_id'] = f"{base_tahun}_boot_{year_synthetic_counts[base_tahun]['bootstrap']}"
            row['is_synthetic'] = True
            row['base_year'] = base_tahun
            
            # Apply small random variation to numeric columns (exclude tahun)
            for col in augmentable_numeric:
                if col in row and pd.notna(row[col]) and col != 'tahun':
                    original_value = float(row[col])
                    # Add small random variation (5-15%)
                    variation = np.random.uniform(0.95, 1.15)
                    row[col] = max(0, original_value * variation)
            
            row['source'] = 'bootstrap_variation_augmentation'
            row['augmentation_timestamp'] = datetime.now().isoformat()
        
        augmented_rows.append(row)
    
    # Add tracking columns to original data
    df_with_tracking = df.copy()
    df_with_tracking['record_id'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    df_with_tracking['synthetic_id'] = df_with_tracking['tahun'].astype(str) + '_original'
    df_with_tracking['is_synthetic'] = False
    df_with_tracking['base_year'] = df_with_tracking['tahun']
    df_with_tracking['source'] = 'original_data'
    df_with_tracking['augmentation_timestamp'] = datetime.now().isoformat()
    
    # Create DataFrame from augmented rows
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        # Combine with original data
        combined_df = pd.concat([df_with_tracking, augmented_df], ignore_index=True)
        return combined_df
    
    return df_with_tracking

def validate_augmented_data(df):
    """
    Validate that augmented data maintains proper structure
    """
    validation_results = {
        'tahun_is_numeric': pd.api.types.is_numeric_dtype(df['tahun']),
        'unique_record_ids': df['record_id'].nunique() == len(df),
        'has_tracking_columns': all(col in df.columns for col in ['is_synthetic', 'base_year', 'synthetic_id']),
        'year_range_preserved': (df['tahun'].min(), df['tahun'].max()),
        'synthetic_count': df['is_synthetic'].sum() if 'is_synthetic' in df.columns else 0,
        'original_count': (~df['is_synthetic']).sum() if 'is_synthetic' in df.columns else len(df),
        'augmentation_methods': df['source'].value_counts().to_dict() if 'source' in df.columns else {}
    }
    
    return validation_results

def create_augmentation_success_message(df, augmented_df):
    """
    Create enhanced success message with validation
    """
    original_size = len(df)
    augmented_size = len(augmented_df)
    added_rows = augmented_size - original_size
    
    # Validate the augmented data
    validation = validate_augmented_data(augmented_df)
    
    # Count augmentation methods used
    source_counts = augmented_df['source'].value_counts()
    augmentation_methods = source_counts[source_counts.index != 'original_data']
    
    method_breakdown = html.Ul([
        html.Li(f"{method.replace('_', ' ').title()}: {count:,} entries") 
        for method, count in augmentation_methods.items()
    ])
    
    # Create validation status
    validation_status = []
    if validation['tahun_is_numeric']:
        validation_status.append(html.Li("✓ 'tahun' preserved as numeric", className="text-success"))
    if validation['unique_record_ids']:
        validation_status.append(html.Li("✓ All record_ids are unique", className="text-success"))
    if validation['has_tracking_columns']:
        validation_status.append(html.Li("✓ All tracking columns added", className="text-success"))
    
    return dbc.Alert([
        html.H5("Augmentasi Data Selesai!", className="alert-heading"),
        html.P(f"Dataset Original: {original_size:,} rows"),
        html.P(f"Entries yang digenerate: {added_rows:,} rows"),
        html.P(f"Total dataset: {augmented_size:,} rows"),
        html.P(f"Besar perubahan: {added_rows/original_size*100:.1f}%", className="fw-bold"),
        html.Hr(),
        html.P("Metode Augmentasi:", className="fw-bold"),
        method_breakdown,
        html.Hr(),
        html.P("Validasi:", className="fw-bold"),
        html.Ul(validation_status),
        html.P(f"Year range: {validation['year_range_preserved'][0]} - {validation['year_range_preserved'][1]}", 
               className="small text-info")
    ], color="success")

def create_data_loading_success_message(df, source_info):
    """
    Create enhanced data loading success message
    """
    # Show information about columns that will be augmented
    exclude_columns = ['id', 'source_collection', 'created_at', 'record_id', 'synthetic_id', 'is_synthetic', 'base_year', 'augmentation_timestamp']
    augmentable_columns = [col for col in df.columns if col not in exclude_columns]
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col not in exclude_columns]
    
    return dbc.Alert([
        html.H5("Data berhasil dimuat!", className="alert-heading"),
        html.P(f"{source_info}"),
        html.P(f"Total baris: {df.shape[0]} | Total kolom: {df.shape[1]}"),
        html.P(f"Kolom yang bisa di augmentasi: {len(augmentable_columns)} | Kolom Numeric untuk divariasi: {len(numeric_columns)}"),
        html.Hr(),
        html.P("Peningkatan ID Strategy:", className="fw-bold text-success"),
        html.Ul([
            html.Li("'tahun' akan tetap numeric untuk analisis time-series"),
            html.Li("'record_id' unik akan ditambahkan untuk setiap record"),
            html.Li("'synthetic_id' akan menunjukkan metode augmentasi"),
            html.Li("'is_synthetic' flag untuk memudahkan filtering"),
            html.Li("'base_year' untuk tracking tahun asli")
        ], className="small"),
    ], color="success")

layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Data Augmentation untuk Crime Rate Prediction", className="text-center mb-4"),
            html.P("Augmentasi data dari semua data collection untuk membantu model training", 
                   className="text-center text-muted mb-4"),
            html.Hr(),
        ], width=12)
    ]),

    # Section 1: Data Source Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pemilihan Sumber Data", className="mb-0")),
                dbc.CardBody([
                    html.P("Upload atau Pilih data dari firebase", 
                           className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Pilih Data Collection:", className="form-label"),
                            dcc.Dropdown(
                                id="collection-dropdown-augmentation",
                                placeholder="Choose a collection...",
                                className="mb-2"
                            ),
                            dbc.Button("Muat Data", 
                                     id="load-collection-btn", 
                                     color="secondary", 
                                     className="w-100"),
                            dcc.Loading(
                                id="loading-collection",
                                type="default",
                                children=html.Div(id="loading-collection-output")
                            )
                        ], width=6)
                    ]),
                    
                    html.Hr(),
                    
                    # File upload
                    html.Label("atau Upload File CSV :", className="form-label"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    dbc.Button("Muat data Yang Diupload", id="load-file-btn", color="info", className="w-100"),
                    dcc.Loading(
                        id="loading-file",
                        type="default",
                        children=html.Div(id="loading-file-output")
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 2: Augmentation Configuration
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Konfigurasi Augmentasi", className="mb-0")),
                dbc.CardBody([
                    html.P("Atur seberapa banyak entries sintesik untuk di generate, dan parameter noisenya.", 
                        className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Banyaknya Synthetic Entries:", className="form-label"),
                            dbc.Input(
                                id="num-entries",
                                type="number",
                                value=1000,
                                min=1,
                                max=10000,
                                step=1,
                                className="mb-2"
                            ),
                            html.P("Masukan beberapa banyak entries untuk di-generate", 
                                className="text-muted small")
                        ], width=6),
                        dbc.Col([
                            html.Label("Noise Level:", className="form-label"),
                            dcc.Slider(
                                id="noise-level",
                                min=0.05,
                                max=0.3,
                                step=0.05,
                                value=0.1,
                                marks={i/20: f'{i/20:.2f}' for i in range(1, 7)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P("Kontrol varisasi noise di dalam data ", 
                                className="text-muted small")
                        ], width=6)
                    ]),
                    
                    dbc.Alert([
                        html.I(className="fas fa-info-circle me-2"),
                        html.Div([
                            html.P("Augmentation Features:", className="fw-bold mb-2"),
                            html.Ul([
                                html.Li("Kolom 'source' akan ditambahkan untuk lebih memperjelas"),
                                html.Li("Tiga metode augmentasi : Gaussian Noise, Interpolation, and Bootstrap Variation")                        ], className="mb-0")
                        ])
                    ], color="info", className="mt-3"),
                    
                    html.Div([
                        dbc.Button("Generate Data Sintesis", 
                                id="augment-btn", 
                                color="success", 
                                size="lg",
                                className="w-100"),
                        dcc.Loading(
                            id="loading-augment",
                            type="default",
                            children=html.Div(id="loading-augment-output")
                        )
                    ], className="mt-4 text-center")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 4: Export Augmented Data
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Aksi Lanjutan", className="mb-0")),
                dbc.CardBody([
                    html.P("Download data atau upload ke firestore", 
                           className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Download CSV", 
                                     id="download-btn", 
                                     color="info", 
                                     disabled=True, 
                                     size="lg",
                                     className="w-100 mb-2"),
                            dcc.Loading(
                                id="loading-download",
                                type="default",
                                children=html.Div(id="loading-download-output")
                            ),
                            html.P("Download sebagai file CSV", 
                                   className="text-muted small text-center")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Upload ke Firestore", 
                                     id="upload-firestore-btn", 
                                     color="warning", 
                                     disabled=True,
                                     size="lg",
                                     className="w-100 mb-2"),
                            dcc.Loading(
                                id="loading-upload",
                                type="default",
                                children=html.Div(id="loading-upload-output")
                            ),
                            html.P("Simpan data ke dalam firestore", 
                                   className="text-muted small text-center")
                        ], width=6)
                    ]),
                    
                    # Modal for Firestore upload configuration
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Upload to Firestore")),
                        dbc.ModalBody([
                            html.P("Konfigurasi upload anda:"),
                            html.Label("Collection Name:", className="form-label"),
                            dbc.Input(
                                id="new-collection-name",
                                placeholder="Enter new collection name (e.g., augmented_crime_data)",
                                className="mb-3"
                            ),
                            dbc.Checklist(
                                id="upload-options",
                                options=[
                                    {"label": "Buat collection baru", "value": "new"},
                                    {"label": "Timpa collection yang ada", "value": "overwrite"},
                                ],
                                value=["new"],
                                className="mb-3"
                            ),
                            html.Div(id="upload-warning", className="mb-3"),
                            dcc.Loading(
                                id="loading-confirm-upload",
                                type="default",
                                children=html.Div(id="loading-confirm-upload-output")
                            )
                        ]),
                        dbc.ModalFooter([
                            dbc.Button("Cancel", id="cancel-upload", className="me-2", color="secondary"),
                            dbc.Button("Confirm Upload", id="confirm-upload", color="warning")
                        ])
                    ], id="upload-modal", is_open=False),
                    
                    dcc.Download(id="download-augmented-data")
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 5: Status and Logs
    dbc.Row([
        dbc.Col([
            html.Div(id="augmentation-status", className="mb-4")
        ])
    ]),

    # Hidden div to store data
    html.Div(id="stored-data", style={"display": "none"}),
    html.Div(id="augmented-data", style={"display": "none"})
])

# Callback to populate collection dropdown
@callback(
    Output("collection-dropdown-augmentation", "options"),
    Input("collection-dropdown-augmentation", "id")
)
def populate_collections(_):
    collections = get_firestore_collections()
    return [{"label": col, "value": col} for col in collections]

# Updated callback for data loading (all collections) - shows new column handling
@callback(
    [Output("stored-data", "children"),
     Output("augmentation-status", "children"),
     Output("loading-all-output", "children")],
    [Input("load-all-btn", "n_clicks")],
    prevent_initial_call=True
)
def load_all_data(n_clicks):
    if not n_clicks:
        return "", "", ""
    
    try:
        data = get_all_collection_data()
        if data:
            df = pd.DataFrame(data)
            collections_count = len(df['source_collection'].unique()) if 'source_collection' in df.columns else 0
            source_info = f"Loaded from ALL Firestore collections ({collections_count} collections)"
            
            return df.to_json(date_format='iso', orient='split'), create_data_loading_success_message(df, source_info), ""
        else:
            return "", dbc.Alert("No data found in any collection.", color="warning"), ""
    except Exception as e:
        logger.error(f"Error loading all data: {e}")
        return "", dbc.Alert(f"Error loading data: {str(e)}", color="danger"), ""

# Updated callback for data loading (specific collection)
@callback(
    [Output("stored-data", "children", allow_duplicate=True),
     Output("augmentation-status", "children", allow_duplicate=True),
     Output("loading-collection-output", "children")],
    [Input("load-collection-btn", "n_clicks")],
    [State("collection-dropdown-augmentation", "value")],
    prevent_initial_call=True
)
def load_collection_data(n_clicks, collection_name):
    if not n_clicks or not collection_name:
        return "", dbc.Alert("Pilih koleksi terlebih dahulu.", color="warning"), ""
    
    try:
        data = get_collection_data(collection_name)
        if data:
            df = pd.DataFrame(data)
            source_info = f"Data dimuat dari collection: {collection_name}"
            
            return df.to_json(date_format='iso', orient='split'), create_data_loading_success_message(df, source_info), ""
        else:
            return "", dbc.Alert("Tidak ada data didalam collection.", color="warning"), ""
    except Exception as e:
        logger.error(f"Error loading collection data: {e}")
        return "", dbc.Alert(f"Error loading data: {str(e)}", color="danger"), ""

# Updated callback for data loading (file upload)
@callback(
    [Output("stored-data", "children", allow_duplicate=True),
     Output("augmentation-status", "children", allow_duplicate=True),
     Output("loading-file-output", "children")],
    [Input("load-file-btn", "n_clicks")],
    [State("upload-data", "contents"),
     State("upload-data", "filename")],
    prevent_initial_call=True
)
def load_file_data(n_clicks, uploaded_content, filename):
    if not n_clicks or not uploaded_content:
        return "", dbc.Alert("Please upload a file first.", color="warning"), ""
    
    try:
        content_type, content_string = uploaded_content.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            source_info = f"Uploaded file: {filename}"
            
            return df.to_json(date_format='iso', orient='split'), create_data_loading_success_message(df, source_info), ""
        else:
            return "", dbc.Alert("Please upload a CSV file.", color="danger"), ""
    except Exception as e:
        logger.error(f"Error loading file data: {e}")
        return "", dbc.Alert(f"Error loading file: {str(e)}", color="danger"), ""

# Updated callback for augmentation status display
@callback(
    [Output("augmented-data", "children"),
     Output("download-btn", "disabled"),
     Output("upload-firestore-btn", "disabled"),
     Output("augmentation-status", "children", allow_duplicate=True),
     Output("loading-augment-output", "children")],
    [Input("augment-btn", "n_clicks")],
    [State("stored-data", "children"),
     State("num-entries", "value"),
     State("noise-level", "value")],
    prevent_initial_call=True
)
def augment_data(n_clicks, stored_data, num_entries, noise_level):
    if not n_clicks or not stored_data:
        return "", True, True, "", ""
    
    try:
        df = pd.read_json(stored_data, orient='split')
        
        if num_entries <= 0:
            return "", True, True, dbc.Alert("Please enter a valid number of entries (greater than 0).", color="warning"), ""
        
        # Apply augmentation
        augmented_df = augment_crime_data(df, num_entries, noise_level)
        
        return (augmented_df.to_json(date_format='iso', orient='split'), 
                False, 
                False, 
                create_augmentation_success_message(df, augmented_df), 
                "")

    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        return "", True, True, dbc.Alert(f"Error during augmentation: {str(e)}", color="danger"), ""

# Updated visualization callback to handle the new source column
@callback(
    [Output("data-summary", "children"),
     Output("data-distribution-graph", "figure"),
     Output("augmentation-comparison-graph", "figure")],
    [Input("stored-data", "children"),
     Input("augmented-data", "children")]
)
def update_visualizations(stored_data, augmented_data):
    if not stored_data:
        return "", {}, {}
    
    try:
        df = pd.read_json(stored_data, orient='split')
        
        # Data summary
        exclude_columns = ['source_collection', 'id' ,'created_at']
        augmentable_numeric = [col for col in df.select_dtypes(include=[np.number]).columns 
                              if col not in exclude_columns and col != 'tahun']
        
        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Dataset Summary", className="card-title"),
                        html.P(f"Total Rows: {df.shape[0]:,}", className="mb-1"),
                        html.P(f"Total Columns: {df.shape[1]}", className="mb-1"),
                        html.P(f"Numeric Columns for Augmentation: {len(augmentable_numeric)}", className="mb-1"),
                    ])
                ], color="light")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Augmentation Info", className="card-title"),
                        html.P("'tahun' → base ID with iterations", className="mb-1 small text-info"),
                        html.P("New 'source' column → tracks methods", className="mb-1 small text-info"),
                        html.P(f"Will vary: {len(augmentable_numeric)} numeric columns", className="mb-1 small text-success"),
                    ])
                ], color="light")
            ], width=6)
        ])
        
        # Distribution graph
        if len(augmentable_numeric) > 0:
            fig1 = px.histogram(df, x=augmentable_numeric[0], 
                              title=f"Distribution of {augmentable_numeric[0]} (Original Data)")
            fig1.update_layout(height=400)
        else:
            fig1 = go.Figure().add_annotation(text="No augmentable numeric columns found", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Comparison graph with source tracking
        if augmented_data and len(augmentable_numeric) > 0:
            aug_df = pd.read_json(augmented_data, orient='split')
            col = augmentable_numeric[0]
            
            fig2 = go.Figure()
            
            # Original data
            original_data = aug_df[aug_df['source'] == 'original_data']
            fig2.add_trace(go.Histogram(x=original_data[col], name="Original", opacity=0.7, nbinsx=30))
            
            # Augmented data by method
            augmentation_methods = aug_df[aug_df['source'] != 'original_data']['source'].unique()
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            
            for i, method in enumerate(augmentation_methods):
                method_data = aug_df[aug_df['source'] == method]
                method_name = method.replace('_', ' ').title()
                fig2.add_trace(go.Histogram(
                    x=method_data[col], 
                    name=method_name, 
                    opacity=0.6, 
                    nbinsx=30,
                    marker_color=colors[i % len(colors)]
                ))
            
            fig2.update_layout(
                title=f"Data Distribution by Source - {col}",
                barmode='overlay',
                height=400
            )
        else:
            fig2 = go.Figure().add_annotation(text="Generate synthetic data to see comparison", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
    
        return summary, fig1, fig2
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return dbc.Alert(f"Error: {str(e)}", color="danger"), {}, {}
    
# Callback for downloading augmented data
@callback(
    [Output("download-augmented-data", "data"),
     Output("loading-download-output", "children")],
    Input("download-btn", "n_clicks"),
    State("augmented-data", "children"),
    prevent_initial_call=True
)
def download_data(n_clicks, augmented_data):
    if n_clicks and augmented_data:
        df = pd.read_json(augmented_data, orient='split')
        return dcc.send_data_frame(df.to_csv, "augmented_crime_data.csv", index=False), ""

# Callbacks for Firestore upload modal
@callback(
    Output("upload-modal", "is_open"),
    [Input("upload-firestore-btn", "n_clicks"),
     Input("cancel-upload", "n_clicks"),
     Input("confirm-upload", "n_clicks")],
    State("upload-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_upload_modal(upload_clicks, cancel_clicks, confirm_clicks, is_open):
    if upload_clicks or cancel_clicks or confirm_clicks:
        return not is_open
    return is_open

@callback(
    Output("upload-warning", "children"),
    Input("upload-options", "value"),
    State("new-collection-name", "value")
)
def update_upload_warning(options, collection_name):
    if not options:
        return ""
    
    if "overwrite" in options:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Warning: This will overwrite all existing data in the collection!"
        ], color="warning")
    else:
        return dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "A new collection will be created with your augmented data."
        ], color="info")

# Callback for actual Firestore upload
@callback(
    [Output("augmentation-status", "children", allow_duplicate=True),
     Output("loading-confirm-upload-output", "children")],
    Input("confirm-upload", "n_clicks"),
    [State("augmented-data", "children"),
     State("new-collection-name", "value"),
     State("upload-options", "value")],
    prevent_initial_call=True
)

def upload_to_firestore_callback(n_clicks, augmented_data, collection_name, options):
    if not n_clicks or not augmented_data or not collection_name:
        return "", ""
    
    try:
        df = pd.read_json(augmented_data, orient='split')
        
        # Validate collection name
        if not collection_name.strip():
            return dbc.Alert("Please enter a valid collection name.", color="danger"), ""
        
        collection_name = collection_name.strip()
        
        # Check if we need to delete existing collection
        if "overwrite" in options:
            try:
                # Delete all documents in the collection
                docs = db.collection(collection_name).stream()
                batch = db.batch()
                count = 0
                
                for doc in docs:
                    batch.delete(doc.reference)
                    count += 1
                    
                    # Commit batch every 500 operations
                    if count % 500 == 0:
                        batch.commit()
                        batch = db.batch()
                
                # Commit remaining operations
                if count % 500 != 0:
                    batch.commit()
                    
                logger.info(f"Deleted {count} existing documents from {collection_name}")
            except Exception as e:
                logger.warning(f"Error clearing collection (might not exist): {e}")
        
        # Upload the augmented data
        success, uploaded_count = upload_to_firestore(df, collection_name)
        
        if success:
            return dbc.Alert([
                html.H5("Upload Successful!", className="alert-heading"),
                html.P(f"Uploaded {uploaded_count:,} documents to collection: {collection_name}"),
                html.P(f"Your augmented crime dataset is now available in Firestore!")
            ], color="success"), ""
        else:
            return dbc.Alert("Failed to upload data to Firestore. Check logs for details.", color="danger"), ""
            
    except Exception as e:
        logger.error(f"Error in upload callback: {e}")
        return dbc.Alert(f"Error uploading to Firestore: {str(e)}", color="danger"), ""