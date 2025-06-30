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
    Enhanced augmentation that produces exactly the specified column structure
    Keeps 'tahun' as base numeric value and generates the exact columns needed
    """
    if df.empty or num_entries <= 0:
        return df
    
    # Define the exact columns we want in the output
    target_columns = [
        'tahun',
        'source',
        'jumlah_bekerja',
        'jumlah_berpendidikan',
        'jumlah_berpendidikan_diatas_15_tahun',
        'jumlah_kriminalitas_diatas_15_tahun',
        'jumlah_kriminalitas_sebenarnya',
        'jumlah_kriminalitas_sebenarnya_diatas_15_tahun',
        'jumlah_miskin',
        'jumlah_miskin_diatas_15_tahun',
        'jumlah_penduduk_seluruh',
        'jumlah_penduduk_u15',
        'jumlah_tidak_bekerja',
        'jumlah_tidak_berpendidikan',
        'jumlah_tidak_berpendidikan_diatas_15_tahun',
        'jumlah_tidak_miskin',
        'jumlah_tidak_miskin_diatas_15_tahun',
        'kriminalitas_1000orang',
        'ratio_bekerja',
        'ratio_berpendidikan',
        'ratio_penduduk_diatas_15_tahun',
        'ratio_tidak_miskin',
        'created_at'
    ]
    
    # Columns that should be augmented (all numeric columns except tahun and created_at)
    augmentable_columns = [col for col in target_columns 
                          if col not in ['tahun', 'source', 'created_at'] 
                          and col in df.columns]
    
    # Ensure original data has the correct structure
    df_clean = df.copy()
    
    # Add missing columns with default values if they don't exist
    for col in target_columns:
        if col not in df_clean.columns:
            if col == 'source':
                df_clean[col] = 'original_data'
            elif col == 'created_at':
                df_clean[col] = pd.Timestamp.now()
            else:
                # For missing numeric columns, use 0 as default
                df_clean[col] = 0
    
    # Ensure tahun is numeric
    if 'tahun' in df_clean.columns:
        df_clean['tahun'] = pd.to_numeric(df_clean['tahun'], errors='coerce').fillna(2020)
    
    # Mark original data
    df_clean['source'] = 'original_data'
    df_clean['created_at'] = pd.Timestamp.now()
    
    augmented_rows = []
    
    # Generate the specified number of synthetic entries
    for i in range(num_entries):
        # Choose augmentation method based on iteration
        if i % 3 == 0:
            # Gaussian noise augmentation
            base_row = df_clean.sample(1).iloc[0]
            row = {}
            
            # Keep original tahun as base
            row['tahun'] = base_row['tahun']
            row['source'] = 'gaussian_noise_augmentation'
            row['created_at'] = pd.Timestamp.now()
            
            # Apply Gaussian noise to augmentable columns
            for col in augmentable_columns:
                if pd.notna(base_row[col]):
                    original_value = float(base_row[col])
                    noise = np.random.normal(0, abs(original_value) * noise_level)
                    row[col] = max(0, original_value + noise)  # Ensure non-negative
                else:
                    row[col] = 0
        
        elif i % 3 == 1:
            # Interpolation-based augmentation
            indices = np.random.choice(df_clean.index, 2, replace=False)
            row1, row2 = df_clean.loc[indices[0]], df_clean.loc[indices[1]]
            
            row = {}
            weight = np.random.uniform(0.3, 0.7)
            
            # Use the first row's tahun as base
            row['tahun'] = row1['tahun']
            row['source'] = 'interpolation_augmentation'
            row['created_at'] = pd.Timestamp.now()
            
            # Interpolate values for augmentable columns
            for col in augmentable_columns:
                if pd.notna(row1[col]) and pd.notna(row2[col]):
                    row[col] = float(row1[col]) * weight + float(row2[col]) * (1 - weight)
                elif pd.notna(row1[col]):
                    row[col] = float(row1[col])
                elif pd.notna(row2[col]):
                    row[col] = float(row2[col])
                else:
                    row[col] = 0
        
        else:
            # Bootstrap with variation
            base_row = df_clean.sample(1).iloc[0]
            row = {}
            
            # Keep original tahun as base
            row['tahun'] = base_row['tahun']
            row['source'] = 'bootstrap_variation_augmentation'
            row['created_at'] = pd.Timestamp.now()
            
            # Apply variation to augmentable columns
            for col in augmentable_columns:
                if pd.notna(base_row[col]):
                    original_value = float(base_row[col])
                    # Add random variation (5-15%)
                    variation = np.random.uniform(0.85, 1.15)
                    row[col] = max(0, original_value * variation)
                else:
                    row[col] = 0
        
        augmented_rows.append(row)
    
    # Create DataFrame from augmented rows
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Combine with original data
        df_output = df_clean.copy()
        combined_df = pd.concat([df_output, augmented_df], ignore_index=True)
        
        # Ensure we only have the target columns in the correct order
        final_df = combined_df[target_columns].copy()
        
        # Ensure data types are correct
        for col in augmentable_columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
        
        final_df['tahun'] = pd.to_numeric(final_df['tahun'], errors='coerce').fillna(2020)
        
        return final_df
    
    # If no augmented rows, still return in correct format
    return df_clean[target_columns].copy()

def validate_augmented_data(df):
    """
    Enhanced validation for the specific column structure
    """
    expected_columns = [
        'tahun', 'source', 'jumlah_bekerja', 'jumlah_berpendidikan',
        'jumlah_berpendidikan_diatas_15_tahun', 'jumlah_kriminalitas_diatas_15_tahun',
        'jumlah_kriminalitas_sebenarnya', 'jumlah_kriminalitas_sebenarnya_diatas_15_tahun',
        'jumlah_miskin', 'jumlah_miskin_diatas_15_tahun', 'jumlah_penduduk_seluruh',
        'jumlah_penduduk_u15', 'jumlah_tidak_bekerja', 'jumlah_tidak_berpendidikan',
        'jumlah_tidak_berpendidikan_diatas_15_tahun', 'jumlah_tidak_miskin',
        'jumlah_tidak_miskin_diatas_15_tahun', 'kriminalitas_1000orang',
        'ratio_bekerja', 'ratio_berpendidikan', 'ratio_penduduk_diatas_15_tahun',
        'ratio_tidak_miskin', 'created_at'
    ]
    
    validation_results = {
        'has_all_required_columns': all(col in df.columns for col in expected_columns),
        'missing_columns': [col for col in expected_columns if col not in df.columns],
        'extra_columns': [col for col in df.columns if col not in expected_columns],
        'tahun_is_numeric': pd.api.types.is_numeric_dtype(df['tahun']),
        'year_range_preserved': (df['tahun'].min(), df['tahun'].max()) if 'tahun' in df.columns else (None, None),
        'synthetic_count': len(df[df['source'] != 'original_data']) if 'source' in df.columns else 0,
        'original_count': len(df[df['source'] == 'original_data']) if 'source' in df.columns else len(df),
        'augmentation_methods': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
        'total_rows': len(df),
        'column_count': len(df.columns)
    }
    
    return validation_results

def create_augmentation_success_message(df, augmented_df):
    """
    Enhanced success message with specific column validation
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
    if validation['has_all_required_columns']:
        validation_status.append(html.Li("✓ Semua kolom tersedia", className="text-success"))
    else:
        validation_status.append(html.Li(f"⚠ Kolom Hilang: {validation['missing_columns']}", className="text-warning"))
    
    if validation['tahun_is_numeric']:
        validation_status.append(html.Li("✓ 'tahun' di simpan sebagai angka numeric", className="text-success"))
    
    if not validation['extra_columns']:
        validation_status.append(html.Li("✓ Tidak ada extra kolom", className="text-success"))
    else:
        validation_status.append(html.Li(f"⚠ Kolom Ekstra ditemukan: {validation['extra_columns']}", className="text-warning"))
    
    return dbc.Alert([
        html.H5("Augmentasi Data Selesai!", className="alert-heading"),
        html.P(f"Dataset Original: {original_size:,} rows"),
        html.P(f"Entries yang digenerate: {added_rows:,} rows"),
        html.P(f"Total dataset: {augmented_size:,} rows"),
        html.P(f"Besar perubahan: {added_rows/original_size*100:.1f}%", className="fw-bold"),
        html.Hr(),
        html.P("Struktur Dataset:", className="fw-bold"),
        html.P(f"Total Columns: {validation['column_count']} | Expected: 23", className="small text-info"),
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
    Enhanced data loading success message for specific columns
    """
    target_columns = [
        'tahun', 'source', 'jumlah_bekerja', 'jumlah_berpendidikan',
        'jumlah_berpendidikan_diatas_15_tahun', 'jumlah_kriminalitas_diatas_15_tahun',
        'jumlah_kriminalitas_sebenarnya', 'jumlah_kriminalitas_sebenarnya_diatas_15_tahun',
        'jumlah_miskin', 'jumlah_miskin_diatas_15_tahun', 'jumlah_penduduk_seluruh',
        'jumlah_penduduk_u15', 'jumlah_tidak_bekerja', 'jumlah_tidak_berpendidikan',
        'jumlah_tidak_berpendidikan_diatas_15_tahun', 'jumlah_tidak_miskin',
        'jumlah_tidak_miskin_diatas_15_tahun', 'kriminalitas_1000orang',
        'ratio_bekerja', 'ratio_berpendidikan', 'ratio_penduduk_diatas_15_tahun',
        'ratio_tidak_miskin', 'created_at'
    ]
    
    # Check which target columns are present
    present_columns = [col for col in target_columns if col in df.columns]
    missing_columns = [col for col in target_columns if col not in df.columns]
    
    # Check for augmentable numeric columns
    numeric_columns = [col for col in present_columns 
                      if col not in ['tahun', 'source', 'created_at'] 
                      and pd.api.types.is_numeric_dtype(df[col])]
    
    return dbc.Alert([
        html.H5("Data berhasil dimuat!", className="alert-heading"),
        html.P(f"{source_info}"),
        html.P(f"Total baris: {df.shape[0]} | Total kolom: {df.shape[1]}"),
        html.P(f"Target columns present: {len(present_columns)}/23 | Numeric columns for augmentation: {len(numeric_columns)}"),
        html.Hr(),
        html.P("Status kolom:", className="fw-bold"),
        html.Ul([
            html.Li(f"✓ Ada: {len(present_columns)} Kolom", className="text-success"),
            html.Li(f"⚠ Kurang: {len(missing_columns)} Kolom" + (f" - {missing_columns[:3]}..." if missing_columns else ""), 
                   className="text-warning" if missing_columns else "text-success"),
            html.Li(f"📊 Akan diaugment: {len(numeric_columns)}  kolom numeric", className="text-info")
        ], className="small"),
        html.Hr(),
        html.P("Fitur augmentasi:", className="fw-bold text-success"),
        html.Ul([
            html.Li("'tahun' akan dipertahankan sebagai base identifier"),
            html.Li("kolom 'source' akan menunjukkan metode augmentasi"),
            html.Li("Kolom kosong akan diisi dengan nilai default"),
            html.Li("'created_at' akan ditambahkan untuk timestamp"),
        ], className="small"),
    ], color="success")


layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Augmentasi data untuk Prediksi Tingkat Kriminalitas", className="text-center mb-4"),
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
                            dbc.Button("Batalkan", id="cancel-upload", className="me-2", color="secondary"),
                            dbc.Button("Confim Upload", id="confirm-upload", color="warning")
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
            return "", dbc.Alert("Tolong Upload File CSV", color="danger"), ""
    except Exception as e:
        logger.error(f"Error dalam memuat data: {e}")
        return "", dbc.Alert(f"Error dalam memuat file: {str(e)}", color="danger"), ""

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
            return "", True, True, dbc.Alert("Masukan angka diatas 0 (nol).", color="warning"), ""
        
        # Apply augmentation
        augmented_df = augment_crime_data(df, num_entries, noise_level)
        
        return (augmented_df.to_json(date_format='iso', orient='split'), 
                False, 
                False, 
                create_augmentation_success_message(df, augmented_df), 
                "")

    except Exception as e:
        logger.error(f"Error saat proses augmentatasi: {e}")
        return "", True, True, dbc.Alert(f"Error saat proses augmentasi: {str(e)}", color="danger"), ""

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
            "Hal ini akan menimpa seluruh colleciton sama!"
        ], color="warning")
    else:
        return dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Collection baru telath di buat dengan data yang sudah di augmentasi"
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
            return dbc.Alert("Masukan nama collection yang valid", color="danger"), ""
        
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
                    
                logger.info(f"Dihapus {count} dari collection {collection_name}")
            except Exception as e:
                logger.warning(f"Error saat membersihkan colletion (mungkin tidak ada): {e}")
        
        # Upload the augmented data
        success, uploaded_count = upload_to_firestore(df, collection_name)
        
        if success:
            return dbc.Alert([
                html.H5("Upload Berhasil!", className="alert-heading"),
                html.P(f"Sudah Di Upload {uploaded_count:,} ke collection: {collection_name}"),
                html.P(f"Data sudah Tersedia di Firestore!")
            ], color="success"), ""
        else:
            return dbc.Alert("Gagal mengupload data.", color="danger"), ""
            
    except Exception as e:
        logger.error(f"Error di upload callback: {e}")
        return dbc.Alert(f"Error upload ke firestore: {str(e)}", color="danger"), ""