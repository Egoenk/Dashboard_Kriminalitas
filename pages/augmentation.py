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
    Augment crime data by generating specified number of synthetic entries
    Excludes 'tahun' column from augmentation
    """
    if df.empty or num_entries <= 0:
        return df
    
    # Exclude 'tahun' and other non-augmentable columns
    exclude_columns = ['tahun', 'id', 'source_collection', 'created_at']
    augmentable_columns = [col for col in df.columns if col not in exclude_columns]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    augmentable_numeric = [col for col in numeric_columns if col not in exclude_columns]
    
    augmented_rows = []
    
    # Generate the specified number of synthetic entries
    for i in range(num_entries):
        # Choose augmentation method based on iteration
        if i % 3 == 0:
            # Gaussian noise augmentation
            row = df.sample(1).copy().iloc[0].to_dict()
            for col in augmentable_numeric:
                if col in row and pd.notna(row[col]):
                    original_value = float(row[col])
                    noise = np.random.normal(0, abs(original_value) * noise_level)
                    row[col] = max(0, original_value + noise)  # Ensure non-negative
        
        elif i % 3 == 1:
            # Interpolation-based augmentation
            indices = np.random.choice(df.index, 2, replace=False)
            row1, row2 = df.loc[indices[0]].to_dict(), df.loc[indices[1]].to_dict()
            
            row = {}
            weight = np.random.uniform(0.3, 0.7)
            
            for col in augmentable_columns:
                if col in augmentable_numeric and col in row1 and col in row2:
                    if pd.notna(row1[col]) and pd.notna(row2[col]):
                        row[col] = float(row1[col]) * weight + float(row2[col]) * (1 - weight)
                    else:
                        row[col] = row1[col] if pd.notna(row1[col]) else row2[col]
                elif col in row1 and col in row2:
                    row[col] = row1[col] if np.random.random() > 0.5 else row2[col]
            
            # Copy non-augmentable columns from first row
            for col in exclude_columns:
                if col in row1:
                    row[col] = row1[col]
        
        else:
            # Bootstrap with slight variation
            base_row = df.sample(1).iloc[0].to_dict()
            row = base_row.copy()
            
            for col in augmentable_numeric:
                if col in row and pd.notna(row[col]):
                    original_value = float(row[col])
                    # Add small random variation (5-15%)
                    variation = np.random.uniform(0.95, 1.15)
                    row[col] = max(0, original_value * variation)
        
        augmented_rows.append(row)
    
    # Create DataFrame from augmented rows
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        # Combine with original data
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        return combined_df
    
    return df

layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Enhanced Data Augmentation for Crime Rate Prediction", className="text-center mb-4"),
            html.P("Generate synthetic crime data entries from all available collections to enhance model training", 
                   className="text-center text-muted mb-4"),
            html.Hr(),
        ], width=12)
    ]),

    # Section 1: Data Source Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Data Source Selection", className="mb-0")),
                dbc.CardBody([
                    html.P("Choose to load all data from Firestore collections or upload a specific CSV file.", 
                           className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Load All Collections Data", 
                                     id="load-all-btn", 
                                     color="primary", 
                                     size="lg",
                                     className="w-100 mb-3"),
                            dcc.Loading(
                                id="loading-all",
                                type="default",
                                children=html.Div(id="loading-all-output")
                            ),
                            html.P("This will load and combine data from all available Firestore collections", 
                                   className="text-muted small text-center")
                        ], width=6),
                        dbc.Col([
                            html.Label("Or Select Specific Collection:", className="form-label"),
                            dcc.Dropdown(
                                id="collection-dropdown-augmentation",
                                placeholder="Choose a collection...",
                                className="mb-2"
                            ),
                            dbc.Button("Load Selected Collection", 
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
                    html.Label("Or Upload CSV File:", className="form-label"),
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
                    dbc.Button("Load Uploaded File", id="load-file-btn", color="info", className="w-100"),
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
                dbc.CardHeader(html.H4("Augmentation Configuration", className="mb-0")),
                dbc.CardBody([
                    html.P("Configure how many synthetic entries to generate and noise parameters.", 
                           className="text-muted mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Number of Synthetic Entries:", className="form-label"),
                            dbc.Input(
                                id="num-entries",
                                type="number",
                                value=1000,
                                min=1,
                                max=10000,
                                step=1,
                                className="mb-2"
                            ),
                            html.P("Enter how many synthetic data entries to generate", 
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
                            html.P("Controls the amount of variation in synthetic data", 
                                   className="text-muted small")
                        ], width=6)
                    ]),
                    
                    dbc.Alert([
                        html.I(className="fas fa-info-circle me-2"),
                        "Note: The 'tahun' column will be preserved and not modified during augmentation."
                    ], color="info", className="mt-3"),
                    
                    html.Div([
                        dbc.Button("Generate Synthetic Data", 
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

    # Section 3: Data Overview and Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Data Overview & Visualization", className="mb-0")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-summary",
                        type="default",
                        children=html.Div(id="data-summary", className="mb-3")
                    ),
                    dcc.Loading(
                        id="loading-distribution",
                        type="default",
                        children=dcc.Graph(id="data-distribution-graph")
                    ),
                    dcc.Loading(
                        id="loading-comparison",
                        type="default",
                        children=dcc.Graph(id="augmentation-comparison-graph")
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 4: Export Augmented Data
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Export Augmented Data", className="mb-0")),
                dbc.CardBody([
                    html.P("Download your augmented dataset or upload it to Firestore for future use.", 
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
                            html.P("Download augmented data as CSV file", 
                                   className="text-muted small text-center")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Upload to Firestore", 
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
                            html.P("Save augmented data to new Firestore collection", 
                                   className="text-muted small text-center")
                        ], width=6)
                    ]),
                    
                    # Modal for Firestore upload configuration
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Upload to Firestore")),
                        dbc.ModalBody([
                            html.P("Configure your Firestore upload:"),
                            html.Label("Collection Name:", className="form-label"),
                            dbc.Input(
                                id="new-collection-name",
                                placeholder="Enter new collection name (e.g., augmented_crime_data)",
                                className="mb-3"
                            ),
                            dbc.Checklist(
                                id="upload-options",
                                options=[
                                    {"label": "Create new collection", "value": "new"},
                                    {"label": "Overwrite existing collection", "value": "overwrite"},
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

# Callback to handle data loading (all collections)
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
            
            # Show information about columns that will be augmented
            exclude_columns = ['tahun', 'id', 'source_collection']
            augmentable_columns = [col for col in df.columns if col not in exclude_columns]
            numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_columns]
            
            return df.to_json(date_format='iso', orient='split'), dbc.Alert([
                html.H5("Data Loaded Successfully!", className="alert-heading"),
                html.P(f"{source_info}"),
                html.P(f"Total rows: {df.shape[0]} | Total columns: {df.shape[1]}"),
                html.P(f"Augmentable columns: {len(augmentable_columns)} | Numeric columns: {len(numeric_columns)}"),
                html.P(f"Excluded from augmentation: {', '.join(exclude_columns)}", className="small text-muted")
            ], color="success"), ""
        else:
            return "", dbc.Alert("No data found in any collection.", color="warning"), ""
    except Exception as e:
        logger.error(f"Error loading all data: {e}")
        return "", dbc.Alert(f"Error loading data: {str(e)}", color="danger"), ""

# Callback to handle data loading (specific collection)
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
        return "", dbc.Alert("Please select a collection first.", color="warning"), ""
    
    try:
        data = get_collection_data(collection_name)
        if data:
            df = pd.DataFrame(data)
            source_info = f"Loaded from Firestore collection: {collection_name}"
            
            # Show information about columns that will be augmented
            exclude_columns = ['tahun', 'id', 'source_collection']
            augmentable_columns = [col for col in df.columns if col not in exclude_columns]
            numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_columns]
            
            return df.to_json(date_format='iso', orient='split'), dbc.Alert([
                html.H5("Data Loaded Successfully!", className="alert-heading"),
                html.P(f"{source_info}"),
                html.P(f"Total rows: {df.shape[0]} | Total columns: {df.shape[1]}"),
                html.P(f"Augmentable columns: {len(augmentable_columns)} | Numeric columns: {len(numeric_columns)}"),
                html.P(f"Excluded from augmentation: {', '.join(exclude_columns)}", className="small text-muted")
            ], color="success"), ""
        else:
            return "", dbc.Alert("No data found in selected collection.", color="warning"), ""
    except Exception as e:
        logger.error(f"Error loading collection data: {e}")
        return "", dbc.Alert(f"Error loading data: {str(e)}", color="danger"), ""

# Callback to handle data loading (file upload)
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
            
            # Show information about columns that will be augmented
            exclude_columns = ['tahun', 'id', 'source_collection']
            augmentable_columns = [col for col in df.columns if col not in exclude_columns]
            numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_columns]
            
            return df.to_json(date_format='iso', orient='split'), dbc.Alert([
                html.H5("Data Loaded Successfully!", className="alert-heading"),
                html.P(f"{source_info}"),
                html.P(f"Total rows: {df.shape[0]} | Total columns: {df.shape[1]}"),
                html.P(f"Augmentable columns: {len(augmentable_columns)} | Numeric columns: {len(numeric_columns)}"),
                html.P(f"Excluded from augmentation: {', '.join(exclude_columns)}", className="small text-muted")
            ], color="success"), ""
        else:
            return "", dbc.Alert("Please upload a CSV file.", color="danger"), ""
    except Exception as e:
        logger.error(f"Error loading file data: {e}")
        return "", dbc.Alert(f"Error loading file: {str(e)}", color="danger"), ""

# Callback for data augmentation
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
        
        original_size = len(df)
        augmented_size = len(augmented_df)
        added_rows = augmented_size - original_size
        
        return (augmented_df.to_json(date_format='iso', orient='split'), 
                False, 
                False, 
                dbc.Alert([
                    html.H5("Data Augmentation Complete!", className="alert-heading"),
                    html.P(f"Original dataset: {original_size:,} rows"),
                    html.P(f"Generated synthetic entries: {added_rows:,} rows"),
                    html.P(f"Total augmented dataset: {augmented_size:,} rows"),
                    html.P(f"Increase: {added_rows/original_size*100:.1f}%", className="fw-bold")
                ], color="success"), 
                "")

    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        return "", True, True, dbc.Alert(f"Error during augmentation: {str(e)}", color="danger"), ""

# Callback for data visualization
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
        exclude_columns = ['tahun', 'id', 'source_collection']
        augmentable_numeric = [col for col in df.select_dtypes(include=[np.number]).columns 
                              if col not in exclude_columns]
        
        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Dataset Summary", className="card-title"),
                        html.P(f"Total Rows: {df.shape[0]:,}", className="mb-1"),
                        html.P(f"Total Columns: {df.shape[1]}", className="mb-1"),
                        html.P(f"Augmentable Numeric Columns: {len(augmentable_numeric)}", className="mb-1"),
                    ])
                ], color="light")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Column Information", className="card-title"),
                        html.P("Excluded from augmentation:", className="mb-1 small text-muted"),
                        html.P("tahun, id, source_collection", className="mb-1 small"),
                        html.P(f"Will augment: {len(augmentable_numeric)} numeric columns", className="mb-1 small text-success"),
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
        
        # Comparison graph
        if augmented_data and len(augmentable_numeric) > 0:
            aug_df = pd.read_json(augmented_data, orient='split')
            col = augmentable_numeric[0]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df[col], name="Original", opacity=0.7, nbinsx=30))
            fig2.add_trace(go.Histogram(x=aug_df[col], name="Augmented", opacity=0.7, nbinsx=30))
            fig2.update_layout(
                title=f"Original vs Augmented Data Distribution - {col}",
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