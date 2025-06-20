# pages/upload.py
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
import base64
import io
from server import db
import logging
from pathlib import Path

dash.register_page(__name__, path="/upload", name="Upload")

logger = logging.getLogger(__name__)

def convert_value(val):
    """Convert string values to appropriate numeric types"""
    if isinstance(val, str):
        val = val.replace(",", ".")  # replace comma with dot
        try:
            # Try converting to float or int
            if "." in val:
                return float(val)
            else:
                return int(val)
        except ValueError:
            return val  # return original string if not a number
    return val  # return original if not a string

def clean_field_name(field_name):
    """Clean field names for better Firestore compatibility"""
    if not isinstance(field_name, str):
        field_name = str(field_name)
    
    # Remove extra whitespace and convert to title case for consistency
    cleaned = field_name.strip()
    
    # Handle common variations of "Tahun" (Year in Indonesian)
    tahun_variations = ['tahun', 'Tahun', 'TAHUN', 'year', 'Year', 'YEAR']
    if cleaned in tahun_variations:
        return 'Tahun'
    
    # Handle ID variations
    id_variations = ['id', 'Id', 'ID', 'iD']
    if cleaned in id_variations:
        return 'ID'
    
    return cleaned

def upload_to_firestore(data, collection_name):
    """Upload data to Firestore collection"""
    try:
        collection_ref = db.collection(collection_name)
        uploaded_count = 0
        errors = []
        
        for i, record in enumerate(data):
            try:
                # Clean field names and convert values
                cleaned_record = {}
                for key, value in record.items():
                    clean_key = clean_field_name(key)
                    clean_value = convert_value(value)
                    # Skip empty/null values
                    if clean_value is not None and clean_value != '':
                        cleaned_record[clean_key] = clean_value
                
                # Check for document ID field (look for year/id fields)
                doc_id = None
                id_field_name = None
                
                # Look for various year/id field patterns (case-insensitive)
                possible_id_fields = ['Tahun', 'tahun', 'TAHUN', 'Year', 'year', 'YEAR', 
                                    'ID', 'id', 'Id', 'iD']
                
                for field_name in cleaned_record.keys():
                    # Check exact matches first
                    if field_name in possible_id_fields:
                        doc_id = str(cleaned_record[field_name])
                        id_field_name = field_name
                        break
                    
                    # Check if field contains year/id patterns
                    field_lower = field_name.lower()
                    if any(pattern in field_lower for pattern in ['tahun', 'year', 'id']):
                        doc_id = str(cleaned_record[field_name])
                        id_field_name = field_name
                        break
                
                # If no ID field found, look in original record keys (before cleaning)
                if not doc_id:
                    for key, value in record.items():
                        key_lower = str(key).lower().strip()
                        if key_lower in ['tahun', 'year', 'id'] or 'tahun' in key_lower or 'year' in key_lower:
                            doc_id = str(convert_value(value))
                            id_field_name = clean_field_name(key)
                            break
                
                if doc_id and id_field_name:
                    # Use the found field as document ID
                    # Remove the ID field from document data to avoid duplication
                    doc_data = {k: v for k, v in cleaned_record.items() if k != id_field_name}
                    logger.info(f"Using document ID: {doc_id} from field: {id_field_name}")
                else:
                    # Auto-generate document ID
                    doc_id = None
                    doc_data = cleaned_record
                    logger.warning(f"No ID field found in record {i+1}, using auto-generated ID")
                
                # Add metadata (only created_at)
                doc_data['created_at'] = pd.Timestamp.now()
                
                if doc_id:
                    collection_ref.document(doc_id).set(doc_data)
                else:
                    collection_ref.add(doc_data)
                
                uploaded_count += 1
                
            except Exception as e:
                errors.append(f"Error uploading record {i+1}: {str(e)}")
                logger.error(f"Error uploading record {i+1}: {e}")
        
        return {
            'success': True,
            'uploaded_count': uploaded_count,
            'total_records': len(data),
            'errors': errors
        }
    
    except Exception as e:
        logger.error(f"Error uploading to Firestore: {e}")
        return {
            'success': False,
            'error': str(e),
            'uploaded_count': 0,
            'total_records': len(data) if data else 0,
            'errors': []
        }

def parse_uploaded_file(contents, filename):
    """Parse uploaded file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if filename.endswith('.json'):
            # JSON file
            data = json.loads(decoded.decode('utf-8'))
            return data, None
        
        elif filename.endswith('.csv'):
            # CSV file with robust parsing
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), 
                sep=None,  # Auto-detect separator
                engine='python',  # More flexible parsing
                skipinitialspace=True,  # Skip whitespace after delimiter
                na_values=['', 'N/A', 'NULL', 'null', 'NaN'],  # Handle missing values
                keep_default_na=True
            )
            
            # Clean column names
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
            
            # Convert to records and handle NaN values
            data = df.where(pd.notnull(df), None).to_dict('records')
            return data, None
        
        elif filename.endswith(('.xlsx', '.xls')):
            # Excel file
            df = pd.read_excel(io.BytesIO(decoded))
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert to records and handle NaN values
            data = df.where(pd.notnull(df), None).to_dict('records')
            return data, None
        
        else:
            return None, f"Unsupported file type: {filename}"
    
    except Exception as e:
        return None, f"Error parsing file {filename}: {str(e)}"

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Upload Dataset", className="text-center mb-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    # Upload section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Upload File", className="mb-0")
                ]),
                dbc.CardBody([
                    html.P("Upload your dataset file (JSON, CSV, or Excel format)", 
                           className="text-muted mb-3"),
                    
                    # File upload component
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-3"),
                            html.Br(),
                            html.P("Drag and Drop or Click to Select Files"),
                            html.P("Supported formats: JSON, CSV, Excel", 
                                   className="text-muted small")
                        ]),
                        style={
                            'width': '100%',
                            'height': '200px',
                            'lineHeight': '200px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '15px',
                            'textAlign': 'center',
                            'backgroundColor': '#f8f9fa',
                            'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    
                    html.Div(id='upload-status', className="mt-3"),
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Configuration section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Dataset Configuration", className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Dataset Name:", className="fw-bold mb-2"),
                            dbc.Input(
                                id="dataset-name-input",
                                placeholder="Enter dataset collection name (e.g., crime_data_2024)",
                                type="text",
                                className="mb-3"
                            ),
                            html.Small("This will be the name of the Firestore collection", 
                                     className="text-muted")
                        ], width=6),
                        dbc.Col([
                            html.Label("Upload Options:", className="fw-bold mb-2"),
                            dbc.Checklist(
                                id="upload-options",
                                options=[
                                    {"label": "Overwrite existing data", "value": "overwrite"},
                                ],
                                value=[],
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Button(
                        "Upload to Firestore",
                        id="upload-to-firestore-btn",
                        color="primary",
                        size="lg",
                        disabled=True,
                        className="w-100"
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Preview section
    dbc.Row([
        dbc.Col([
            html.Div(id="data-preview-section", className="mb-4")
        ], width=12)
    ]),
    
    # Results section
    dbc.Row([
        dbc.Col([
            html.Div(id="upload-results", className="mb-4")
        ], width=12)
    ]),
    
    # Store components
    dcc.Store(id="uploaded-data-store"),
    dcc.Store(id="file-info-store"),
])

# Callback for file upload
@callback(
    [Output('upload-status', 'children'),
     Output('uploaded-data-store', 'data'),
     Output('file-info-store', 'data'),
     Output('data-preview-section', 'children'),
     Output('upload-to-firestore-btn', 'disabled')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filename):
    if contents is None:
        return "", None, None, "", True
    
    try:
        # Parse the uploaded file
        data, error = parse_uploaded_file(contents, filename)
        
        if error:
            status = dbc.Alert(error, color="danger")
            return status, None, None, "", True
        
        if not data:
            status = dbc.Alert("No data found in the uploaded file", color="warning")
            return status, None, None, "", True
        
        # Create status message
        status = dbc.Alert([
            html.H5("File uploaded successfully!", className="alert-heading"),
            html.P(f"File: {filename}"),
            html.P(f"Records found: {len(data)}")
        ], color="success")
        
        # Store file info
        file_info = {
            'filename': filename,
            'record_count': len(data),
            'columns': list(data[0].keys()) if data else []
        }
        
        # Create preview
        preview = create_data_preview(data, filename)
        
        return status, data, file_info, preview, False
        
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        status = dbc.Alert(f"Error processing file: {str(e)}", color="danger")
        return status, None, None, "", True

def create_data_preview(data, filename):
    """Create a preview of the uploaded data"""
    if not data:
        return ""
    
    # Create a preview of first 5 records
    preview_data = data[:5]
    df = pd.DataFrame(preview_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Data Preview", className="mb-0")
        ]),
        dbc.CardBody([
            html.P(f"Showing first 5 records from {filename}", className="text-muted mb-3"),
            html.P(f"Total records: {len(data)} | Columns: {len(df.columns)}", className="fw-bold"),
            html.P(f"Columns: {', '.join(df.columns)}", className="text-muted small mb-3"),
            
            # Data table
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontFamily': 'Arial',
                        'fontSize': '12px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ])
        ])
    ])

# Callback to enable upload button
@callback(
    Output('upload-to-firestore-btn', 'disabled', allow_duplicate=True),
    [Input('dataset-name-input', 'value'),
     Input('uploaded-data-store', 'data')],
    prevent_initial_call=True
)
def enable_upload_button(dataset_name, uploaded_data):
    """Enable upload button when both dataset name and data are available"""
    return not (dataset_name and dataset_name.strip() and uploaded_data)

# Callback for uploading to Firestore
@callback(
    Output('upload-results', 'children'),
    Input('upload-to-firestore-btn', 'n_clicks'),
    [State('uploaded-data-store', 'data'),
     State('dataset-name-input', 'value'),
     State('upload-options', 'value'),
     State('file-info-store', 'data')]
)
def upload_to_firebase(n_clicks, data, dataset_name, options, file_info):
    if not n_clicks or not data or not dataset_name:
        return ""
    
    try:
        # Upload to Firestore
        result = upload_to_firestore(data, dataset_name.strip())
        
        if result['success']:
            # Success message
            success_content = [
                html.H4("Upload Successful!", className="alert-heading"),
                html.P(f"Dataset uploaded to collection: '{dataset_name}'"),
                html.P(f"Records uploaded: {result['uploaded_count']}/{result['total_records']}"),
            ]
            
            if result['errors']:
                success_content.append(html.Hr())
                success_content.append(html.H6("Errors encountered:"))
                for error in result['errors']:
                    success_content.append(html.P(error, className="text-warning small"))
            
            return dbc.Alert(success_content, color="success", dismissable=True)
        
        else:
            # Error message
            error_content = [
                html.H4("Upload Failed!", className="alert-heading"),
                html.P(f"Error: {result.get('error', 'Unknown error')}"),
                html.P(f"Records processed: {result['uploaded_count']}/{result['total_records']}")
            ]
            
            return dbc.Alert(error_content, color="danger", dismissable=True)
    
    except Exception as e:
        logger.error(f"Error in upload callback: {e}")
        return dbc.Alert(f"Unexpected error: {str(e)}", color="danger", dismissable=True)