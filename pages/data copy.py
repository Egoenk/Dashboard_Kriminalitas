# pages/data.py - Enhanced with better formatting and presentation
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from server import db
import logging

dash.register_page(__name__, path="/data", name="Data")

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
            doc_data['id'] = doc.id  # Add document ID
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

def format_columns_for_display(data):
    """Format columns with proper ordering and data types"""
    if not data:
        return [], []
    
    # Get all column names
    all_columns = list(data[0].keys())
    
    # Define column ordering priority
    priority_columns = ['tahun', 'year', 'tanggal', 'date']
    
    # Separate columns by priority
    ordered_columns = []
    remaining_columns = all_columns.copy()
    
    # Add priority columns first (tahun/year at leftmost)
    for priority in priority_columns:
        for col in all_columns:
            if col.lower() == priority and col in remaining_columns:
                ordered_columns.append(col)
                remaining_columns.remove(col)
                break
    
    # Add ID column after priority columns but before others
    if 'id' in remaining_columns:
        ordered_columns.append('id')
        remaining_columns.remove('id')
    
    # Add remaining columns
    ordered_columns.extend(remaining_columns)
    
    # Create column definitions with proper formatting
    columns = []
    for col in ordered_columns:
        # Determine column type and formatting
        sample_values = [row.get(col) for row in data[:5] if row.get(col) is not None]
        
        column_def = {
            "name": col.replace('_', ' ').title(),
            "id": col,
            "deletable": False
        }
        
        # Set editability
        if col == 'id':
            column_def["editable"] = False
        else:
            column_def["editable"] = True
        
        # Determine data type and formatting
        if sample_values:
            # Check if column contains numeric data
            numeric_values = []
            for val in sample_values:
                try:
                    numeric_val = float(val)
                    numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    pass
            
            if len(numeric_values) > 0:
                # Check if values are percentages (between 0 and 1 or contain %)
                is_percentage = (
                    any(0 <= val <= 1 for val in numeric_values if val != 0) or
                    any(str(val).endswith('%') for val in sample_values)
                )
                
                if is_percentage:
                    column_def.update({
                        "type": "numeric",
                        "format": {
                            "specifier": ".2%"
                        }
                    })
                else:
                    # Use thousand separator for regular numbers
                    column_def.update({
                        "type": "numeric",
                        "format": {
                            "specifier": ",.0f"
                        }
                    })
            else:
                column_def["type"] = "text"
        else:
            column_def["type"] = "text"
        
        columns.append(column_def)
    
    # Reorder data to match column order
    reordered_data = []
    for row in data:
        new_row = {}
        for col in ordered_columns:
            new_row[col] = row.get(col, "")
        reordered_data.append(new_row)
    
    return reordered_data, columns

# Layout for the data page
layout = html.Div([
    # Page header
    dbc.Row([
        dbc.Col([
            html.H2("Data Management", className="mb-4"),
        ])
    ]),
    
    # Collection selection and controls row
    dbc.Row([
        dbc.Col([
            html.Label("Pilih Koleksi Data:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='data-collection-dropdown',
                placeholder="Pilih koleksi dari Firestore...",
                className="mb-3",
                value='crime_data'
            ),
        ], width=6),
        dbc.Col([
            html.Label("Actions:", className="fw-bold mb-2"),
            html.Div([
                dbc.Button("Load Data", id="load-data-button", 
                          color="info", className="me-2"),
                dbc.Button("Add New Row", id="add-row-button", 
                          color="success", className="me-2", disabled=True),
                dbc.Button("Save Changes", id="save-changes-button", 
                          color="warning", className="me-2", disabled=True),
            ],ClassName="mb-3")
        ], width=6)
    ]),
    
    # Status and info row
    dbc.Row([
        dbc.Col([
            html.Div(id="data-status-message", className="mb-3"),
            html.Div(id="data-info", className="mb-3"),
        ])
    ]),
    
    # Data table
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-data-table",
                children=[
                    dash_table.DataTable(
                        id='firebase-data-table',
                        editable=True,
                        row_deletable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_action="native",
                        page_size=15,
                        style_table={
                            'overflowX': 'auto',
                            'border': '1px solid #dee2e6',
                            'borderRadius': '0.375rem'
                        },
                        style_cell={
                            'minWidth': '100px', 
                            'width': '150px', 
                            'maxWidth': '300px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'padding': '12px',
                            'fontFamily': 'Arial, sans-serif',
                            'fontSize': '14px'
                        },
                        style_header={
                            'backgroundColor': '#495057',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'border': '1px solid #dee2e6'
                        },
                        style_data={
                            'backgroundColor': 'white',
                            'color': 'black',
                            'border': '1px solid #dee2e6'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa'
                            },
                            # Style for numeric columns
                            {
                                'if': {'column_type': 'numeric'},
                                'textAlign': 'right',
                                'paddingRight': '20px'
                            },
                            # Style for year/tahun columns
                            {
                                'if': {'column_id': ['tahun', 'year']},
                                'backgroundColor': '#e3f2fd',
                                'fontWeight': 'bold'
                            }
                        ],
                        # Custom CSS for delete button positioning
                        css=[
                            {
                                'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table',
                                'rule': 'table-layout: auto;'
                            },
                            {
                                'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner .dash-spreadsheet-inner .row-delete-button',
                                'rule': 'order: -1; margin-right: 10px;'
                            }
                        ],
                        data=[],
                        columns=[]
                    ),
                ],
                type="default",
            )
        ])
    ]),
    
    # Store components for data
    dcc.Store(id="table-data-store"),
    dcc.Store(id="selected-collection-store"),
])

# Callback to populate collection dropdown
@callback(
    Output('data-collection-dropdown', 'options'),
    Input('data-collection-dropdown', 'id')
)
def update_collection_options(_):
    """Update collection dropdown options"""
    try:
        collections = get_firestore_collections()
        options = [{'label': col, 'value': col} for col in collections]
        return options
    except Exception as e:
        logger.error(f"Error updating collection options: {e}")
        return []

# Callback to enable/disable buttons based on collection selection
@callback(
    [Output('load-data-button', 'disabled'),
     Output('selected-collection-store', 'data')],
    Input('data-collection-dropdown', 'value')
)
def enable_buttons(collection_name):
    """Enable buttons when collection is selected"""
    disabled = collection_name is None
    return False, collection_name  # Always enable Load Data button

# Callback to load data from selected collection
@callback(
    [Output("table-data-store", "data"),
     Output("data-status-message", "children"),
     Output("data-info", "children"),
     Output('add-row-button', 'disabled'),
     Output('save-changes-button', 'disabled')],
    Input("load-data-button", "n_clicks"),
    State("selected-collection-store", "data")
)
def load_data(n_clicks, collection_name):
    if not n_clicks or not collection_name:
        return [], "", "", True, True
    
    try:
        data = get_collection_data(collection_name)
        
        if not data:
            status_msg = dbc.Alert(
                f"Koleksi '{collection_name}' tidak memiliki data atau tidak ditemukan.", 
                color="warning"
            )
            return [], status_msg, "", True, True
        
        df = pd.DataFrame(data)
        
        # Create status message
        status_msg = dbc.Alert(
            f"Berhasil memuat {len(df)} baris data dari koleksi '{collection_name}'", 
            color="success"
        )
        
        # Create info about the data
        info_content = [
            html.H5("Informasi Data:", className="mb-2"),
            html.P(f"Jumlah baris: {len(df):,}"),
            html.P(f"Jumlah kolom: {len(df.columns)}"),
            html.P(f"Kolom tersedia: {', '.join(df.columns.tolist())}")
        ]
        info_div = dbc.Card(
            dbc.CardBody(info_content),
            color="light",
            className="mb-3"
        )
        
        return df.to_dict('records'), status_msg, info_div, False, False
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error memuat data: {str(e)}", color="danger")
        return [], error_msg, "", True, True

# Callback to update table from store
@callback(
    [Output("firebase-data-table", "data"),
     Output("firebase-data-table", "columns")],
    Input("table-data-store", "data")
)
def update_table(data):
    if not data:
        # Return empty table
        return [], []
    
    # Format columns with proper ordering and formatting
    formatted_data, columns = format_columns_for_display(data)
    
    return formatted_data, columns

# Callback to add new row
@callback(
    Output("table-data-store", "data", allow_duplicate=True),
    Input("add-row-button", "n_clicks"),
    [State("table-data-store", "data"),
     State("selected-collection-store", "data")],
    prevent_initial_call=True
)
def add_row(n_clicks, current_data, collection_name):
    if not collection_name:
        return current_data or []
    
    if not current_data:
        # If no data yet, create a basic structure
        return [{"id": "new_row_1", "field1": "", "field2": "", "field3": ""}]
    
    # Create a new row with the same columns as existing data
    new_row = {key: "" for key in current_data[0].keys()}
    new_row["id"] = f"new_row_{n_clicks}"  # Temporary ID until saved
    current_data.append(new_row)
    return current_data

# Callback to save changes
@callback(
    [Output("data-status-message", "children", allow_duplicate=True),
     Output("table-data-store", "data", allow_duplicate=True)],
    Input("save-changes-button", "n_clicks"),
    [State("firebase-data-table", "data"),
     State("selected-collection-store", "data")],
    prevent_initial_call=True
)
def save_changes(n_clicks, table_data, collection_name):
    if not table_data or not collection_name:
        return dbc.Alert("Tidak ada data untuk disimpan", color="warning"), []
    
    try:
        saved_count = 0
        updated_count = 0
        
        # For each row in the table
        for row in table_data:
            # Get the document ID
            doc_id = row.get('id', None)
            
            # Create a copy of row data without the id field
            row_data = {k: v for k, v in row.items() if k != 'id'}
            
            # Skip empty rows
            if not any(str(v).strip() for v in row_data.values()):
                continue
            
            # Add metadata
            if doc_id and doc_id.startswith('new_row_'):
                # This is a newly added row
                row_data['created_at'] = pd.Timestamp.now()
                doc_ref = db.collection(collection_name).document()
                doc_ref.set(row_data)
                saved_count += 1
            elif doc_id:
                # This is an existing row
                row_data['updated_at'] = pd.Timestamp.now()
                doc_ref = db.collection(collection_name).document(doc_id)
                doc_ref.set(row_data, merge=True)
                updated_count += 1
        
        # Refresh data after saving
        updated_data = get_collection_data(collection_name)
        df = pd.DataFrame(updated_data) if updated_data else pd.DataFrame()
        
        success_msg = dbc.Alert([
            html.H5("Perubahan berhasil disimpan!", className="alert-heading"),
            html.P(f"Data baru: {saved_count:,} baris"),
            html.P(f"Data diperbarui: {updated_count:,} baris"),
            html.P(f"Total data di '{collection_name}': {len(df):,} baris")
        ], color="success", dismissable=True)
        
        return success_msg, df.to_dict('records') if not df.empty else []
    
    except Exception as e:
        error_msg = dbc.Alert(
            f"Error menyimpan perubahan: {str(e)}", 
            color="danger", 
            dismissable=True
        )
        return error_msg, table_data or []