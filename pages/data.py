import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from server import db
import logging

dash.register_page(__name__, path="/data", name="Data")

logger = logging.getLogger(__name__)

def get_firestore_collections():
    """Ambil semua collection di database"""
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        return collection_names
    except Exception as e:
        logger.error(f"Error dalam mengambil 'collections': {e}")
        return []

def get_collection_data(collection_name):
    """Ambil data dari collection dan sort berdasarkan tahun"""
    try:
        docs = db.collection(collection_name).stream()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            data.append(doc_data)
        
        # Sort data berdasarkan tahun (ascending)
        if data:
            # Convert tahun to numeric for proper sorting, handle missing/invalid values
            def get_year_key(item):
                year = item.get('tahun')
                if year is None:
                    return float('inf')  # Put None values at the end
                try:
                    return int(year)
                except (ValueError, TypeError):
                    return float('inf')  # Put invalid values at the end
            
            data.sort(key=get_year_key)
        
        return data
    except Exception as e:
        logger.error(f"Error dalam mengambil data dari collection {collection_name}: {e}")
        return []

def format_columns_for_display(data):
    """Formatting columns with specific order and formatting rules"""
    if not data:
        return [], []
    
    # 1. Define our strict column order
    priority_columns = ['tahun', 'source']
    mandatory_end_column = 'created_at'  
    
    # 2. Get all available columns (excluding internal 'id')
    available_columns = [col for col in data[0].keys() if col != 'id']
    
    # 3. Build the final column order
    final_order = []
    
    # Add priority columns first (if they exist)
    for col in priority_columns:
        if col in available_columns:
            final_order.append(col)
    
    # Add remaining columns (excluding already added and created_at)
    remaining_columns = [
        col for col in available_columns 
        if col not in final_order and col != mandatory_end_column
    ]
    final_order.extend(sorted(remaining_columns))
    
    # Add created_at at the end if it exists
    if mandatory_end_column in available_columns:
        final_order.append(mandatory_end_column)
    
    columns = []
    for col in final_order:
        column_def = {
            "name": col.replace('_', ' ').title(),
            "id": col,
            "deletable": False,
            "editable": True if col != mandatory_end_column else False
        }
        
        # SPECIAL FORMATTING RULES:
        
        # 1. Tahun - plain number with no formatting
        if col == 'tahun':
            column_def.update({
                "type": "numeric",
                "format": {
                    "specifier": ".0f"
                }
            })
        
        # 2. Numeric columns (check first 10 rows for numeric values)
        elif any(isinstance(row.get(col), (int, float)) and not isinstance(row.get(col), bool)
               for row in data[:10] if row.get(col) is not None):
            
            # Check if values have decimals
            has_decimals = any(
                isinstance(row.get(col), float) and not row.get(col).is_integer()
                for row in data[:10] if row.get(col) is not None
            )
            
            if has_decimals:
                # Decimal numbers: 1.234,56 format
                column_def.update({
                    "type": "numeric",
                    "format": {
                        "locale": {
                            "decimal": ",",
                            "group": "."
                        },
                        "specifier": ",.2f"  # 2 decimal places
                    }
                })
            else:
                # Whole numbers: 1.234 format
                column_def.update({
                    "type": "numeric",
                    "format": {
                        "locale": {
                            "group": "."
                        },
                        "specifier": ",.0f"
                    }
                })
        else:
            column_def["type"] = "text"
        
        columns.append(column_def)
    
    # Reorder the data according to our column order
    reordered_data = []
    for row in data:
        new_row = {}
        for col in final_order:
            new_row[col] = row.get(col)
        reordered_data.append(new_row)
    
    return reordered_data, columns

# Confirmation modal for save changes
save_confirm_modal = dbc.Modal(
    [
        dbc.ModalHeader("Konfirmasi Simpan Perubahan"),
        dbc.ModalBody("Anda yakin ingin menyimpan perubahan ke database?"),
        dbc.ModalFooter([
            dbc.Button("Batal", id="cancel-save", className="ms-auto"),
            dbc.Button("Simpan", id="confirm-save", color="primary"),
        ]),
    ],
    id="save-confirm-modal",
    centered=True,
)

# Confirmation modal for delete collection
delete_confirm_modal = dbc.Modal(
    [
        dbc.ModalHeader("Konfirmasi Hapus Koleksi"),
        dbc.ModalBody("Anda yakin ingin menghapus koleksi ini? Data tidak dapat dikembalikan!"),
        dbc.ModalFooter([
            dbc.Button("Batal", id="cancel-delete", className="ms-auto"),
            dbc.Button("Hapus", id="confirm-delete", color="danger"),
        ]),
    ],
    id="delete-confirm-modal",
    centered=True,
)

# Layout for the data page
layout = html.Div([
    # Page header
    dbc.Row([
        dbc.Col([
            html.H2("Managemen Data", className="mb-4"),
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
            html.Label("Aksi:", className="fw-bold mb-2"),
            html.Div([
                dbc.Button("Muat Data", id="load-data-button", 
                          color="info", className="me-2"),
                dbc.Button("Tambah Baris Baru", id="add-row-button", 
                          color="success", className="me-2", disabled=True),
                dbc.Button("Simpan Perubahan", id="save-changes-button", 
                          color="warning", className="me-2", disabled=True),
                dbc.Button("Hapus Koleksi", id="delete-collection-button", 
                          color="danger", className="me-2", disabled=True),
            ], className="mb-3")
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
                            # Style for tahun column - special highlighting
                            {
                                'if': {'column_id': 'tahun'},
                                'backgroundColor': '#e3f2fd',
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            },
                            # Style for source column
                            {
                                'if': {'column_id': 'source'},
                                'backgroundColor': '#f0f8ff',
                                'fontStyle': 'italic'
                            },
                            # Style for created_at column
                            {
                                'if': {'column_id': 'created_at'},
                                'backgroundColor': '#f5f5f5',
                                'fontSize': '12px',
                                'color': '#666'
                            }
                        ],
                        css=[
                            {
                                'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table',
                                'rule': 'table-layout: auto;'
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
    
    # Confirmation modals
    save_confirm_modal,
    delete_confirm_modal,
    
    # Interval component for dynamic table updates
    dcc.Interval(
        id='table-update-interval',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0,
        disabled=True
    )
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
     Output('delete-collection-button', 'disabled'),
     Output('selected-collection-store', 'data'),
     Output('table-update-interval', 'disabled')],
    Input('data-collection-dropdown', 'value')
)
def enable_buttons(collection_name):
    """Button nyala setelah memilih koleksi dan enable interval update"""
    disabled = collection_name is None
    interval_disabled = collection_name is None
    return False, disabled, collection_name, interval_disabled

# Callback to load data from selected collection (including interval updates)
@callback(
    [Output("table-data-store", "data"),
     Output("data-status-message", "children"),
     Output("data-info", "children"),
     Output('add-row-button', 'disabled'),
     Output('save-changes-button', 'disabled')],
    [Input("load-data-button", "n_clicks"),
     Input('table-update-interval', 'n_intervals')],
    State("selected-collection-store", "data")
)
def load_data(n_clicks, n_intervals, collection_name):
    # Only load if button clicked or interval triggered (and collection selected)
    if (not n_clicks and n_intervals == 0) or not collection_name:
        return [], "", "", True, True
    
    try:
        data = get_collection_data(collection_name)
        
        if not data:
            status_msg = dbc.Alert(
                f"Koleksi '{collection_name}' tidak memiliki data atau tidak ditemukan.", 
                color="warning"
            )
            return [], status_msg, "", True, True
        
        # Create status message
        ctx = dash.callback_context
        triggered_by = "interval" if ctx.triggered and "interval" in ctx.triggered[0]['prop_id'] else "button"
        
        if triggered_by == "interval":
            status_msg = dbc.Alert(
                f"Data diperbarui otomatis - {len(data)} baris dari '{collection_name}' (diurutkan berdasarkan tahun)", 
                color="info",
                dismissable=True
            )
        else:
            status_msg = dbc.Alert(
                f"Berhasil memuat {len(data)} baris data dari koleksi '{collection_name}' (diurutkan berdasarkan tahun)", 
                color="success"
            )
        
        # Create info about the data
        columns = list(data[0].keys()) if data else []
        
        # Get year range for display
        years = [row.get('tahun') for row in data if row.get('tahun') is not None]
        year_info = ""
        if years:
            try:
                numeric_years = [int(y) for y in years if str(y).isdigit()]
                if numeric_years:
                    year_info = f"Rentang tahun: {min(numeric_years)} - {max(numeric_years)}"
            except:
                year_info = "Rentang tahun: Data campuran"
        
        info_content = [
            html.H5("Informasi Data:", className="mb-2"),
            html.P(f"Jumlah baris: {len(data):,}"),
            html.P(f"Jumlah kolom: {len(columns)}"),
            html.P(year_info) if year_info else html.Span(),
            html.P(f"Kolom tersedia: {', '.join([col for col in columns if col != 'id'])}")
        ]
        info_div = dbc.Card(
            dbc.CardBody(info_content),
            color="light",
            className="mb-3"
        )
        
        return data, status_msg, info_div, False, False
        
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
        new_row = {"tahun": "", "source": "", "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        return [new_row]
    
    # Create a new row with the same columns as existing data (excluding id)
    new_row = {key: "" for key in current_data[0].keys() if key != 'id'}
    new_row["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add new row and re-sort by tahun
    updated_data = current_data + [new_row]
    
    # Sort the updated data by tahun
    def get_year_key(item):
        year = item.get('tahun')
        if year is None or year == "":
            return float('inf')  # Put empty values at the end
        try:
            return int(year)
        except (ValueError, TypeError):
            return float('inf')  # Put invalid values at the end
    
    updated_data.sort(key=get_year_key)
    
    return updated_data

# Callback to show save confirmation modal
@callback(
    Output("save-confirm-modal", "is_open"),
    [Input("save-changes-button", "n_clicks"),
     Input("confirm-save", "n_clicks"),
     Input("cancel-save", "n_clicks")],
    State("save-confirm-modal", "is_open")
)
def toggle_save_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

# Callback to show delete confirmation modal
@callback(
    Output("delete-confirm-modal", "is_open"),
    [Input("delete-collection-button", "n_clicks"),
     Input("confirm-delete", "n_clicks"),
     Input("cancel-delete", "n_clicks")],
    State("delete-confirm-modal", "is_open")
)
def toggle_delete_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

# Combined callback for both save and delete operations
@callback(
    [Output("data-status-message", "children", allow_duplicate=True),
     Output("table-data-store", "data", allow_duplicate=True),
     Output("data-collection-dropdown", "options", allow_duplicate=True)],
    [Input("confirm-save", "n_clicks"),
     Input("confirm-delete", "n_clicks")],
    [State("firebase-data-table", "data"),
     State("selected-collection-store", "data"),
     State("data-collection-dropdown", "options")],
    prevent_initial_call=True
)
def handle_data_operations(save_clicks, delete_clicks, table_data, collection_name, current_options):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == "confirm-save":
        if not table_data or not collection_name:
            return dbc.Alert("Tidak ada data untuk disimpan", color="warning"), [], current_options
        
        try:
            saved_count = 0
            updated_count = 0
            
            for row in table_data:
                doc_id = None
                if 'id' in row:
                    doc_id = row['id']
                    del row['id']
                
                if not any(str(v).strip() for v in row.values() if v is not None):
                    continue
                
                if doc_id:
                    row['updated_at'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    doc_ref = db.collection(collection_name).document(doc_id)
                    doc_ref.set(row, merge=True)
                    updated_count += 1
                else:
                    row['created_at'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    doc_ref = db.collection(collection_name).document()
                    doc_ref.set(row)
                    saved_count += 1
            
            # Get updated data with proper sorting
            updated_data = get_collection_data(collection_name)
            
            success_msg = dbc.Alert([
                html.H5("Perubahan berhasil disimpan!", className="alert-heading"),
                html.P(f"Data baru: {saved_count:,} baris"),
                html.P(f"Data diperbarui: {updated_count:,} baris"),
                html.P(f"Total data di '{collection_name}': {len(updated_data):,} baris"),
                html.P("Data telah diurutkan berdasarkan tahun")
            ], color="success", dismissable=True)
            
            return success_msg, updated_data, dash.no_update
            
        except Exception as e:
            error_msg = dbc.Alert(
                f"Error menyimpan perubahan: {str(e)}", 
                color="danger", 
                dismissable=True
            )
            return error_msg, table_data or [], current_options
    
    elif triggered_id == "confirm-delete":
        if not collection_name:
            return dbc.Alert("Tidak ada koleksi yang dipilih", color="warning"), [], current_options
        
        try:
            docs = db.collection(collection_name).stream()
            deleted_count = 0
            for doc in docs:
                doc.reference.delete()
                deleted_count += 1
            
            updated_options = [opt for opt in current_options if opt['value'] != collection_name]
            
            success_msg = dbc.Alert([
                html.H5("Koleksi berhasil dihapus!", className="alert-heading"),
                html.P(f"Koleksi '{collection_name}' telah dihapus"),
                html.P(f"Dokumen yang dihapus: {deleted_count:,}")
            ], color="success", dismissable=True)
            
            return success_msg, [], updated_options
        
        except Exception as e:
            error_msg = dbc.Alert(
                f"Error menghapus koleksi: {str(e)}", 
                color="danger", 
                dismissable=True
            )
            return error_msg, [], current_options
    
    return dash.no_update, dash.no_update, dash.no_update