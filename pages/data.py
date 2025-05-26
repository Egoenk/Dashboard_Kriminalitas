# pages/data.py - Updated with authentication
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from server import db
from auth import auth_manager

dash.register_page(__name__, path="/data", name="Data")

# Function to fetch all data from Firestore
def get_all_data():
    try:
        # Reference to the collection (adjust your collection name as needed)
        collection_ref = db.collection('crime_data')
        docs = collection_ref.stream()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Add document ID to the data
            data.append(doc_data)
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Layout for the data page
layout = html.Div([
    dcc.Store(id="auth-check", data={"check": True}),
    html.Div(id="data-page-content")
])

# Authentication check callback
@callback(
    Output("data-page-content", "children"),
    [Input("auth-check", "data")],
    [State("login-status-store", "data")]
)
def check_auth_and_display(auth_check, login_data):
    # Verify authentication
    if not login_data or not login_data.get("logged_in"):
        return html.Div([
            dcc.Location(pathname="/login", id="redirect-to-login")
        ])
    
    session_id = login_data.get("session_id")
    is_valid, session_data = auth_manager.validate_session(session_id)
    
    if not is_valid:
        return html.Div([
            dcc.Location(pathname="/login", id="redirect-to-login")
        ])
    
    # User is authenticated, show the data page
    username = session_data.get("username")
    role = session_data.get("role")
    
    return html.Div([
        # Page header with user info
        dbc.Row([
            dbc.Col([
                html.H2("Data Management", className="mb-1"),
                html.P(f"Logged in as: {username} ({role})", 
                       className="text-muted mb-4")
            ])
        ]),
        
        # Action buttons row
        dbc.Row([
            dbc.Col([
                dbc.Button("Refresh Data", id="refresh-button", 
                          color="primary", className="me-2"),
                dbc.Button("Add New Row", id="add-row-button", 
                          color="success", className="me-2"),
                dbc.Button("Save Changes", id="save-changes-button", 
                          color="warning", className="me-2"),
            ], className="mb-3")
        ]),
        
        # Data table
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='firebase-data-table',
                    editable=True,
                    row_deletable=True if role == 'admin' else False,
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
                        'maxWidth': '200px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'padding': '10px',
                        'fontFamily': 'Arial, sans-serif'
                    },
                    style_header={
                        'backgroundColor': '#495057',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data={
                        'backgroundColor': 'white',
                        'color': 'black',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                ),
            ])
        ]),
        
        # Status message
        dbc.Row([
            dbc.Col([
                html.Div(id="status-message", className="mt-3")
            ])
        ]),
        
        # Store component for data
        dcc.Store(id="table-data-store"),
    ])

# Callback to load data initially and on refresh
@callback(
    Output("table-data-store", "data"),
    [Input("refresh-button", "n_clicks")],
    [State("login-status-store", "data")],
    prevent_initial_call=False
)
def load_data(n_clicks, login_data):
    # Check authentication
    if not login_data or not login_data.get("logged_in"):
        return []
    
    session_id = login_data.get("session_id")
    is_valid, session_data = auth_manager.validate_session(session_id)
    
    if not is_valid:
        return []
    
    df = get_all_data()
    return df.to_dict('records') if not df.empty else []

# Callback to update table from store
@callback(
    [Output("firebase-data-table", "data"),
     Output("firebase-data-table", "columns")],
    Input("table-data-store", "data")
)
def update_table(data):
    if not data:
        # Return empty table with basic structure
        return [], [
            {"name": "Date", "id": "date"},
            {"name": "Crime Type", "id": "crime_type"},
            {"name": "District", "id": "district"},
            {"name": "Count", "id": "count"}
        ]
    
    # Create columns based on data keys, excluding the id column from display
    columns = [{"name": key.replace('_', ' ').title(), "id": key} 
              for key in data[0].keys() if key != 'id']
    return data, columns

# Callback to add new row
@callback(
    Output("table-data-store", "data", allow_duplicate=True),
    Input("add-row-button", "n_clicks"),
    [State("table-data-store", "data"),
     State("login-status-store", "data")],
    prevent_initial_call=True
)
def add_row(n_clicks, current_data, login_data):
    # Check authentication
    if not login_data or not login_data.get("logged_in"):
        return current_data or []
    
    session_id = login_data.get("session_id")
    is_valid, session_data = auth_manager.validate_session(session_id)
    
    if not is_valid:
        return current_data or []
    
    if not current_data:
        # If no data yet, create a table with placeholder columns
        return [{"id": "new_row", "date": "", "crime_type": "", "district": "", "count": ""}]
    
    # Create a new row with the same columns as existing data
    new_row = {key: "" for key in current_data[0].keys()}
    new_row["id"] = f"new_row_{n_clicks}"  # Temporary ID until saved
    current_data.append(new_row)
    return current_data

# Callback to save changes
@callback(
    [Output("status-message", "children"),
     Output("table-data-store", "data", allow_duplicate=True)],
    Input("save-changes-button", "n_clicks"),
    [State("firebase-data-table", "data"),
     State("login-status-store", "data")],
    prevent_initial_call=True
)
def save_changes(n_clicks, table_data, login_data):
    # Check authentication
    if not login_data or not login_data.get("logged_in"):
        return dbc.Alert("Authentication required", color="danger"), []
    
    session_id = login_data.get("session_id")
    is_valid, session_data = auth_manager.validate_session(session_id)
    
    if not is_valid:
        return dbc.Alert("Session expired", color="danger"), []
    
    if not table_data:
        return dbc.Alert("No data to save", color="warning"), []
    
    try:
        username = session_data.get("username")
        
        # For each row in the table
        for row in table_data:
            # Get the document ID
            doc_id = row.get('id', None)
            
            # Create a copy of row data without the id field
            row_data = {k: v for k, v in row.items() if k != 'id'}
            
            # Add metadata
            if doc_id and doc_id.startswith('new_row_'):
                # This is a newly added row
                row_data['created_by'] = username
                row_data['created_at'] = pd.Timestamp.now()
                doc_ref = db.collection('crime_data').document()
                doc_ref.set(row_data)
            elif doc_id:
                # This is an existing row
                row_data['updated_by'] = username
                row_data['updated_at'] = pd.Timestamp.now()
                doc_ref = db.collection('crime_data').document(doc_id)
                doc_ref.set(row_data, merge=True)
        
        # Refresh data after saving
        df = get_all_data()
        success_msg = dbc.Alert(
            f"Changes saved successfully by {username}!", 
            color="success", 
            dismissable=True
        )
        return success_msg, df.to_dict('records')
    
    except Exception as e:
        error_msg = dbc.Alert(
            f"Error saving changes: {str(e)}", 
            color="danger", 
            dismissable=True
        )
        return error_msg, table_data or []