# pages/data.py
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from server import db  # Import Firestore from server.py

dash.register_page(__name__, path="/data", name="Data")

# Function to fetch all data from Firestore
def get_all_data():
    try:
        # Reference to the collection (adjust your collection name as needed)
        collection_ref = db.collection('table_dummy')
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
    html.H2("Data Management", className="mb-4"),
    
    # Refresh button
    dbc.Button("Refresh Data", id="refresh-button", color="primary", className="mb-3"),
    
    # Data table
    dash_table.DataTable(
        id='firebase-data-table',
        editable=True,
        row_deletable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
    ),
    
    # Add new row button
    dbc.Button("Add New Row", id="add-row-button", color="success", className="mt-3"),
    
    # Save changes button
    dbc.Button("Save Changes", id="save-changes-button", color="warning", className="mt-3 ms-2"),
    
    # Status message
    html.Div(id="status-message", className="mt-3"),
    
    # Store component for data
    dcc.Store(id="table-data-store"),
])

# Callback to load data initially and on refresh
@callback(
    Output("table-data-store", "data"),
    [Input("refresh-button", "n_clicks")],
    prevent_initial_call=False
)
def load_data(n_clicks):
    df = get_all_data()
    return df.to_dict('records') if not df.empty else []

# Callback to update table from store
@callback(
    Output("firebase-data-table", "data"),
    Output("firebase-data-table", "columns"),
    Input("table-data-store", "data")
)
def update_table(data):
    if not data:
        return [], []
    
    # Create columns based on data keys
    columns = [{"name": i, "id": i} for i in data[0].keys()]
    return data, columns

# Callback to add new row
@callback(
    Output("table-data-store", "data", allow_duplicate=True),
    Input("add-row-button", "n_clicks"),
    State("table-data-store", "data"),
    prevent_initial_call=True
)
def add_row(n_clicks, current_data):
    if not current_data:
        # If no data yet, create a table with placeholder columns
        return [{"id": "new_row", "column1": "", "column2": ""}]
    
    # Create a new row with the same columns as existing data
    new_row = {key: "" for key in current_data[0].keys()}
    new_row["id"] = f"new_row_{n_clicks}"  # Temporary ID until saved
    current_data.append(new_row)
    return current_data

# Callback to save changes
@callback(
    Output("status-message", "children"),
    Output("table-data-store", "data", allow_duplicate=True),
    Input("save-changes-button", "n_clicks"),
    State("firebase-data-table", "data"),
    prevent_initial_call=True
)
def save_changes(n_clicks, table_data):
    if not table_data:
        return "No data to save", []
    
    try:
        # For each row in the table
        for row in table_data:
            # Get the document ID
            doc_id = row.pop('id', None)
            
            if doc_id and doc_id.startswith('new_row_'):
                # This is a newly added row
                doc_ref = db.collection('crime_data').document()
                doc_ref.set(row)
            elif doc_id:
                # This is an existing row
                doc_ref = db.collection('crime_data').document(doc_id)
                doc_ref.set(row)
        
        # Refresh data after saving
        df = get_all_data()
        return html.Div("Changes saved successfully!", style={"color": "green"}), df.to_dict('records')
    
    except Exception as e:
        return html.Div(f"Error saving changes: {str(e)}", style={"color": "red"}), table_data