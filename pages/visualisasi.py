import dash
from dash import html, dcc, callback, Input, Output, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from server import db
import logging

# Register the page
dash.register_page(__name__, path="/visualisasi", name="Visualisasi")

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

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Visualisasi Data Kriminalitas", className="text-center mb-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Pilih Koleksi Data:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='collection-dropdown',
                placeholder="Pilih koleksi dari Firestore...",
                className="mb-3"
            ),
        ], width=4),
        dbc.Col([
            html.Label("Pilih Jenis Grafik:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Line Chart', 'value': 'line'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Histogram', 'value': 'histogram'}
                ],
                value='bar',
                className="mb-3"
            ),
        ], width=4),
        dbc.Col([
            html.Label("Pilih Variabel:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='variable-dropdown',
                placeholder="Pilih variabel untuk divisualisasikan...",
                className="mb-3",
                disabled=True
            ),
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Muat Data", 
                id="load-data-btn", 
                color="primary", 
                className="mb-3",
                disabled=True
            ),
            html.Div(id="data-status", className="mb-3"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-charts",
                children=[html.Div(id="charts-container")],
                type="default",
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Data Table", className="mt-4 mb-3"),
            dcc.Loading(
                id="loading-table",
                children=[html.Div(id="data-table-container")],
                type="default",
            )
        ], width=12)
    ])
], fluid=True)

# Callbacks
@callback(
    Output('collection-dropdown', 'options'),
    Input('collection-dropdown', 'id')
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

@callback(
    [Output('load-data-btn', 'disabled'),
     Output('variable-dropdown', 'disabled')],
    Input('collection-dropdown', 'value')
)
def enable_controls(collection_name):
    """Enable controls when collection is selected"""
    is_disabled = collection_name is None
    return is_disabled, is_disabled

@callback(
    Output('variable-dropdown', 'options'),
    [Input('collection-dropdown', 'value')]
)
def update_variable_options(collection_name):
    """Update variable options based on selected collection"""
    if not collection_name:
        return []
    
    try:
        # Get sample data to determine available columns
        data = get_collection_data(collection_name)
        if not data:
            return []
        
        df = pd.DataFrame(data)
        
        # Variable options (focus on crime-related variables)
        options = []
        
        # Prioritize common crime-related columns
        priority_columns = ['jumlah_kasus', 'total_kasus', 'kasus', 'tahun', 'jenis_kejahatan', 
                           'lokasi', 'kategori', 'kecamatan', 'kelurahan']
        
        for col in df.columns:
            # Skip document ID and non-relevant columns
            if col.lower() in ['id', 'timestamp', 'created_at', 'updated_at']:
                continue
                
            # Add priority columns first
            if col.lower() in priority_columns:
                if df[col].dtype in ['object', 'category']:
                    options.append({'label': f"{col} (Kategorikal)", 'value': col})
                else:
                    options.append({'label': f"{col} (Numerik)", 'value': col})
        
        # Add remaining columns
        for col in df.columns:
            if col not in [opt['value'] for opt in options] and col.lower() not in ['id', 'timestamp']:
                if df[col].dtype in ['object', 'category']:
                    options.append({'label': f"{col} (Kategorikal)", 'value': col})
                else:
                    options.append({'label': f"{col} (Numerik)", 'value': col})
        
        return options
        
    except Exception as e:
        logger.error(f"Error updating variable options: {e}")
        return []

@callback(
    [Output('charts-container', 'children'),
     Output('data-table-container', 'children'),
     Output('data-status', 'children')],
    [Input('load-data-btn', 'n_clicks')],
    [State('collection-dropdown', 'value'),
     State('chart-type-dropdown', 'value'),
     State('variable-dropdown', 'value')]
)
def update_visualizations(n_clicks, collection_name, chart_type, selected_variable):
    """Update visualizations based on selected options"""
    if not n_clicks or not collection_name or not selected_variable:
        return [], [], ""
    
    try:
        # Get data from Firestore
        raw_data = get_collection_data(collection_name)
        
        if not raw_data:
            return [], [], dbc.Alert("Tidak ada data ditemukan dalam koleksi ini.", color="warning")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Clean data - convert 'tahun' to numeric if exists
        if 'tahun' in df.columns:
            df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
            df = df.dropna(subset=['tahun'])
            df['tahun'] = df['tahun'].astype(int)
        
        # Status message
        status_msg = dbc.Alert(
            f"Berhasil memuat {len(df)} baris data dari koleksi '{collection_name}'", 
            color="success"
        )
        
        # Create visualization
        chart = create_chart(df, chart_type, selected_variable, collection_name)
        
        # Create data table
        table = create_data_table(df)
        
        return [chart], table, status_msg
        
    except Exception as e:
        logger.error(f"Error updating visualizations: {e}")
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return [], [], error_msg

def create_chart(df, chart_type, variable, collection_name):
    """Create a single chart based on the selected type and variable"""
    try:
        # Prepare data
        if variable not in df.columns:
            return dbc.Alert(f"Variabel '{variable}' tidak ditemukan dalam data.", color="warning")
        
        # Remove rows with NaN values in the selected variable
        df_clean = df.dropna(subset=[variable])
        
        if df_clean.empty:
            return dbc.Alert(f"Tidak ada data valid untuk variabel '{variable}'.", color="warning")
        
        # Create chart based on type
        fig = None
        
        if chart_type == 'bar':
            if df_clean[variable].dtype in ['object', 'category']:
                # Group by year if available
                if 'tahun' in df_clean.columns:
                    grouped = df_clean.groupby(['tahun', variable]).size().reset_index(name='count')
                    fig = px.bar(grouped, x='tahun', y='count', color=variable,
                               title=f'Kasus Kriminal per Tahun - {variable}',
                               labels={'tahun': 'Tahun', 'count': 'Jumlah Kasus'})
                else:
                    value_counts = df_clean[variable].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Bar Chart - {variable}',
                               labels={'x': variable, 'y': 'Count'})
            else:
                # Numeric bar chart (group by year if available)
                if 'tahun' in df_clean.columns:
                    grouped = df_clean.groupby('tahun')[variable].sum().reset_index()
                    fig = px.bar(grouped, x='tahun', y=variable,
                               title=f'Total {variable} per Tahun',
                               labels={'tahun': 'Tahun', variable: f'Total {variable}'})
                else:
                    fig = px.histogram(df_clean, x=variable, 
                                     title=f'Bar Chart - {variable}')
        
        elif chart_type == 'line':
            if 'tahun' in df_clean.columns:
                if df_clean[variable].dtype in ['object', 'category']:
                    # For categorical variables, count occurrences per year
                    grouped = df_clean.groupby(['tahun', variable]).size().reset_index(name='count')
                    fig = px.line(grouped, x='tahun', y='count', color=variable,
                                title=f'Tren Kasus Kriminal per Tahun - {variable}',
                                labels={'tahun': 'Tahun', 'count': 'Jumlah Kasus'})
                else:
                    # For numeric variables, sum per year
                    grouped = df_clean.groupby('tahun')[variable].sum().reset_index()
                    fig = px.line(grouped, x='tahun', y=variable,
                                title=f'Tren {variable} per Tahun',
                                labels={'tahun': 'Tahun', variable: f'Total {variable}'})
            else:
                return dbc.Alert("Line chart memerlukan kolom 'tahun' untuk tren waktu.", color="warning")
        
        elif chart_type == 'pie':
            if df_clean[variable].dtype in ['object', 'category']:
                value_counts = df_clean[variable].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Distribusi Kasus - {variable}')
            else:
                return dbc.Alert("Pie chart hanya dapat digunakan untuk data kategorikal.", color="warning")
        
        elif chart_type == 'histogram':
            if df_clean[variable].dtype in ['number']:
                fig = px.histogram(df_clean, x=variable,
                                 title=f'Distribusi {variable}')
            else:
                return dbc.Alert("Histogram hanya dapat digunakan untuk data numerik.", color="warning")
        
        if fig is None:
            return dbc.Alert("Tidak dapat membuat grafik dengan tipe yang dipilih.", color="warning")
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=True,
            annotations=[
                dict(
                    text=f"Data: {collection_name}",
                    xref="paper", yref="paper",
                    x=1, y=-0.1, xanchor='right', yanchor='top',
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
            ]
        )
        
        return dcc.Graph(figure=fig, style={'margin-bottom': '20px'})
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return dbc.Alert(f"Error membuat grafik: {str(e)}", color="danger")

def create_data_table(df):
    """Create a data table from DataFrame"""
    try:
        # Limit to first 100 rows for performance
        display_df = df.head(100)
        
        # Select only relevant columns for display
        relevant_cols = ['tahun', 'jenis_kejahatan', 'lokasi', 'jumlah_kasus', 'kecamatan', 'kelurahan']
        display_cols = [col for col in relevant_cols if col in display_df.columns]
        
        # Add any remaining columns that aren't in the relevant list
        other_cols = [col for col in display_df.columns if col not in relevant_cols]
        display_cols.extend(other_cols)
        
        return dash_table.DataTable(
            data=display_df[display_cols].to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_cols],
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
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            page_size=20,
            sort_action="native",
            filter_action="native",
            export_format="xlsx",
            export_headers="display"
        )
    except Exception as e:
        logger.error(f"Error creating data table: {e}")
        return dbc.Alert(f"Error membuat tabel: {str(e)}", color="danger")