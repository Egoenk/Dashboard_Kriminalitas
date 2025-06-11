import dash
from dash import html, dcc, callback, Input, Output, dash_table
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
        ], width=6),
        dbc.Col([
            html.Label("Pilih Jenis Visualisasi:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Tren Kriminalitas per Tahun', 'value': 'trend'},
                    {'label': 'Korelasi Variabel', 'value': 'correlation'},
                    {'label': 'Distribusi Data', 'value': 'distribution'},
                    {'label': 'Perbandingan Rasio', 'value': 'ratio_comparison'},
                    {'label': 'Scatter Plot Analysis', 'value': 'scatter'}
                ],
                value='trend',
                className="mb-3"
            ),
        ], width=6)
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
    Output('load-data-btn', 'disabled'),
    Input('collection-dropdown', 'value')
)
def enable_load_button(collection_name):
    """Enable load button when collection is selected"""
    return collection_name is None

@callback(
    [Output('charts-container', 'children'),
     Output('data-table-container', 'children'),
     Output('data-status', 'children')],
    [Input('load-data-btn', 'n_clicks')],
    [Input('collection-dropdown', 'value'),
     Input('chart-type-dropdown', 'value')]
)
def update_visualizations(n_clicks, collection_name, chart_type):
    """Update visualizations based on selected collection and chart type"""
    if not n_clicks or not collection_name:
        return [], [], ""
    
    try:
        # Get data from Firestore
        data = get_collection_data(collection_name)
        
        if not data:
            return [], [], dbc.Alert("Tidak ada data ditemukan dalam koleksi ini.", color="warning")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Status message
        status_msg = dbc.Alert(
            f"Berhasil memuat {len(df)} baris data dari koleksi '{collection_name}'", 
            color="success"
        )
        
        # Create visualizations based on chart type
        charts = create_charts(df, chart_type)
        
        # Create data table
        table = create_data_table(df)
        
        return charts, table, status_msg
        
    except Exception as e:
        logger.error(f"Error updating visualizations: {e}")
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return [], [], error_msg

def create_charts(df, chart_type):
    """Create charts based on the selected type"""
    charts = []
    
    if chart_type == 'trend' and 'Tahun' in df.columns:
        # Crime trend over years
        if 'kriminalitas_1000orang' in df.columns:
            fig1 = px.line(df, x='Tahun', y='kriminalitas_1000orang', 
                          title='Tren Kriminalitas per 1000 Orang',
                          markers=True)
            fig1.update_layout(height=400)
            charts.append(dcc.Graph(figure=fig1))
        
        # Population trend
        if 'jumlah_penduduk_seluruh' in df.columns:
            fig2 = px.line(df, x='Tahun', y='jumlah_penduduk_seluruh',
                          title='Tren Jumlah Penduduk',
                          markers=True)
            fig2.update_layout(height=400)
            charts.append(dcc.Graph(figure=fig2))
    
    elif chart_type == 'correlation':
        # Correlation matrix of numeric variables
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title='Matriks Korelasi Variabel',
                           color_continuous_scale='RdBu',
                           aspect='auto')
            fig.update_layout(height=600)
            charts.append(dcc.Graph(figure=fig))
    
    elif chart_type == 'distribution':
        # Distribution of key variables
        key_vars = ['kriminalitas_1000orang', 'jumlah_penduduk_seluruh', 'jumlah_miskin']
        for var in key_vars:
            if var in df.columns:
                fig = px.histogram(df, x=var, title=f'Distribusi {var}')
                fig.update_layout(height=300)
                charts.append(dcc.Graph(figure=fig))
    
    elif chart_type == 'ratio_comparison':
        # Compare different ratios
        ratio_cols = [col for col in df.columns if 'ratio' in col.lower()]
        if ratio_cols and 'Tahun' in df.columns:
            for ratio_col in ratio_cols:
                fig = px.bar(df, x='Tahun', y=ratio_col, 
                           title=f'Perbandingan {ratio_col}')
                fig.update_layout(height=400)
                charts.append(dcc.Graph(figure=fig))
    
    elif chart_type == 'scatter':
        # Scatter plots for analysis
        if 'kriminalitas_1000orang' in df.columns:
            # Crime vs Employment
            if 'jumlah_bekerja' in df.columns:
                fig1 = px.scatter(df, x='jumlah_bekerja', y='kriminalitas_1000orang',
                                title='Kriminalitas vs Jumlah Bekerja',
                                trendline='ols')
                fig1.update_layout(height=400)
                charts.append(dcc.Graph(figure=fig1))
            
            # Crime vs Poverty
            if 'jumlah_miskin' in df.columns:
                fig2 = px.scatter(df, x='jumlah_miskin', y='kriminalitas_1000orang',
                                title='Kriminalitas vs Jumlah Miskin',
                                trendline='ols')
                fig2.update_layout(height=400)
                charts.append(dcc.Graph(figure=fig2))
    
    if not charts:
        charts.append(dbc.Alert("Tidak dapat membuat visualisasi dengan data yang tersedia.", color="info"))
    
    return charts

def create_data_table(df):
    """Create a data table from DataFrame"""
    # Limit to first 100 rows for performance
    display_df = df.head(100)
    
    return dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in display_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=20,
        sort_action="native",
        filter_action="native"
    )