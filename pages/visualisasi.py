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
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Area Chart', 'value': 'area'},
                    {'label': 'Heatmap', 'value': 'heatmap'},
                    {'label': 'Violin Plot', 'value': 'violin'}
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
    
    # Filter Section
    dbc.Row([
        dbc.Col([
            html.H5("Filter Data", className="fw-bold mb-3"),
            html.Div(id="filter-controls", className="mb-3"),
        ], width=12)
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
    [Output('variable-dropdown', 'options'),
     Output('filter-controls', 'children')],
    [Input('collection-dropdown', 'value')]
)
def update_variable_options_and_filters(collection_name):
    """Update variable options and create filter controls"""
    if not collection_name:
        return [], []
    
    try:
        # Get sample data to determine available columns
        data = get_collection_data(collection_name)
        if not data:
            return [], []
        
        df = pd.DataFrame(data)
        
        # Variable options (numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        variable_options = []
        for col in numeric_cols:
            variable_options.append({'label': f"{col} (Numeric)", 'value': col})
        for col in categorical_cols:
            variable_options.append({'label': f"{col} (Categorical)", 'value': col})
        
        # Create filter controls
        filter_controls = []
        
        # Numeric filters
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                try:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    
                    # Skip if min and max are the same
                    if min_val == max_val:
                        continue
                        
                    filter_controls.append(
                        dbc.Row([
                            dbc.Col([
                                html.Label(f"Filter {col}:", className="fw-bold"),
                                dcc.RangeSlider(
                                    id={'type': 'range-filter', 'index': col},
                                    min=min_val,
                                    max=max_val,
                                    value=[min_val, max_val],
                                    marks={
                                        min_val: {'label': f'{min_val:.1f}', 'style': {'fontSize': '10px'}},
                                        max_val: {'label': f'{max_val:.1f}', 'style': {'fontSize': '10px'}}
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=6)
                        ], className="mb-3")
                    )
                except (ValueError, TypeError):
                    # Skip columns that can't be converted to numeric
                    continue
        
        # Categorical filters
        for col in categorical_cols:
            if col in df.columns and not df[col].isna().all():
                try:
                    unique_values = [str(val) for val in df[col].dropna().unique().tolist()]
                    if len(unique_values) > 20:  # Limit options for performance
                        unique_values = unique_values[:20]
                    
                    filter_controls.append(
                        dbc.Row([
                            dbc.Col([
                                html.Label(f"Filter {col}:", className="fw-bold"),
                                dcc.Dropdown(
                                    id={'type': 'dropdown-filter', 'index': col},
                                    options=[{'label': val, 'value': val} for val in unique_values],
                                    value=unique_values,
                                    multi=True,
                                    placeholder=f"Pilih {col}..."
                                )
                            ], width=6)
                        ], className="mb-3")
                    )
                except Exception:
                    # Skip problematic columns
                    continue
        
        if not filter_controls:
            filter_controls = [
                dbc.Alert("Tidak ada filter yang tersedia untuk data ini.", color="info", className="mb-3")
            ]
        
        return variable_options, filter_controls
        
    except Exception as e:
        logger.error(f"Error updating variable options: {e}")
        return [], [dbc.Alert(f"Error loading filters: {str(e)}", color="danger")]

@callback(
    [Output('charts-container', 'children'),
     Output('data-table-container', 'children'),
     Output('data-status', 'children')],
    [Input('load-data-btn', 'n_clicks')],
    [State('collection-dropdown', 'value'),
     State('chart-type-dropdown', 'value'),
     State('variable-dropdown', 'value'),
     State({'type': 'range-filter', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'dropdown-filter', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'range-filter', 'index': dash.dependencies.ALL}, 'id'),
     State({'type': 'dropdown-filter', 'index': dash.dependencies.ALL}, 'id')]
)
def update_visualizations(n_clicks, collection_name, chart_type, selected_variable, 
                         range_values, dropdown_values, range_ids, dropdown_ids):
    """Update visualizations based on selected options and filters"""
    if not n_clicks or not collection_name or not selected_variable:
        return [], [], ""
    
    try:
        # Get data from Firestore
        raw_data = get_collection_data(collection_name)
        
        if not raw_data:
            return [], [], dbc.Alert("Tidak ada data ditemukan dalam koleksi ini.", color="warning")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Apply filters
        filtered_df = apply_dynamic_filters(df, range_values, dropdown_values, range_ids, dropdown_ids)
        
        # Status message
        rows_filtered = len(df) - len(filtered_df)
        status_msg = dbc.Alert(
            f"Berhasil memuat {len(filtered_df)} baris data dari koleksi '{collection_name}'" + 
            (f" ({rows_filtered} baris difilter)" if rows_filtered > 0 else ""), 
            color="success"
        )
        
        # Create visualization
        chart = create_chart(filtered_df, chart_type, selected_variable, collection_name)
        
        # Create data table
        table = create_data_table(filtered_df)
        
        return [chart], table, status_msg
        
    except Exception as e:
        logger.error(f"Error updating visualizations: {e}")
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return [], [], error_msg

def apply_dynamic_filters(df, range_values, dropdown_values, range_ids, dropdown_ids):
    """Apply dynamic filters to the dataframe"""
    filtered_df = df.copy()
    
    try:
        # Apply range filters
        for i, (filter_range, filter_id) in enumerate(zip(range_values, range_ids)):
            if filter_range and len(filter_range) == 2:
                column_name = filter_id['index']
                if column_name in filtered_df.columns:
                    min_val, max_val = filter_range
                    filtered_df = filtered_df[
                        (filtered_df[column_name] >= min_val) & 
                        (filtered_df[column_name] <= max_val)
                    ]
        
        # Apply dropdown filters
        for i, (filter_values, filter_id) in enumerate(zip(dropdown_values, dropdown_ids)):
            if filter_values:
                column_name = filter_id['index']
                if column_name in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[column_name].isin(filter_values)]
        
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        # Return original dataframe if filtering fails
        return df
    
    return filtered_df

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
                # Categorical bar chart
                value_counts = df_clean[variable].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'Bar Chart - {variable}',
                           labels={'x': variable, 'y': 'Count'})
            else:
                # Numeric bar chart (binned)
                fig = px.histogram(df_clean, x=variable, 
                                 title=f'Bar Chart - {variable}')
        
        elif chart_type == 'line':
            if 'Tahun' in df_clean.columns:
                fig = px.line(df_clean, x='Tahun', y=variable,
                            title=f'Line Chart - {variable} over Years',
                            markers=True)
            else:
                # Use index if no time column
                fig = px.line(df_clean, y=variable, 
                            title=f'Line Chart - {variable}')
        
        elif chart_type == 'scatter':
            # Find another numeric column for x-axis
            numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                x_var = [col for col in numeric_cols if col != variable][0]
                fig = px.scatter(df_clean, x=x_var, y=variable,
                               title=f'Scatter Plot - {variable} vs {x_var}',
                               trendline='ols')
            else:
                fig = px.scatter(df_clean, y=variable,
                               title=f'Scatter Plot - {variable}')
        
        elif chart_type == 'box':
            fig = px.box(df_clean, y=variable,
                        title=f'Box Plot - {variable}')
        
        elif chart_type == 'histogram':
            fig = px.histogram(df_clean, x=variable,
                             title=f'Histogram - {variable}')
        
        elif chart_type == 'pie':
            if df_clean[variable].dtype in ['object', 'category']:
                value_counts = df_clean[variable].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Pie Chart - {variable}')
            else:
                return dbc.Alert("Pie chart hanya dapat digunakan untuk data kategorikal.", color="warning")
        
        elif chart_type == 'area':
            if 'Tahun' in df_clean.columns:
                fig = px.area(df_clean, x='Tahun', y=variable,
                            title=f'Area Chart - {variable} over Years')
            else:
                fig = px.area(df_clean, y=variable,
                            title=f'Area Chart - {variable}')
        
        elif chart_type == 'heatmap':
            # Create correlation heatmap for numeric variables
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df_clean[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              title='Heatmap - Correlation Matrix',
                              color_continuous_scale='RdBu',
                              aspect='auto')
            else:
                return dbc.Alert("Heatmap memerlukan lebih dari satu variabel numerik.", color="warning")
        
        elif chart_type == 'violin':
            fig = px.violin(df_clean, y=variable,
                          title=f'Violin Plot - {variable}')
        
        if fig is None:
            return dbc.Alert("Tidak dapat membuat grafik dengan tipe yang dipilih.", color="warning")
        
        # Update layout and add collection ID annotation
        fig.update_layout(
            height=500,
            showlegend=True,
            annotations=[
                dict(
                    text=f"Collection ID: {collection_name}",
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
        
        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_df.columns],
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