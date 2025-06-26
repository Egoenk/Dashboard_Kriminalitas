import dash
from dash import html, dcc, callback, Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from server import db
import logging
import numpy as np

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
            doc_data['id'] = doc.id
            data.append(doc_data)
        return data
    except Exception as e:
        logger.error(f"Error getting data from collection {collection_name}: {e}")
        return []

def prepare_timeseries_data(df):
    """Prepare data for time series analysis"""
    if 'tahun' not in df.columns:
        return None, []
    
    # Convert tahun to numeric
    df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
    df = df.dropna(subset=['tahun'])
    df['tahun'] = df['tahun'].astype(int)
    
    # Get numeric columns (excluding tahun and id)
    numeric_cols = []
    for col in df.columns:
        if col.lower() in ['id', 'tahun']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype in ['int64', 'float64']:
            # Check if column has meaningful variation
            if df[col].nunique() > 1:
                numeric_cols.append(col)
    
    return df, numeric_cols

layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H2("Visualisasi data kriminalistas", className="text-center mb-4"),
            html.Hr(),
        ], width=12)
    ]),

    # Section 1: Data Selection and Configuration
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Konfigurasi data", className="mb-0")),
                dbc.CardBody([
                    html.P("Pilih sumber data dan atur configurasi visualisasinya.", className="text-muted mb-3"),
                    
                    # Collection Selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Data Collection:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='ts-collection-dropdown',
                                placeholder="Pilih Firestore collection...",
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Tipe Chart:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='ts-chart-type-dropdown',
                                options=[
                                    {'label': 'Line Chart', 'value': 'line'},
                                    {'label': 'Bar Chart', 'value': 'bar'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'},
                                    {'label': 'Box Plot', 'value': 'box'},
                                    {'label': 'Heatmap', 'value': 'heatmap'}
                                ],
                                value='line',
                                className="mb-3"
                            ),
                        ], width=6)
                    ]),
                    
                    # Feature Selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Feature yang dilihat:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='ts-features-dropdown',
                                placeholder="Pilih features untuk di visualisasikan...",
                                multi=True,
                                className="mb-3",
                                disabled=True
                            ),
                        ], width=8),
                        dbc.Col([
                            html.Label("Rentang Tahun:", className="fw-bold mb-2"),
                            dcc.RangeSlider(
                                id='ts-year-range-slider',
                                marks={},
                                step=1,
                                disabled=True,
                                className="mb-3"
                            ),
                        ], width=4)
                    ]),
                    
                    # Control Buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Muat Data", id="ts-load-btn", color="primary", disabled=True),
                                dbc.Button("Reset View", id="ts-reset-btn", color="secondary", disabled=True),
                                dbc.Button("Pilih Semua Fitur", id="ts-select-all-btn", color="info", disabled=True)
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 2: Visualization Output
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Visualisasi data krimanlitas", className="mb-0 d-inline-block"),
                    html.Small(id="ts-chart-info", className="text-muted ms-3")
                ]),
                dbc.CardBody([
                    html.P("Visualisasi akan ditampilkan setelah data dimuat.", className="text-muted mb-3"),
                    dcc.Loading(
                        id="ts-loading",
                        children=[
                            html.Div(id="ts-chart-container"),
                        ],
                        type="default",
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Section 3: Status and Data Summary
    dbc.Row([
        dbc.Col([
            html.Div(id="ts-status-output", className="mb-4"),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("Data Summary", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="ts-data-summary")
                ])
            ])
        ], width=4)
    ])
], fluid=True)

# Callbacks
@callback(
    Output('ts-collection-dropdown', 'options'),
    Input('ts-collection-dropdown', 'id')
)
def update_ts_collection_options(_):
    """Update collection dropdown options"""
    try:
        collections = get_firestore_collections()
        options = [{'label': col, 'value': col} for col in collections]
        return options
    except Exception as e:
        logger.error(f"Error updating collection options: {e}")
        return []

@callback(
    [Output('ts-features-dropdown', 'options'),
     Output('ts-features-dropdown', 'value'),
     Output('ts-features-dropdown', 'disabled'),
     Output('ts-year-range-slider', 'min'),
     Output('ts-year-range-slider', 'max'),
     Output('ts-year-range-slider', 'value'),
     Output('ts-year-range-slider', 'marks'),
     Output('ts-year-range-slider', 'disabled'),
     Output('ts-load-btn', 'disabled'),
     Output('ts-chart-container', 'children'),
     Output('ts-status-output', 'children'),
     Output('ts-data-summary', 'children')],
    Input('ts-collection-dropdown', 'value')
)
def load_initial_data(collection_name):
    """Load initial data and setup controls"""
    if not collection_name:
        return [], [], True, 2020, 2024, [2020, 2024], {}, True, True, [], "", ""
    
    try:
        # Get data from Firestore
        raw_data = get_collection_data(collection_name)
        
        if not raw_data:
            status = dbc.Alert("Tidak ada data di dalam colleciton.", color="warning")
            return [], [], True, 2020, 2024, [2020, 2024], {}, True, True, [], status, ""
        
        # Convert to DataFrame and prepare for time series
        df = pd.DataFrame(raw_data)
        df_clean, numeric_cols = prepare_timeseries_data(df)
        
        if df_clean is None or not numeric_cols:
            status = dbc.Alert("Tidak ada data time-series. kolom 'tahun' diperlukan.", color="warning")
            return [], [], True, 2020, 2024, [2020, 2024], {}, True, True, [], status, ""
        
        # Setup feature options
        feature_options = [{'label': col.replace('_', ' ').title(), 'value': col} for col in numeric_cols]
        
        # Setup year range
        min_year = int(df_clean['tahun'].min())
        max_year = int(df_clean['tahun'].max())
        year_marks = {year: str(year) for year in range(min_year, max_year + 1, max(1, (max_year - min_year) // 10))}
        
        # Create default visualization with all features
        default_chart = create_default_timeseries_chart(df_clean, numeric_cols, collection_name)
        
        # Data summary
        summary = create_data_summary(df_clean, numeric_cols)
        
        status = dbc.Alert(f"Berhasil dimuat {len(df_clean)} records dari '{collection_name}'", color="success")
        
        return (feature_options, numeric_cols, False, min_year, max_year, [min_year, max_year], 
                year_marks, False, False, [default_chart], status, summary)
        
    except Exception as e:
        logger.error(f"Error loading initial data: {e}")
        status = dbc.Alert(f"Error saat memuat data: {str(e)}", color="danger")
        return [], [], True, 2020, 2024, [2020, 2024], {}, True, True, [], status, ""

@callback(
    [Output('ts-reset-btn', 'disabled'),
     Output('ts-select-all-btn', 'disabled')],
    Input('ts-features-dropdown', 'options')
)
def enable_control_buttons(feature_options):
    """Enable control buttons when features are available"""
    is_disabled = len(feature_options) == 0
    return is_disabled, is_disabled

@callback(
    Output('ts-features-dropdown', 'value', allow_duplicate=True),
    Input('ts-select-all-btn', 'n_clicks'),
    State('ts-features-dropdown', 'options'),
    prevent_initial_call=True
)
def select_all_features(n_clicks, options):
    """Select all available features"""
    if n_clicks and options:
        return [opt['value'] for opt in options]
    return []

@callback(
    Output('ts-features-dropdown', 'value', allow_duplicate=True),
    Input('ts-reset-btn', 'n_clicks'),
    State('ts-features-dropdown', 'options'),
    prevent_initial_call=True
)
def reset_feature_selection(n_clicks, options):
    """Reset to show all features"""
    if n_clicks and options:
        return [opt['value'] for opt in options]
    return []

@callback(
    [Output('ts-chart-container', 'children', allow_duplicate=True),
     Output('ts-chart-info', 'children')],
    [Input('ts-load-btn', 'n_clicks'),
     Input('ts-features-dropdown', 'value'),
     Input('ts-chart-type-dropdown', 'value'),
     Input('ts-year-range-slider', 'value')],
    [State('ts-collection-dropdown', 'value')],
    prevent_initial_call=True
)
def update_timeseries_chart(n_clicks, selected_features, chart_type, year_range, collection_name):
    """Update the time series chart based on user selections"""
    if not collection_name or not selected_features:
        return [], ""
    
    try:
        # Get data
        raw_data = get_collection_data(collection_name)
        df = pd.DataFrame(raw_data)
        df_clean, _ = prepare_timeseries_data(df)
        
        if df_clean is None:
            return [dbc.Alert("Tidak ada time-series yang valid.", color="warning")], ""
        
        # Filter by year range
        if year_range:
            df_filtered = df_clean[(df_clean['tahun'] >= year_range[0]) & (df_clean['tahun'] <= year_range[1])]
        else:
            df_filtered = df_clean
        
        # Create chart
        chart = create_timeseries_chart(df_filtered, selected_features, chart_type, collection_name)
        
        # Chart info
        info_text = f"Menunjukan {len(selected_features)} fitur | {len(df_filtered)} data points | Tampilan {chart_type.title()}"
        
        return [chart], info_text
        
    except Exception as e:
        logger.error(f"Error mengupdat chart: {e}")
        return [dbc.Alert(f"Error dalam memnuat chart: {str(e)}", color="danger")], ""

def create_default_timeseries_chart(df, numeric_cols, collection_name):
    """Create default time series chart showing all numeric features"""
    try:
        # Aggregate data by year
        yearly_data = df.groupby('tahun')[numeric_cols].sum().reset_index()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(numeric_cols[:10]):  # Limit to first 10 features for readability
            fig.add_trace(go.Scatter(
                x=yearly_data['tahun'],
                y=yearly_data[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>%{{fullData.name}}</b><br>Year: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Time Series Overview - {collection_name}',
            xaxis_title='Year',
            yaxis_title='Values',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80)
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
        
    except Exception as e:
        logger.error(f"Error creating default chart: {e}")
        return dbc.Alert(f"Error creating default chart: {str(e)}", color="danger")

def create_timeseries_chart(df, selected_features, chart_type, collection_name):
    """Create time series chart based on selected parameters"""
    try:
        if not selected_features:
            return dbc.Alert("Please select at least one feature to display.", color="info")
        
        # Aggregate data by year
        yearly_data = df.groupby('tahun')[selected_features].sum().reset_index()
        
        if chart_type == 'line':
            fig = create_line_chart(yearly_data, selected_features)
        elif chart_type == 'bar':
            fig = create_bar_chart(yearly_data, selected_features)
        elif chart_type == 'scatter':
            fig = create_scatter_chart(yearly_data, selected_features)
        elif chart_type == 'box':
            fig = create_box_chart(df, selected_features)
        elif chart_type == 'heatmap':
            fig = create_heatmap_chart(yearly_data, selected_features)
        else:
            fig = create_line_chart(yearly_data, selected_features)
        
        fig.update_layout(
            title=f'{chart_type.title()} Chart - {collection_name}',
            height=600,
            margin=dict(t=80)
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return dbc.Alert(f"Error creating chart: {str(e)}", color="danger")

def create_line_chart(df, features):
    """Create line chart for time series"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for i, feature in enumerate(features):
        fig.add_trace(go.Scatter(
            x=df['tahun'],
            y=df[feature],
            mode='lines+markers',
            name=feature.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>%{{fullData.name}}</b><br>Year: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Records',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_bar_chart(df, features):
    """Create bar chart for yearly comparison"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    x_offset = np.linspace(-0.3, 0.3, len(features))
    
    for i, feature in enumerate(features):
        fig.add_trace(go.Bar(
            x=df['tahun'] + x_offset[i],
            y=df[feature],
            name=feature.replace('_', ' ').title(),
            marker_color=colors[i % len(colors)],
            width=0.6/len(features),
            hovertemplate=f'<b>%{{fullData.name}}</b><br>Year: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Records',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_scatter_chart(df, features):
    """Create scatter plot for correlation view"""
    if len(features) < 2:
        return create_line_chart(df, features)
    
    fig = px.scatter(df, x=features[0], y=features[1], 
                     color='tahun', size_max=10,
                     title=f'Korelasi: {features[0]} vs {features[1]}')
    
    return fig

def create_box_chart(df, features):
    """Create box plot for value distribution"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for i, feature in enumerate(features):
        fig.add_trace(go.Box(
            y=df[feature],
            name=feature.replace('_', ' ').title(),
            marker_color=colors[i % len(colors)],
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        yaxis_title='Records',
        showlegend=True
    )
    
    return fig

def create_heatmap_chart(df, features):
    """Create heatmap for correlation matrix"""
    if len(features) < 2:
        return create_line_chart(df, features)
    
    # Create correlation matrix
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[f.replace('_', ' ').title() for f in features],
        y=[f.replace('_', ' ').title() for f in features],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Korelasi: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig

def create_data_summary(df, numeric_cols):
    """Create data summary statistics"""
    try:
        summary_stats = []
        
        # Basic info
        summary_stats.append(html.P([html.Strong("Total Records: "), f"{len(df):,}"]))
        summary_stats.append(html.P([html.Strong("Rentang Tahun: "), f"{int(df['tahun'].min())} - {int(df['tahun'].max())}"]))
        summary_stats.append(html.P([html.Strong("Feature: "), f"{len(numeric_cols)}"]))
        
        # Feature list
        if numeric_cols:
            summary_stats.append(html.Hr())
            summary_stats.append(html.P(html.Strong("Feature tersedia:")))
            for col in numeric_cols[:5]:  # Show first 5
                avg_val = df[col].mean()
                summary_stats.append(html.P([
                    f"• {col.replace('_', ' ').title()}: ",
                    html.Small(f"rata-rata {avg_val:,.1f}", className="text-muted")
                ], className="mb-1"))
            
            if len(numeric_cols) > 5:
                summary_stats.append(html.P(f"... and {len(numeric_cols) - 5} more", className="text-muted"))
        
        return summary_stats
        
    except Exception as e:
        logger.error(f"Error dalam membuat ringkasan: {e}")
        return [html.P("Error dalam mengenrate ringkasan", className="text-danger")]