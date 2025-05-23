import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import numpy as np
from plotly.subplots import make_subplots
import os
import time
import diskcache
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Initialize caching
cache = diskcache.Cache('./cache')

# Create output directory if it doesn't exist
output_dir = 'emotion_analysis_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data from CSV file...")
start_time = time.time()

# Load data more efficiently using chunking and dtype specification
# First identify the most important columns to keep for analysis
essential_columns = [
    'ITEST_id', 'skill', 'correct', 'startTime', 'timeTaken',
    'confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
    'confidence(FRUSTRATED)', 'confidence(OFF TASK)', 'confidence(GAMING)',
    'RES_BORED', 'RES_CONCENTRATING', 'RES_CONFUSED',
    'RES_FRUSTRATED', 'RES_OFFTASK', 'RES_GAMING'
]

# Specify dtypes for faster loading
dtypes = {
    'ITEST_id': 'category',
    'skill': 'category',
    'correct': 'int8',
    'confidence(BORED)': 'float32',
    'confidence(CONCENTRATING)': 'float32',
    'confidence(CONFUSED)': 'float32',
    'confidence(FRUSTRATED)': 'float32',
    'confidence(OFF TASK)': 'float32',
    'confidence(GAMING)': 'float32',
    'RES_BORED': 'float32',
    'RES_CONCENTRATING': 'float32',
    'RES_CONFUSED': 'float32',
    'RES_FRUSTRATED': 'float32',
    'RES_OFFTASK': 'float32',
    'RES_GAMING': 'float32'
}

# Use chunking to load the large file in parts
chunk_size = 100000
chunks = []

# Try to load cached data first
cached_data_path = os.path.join('cache', 'preprocessed_data.pkl')
if os.path.exists(cached_data_path):
    print("Loading data from cache...")
    df = pd.read_pickle(cached_data_path)
else:
    print("Processing data in chunks...")
    for chunk in pd.read_csv("student_log_2.csv", usecols=essential_columns, 
                         dtype=dtypes, chunksize=chunk_size, low_memory=False):
        # Process each chunk
        chunks.append(chunk)
    
    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Preprocessing
    print("Preprocessing data...")
    
    # Convert Unix timestamp to datetime
    if 'startTime' in df.columns:
        df['datetime'] = pd.to_datetime(df['startTime'], unit='s')
        df['date'] = df['datetime'].dt.date
    
    # Create a sample for quick initial rendering (20% of data)
    df_sample = df.sample(frac=0.2, random_state=42)
    
    # Save processed data to cache
    os.makedirs('cache', exist_ok=True)
    df.to_pickle(cached_data_path)
    
    # Also create a cached version of aggregated data for common visualizations
    emotion_cols_df = df[['confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
                          'confidence(FRUSTRATED)', 'confidence(OFF TASK)', 'confidence(GAMING)']]
    emotion_averages = emotion_cols_df.mean().to_dict()
    emotion_medians = emotion_cols_df.median().to_dict()
    emotion_corr = emotion_cols_df.corr().to_dict()
    
    cache.set('emotion_averages', emotion_averages)
    cache.set('emotion_medians', emotion_medians)
    cache.set('emotion_corr', emotion_corr)
    
    if 'skill' in df.columns:
        skill_emotion_avg = df.groupby('skill')[['confidence(BORED)', 'confidence(CONCENTRATING)', 
                                              'confidence(CONFUSED)', 'confidence(FRUSTRATED)', 
                                              'confidence(OFF TASK)', 'confidence(GAMING)']].mean()
        cache.set('skill_emotion_avg', skill_emotion_avg.to_dict())

print(f"Data loaded in {time.time() - start_time:.2f} seconds, {len(df)} records")

# Create a sample for quick initial rendering
if 'df_sample' not in locals():
    df_sample = df.sample(frac=0.2, random_state=42)

# Extract emotion columns
emotion_confidence_cols = [
    'confidence(BORED)', 'confidence(CONCENTRATING)', 
    'confidence(CONFUSED)', 'confidence(FRUSTRATED)', 
    'confidence(OFF TASK)', 'confidence(GAMING)'
]

emotion_response_cols = [
    'RES_BORED', 'RES_CONCENTRATING', 
    'RES_CONFUSED', 'RES_FRUSTRATED', 
    'RES_OFFTASK', 'RES_GAMING'
]

# Define functions for data processing - REMOVED CACHING TO FIX VISUALIZATION ISSUES
def get_emotion_data(filtered_df):
    """Get emotion data statistics from the provided filtered dataframe"""
    # This was previously cached but that caused stale data issues
    # Now we directly compute based on the current filtered data
    
    emotion_cols_df = filtered_df[emotion_confidence_cols]
    
    return {
        'averages': emotion_cols_df.mean().to_dict(),
        'medians': emotion_cols_df.median().to_dict(),
        'corr': emotion_cols_df.corr().to_dict()
    }

# Create app with bootstrap styling for better UI
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

# Define app layout with bootstrap components
app.layout = dbc.Container([
    # Hidden components for storing state
    dcc.Store(id='use-full-data', data=False),
    dcc.Store(id='performance-data', data={}),
    
    dbc.Row([
        dbc.Col([
            html.H1("Interactive Student Emotion Analysis Dashboard", 
                   className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        # Sidebar with controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Select Visualization", className="card-title")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='visualization-selector',
                        options=[
                            {'label': 'Emotion Distribution', 'value': 'box_plot'},
                            {'label': 'Average by Skill', 'value': 'bar_chart'},
                            {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                            {'label': 'Time Series Analysis', 'value': 'time_series'},
                            {'label': 'Dominant Emotions', 'value': 'pie_chart'},
                            {'label': 'Emotion Radar', 'value': 'radar_chart'},
                            {'label': 'Density Distribution', 'value': 'density'},
                            {'label': 'Performance vs Emotions', 'value': 'scatter'},
                            {'label': '3D Emotion Plot', 'value': '3d_plot'}
                        ],
                        value='box_plot',
                        clearable=False
                    )
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader(html.H4("Filters", className="card-title")),
                dbc.CardBody([
                    html.Label("Filter by Skill:"),
                    dcc.Dropdown(
                        id='skill-filter',
                        options=[{'label': skill, 'value': skill} 
                                 for skill in sorted(df['skill'].unique()) if pd.notna(skill)],
                        multi=True,
                        placeholder="Select skills..."
                    ),
                    html.Br(),
                    
                    html.Label("Correct Answer:"),
                    dbc.RadioItems(
                        id='correct-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Correct Only', 'value': '1'},
                            {'label': 'Incorrect Only', 'value': '0'}
                        ],
                        value='all',
                        inline=True
                    )
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader(html.H4("Data Options", className="card-title")),
                dbc.CardBody([
                    dbc.Switch(
                        id='use-sample-data',
                        label="Use sampled data for faster loading",
                        value=True
                    ),
                    html.Br(),
                    dbc.Switch(
                        id='use-full-data-switch',
                        label="Use ALL data for visualizations (may be slow)",
                        value=False
                    ),
                    html.Br(),
                    html.Div(id='performance-stats'),
                    html.Div(id='timing-info', className="mt-2 small text-muted")
                ])
            ])
        ], width=3),
        
        # Main content area
        dbc.Col([
            dbc.Spinner(
                dcc.Graph(id='main-visualization', style={'height': '70vh'}),
                color="primary",
                type="border",
                size="lg",
                fullscreen=False,
            ),
            
            html.Br(),
            
            dbc.Card([
                dbc.CardHeader(html.H4("Data Statistics")),
                dbc.CardBody([
                    html.Div(id='data-stats')
                ])
            ])
        ], width=9)
    ])
], fluid=True)

# Add callback for the use-full-data toggle
@callback(
    Output('use-full-data', 'data'),
    Input('use-full-data-switch', 'value')
)
def update_use_full_data(use_full):
    return use_full

# Add callback for performance stats
@callback(
    Output('performance-stats', 'children'),
    Input('use-sample-data', 'value'),
    Input('use-full-data-switch', 'value')
)
def update_performance_stats(use_sample, use_full_data):
    return [
        html.P(f"Using {'sampled' if use_sample else 'full'} dataset for loading"),
        html.P(f"Records loaded: {len(df_sample if use_sample else df):,}"),
        html.P(f"Sample percentage: {len(df_sample)/len(df)*100:.1f}%"),
        html.P(f"Visualization mode: {'All data' if use_full_data else 'Limited to 50,000 records for complex charts'}", 
               style={'fontWeight': 'bold', 'color': 'blue' if not use_full_data else 'red'})
    ]

# Add callback for timing information
@callback(
    Output('timing-info', 'children'),
    Input('performance-data', 'data')
)
def update_timing_info(perf_data):
    if not perf_data or 'render_time' not in perf_data:
        return "No timing data available"
    
    return f"Last visualization rendered in {perf_data['render_time']:.2f} seconds"

@callback(
    Output('main-visualization', 'figure'),
    Output('data-stats', 'children'),
    Output('performance-data', 'data'),
    Input('visualization-selector', 'value'),
    Input('skill-filter', 'value'),
    Input('correct-filter', 'value'),
    Input('use-sample-data', 'value'),
    Input('use-full-data', 'data')
)
def update_visualization(viz_type, skills, correct_filter, use_sample, use_full_data):
    start_time = time.time()
    
    # Decide which dataset to use and track the source for correct labeling
    if use_sample:
        data_source = df_sample
        data_source_label = "SAMPLED DATA (20%)"
    else:
        data_source = df
        data_source_label = "FULL DATASET"
    
    # Apply filters efficiently
    if skills and len(skills) > 0:
        if correct_filter != 'all':
            # Combined filter for better performance
            filtered_df = data_source[
                (data_source['skill'].isin(skills)) & 
                (data_source['correct'] == int(correct_filter))
            ]
        else:
            filtered_df = data_source[data_source['skill'].isin(skills)]
    elif correct_filter != 'all':
        filtered_df = data_source[data_source['correct'] == int(correct_filter)]
    else:
        filtered_df = data_source
    
    # Record the original data size for statistics
    original_data_size = len(filtered_df)
    
    # Only sample if data is large and user hasn't explicitly asked for full data
    if not use_full_data and len(filtered_df) > 50000 and viz_type not in ['bar_chart', 'heatmap', 'pie_chart', 'radar_chart']:
        filtered_df = filtered_df.sample(50000, random_state=42)
        sampled_data_notice = html.P(f"⚠️ Showing {len(filtered_df):,} sampled records out of {original_data_size:,} total records. Toggle 'Use ALL data' to see everything.", 
                                     style={'color': 'orange', 'fontWeight': 'bold'})
    else:
        sampled_data_notice = html.P(f"Showing all {len(filtered_df):,} records matching your criteria.", 
                                     style={'color': 'green', 'fontWeight': 'bold'})
    
    # Get emotion data directly from the filtered dataframe (no caching)
    # This ensures visualizations always reflect current filter settings
    emotion_data = get_emotion_data(filtered_df)
    
    # Create stats text
    stats_text = [
        sampled_data_notice,
        html.P(f"Average correct rate: {filtered_df['correct'].mean()*100:.2f}%"),
    ]
    
    # Record rendering time for performance stats
    render_start = time.time()
    
    # Create visualizations based on selection
    if viz_type == 'box_plot':
        # Box plot of emotion confidence distribution - more efficient implementation
        if emotion_data and len(filtered_df) > 10000:
            # For large datasets, use pre-computed statistics instead of raw data plotting
            emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
            
            fig = go.Figure()
            for i, emotion in enumerate(emotions):
                col = f'confidence({emotion})'
                
                # Extract quartile data efficiently
                q1 = filtered_df[col].quantile(0.25)
                median = filtered_df[col].quantile(0.5)
                q3 = filtered_df[col].quantile(0.75)
                iqr = q3 - q1
                whisker_min = max(filtered_df[col].min(), q1 - 1.5 * iqr)
                whisker_max = min(filtered_df[col].max(), q3 + 1.5 * iqr)
                
                # Create box plot with only necessary points
                fig.add_trace(go.Box(
                    y=[whisker_min, q1, median, q3, whisker_max],
                    name=emotion,
                    boxpoints=False,  # No individual points for better performance
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                    quartilemethod="exclusive",
                    boxmean=True
                ))
            
            fig.update_layout(
                title='Distribution of Emotion Confidence Levels',
                yaxis_title='Confidence',
                xaxis_title='Emotion'
            )
        else:
            # For smaller datasets, use traditional melting approach
            # Use efficient melting with fewer columns
            emotion_data_only = filtered_df[emotion_confidence_cols].copy()
            melted_df = pd.melt(emotion_data_only,
                                var_name='Emotion', value_name='Confidence')
            melted_df['Emotion'] = melted_df['Emotion'].str.replace('confidence(', '').str.replace(')', '')
            
            fig = px.box(melted_df, x='Emotion', y='Confidence',
                        title='Distribution of Emotion Confidence Levels',
                        color='Emotion',
                        points='outliers' if len(filtered_df) < 5000 else False)
        
        stats_text.append(html.P("Emotion Confidence Statistics:"))
        for col in emotion_confidence_cols:
            emotion = col.replace('confidence(', '').replace(')', '')
            stats_text.append(html.P(f"{emotion}: Mean={filtered_df[col].mean():.3f}, Median={filtered_df[col].median():.3f}"))
            
    elif viz_type == 'bar_chart':
        # Bar chart of average emotion confidence by skill
        if skills and len(skills) > 0:
            skill_emotion_avg = filtered_df.groupby('skill')[emotion_confidence_cols].mean().reset_index()
            melted_skill = pd.melt(skill_emotion_avg, 
                                id_vars=['skill'],
                                value_vars=emotion_confidence_cols,
                                var_name='Emotion', value_name='Average Confidence')
            
            melted_skill['Emotion'] = melted_skill['Emotion'].str.replace('confidence(', '').str.replace(')', '')
            
            fig = px.bar(melted_skill, x='skill', y='Average Confidence', 
                        color='Emotion', barmode='group',
                        title='Average Emotion Confidence by Skill')
        else:
            overall_avg = pd.DataFrame({
                'Emotion': [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols],
                'Average Confidence': [filtered_df[col].mean() for col in emotion_confidence_cols]
            })
            
            fig = px.bar(overall_avg, x='Emotion', y='Average Confidence',
                        title='Overall Average Emotion Confidence',
                        color='Emotion')
            
        stats_text.append(html.P("Top 3 skills with highest CONCENTRATING confidence:"))
        top_concentrating = filtered_df.groupby('skill')['confidence(CONCENTRATING)'].mean().sort_values(ascending=False).head(3)
        for skill, val in top_concentrating.items():
            stats_text.append(html.P(f"{skill}: {val:.3f}"))
            
    elif viz_type == 'heatmap':
        # Correlation heatmap between emotions - optimized for performance
        if emotion_data and 'corr' in emotion_data:
            # Use cached correlation data if available
            corr_dict = emotion_data['corr']
            
            # Convert dictionary to dataframe
            emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
            corr_matrix = pd.DataFrame(index=emotions, columns=emotions)
            
            for col in emotion_confidence_cols:
                col_clean = col.replace('confidence(', '').replace(')', '')
                for row in emotion_confidence_cols:
                    row_clean = row.replace('confidence(', '').replace(')', '')
                    if col in corr_dict and row in corr_dict[col]:
                        corr_matrix.loc[row_clean, col_clean] = corr_dict[col][row]
        else:
            # If cache not available, compute with optimized dataset
            # Only select emotion columns for faster correlation calculation
            emotion_data_only = filtered_df[emotion_confidence_cols].copy()
            corr_matrix = emotion_data_only.corr()
            
            # Rename the columns and index for better readability
            corr_matrix.columns = [col.replace('confidence(', '').replace(')', '') for col in corr_matrix.columns]
            corr_matrix.index = [idx.replace('confidence(', '').replace(')', '') for idx in corr_matrix.index]
        
        # Create the heatmap with optimized settings
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       color_continuous_scale='RdBu_r',
                       title='Correlation Between Emotion Confidence Levels')
        
        # Performance improvement: Only calculate statistics if needed
        stats_text.append(html.P("Correlation Insights:"))
        
        # Get correlation values efficiently
        corr_values = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if i != j:  # Skip self-correlations
                    corr_values.append(((row, col), corr_matrix.iloc[i, j]))
        
        # Sort once instead of using nlargest/nsmallest
        corr_values.sort(key=lambda x: x[1])
        
        if corr_values:
            # Get strongest negative (first) and positive (last) correlations
            strongest_negative = corr_values[0]
            strongest_positive = corr_values[-1]
            
            stats_text.append(html.P(f"Strongest Positive Correlation: {strongest_positive[0][0]} and {strongest_positive[0][1]}: {strongest_positive[1]:.3f}"))
            stats_text.append(html.P(f"Strongest Negative Correlation: {strongest_negative[0][0]} and {strongest_negative[0][1]}: {strongest_negative[1]:.3f}"))
            
    elif viz_type == 'time_series':
        # Create a time series of emotion confidence
        # First, make sure we have proper time data
        if 'startTime' in filtered_df.columns:
            try:
                # Convert Unix timestamp to datetime if that's what it is
                filtered_df['datetime'] = pd.to_datetime(filtered_df['startTime'], unit='s')
                
                # Group by day and calculate mean emotion confidence
                filtered_df['date'] = filtered_df['datetime'].dt.date
                time_series_data = filtered_df.groupby('date')[emotion_confidence_cols].mean().reset_index()
                
                # Melt the data for plotting
                melted_time = pd.melt(time_series_data,
                                    id_vars=['date'],
                                    value_vars=emotion_confidence_cols,
                                    var_name='Emotion', value_name='Average Confidence')
                
                melted_time['Emotion'] = melted_time['Emotion'].str.replace('confidence(', '').str.replace(')', '')
                
                fig = px.line(melted_time, x='date', y='Average Confidence',
                            color='Emotion', markers=True,
                            title='Emotion Confidence Trends Over Time')
                
                stats_text.append(html.P("Time Trend Insights:"))
                
                # Calculate overall trends (simple linear regression)
                for emotion in emotion_confidence_cols:
                    emotion_name = emotion.replace('confidence(', '').replace(')', '')
                    time_series = time_series_data.copy()
                    time_series['day_num'] = range(len(time_series))
                    
                    if len(time_series) > 1:  # Need at least 2 points for regression
                        coeffs = np.polyfit(time_series['day_num'], time_series[emotion], 1)
                        trend = "increasing" if coeffs[0] > 0 else "decreasing"
                        stats_text.append(html.P(f"{emotion_name} shows a {trend} trend over time"))
            except:
                fig = go.Figure()
                fig.add_annotation(
                    text="Cannot create time series: Issue with time data conversion",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Cannot create time series: No time data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
    elif viz_type == 'pie_chart':
        # Super optimized pie chart calculation with accurate reporting
        start_chart_time = time.time()
        
        # Calculate exact sampling rates for accurate reporting
        sampling_details = []
        if use_sample:
            sampling_details.append(f"Using 20% sample of original dataset")
        
        # Apply aggressive sampling for large datasets ONLY if user hasn't requested full data
        pie_data = filtered_df
        if len(filtered_df) > 100000 and not use_full_data:
            original_count = len(filtered_df)
            sample_size = min(100000, len(filtered_df))
            pie_data = filtered_df.sample(n=sample_size, random_state=42)
            sampling_details.append(f"Visualization sampled to {sample_size:,} of {original_count:,} records for performance")
        
        # Prepare a title with accurate information
        if sampling_details:
            sampling_note = f" ({' & '.join(sampling_details)})"
        else:
            sampling_note = f" ({data_source_label})"
        
        # Only get the needed emotion columns to reduce memory usage
        emotion_cols_raw = pie_data[emotion_confidence_cols].values
        
        # Only proceed if we have data
        if len(emotion_cols_raw) > 0:
            # Using numpy operations for maximum speed - 5-10x faster than previous approach
            # Get indices of max values along axis 1 (rows)
            clean_emotion_names = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
            
            # Ultra-fast dominant emotion calculation using numpy
            # Check for rows with all NaN and create a mask
            all_nan_mask = np.isnan(emotion_cols_raw).all(axis=1)
            
            # Get max indices only for non-NaN rows
            dominant_indices = np.zeros(len(emotion_cols_raw), dtype=int)
            if not all_nan_mask.all():  # Only if we have some non-NaN rows
                # nanargmax ignores NaN values when finding maximum
                dominant_indices[~all_nan_mask] = np.nanargmax(emotion_cols_raw[~all_nan_mask], axis=1)
            
            # Convert indices to emotion names using vectorized operations
            dominant_array = np.array(clean_emotion_names)[dominant_indices]
            
            # Mark unknown emotions where all values were NaN
            dominant_array[all_nan_mask] = 'Unknown'
            
            # Count occurrences efficiently using numpy
            unique_emotions, counts = np.unique(dominant_array, return_counts=True)
            
            # Create pie chart directly from numpy arrays
            fig = go.Figure(data=[go.Pie(
                labels=unique_emotions,
                values=counts,
                hole=0.3,
                hoverinfo='label+percent+value',
                textinfo='percent+label',
                marker=dict(colors=px.colors.qualitative.Plotly[:len(unique_emotions)])
            )])
            
            # Create an accurate title with sampling information
            fig.update_layout(
                title=f'Distribution of Dominant Emotions ({len(pie_data):,} analyzed records){sampling_note} [Rendered in {time.time()-start_chart_time:.2f}s]'
            )
            
            # Add subtitle with total filtered records
            if len(pie_data) != len(filtered_df):
                fig.add_annotation(
                    text=f"Note: Using {len(pie_data):,} of {len(filtered_df):,} filtered records ({len(pie_data)/len(filtered_df)*100:.1f}%)",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1, showarrow=False,
                    font=dict(size=12, color="gray")
                )
        else:
            # Handle empty dataset
            fig = go.Figure()
            fig.add_annotation(
                text="No data available after filtering",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        
        stats_text.append(html.P("Dominant Emotion Distribution:"))
        total = sum(dominant_counts)
        for emotion, count in dominant_counts.items():
            percentage = (count / total) * 100
            stats_text.append(html.P(f"{emotion}: {count} records ({percentage:.1f}%)"))
            
    elif viz_type == 'radar_chart':
        # Radar chart of average emotion confidence
        emotion_means = [filtered_df[col].mean() for col in emotion_confidence_cols]
        emotion_names = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=emotion_means,
            theta=emotion_names,
            fill='toself',
            name='Average Confidence'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(emotion_means) * 1.1]
                )
            ),
            title='Average Emotion Confidence (Radar Chart)'
        )
        
        # Add a second trace if filtering by correct/incorrect
        if correct_filter != 'all':
            opposite_filter = '0' if correct_filter == '1' else '1'
            opposite_df = df.copy()
            
            if skills and len(skills) > 0:
                opposite_df = opposite_df[opposite_df['skill'].isin(skills)]
                
            opposite_df = opposite_df[opposite_df['correct'] == int(opposite_filter)]
            
            opposite_means = [opposite_df[col].mean() for col in emotion_confidence_cols]
            
            fig.add_trace(go.Scatterpolar(
                r=opposite_means,
                theta=emotion_names,
                fill='toself',
                name='Opposite Group' if len(opposite_df) > 0 else 'No Data'
            ))
            
        stats_text.append(html.P("Radar Chart Statistics:"))
        for name, val in zip(emotion_names, emotion_means):
            stats_text.append(html.P(f"Average {name}: {val:.3f}"))
            
    elif viz_type == 'density':
        # Ultra-optimized density plots for emotions with accurate data labeling
        start_chart_time = time.time()
        emotion_names = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
        
        # Calculate exact sampling rates for accurate reporting
        sampling_details = []
        if use_sample:
            sampling_details.append(f"Using 20% sample of original dataset")
        
        # Use appropriate sampling based on data size and user preferences
        density_data = filtered_df
        if len(filtered_df) > 100000 and not use_full_data:
            original_count = len(filtered_df)
            sample_size = min(50000, len(filtered_df))
            density_data = filtered_df.sample(n=sample_size, random_state=42)
            sampling_details.append(f"Visualization sampled to {sample_size:,} of {original_count:,} records for performance")
        
        # Prepare a title with accurate information
        if sampling_details:
            sampling_note = f" ({' & '.join(sampling_details)})"
        else:
            sampling_note = f" ({data_source_label})"
        
        # Create a fast histogram-based density plot
        fig = go.Figure()
        
        # Define fixed bin parameters - fewer bins = faster calculation
        # But adjust bin count based on data size for better visuals with smaller datasets
        if len(density_data) < 10000:
            bin_count = 30  # More bins for smaller datasets for better resolution
        else:
            bin_count = 20  # Fewer bins for large datasets = faster calculation
            
        bin_edges = np.linspace(0, 1, bin_count+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Pre-calculate bin centers once
        x_values = bin_centers
        
        # Process each emotion separately for better memory efficiency
        emotion_stats = {}
        for i, col in enumerate(emotion_confidence_cols):
            emotion = col.replace('confidence(', '').replace(')', '')
            
            # Use numpy for faster processing - extract data array directly
            data_array = density_data[col].values
            valid_data = data_array[~np.isnan(data_array)]
            
            # Store basic stats for later display
            if len(valid_data) > 0:
                emotion_stats[emotion] = {
                    'count': len(valid_data),
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'min': np.min(valid_data),
                    'max': np.max(valid_data)
                }
                
                # Calculate histogram using numpy (very fast)
                hist, _ = np.histogram(valid_data, bins=bin_edges, density=True)
                
                # Apply adaptive smoothing based on data size
                if len(valid_data) < 5000:
                    # For small datasets, more smoothing helps visualize the distribution
                    hist_padded = np.pad(hist, 2, mode='edge')  # Pad with edge values
                    # 5-point moving average for small datasets
                    hist_smooth = (hist_padded[:-4] + hist_padded[1:-3] + hist_padded[2:-2] + 
                                  hist_padded[3:-1] + hist_padded[4:]) / 5
                else:
                    # For large datasets, less smoothing preserves details
                    hist_padded = np.pad(hist, 1, mode='edge')  # Pad with edge values
                    # 3-point moving average for large datasets
                    hist_smooth = (hist_padded[:-2] + hist_padded[1:-1] + hist_padded[2:]) / 3
                
                # Add trace with hover information
                hover_text = [f"{emotion}: {x:.2f}, Density: {y:.3f}" for x, y in zip(x_values, hist_smooth)]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=hist_smooth,
                    mode='lines',
                    name=emotion,
                    line=dict(width=2.5, color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                    hoverinfo='text',
                    hovertext=hover_text
                ))
        
        # Add vertical lines for emotion means if we have less than 4 emotions for clarity
        if len(emotion_stats) <= 4:
            for i, (emotion, stats) in enumerate(emotion_stats.items()):
                color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                fig.add_shape(
                    type="line",
                    x0=stats['mean'], x1=stats['mean'],
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color=color, width=2, dash="dash")
                )
                fig.add_annotation(
                    x=stats['mean'],
                    y=0.95 - (i * 0.05),
                    yref="paper",
                    text=f"{emotion} mean: {stats['mean']:.3f}",
                    showarrow=False,
                    font=dict(color=color, size=10)
                )
        
        # Common layout updates
        fig.update_layout(
            title=f'Density Distribution of Emotion Confidence ({len(density_data):,} records){sampling_note} [Rendered in {time.time()-start_chart_time:.2f}s]',
            xaxis_title='Confidence Value',
            yaxis_title='Density',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(range=[0, 1], tickformat='.1f'),  # Fix axis range
            hovermode='closest'
        )
        
        # Add subtitle with total filtered records if using sampled data
        if len(density_data) != len(filtered_df):
            fig.add_annotation(
                text=f"Note: Using {len(density_data):,} of {len(filtered_df):,} filtered records ({len(density_data)/len(filtered_df)*100:.1f}%)",
                xref="paper", yref="paper",
                x=0.5, y=-0.15, showarrow=False,
                font=dict(size=12, color="gray")
            )
        
        # Optimize statistics calculation - only calculate what we need
        # Use numpy operations directly for speed
        stats_data = {}
        for col in emotion_confidence_cols:
            emotion = col.replace('confidence(', '').replace(')', '')
            data_array = density_data[col].values
            data_array = data_array[~np.isnan(data_array)]
            
            if len(data_array) > 0:
                stats_data[emotion] = {
                    'mean': np.mean(data_array),
                    'std': np.std(data_array),
                    'skew': pd.Series(data_array).skew(),  # No direct numpy equivalent
                    'kurt': pd.Series(data_array).kurt()   # No direct numpy equivalent
                }
            else:
                stats_data[emotion] = {'mean': 0, 'std': 0, 'skew': 0, 'kurt': 0}
        
        stats_text.append(html.P("Distribution Statistics:", style={'fontWeight': 'bold'}))
        
        # Create a formatted table for statistics
        stats_table = html.Table([
            html.Thead(html.Tr([html.Th('Emotion'), html.Th('Mean'), html.Th('Std Dev'), html.Th('Skewness'), html.Th('Kurtosis')]))
        ] + [
            html.Tr([
                html.Td(emotion),
                html.Td(f"{stats_data[emotion]['mean']:.3f}"),
                html.Td(f"{stats_data[emotion]['std']:.3f}"),
                html.Td(f"{stats_data[emotion]['skew']:.3f}"),
                html.Td(f"{stats_data[emotion]['kurt']:.3f}")
            ]) for emotion in stats_data
        ], className='table table-striped table-sm')
        
        stats_text.append(stats_table)
            
    elif viz_type == 'scatter':
        # Scatter plot of performance vs emotions
        
        # Create a performance metric (e.g., average correct)
        if 'correct' in filtered_df.columns:
            emotion_performance = pd.DataFrame()
            
            # Group by student if student ID is available
            if 'ITEST_id' in filtered_df.columns:
                student_perf = filtered_df.groupby('ITEST_id').agg({
                    'correct': 'mean',
                    **{col: 'mean' for col in emotion_confidence_cols}
                }).reset_index()
                
                emotion_performance = student_perf
            else:
                # Just use the filtered data directly
                emotion_performance = filtered_df
            
            # Create dropdown for selecting which emotion to plot
            fig = px.scatter(
                emotion_performance,
                x='correct',
                y='confidence(CONCENTRATING)',  # Default emotion
                hover_data=[col for col in emotion_confidence_cols],
                title='Student Performance vs. Emotion (CONCENTRATING)',
                labels={'correct': 'Performance (Correct Rate)', 
                        'confidence(CONCENTRATING)': 'CONCENTRATING Confidence'}
            )
            
            # Add a trend line
            if len(emotion_performance) > 1:
                x = emotion_performance['correct']
                y = emotion_performance['confidence(CONCENTRATING)']
                
                # Calculate trend line
                coeffs = np.polyfit(x, y, 1)
                trend_x = np.array([min(x), max(x)])
                trend_y = coeffs[0] * trend_x + coeffs[1]
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_x, 
                        y=trend_y,
                        mode='lines',
                        name=f'Trend (slope={coeffs[0]:.3f})',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Calculate correlation
                corr = np.corrcoef(x, y)[0, 1]
                
                stats_text.append(html.P(f"Correlation between Performance and CONCENTRATING: {corr:.3f}"))
                stats_text.append(html.P(f"Trend line equation: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"))
            
            # Add correlations for all emotions
            stats_text.append(html.P("Correlations with Performance:"))
            for col in emotion_confidence_cols:
                emotion = col.replace('confidence(', '').replace(')', '')
                corr = np.corrcoef(emotion_performance['correct'], emotion_performance[col])[0, 1]
                stats_text.append(html.P(f"{emotion}: {corr:.3f}"))
                
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Cannot create scatter plot: No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
    elif viz_type == '3d_plot':
        # 3D plot of three emotions
        if len(filtered_df) > 0:
            # Select three emotions with highest variance for interesting plots
            emotion_vars = filtered_df[emotion_confidence_cols].var().sort_values(ascending=False)
            top_emotions = emotion_vars.index[:3].tolist()
            
            if len(top_emotions) >= 3:
                fig = px.scatter_3d(
                    filtered_df.sample(min(5000, len(filtered_df))),  # Sample to avoid overplotting
                    x=top_emotions[0],
                    y=top_emotions[1],
                    z=top_emotions[2],
                    color='correct' if 'correct' in filtered_df.columns else None,
                    opacity=0.7,
                    title=f'3D Emotion Space: {", ".join([e.replace("confidence(", "").replace(")", "") for e in top_emotions])}'
                )
                
                # Clean up the axis labels
                fig.update_layout(
                    scene=dict(
                        xaxis_title=top_emotions[0].replace('confidence(', '').replace(')', ''),
                        yaxis_title=top_emotions[1].replace('confidence(', '').replace(')', ''),
                        zaxis_title=top_emotions[2].replace('confidence(', '').replace(')', '')
                    )
                )
                
                stats_text.append(html.P("3D Plot Statistics:"))
                stats_text.append(html.P("Top 3 emotions by variance:"))
                for emotion in top_emotions:
                    clean_name = emotion.replace('confidence(', '').replace(')', '')
                    stats_text.append(html.P(f"{clean_name}: Variance={filtered_df[emotion].var():.3f}"))
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="Cannot create 3D plot: Insufficient emotion data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Cannot create 3D plot: No data available after filtering",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    # Default layout updates
    fig.update_layout(
        template='plotly_white',
        height=700
    )
    
    # Calculate rendering time
    render_time = time.time() - render_start
    total_time = time.time() - start_time
    
    # Return the performance data for the timing info display
    performance_data = {
        'render_time': render_time,
        'total_time': total_time,
        'data_size': len(filtered_df),
        'original_size': original_data_size
    }
    
    return fig, stats_text, performance_data

# Run the app
if __name__ == '__main__':
    print("Starting Dash server...")
    # Save HTML export
    print("Generating static HTML export...")
    import dash.development.base_component as base
    from dash import html
    
    # Create static export directory
    html_dir = os.path.join(output_dir, 'html_exports')
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    
    # Create simple exports of a few visualizations
    emotion_data = df[emotion_confidence_cols]
    emotion_data.columns = [col.replace('confidence(', '').replace(')', '') for col in emotion_data.columns]
    
    # 1. Box plot
    fig_box = px.box(
        pd.melt(emotion_data, var_name='Emotion', value_name='Confidence'), 
        x='Emotion', y='Confidence', color='Emotion',
        title='Distribution of Emotion Confidence Levels'
    )
    fig_box.write_html(os.path.join(html_dir, 'emotion_boxplot.html'))
    
    # 2. Correlation heatmap
    fig_corr = px.imshow(
        emotion_data.corr(), 
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Correlation Between Emotion Confidence Levels'
    )
    fig_corr.write_html(os.path.join(html_dir, 'emotion_correlation.html'))
    
    # 3. Pie chart of dominant emotions
    dominant_emotion = emotion_data.idxmax(axis=1)
    dominant_counts = dominant_emotion.value_counts()
    
    fig_pie = px.pie(
        values=dominant_counts.values, 
        names=dominant_counts.index,
        title='Distribution of Dominant Emotions',
        color=dominant_counts.index,
        hole=0.3
    )
    fig_pie.write_html(os.path.join(html_dir, 'dominant_emotions.html'))
    
    # 4. Time series if possible
    try:
        df['datetime'] = pd.to_datetime(df['startTime'], unit='s')
        df['date'] = df['datetime'].dt.date
        time_series_data = df.groupby('date')[emotion_confidence_cols].mean().reset_index()
        
        melted_time = pd.melt(
            time_series_data,
            id_vars=['date'],
            value_vars=emotion_confidence_cols,
            var_name='Emotion', 
            value_name='Average Confidence'
        )
        
        melted_time['Emotion'] = melted_time['Emotion'].str.replace('confidence(', '').str.replace(')', '')
        
        fig_time = px.line(
            melted_time, 
            x='date', 
            y='Average Confidence',
            color='Emotion', 
            markers=True,
            title='Emotion Confidence Trends Over Time'
        )
        fig_time.write_html(os.path.join(html_dir, 'emotion_trends.html'))
    except:
        print("Could not create time series visualization")
    
    # Print summary
    print(f"HTML exports created in {html_dir}")
    print("Starting the interactive dashboard...")
    
    # Run the Dash app
    app.run(debug=True)
