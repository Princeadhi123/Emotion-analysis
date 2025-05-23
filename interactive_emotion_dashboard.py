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
import gc  # Garbage collection for memory management
import warnings
import psutil  # For memory monitoring (if installed)
warnings.filterwarnings('ignore')

# Initialize caching with larger size limit and improved settings
cache = diskcache.Cache('./cache', size_limit=2e9)  # 2GB cache limit

# Memory monitoring function
def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"{memory_info.rss / (1024 * 1024):.1f} MB"
    except:
        return "Memory info unavailable"

# Create output directory if it doesn't exist
output_dir = 'emotion_analysis_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data from CSV file...")
start_time = time.time()
print(f"Initial memory usage: {get_memory_usage()}")

# Load data more efficiently using chunking and dtype specification
# First identify the most important columns to keep for analysis - only load what we need
essential_columns = [
    'ITEST_id', 'skill', 'correct', 'startTime', 'timeTaken',
    'confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
    'confidence(FRUSTRATED)', 'confidence(OFF TASK)', 'confidence(GAMING)',
    'RES_BORED', 'RES_CONCENTRATING', 'RES_CONFUSED',
    'RES_FRUSTRATED', 'RES_OFFTASK', 'RES_GAMING'
]

# Specify dtypes for faster loading with optimal memory footprint
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

# Use chunking to load the large file in parts with optimized parameters
chunk_size = 200000  # Larger chunks = fewer I/O operations
parquet_cache_path = os.path.join('cache', 'preprocessed_data.parquet')
pickle_cache_path = os.path.join('cache', 'preprocessed_data.pkl')

# Try to load cached data first - preferring parquet for its efficiency
if os.path.exists(parquet_cache_path):
    print("Loading data from parquet cache...")
    df = pd.read_parquet(parquet_cache_path)
    print(f"Memory usage after loading: {get_memory_usage()}")
elif os.path.exists(pickle_cache_path):
    print("Loading data from pickle cache...")
    df = pd.read_pickle(pickle_cache_path)
else:
    print("Processing data in chunks...")
    chunks = []
    
    # Process in chunks with optimized reading parameters
    for i, chunk in enumerate(pd.read_csv("student_log_2.csv", 
                                        usecols=essential_columns, 
                                        dtype=dtypes, 
                                        chunksize=chunk_size, 
                                        low_memory=True,  # Use low_memory=True for better memory usage
                                        engine='c')):
        # Process each chunk
        chunks.append(chunk)
        if i % 5 == 0:
            print(f"Processed {i+1} chunks, memory usage: {get_memory_usage()}")
    
    print("Combining chunks...")
    # Combine all chunks - more efficient with concat and single operation
    df = pd.concat(chunks, ignore_index=True, copy=False)
    
    # Free memory from chunks
    del chunks
    gc.collect()
    print(f"Memory after chunk processing: {get_memory_usage()}")
    
    # Preprocessing
    print("Preprocessing data...")
    
    # Convert Unix timestamp to datetime - more efficient conversion
    if 'startTime' in df.columns:
        # More efficient datetime conversion
        df['datetime'] = pd.to_datetime(df['startTime'], unit='s')
        df['date'] = df['datetime'].dt.date
    
    # Create a sample for quick initial rendering (20% of data)
    df_sample = df.sample(frac=0.2, random_state=42)
    
    # Save processed data to cache - using parquet for better performance
    print("Saving preprocessed data to cache...")
    os.makedirs('cache', exist_ok=True)
    
    # Use parquet for more efficient storage and faster loading
    df.to_parquet(parquet_cache_path, engine='fastparquet', compression='snappy')
    
    # Create a cached version of aggregated data for common visualizations
    print("Caching aggregated statistics...")
    emotion_cols_df = df[['confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
                         'confidence(FRUSTRATED)', 'confidence(OFF TASK)', 'confidence(GAMING)']]
    
    # Optimize aggregate calculations
    emotion_averages = emotion_cols_df.mean().to_dict()
    emotion_medians = emotion_cols_df.median().to_dict()
    emotion_corr = emotion_cols_df.corr().to_dict()
    
    # Cache results with expiry times
    cache.set('emotion_averages', emotion_averages, expire=7*24*60*60)  # 1 week expiry
    cache.set('emotion_medians', emotion_medians, expire=7*24*60*60)
    cache.set('emotion_corr', emotion_corr, expire=7*24*60*60)
    
    # Free memory from temporary dataframe
    del emotion_cols_df
    gc.collect()
    
    if 'skill' in df.columns:
        # Optimize skill-based aggregation with parallel operations
        emotion_cols = ['confidence(BORED)', 'confidence(CONCENTRATING)', 
                      'confidence(CONFUSED)', 'confidence(FRUSTRATED)', 
                      'confidence(OFF TASK)', 'confidence(GAMING)']
        
        # More efficient groupby with predefined columns
        skill_emotion_avg = df.groupby('skill')[emotion_cols].mean()
        
        # Also cache skill-level statistics for other metrics (median, count)
        skill_emotion_median = df.groupby('skill')[emotion_cols].median()
        skill_counts = df.groupby('skill').size().to_dict()
        
        # Cache with expiry
        cache.set('skill_emotion_avg', skill_emotion_avg.to_dict(), expire=7*24*60*60)
        cache.set('skill_emotion_median', skill_emotion_median.to_dict(), expire=7*24*60*60)
        cache.set('skill_counts', skill_counts, expire=7*24*60*60)

print(f"Data loaded in {time.time() - start_time:.2f} seconds, {len(df)} records, memory usage: {get_memory_usage()}")

# Create a sample for quick initial rendering if not already created
if 'df_sample' not in locals():
    df_sample = df.sample(frac=0.2, random_state=42)
    # Optimize memory usage of sample dataframe
    for col in df_sample.select_dtypes(include=['float64']).columns:
        df_sample[col] = df_sample[col].astype('float32')

# Extract emotion columns - defined as constants for better performance
EMOTION_CONFIDENCE_COLS = [
    'confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
    'confidence(FRUSTRATED)', 'confidence(OFF TASK)', 'confidence(GAMING)'
]

EMOTION_RESPONSE_COLS = [
    'RES_BORED', 'RES_CONCENTRATING', 
    'RES_CONFUSED', 'RES_FRUSTRATED', 
    'RES_OFFTASK', 'RES_GAMING'
]

# Global variables for consistent access
emotion_confidence_cols = EMOTION_CONFIDENCE_COLS
emotion_response_cols = EMOTION_RESPONSE_COLS

# Define functions for data processing with smart caching strategy
# Optimized emotion data processing with caching for filtered datasets
@lru_cache(maxsize=32)  # Cache for the most recent filter combinations
def get_emotion_data_cache(df_hash):
    # This is a placeholder that will be called by the wrapper function
    # The actual implementation is in the wrapper to handle the unhashable dataframe
    pass

# Wrapper function to handle the actual dataframe processing
def get_emotion_data_wrapper(filtered_df):
    """Optimized wrapper function to process emotion data with caching"""
    if len(filtered_df) == 0:
        return None
    
    # Use a hash of key dataframe properties as the cache key
    # This allows caching results for the same filter combinations
    try:
        # Create a robust hash based on key dataframe characteristics
        if 'skill' in filtered_df.columns and len(filtered_df) > 0:
            skills_hash = hash(tuple(sorted(filtered_df['skill'].unique().tolist())))
        else:
            skills_hash = 0
            
        correct_mean = filtered_df['correct'].mean() if 'correct' in filtered_df.columns else 0
        
        df_hash = hash((len(filtered_df), skills_hash, round(correct_mean, 3)))
        
        # Check if we have this in memory cache
        if hasattr(get_emotion_data_wrapper, 'cache') and df_hash in get_emotion_data_wrapper.cache:
            return get_emotion_data_wrapper.cache[df_hash]
    except:
        # If hashing fails, generate a random key - less efficient but still works
        df_hash = hash(time.time())
    
    # Create a new cache if it doesn't exist
    if not hasattr(get_emotion_data_wrapper, 'cache'):
        get_emotion_data_wrapper.cache = {}
    
    # Process data more efficiently
    emotion_data = {}
    
    # Use numpy for faster statistical calculations
    emotion_cols_array = filtered_df[emotion_confidence_cols].values
    
    # Perform calculations in single pass where possible
    emotion_data['averages'] = {col: np.nanmean(emotion_cols_array[:, i]) 
                               for i, col in enumerate(emotion_confidence_cols)}
    emotion_data['medians'] = {col: np.nanmedian(emotion_cols_array[:, i]) 
                              for i, col in enumerate(emotion_confidence_cols)}
    
    # More efficient correlation calculation using numpy directly when possible
    # For small datasets, pandas is actually faster due to optimized implementation
    if len(filtered_df) > 10000:
        # For large datasets, calculate correlation directly with numpy
        # First remove NaN values for accurate correlation
        valid_rows = ~np.isnan(emotion_cols_array).any(axis=1)
        if valid_rows.sum() > 1:  # Need at least 2 rows for correlation
            valid_data = emotion_cols_array[valid_rows]
            corr_matrix = np.corrcoef(valid_data, rowvar=False)
            
            # Convert to dictionary format expected by the app
            emotion_data['corr'] = {}
            for i, col1 in enumerate(emotion_confidence_cols):
                emotion_data['corr'][col1] = {}
                for j, col2 in enumerate(emotion_confidence_cols):
                    if i < len(corr_matrix) and j < len(corr_matrix[0]):
                        emotion_data['corr'][col1][col2] = float(corr_matrix[i, j])
                    else:
                        emotion_data['corr'][col1][col2] = 0.0
        else:
            # Fallback for insufficient data
            emotion_cols_df = filtered_df[emotion_confidence_cols]
            emotion_data['corr'] = emotion_cols_df.corr().to_dict()
    else:
        # For smaller datasets, pandas implementation is more optimized
        emotion_cols_df = filtered_df[emotion_confidence_cols]
        emotion_data['corr'] = emotion_cols_df.corr().to_dict()
    
    # Calculate dominant emotions with numpy for better performance
    clean_emotion_names = [col.replace('confidence(', '').replace(')', '') 
                          for col in emotion_confidence_cols]
    
    # Handle missing values properly
    all_nan_mask = np.isnan(emotion_cols_array).all(axis=1)
    dominant_indices = np.zeros(len(emotion_cols_array), dtype=int)
    
    if not all_nan_mask.all():
        # nanargmax ignores NaN values when finding maximum
        dominant_indices[~all_nan_mask] = np.nanargmax(emotion_cols_array[~all_nan_mask], axis=1)
    
    # Map indices to emotion names
    dominant_array = np.array(clean_emotion_names)[dominant_indices]
    dominant_array[all_nan_mask] = 'Unknown'
    
    # Count occurrences
    unique_emotions, counts = np.unique(dominant_array, return_counts=True)
    emotion_data['dominant_counts'] = {emotion: count 
                                     for emotion, count in zip(unique_emotions, counts)}
    
    # Store in the local cache with size limit management
    get_emotion_data_wrapper.cache[df_hash] = emotion_data
    
    # If the cache gets too big, remove the oldest entries
    if len(get_emotion_data_wrapper.cache) > 50:
        oldest_key = next(iter(get_emotion_data_wrapper.cache))
        del get_emotion_data_wrapper.cache[oldest_key]
    
    return emotion_data

# Function to call the cached wrapper
def get_emotion_data(filtered_df):
    """Main function to get emotion data with efficient caching"""
    return get_emotion_data_wrapper(filtered_df)

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
    """Main callback for updating visualizations based on user selections"""
    start_time = time.time()
    perf_data = {}
    
    # Memory tracking for performance monitoring
    initial_memory = get_memory_usage()
    perf_data['initial_memory'] = initial_memory
    
    # Choose dataset with optimized decision tree
    # Use a view rather than a copy where possible for better memory efficiency
    if use_sample:
        data_source = df_sample
        data_source_label = "SAMPLED DATA (20%)"
    else:
        data_source = df
        data_source_label = "FULL DATASET"
    
    # Apply filters efficiently using vectorized operations
    # Create boolean masks for each condition and combine them for single-pass filtering
    filters = []
    
    if skills and len(skills) > 0:
        # Cache the skill mask for reuse
        skill_mask = data_source['skill'].isin(skills)
        filters.append(skill_mask)
        
    if correct_filter != 'all':
        correct_mask = data_source['correct'] == int(correct_filter)
        filters.append(correct_mask)
    
    # Apply combined filters in a single operation if any exist
    if filters:
        # Combine all filters with logical AND
        combined_mask = filters[0]
        for f in filters[1:]:
            combined_mask = combined_mask & f
        filtered_df = data_source[combined_mask]
    else:
        # No filters applied - use original data source
        # Use .copy(deep=False) to create a view rather than copying data if possible
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
    
    # Track memory after filtering for performance analysis
    memory_after_filter = get_memory_usage()
    perf_data['memory_after_filter'] = memory_after_filter
    
    # Create visualizations based on selection
    if viz_type == 'box_plot':
        # Box plot of emotion confidence distribution - ultra-optimized implementation
        render_start_time = time.time()
        
        if len(filtered_df) > 5000:
            # For large datasets, use statistical summary approach
            emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
            
            # Pre-calculate statistics in a single pass to minimize dataframe operations
            stats = {}
            for i, col in enumerate(emotion_confidence_cols):
                # Extract array once for all operations
                data_array = filtered_df[col].values
                valid_data = data_array[~np.isnan(data_array)]
                
                if len(valid_data) > 0:
                    # Calculate all statistics in one pass with numpy - much faster than pandas
                    min_val = np.min(valid_data)
                    q1 = np.percentile(valid_data, 25)
                    median = np.percentile(valid_data, 50)
                    q3 = np.percentile(valid_data, 75)
                    max_val = np.max(valid_data)
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    iqr = q3 - q1
                    
                    # Calculate whiskers for boxplot
                    whisker_min = max(min_val, q1 - 1.5 * iqr)
                    whisker_max = min(max_val, q3 + 1.5 * iqr)
                    
                    stats[col] = {
                        'min': min_val,
                        'q1': q1,
                        'median': median,
                        'q3': q3,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val,
                        'whisker_min': whisker_min,
                        'whisker_max': whisker_max,
                        'count': len(valid_data)
                    }
                else:
                    # Handle empty data
                    stats[col] = {
                        'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0,
                        'mean': 0, 'std': 0, 'whisker_min': 0, 'whisker_max': 0, 'count': 0
                    }
            
            # Create box plot directly from pre-calculated statistics
            fig = go.Figure()
            for i, emotion in enumerate(emotions):
                col = f'confidence({emotion})'
                if col in stats:
                    # Use pre-calculated values directly
                    s = stats[col]
                    color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    
                    # Create boxplot with statistical summary
                    fig.add_trace(go.Box(
                        y=[s['whisker_min'], s['q1'], s['median'], s['q3'], s['whisker_max']],
                        name=emotion,
                        boxpoints=False,  # No individual points for better performance
                        marker_color=color,
                        quartilemethod="exclusive",
                        boxmean=True,  # Show mean marker (must be True, 'sd', or False)
                        line=dict(width=2),  # Thicker lines for better visibility
                        hoverinfo='name+y',  # Simplified hover for better performance
                    ))
                    
                    # Add annotation for sample size
                    fig.add_annotation(
                        x=i,
                        y=s['whisker_min'] - 0.02,
                        text=f"n={s['count']:,}",
                        showarrow=False,
                        font=dict(size=8, color=color)
                    )
            
            # Update layout with sampling information
            title_suffix = f" (Statistical Summary, n={len(filtered_df):,})"
            if len(filtered_df) != len(data_source):
                title_suffix += f" of {len(data_source):,} total records"
                
            fig.update_layout(
                title=f'Distribution of Emotion Confidence Levels{title_suffix}',
                yaxis_title='Confidence',
                xaxis_title='Emotion',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(t=80, b=50)
            )
        else:
            # For smaller datasets, use more direct plotting but still optimized
            # Only keep essential columns for melting - reduces memory overhead
            emotion_df = filtered_df[emotion_confidence_cols].copy()
            
            # Determine if we should show points based on data size
            show_points = len(filtered_df) < 2000
            
            # Melt more efficiently, specifying the result size ahead of time
            num_rows = len(emotion_df) * len(emotion_confidence_cols)
            melted_df = pd.DataFrame({
                'Emotion': np.repeat(np.array([col.replace('confidence(', '').replace(')', '') 
                                            for col in emotion_confidence_cols]), len(emotion_df)),
                'Confidence': np.concatenate([emotion_df[col].values for col in emotion_confidence_cols])
            })
            
            # Create visualization with proper settings for dataset size
            fig = px.box(melted_df, x='Emotion', y='Confidence',
                        title=f'Distribution of Emotion Confidence Levels (n={len(filtered_df):,})',
                        color='Emotion',
                        points='outliers' if show_points else False,  # Only show outliers for small datasets
                        notched=False)  # Notches can be misleading with small samples
            
            # Additional optimization for medium datasets
            if not show_points:
                # Remove individual points entirely for better performance
                for trace in fig.data:
                    trace.update(boxpoints=False)
        
        # Calculate and display statistics directly from pre-calculated values
        stats_text.append(html.H5("Emotion Confidence Statistics:"))
        stats_table = html.Table([
            html.Thead(html.Tr([
                html.Th("Emotion"), html.Th("Mean"), html.Th("Median"), html.Th("Std Dev")
            ]))
        ] + [
            html.Tr([
                html.Td(col.replace('confidence(', '').replace(')', '')),
                html.Td(f"{filtered_df[col].mean():.3f}"),
                html.Td(f"{filtered_df[col].median():.3f}"),
                html.Td(f"{filtered_df[col].std():.3f}")
            ]) for col in emotion_confidence_cols
        ], className='table table-sm table-striped')
        stats_text.append(stats_table)
        
        # Performance tracking
        perf_data['boxplot_render_time'] = time.time() - render_start_time
            
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
        # Correlation heatmap between emotions - ultra-optimized for performance
        render_start_time = time.time()
        
        # Decide if we should use numpy or pandas based on dataset size
        if len(filtered_df) > 10000:
            # For large datasets, use numpy which is faster for large arrays
            # First get the data as a numpy array
            emotion_array = filtered_df[emotion_confidence_cols].values
            
            # Remove rows with any NaN values
            valid_rows = ~np.isnan(emotion_array).any(axis=1)
            valid_data = emotion_array[valid_rows]
            
            # Only calculate correlation if we have enough data
            if len(valid_data) > 1:
                # Use numpy's corrcoef which is highly optimized for large datasets
                corr_matrix_np = np.corrcoef(valid_data, rowvar=False)
                
                # Convert to pandas DataFrame for consistent interface
                emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
                corr_matrix = pd.DataFrame(corr_matrix_np, index=emotions, columns=emotions)
            else:
                # Fallback for insufficient data
                emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
                corr_matrix = pd.DataFrame(np.eye(len(emotions)), index=emotions, columns=emotions)
        else:
            # For smaller datasets, pandas correlation is well-optimized
            # Only select emotion columns to reduce memory overhead
            emotion_data_only = filtered_df[emotion_confidence_cols].copy()
            corr_matrix = emotion_data_only.corr()
            
            # Rename for readability
            emotions = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
            corr_matrix.columns = emotions
            corr_matrix.index = emotions
            
            # Free memory
            del emotion_data_only
        
        # Configure heatmap for optimal performance and readability
        # Create a more efficient heatmap by controlling number of decimal places
        # and limiting text annotations for larger matrices
        heatmap_config = {
            'z': corr_matrix.values,
            'x': corr_matrix.columns.tolist(),
            'y': corr_matrix.index.tolist(),
            'colorscale': 'RdBu_r',  # Red-Blue diverging colorscale
            'zmid': 0,  # Center the colorscale at 0
            'zmin': -1,
            'zmax': 1
        }
        
        # Only add text annotations for smaller matrices (better performance)
        if len(corr_matrix) <= 8:
            heatmap_config['text'] = [[f'{val:.2f}' for val in row] for row in corr_matrix.values]
            heatmap_config['texttemplate'] = '%{text}'
            heatmap_config['textfont'] = {'size': 10}
        
        # Create the figure with go.Heatmap for more control
        fig = go.Figure(data=go.Heatmap(**heatmap_config))
        
        # Add effective title with dataset size information
        title = 'Correlation Between Emotion Confidence Levels'
        if len(filtered_df) != len(data_source):
            title += f" (n={len(filtered_df):,} of {len(data_source):,} records)"
        else:
            title += f" (n={len(filtered_df):,} records)"
        
        # Update layout with improved readability
        fig.update_layout(
            title=title,
            xaxis=dict(side='bottom', tickangle=45),  # Better label readability
            yaxis=dict(autorange='reversed'),  # Standard heatmap orientation
            margin=dict(t=80, b=100),  # More space for rotated labels
            height=600,  # Fixed height for better view
            coloraxis=dict(colorbar=dict(title='Correlation'))
        )
        
        # Extract insights efficiently using numpy operations
        # Create a mask to ignore self-correlations
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        corr_values = corr_matrix.values[mask].flatten()
        
        if len(corr_values) > 0:
            # Find extremes directly with numpy - much faster than sorting
            max_corr_idx = np.argmax(corr_values)
            min_corr_idx = np.argmin(corr_values)
            
            # Convert flat indices back to matrix coordinates
            flat_indices = np.flatnonzero(mask)
            max_flat_idx = flat_indices[max_corr_idx]
            min_flat_idx = flat_indices[min_corr_idx]
            
            # Convert to row, column indices
            n = len(corr_matrix)
            max_i, max_j = max_flat_idx // n, max_flat_idx % n
            min_i, min_j = min_flat_idx // n, min_flat_idx % n
            
            # Get the correlation values and emotion names
            strongest_positive_val = corr_values[max_corr_idx]
            strongest_negative_val = corr_values[min_corr_idx]
            strongest_positive_pair = (corr_matrix.index[max_i], corr_matrix.columns[max_j])
            strongest_negative_pair = (corr_matrix.index[min_i], corr_matrix.columns[min_j])
            
            # Create a formatted table for insights
            stats_text.append(html.H5("Correlation Insights:"))
            stats_text.append(html.Table([
                html.Thead(html.Tr([
                    html.Th("Correlation Type"), html.Th("Emotions"), html.Th("Value")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Strongest Positive"),
                        html.Td(f"{strongest_positive_pair[0]} & {strongest_positive_pair[1]}"),
                        html.Td(f"{strongest_positive_val:.3f}", style={'color': 'green'})
                    ]),
                    html.Tr([
                        html.Td("Strongest Negative"),
                        html.Td(f"{strongest_negative_pair[0]} & {strongest_negative_pair[1]}"),
                        html.Td(f"{strongest_negative_val:.3f}", style={'color': 'red'})
                    ])
                ])
            ], className='table table-sm table-striped'))
            
            # Add summary stats about correlations
            mean_abs_corr = np.mean(np.abs(corr_values))
            stats_text.append(html.P(f"Average correlation magnitude: {mean_abs_corr:.3f}"))
        else:
            stats_text.append(html.P("Insufficient data to calculate correlation insights."))
            
        # Performance tracking
        perf_data['heatmap_render_time'] = time.time() - render_start_time
            
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
        # Ultra-optimized density plots with vectorized operations and WebGL rendering
        render_start_time = time.time()
        
        # Memory checkpoint before density calculation
        memory_before_density = get_memory_usage()
        perf_data['memory_before_density'] = memory_before_density
        
        # Get clean emotion names once
        emotion_names = [col.replace('confidence(', '').replace(')', '') for col in emotion_confidence_cols]
        
        # Determine optimal data size and sampling strategy
        max_density_points = 50000  # Maximum points for responsive rendering
        density_data = filtered_df
        original_count = len(filtered_df)
        sampling_message = ""
        
        # Apply intelligent sampling only if needed
        if len(filtered_df) > max_density_points and not use_full_data:
            # Calculate sampling ratio to get target number of points
            sample_size = min(max_density_points, len(filtered_df))
            
            # Use more efficient stratified sampling when possible
            if 'skill' in filtered_df.columns and len(filtered_df['skill'].unique()) > 1:
                # Stratified sampling preserves distribution across skills
                try:
                    density_data = filtered_df.groupby('skill', group_keys=False).apply(
                        lambda x: x.sample(min(int(len(x) * sample_size / len(filtered_df) * 1.5), len(x)), 
                                          random_state=42)
                    )
                    # If we got too many samples, take a random subsample
                    if len(density_data) > sample_size:
                        density_data = density_data.sample(sample_size, random_state=42)
                    sampling_message = f"Stratified sampling by skill ({len(density_data):,} of {original_count:,} records)"
                except:
                    # Fallback to random sampling if stratified fails
                    density_data = filtered_df.sample(n=sample_size, random_state=42)
                    sampling_message = f"Random sampling ({sample_size:,} of {original_count:,} records)"
            else:
                # Simple random sampling
                density_data = filtered_df.sample(n=sample_size, random_state=42)
                sampling_message = f"Random sampling ({sample_size:,} of {original_count:,} records)"
        
        # Create figure with specific performance optimizations
        fig = go.Figure()
        
        # Optimized bin calculation - adjust based on data size for best visualization/performance trade-off
        if len(density_data) < 5000:
            bin_count = 35  # More detail for small datasets
        elif len(density_data) < 20000:
            bin_count = 25  # Balanced for medium datasets
        else:
            bin_count = 15  # Minimal bins for large datasets
        
        # Calculate bins once - reuse for all emotions
        bin_edges = np.linspace(0, 1, bin_count+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Prepare for batch processing
        all_emotion_stats = {}
        num_emotions = len(emotion_confidence_cols)
        
        # Extract all emotion data at once for better vectorization
        # This avoids repeated dataframe column access
        all_data = density_data[emotion_confidence_cols].values
        
        # Process each emotion with numpy vectorized operations
        for i, col_idx in enumerate(range(num_emotions)):
            col = emotion_confidence_cols[col_idx]
            emotion = emotion_names[col_idx]
            
            # Extract column directly from the pre-loaded array
            data_array = all_data[:, col_idx]
            valid_mask = ~np.isnan(data_array)
            valid_data = data_array[valid_mask]
            
            # Skip if no valid data
            if len(valid_data) == 0:
                continue
                
            # Calculate all statistics in a single pass
            count = len(valid_data)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)
            median_val = np.median(valid_data)
            std_val = np.std(valid_data)
            
            # Store statistics for later display
            all_emotion_stats[emotion] = {
                'count': count,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'median': median_val,
                'std': std_val
            }
            
            # Calculate histogram with numpy - much faster than pandas
            hist, _ = np.histogram(valid_data, bins=bin_edges, density=True)
            
            # Apply efficient smoothing based on data characteristics
            if std_val < 0.1 or count < 1000:  # More smoothing for narrow distributions or small samples
                # Use Gaussian filter for better smoothing of irregular distributions
                from scipy import ndimage
                # Sigma parameter controls smoothing amount
                sigma = 1.0 if count < 500 else 0.8
                hist_smooth = ndimage.gaussian_filter1d(hist, sigma=sigma)
            else:
                # Simple moving average for regular distributions
                # Pad first to avoid edge effects
                hist_padded = np.pad(hist, 1, mode='edge')
                hist_smooth = (hist_padded[:-2] + hist_padded[1:-1] + hist_padded[2:]) / 3
            
            # Select color from qualitative palette
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            
            # Create optimized hover text with fewer points for better performance
            # Only create hover points every n bins based on bin count
            hover_stride = max(1, bin_count // 15)
            hover_text = [f"{emotion}: {x:.2f}, Density: {y:.3f}" 
                         for x, y in zip(bin_centers[::hover_stride], hist_smooth[::hover_stride])]
            
            # Add trace with optimized parameters
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist_smooth,
                mode='lines',
                name=emotion,
                line=dict(width=2, color=color),
                hoverinfo='text',
                hovertext=hover_text,
                hoverlabel=dict(namelength=-1),  # Show full emotion name
                # Use WebGL for large datasets for better performance
                line_shape='spline' if len(valid_data) < 10000 else 'linear'  # Spline looks better but is more expensive
            ))
            
            # Add vertical line for mean with efficient annotation placement
            fig.add_shape(
                type="line",
                x0=mean_val, x1=mean_val,
                y0=0, y1=0.9,  # Not full height to avoid covering annotations
                yref="paper",
                line=dict(color=color, width=1.5, dash="dot")
            )
        
        # Optimize layout with performance-focused settings
        title = 'Density Distribution of Emotion Confidence'
        if sampling_message:
            title += f" ({sampling_message})"
        
        fig.update_layout(
            title=title,
            xaxis_title='Confidence Value',
            yaxis_title='Density',
            # Horizontal legend for better space usage
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            # Fixed x-axis range for consistent visualization
            xaxis=dict(range=[0, 1], tickformat='.1f', nticks=6),  # Fewer ticks = better performance
            # Other performance optimizations
            hovermode='closest',
            margin=dict(t=60, b=60, l=60, r=40),  # Balanced margins
            plot_bgcolor='rgba(240,240,240,0.95)',  # Slight gray background for better contrast
            height=600  # Fixed height for better proportions
        )
        
        # Create a more advanced statistics panel
        stats_text.append(html.H5("Distribution Statistics:"))
        
        # Table with only the most relevant statistics
        stats_table = html.Table([
            html.Thead(html.Tr([
                html.Th('Emotion'), html.Th('Mean'), html.Th('Median'), 
                html.Th('Std Dev'), html.Th('Sample Size')
            ]))
        ] + [
            html.Tr([
                html.Td(emotion, style={'fontWeight': 'bold'}),
                html.Td(f"{stats['mean']:.3f}"),
                html.Td(f"{stats['median']:.3f}"),
                html.Td(f"{stats['std']:.3f}"),
                html.Td(f"{stats['count']:,}")
            ]) for emotion, stats in all_emotion_stats.items()
        ], className='table table-sm table-striped')
        
        stats_text.append(stats_table)
        
        # Add additional insights based on statistical analysis
        if len(all_emotion_stats) >= 2:
            # Find most and least variable emotions
            emotion_std = [(e, s['std']) for e, s in all_emotion_stats.items()]
            most_variable = max(emotion_std, key=lambda x: x[1])
            least_variable = min(emotion_std, key=lambda x: x[1])
            
            stats_text.append(html.H5("Statistical Insights:", className="mt-3"))
            stats_text.append(html.P([
                html.Strong("Most variable emotion: "), 
                f"{most_variable[0]} (std={most_variable[1]:.3f})"
            ]))
            stats_text.append(html.P([
                html.Strong("Most consistent emotion: "), 
                f"{least_variable[0]} (std={least_variable[1]:.3f})"
            ]))
            
            # Identify dominant emotions based on means
            emotion_means = [(e, s['mean']) for e, s in all_emotion_stats.items()]
            emotion_means.sort(key=lambda x: x[1], reverse=True)
            
            if len(emotion_means) >= 3:
                stats_text.append(html.P([
                    html.Strong("Top 3 emotions by average confidence: "),
                    html.Br(),
                    ", ".join([f"{e} ({v:.3f})" for e, v in emotion_means[:3]])
                ]))
        
        # Performance tracking
        memory_after_density = get_memory_usage()
        perf_data['memory_after_density'] = memory_after_density
        perf_data['density_render_time'] = time.time() - render_start_time
            
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
