# Interactive Emotion Analysis Dashboard

This dashboard provides interactive visualizations for student emotion data from the `student_log_2.csv` file.

## Features

- **Multiple Visualization Types**:
  - Box plots of emotion distribution
  - Bar charts of average confidence levels
  - Correlation heatmaps between emotions
  - Time series analysis of emotions
  - Pie charts of dominant emotion distribution
  - Radar charts comparing emotions
  - Density distributions
  - Performance vs. emotion scatter plots
  - 3D emotion space visualization

- **Interactive Filtering**:
  - Filter by skill
  - Filter by correct/incorrect answers
  - Dynamic statistics based on selected filters

- **Static HTML Exports**:
  - Pre-generated visualizations saved in the `emotion_analysis_results/html_exports` directory

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard with:
```
python interactive_emotion_dashboard.py
```

This will:
1. Load the data from `student_log_2.csv`
2. Generate static HTML visualizations in the `emotion_analysis_results/html_exports` directory
3. Start a local web server with the interactive dashboard

The dashboard will be available at http://127.0.0.1:8050/ in your web browser.

## Dashboard Instructions

1. Use the dropdown menu to select different visualization types
2. Apply filters to focus on specific skills or correct/incorrect answers
3. View dynamic statistics that update based on your selections
