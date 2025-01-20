import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_time_series_matrix(df_row, time_series_columns):
    """Convert time series data for a single row into a matrix format"""
    valid_columns = [col for col in time_series_columns 
                    if col != 'Time' 
                    and isinstance(df_row[col], np.ndarray) 
                    and df_row[col].size > 0]
    matrix = np.vstack([df_row[col] for col in valid_columns]).T
    return matrix, valid_columns

def get_global_ranges_robust(df, time_series_columns, lower_percentile=1, upper_percentile=99):
    """Calculate global ranges using percentiles to remove outliers"""
    global_ranges = {}
    for col in time_series_columns:
        if col == 'Time':
            continue
        all_values = np.concatenate([
            arr for arr in df[col].values 
            if isinstance(arr, np.ndarray) and arr.size > 0
        ])
        min_val = np.nanpercentile(all_values, lower_percentile)
        max_val = np.nanpercentile(all_values, upper_percentile)
        global_ranges[col] = (min_val, max_val)
    return global_ranges

def normalize_time_series_data(df, time_series_columns, global_ranges):
    """Create a new dataframe with globally normalized time series data"""
    df_normalized = df.copy()
    for col in time_series_columns:
        if col == 'Time':
            continue
        min_val, max_val = global_ranges[col]
        df_normalized[col] = df_normalized[col].apply(
            lambda x: np.clip((x - min_val) / (max_val - min_val), 0, 1)
            if isinstance(x, np.ndarray) and x.size > 0 
            else x
        )
    return df_normalized

def pad_time_series(df, time_series_columns, max_length):
    """Pad or truncate all time series to the specified length"""
    df_padded = df.copy()
    for col in time_series_columns:
        df_padded[col] = df_padded[col].apply(
            lambda x: (
                np.pad(x[:max_length],
                      (0, max_length - min(len(x), max_length)),
                      mode='constant',
                      constant_values=np.nan)
                if isinstance(x, np.ndarray) and x.size > 0
                else np.full(max_length, np.nan)
            )
        )
    return df_padded

def plot_normalized_heatmap(df_normalized, row_idx, time_series_columns, figsize=(10, 10), 
                          show_labels=True, show_colorbar=True, filepath=None, show_plot=True):
    """Plot heatmap using pre-normalized data with consistent width"""
    row_data = df_normalized.iloc[row_idx]
    matrix, valid_columns = create_time_series_matrix(row_data, time_series_columns)
    
    fig = plt.figure(figsize=figsize)
    
    # Create time labels
    max_length = matrix.shape[0]
    time_step = 5  # assuming 5-second intervals
    time_points = np.arange(0, max_length * time_step, time_step)
    step = len(time_points) // 10
    time_labels = [f'{t:.0f}' if i % step == 0 else '' for i, t in enumerate(time_points)]
    
    # Create mask for NaN values
    mask = np.isnan(matrix)
    
    # Create heatmap
    sns.heatmap(matrix.T,
                mask=mask.T,
                xticklabels=time_labels if show_labels else False,
                yticklabels=valid_columns if show_labels else False,
                cmap='viridis',
                vmin=0,
                vmax=1,
                cbar=show_colorbar)
    
    if show_labels:
        plt.title(f'Time Series Heatmap for Row {row_idx}\n'
                  f'Death: {row_data["Death"]}, Composite Outcome: {row_data["composite_outcome"]}')
        plt.ylabel('Measurements')
        plt.xlabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    else:
        plt.xticks([])
        plt.yticks([])
        plt.title('')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0 if not show_labels else None, dpi=100)
    
    if show_plot:
        plt.show()
        
    plt.close()
