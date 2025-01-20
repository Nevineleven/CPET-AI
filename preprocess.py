import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def filter_time_series_columns(df, time_series_columns, skip_column='Time', 
                             nan_threshold=0.10, empty_list_threshold=0.10):
    """
    Filter time series columns based on NaN and empty list thresholds
    """
    df_filtered = df.copy()
    for col in time_series_columns:
        if col == skip_column:
            continue
        df_filtered[col] = df_filtered[col].apply(
            lambda x: np.array([]) if pd.isna(x).mean() > nan_threshold else x
        )
        empty_array_count = df_filtered[col].apply(lambda x: isinstance(x, np.ndarray) and x.size == 0).sum()
        missing_array_proportion = empty_array_count / len(df_filtered[col])
        if missing_array_proportion > empty_list_threshold:
            df_filtered.drop(columns=[col], inplace=True)
    return df_filtered

def interpolate_time_series(df, time_series_columns, interval=5):
    """
    Interpolate time series data to have regular intervals
    """
    df_interp = df.copy()
    
    for idx in df_interp.index:
        row = df_interp.loc[idx]
        time_data = row['Time']
        
        # Skip if no time data
        if not isinstance(time_data, np.ndarray) or time_data.size == 0:
            continue
            
        # Create regular time points from 0 to max time, every 'interval' seconds
        new_time = np.arange(0, time_data[-1], interval)
        
        # Interpolate each time series
        for col in time_series_columns:
            if col == 'Time':
                continue
                
            series_data = row[col]
            # Skip if empty data
            if not isinstance(series_data, np.ndarray) or series_data.size == 0:
                continue
                
            # Interpolate using numpy
            try:
                interpolated = np.interp(new_time, time_data, series_data)
                df_interp.at[idx, col] = interpolated
                df_interp.at[idx, 'Time'] = new_time
            except ValueError as e:
                print(f"Error interpolating {col} for row {idx}: {e}")
                continue
    
    return df_interp

def preprocess_data(parquet_path):
    """
    Main function to load and preprocess the data
    """
    # Load data
    df = pd.read_parquet(parquet_path)
    
    # Calculate age
    df['Age'] = (df['testDateTime'] - df['DOB']).dt.days / 365
    
    # Define time series columns
    time_series_columns = [
        'Time', 'Rf', 'VT', 'VE', 'VO2', 'VCO2', 'RQ', 'O2exp', 'CO2exp',
        'VE/VO2', 'VE/VCO2', 'VO2/Kg', 'METS', 'HR', 'VO2/HR', 'FeO2', 'FeCO2',
        'FetO2', 'FetCO2', 'FiO2', 'FiCO2', 'SpO2', 'Power', 'Real_Power',
        'Revolution', 'Speed', 'Grade', 'P_Syst', 'P_Diast', 'Phase',
        'Ambient_Temp', 'RH_Amb', 'Analyzer_Pressure', 'PB', 'Ti', 'Te',
        'Dyspnea', 'Leg Pain'
    ]
    
    # Filter columns
    df_filtered = filter_time_series_columns(
        df, 
        time_series_columns, 
        skip_column='Time',
        nan_threshold=0.10,
        empty_list_threshold=0.10
    )
    
    # Get filtered column list
    filtered_time_series_columns = [
        col for col in time_series_columns 
        if col in df_filtered.columns and col != 'Phase'
    ]
    
    # Interpolate time series
    df_interpolated = interpolate_time_series(
        df_filtered, 
        filtered_time_series_columns, 
        interval=5
    )
    
    return df_interpolated, filtered_time_series_columns

# Usage example:
if __name__ == "__main__":
    # Path to your parquet file
    parquet_path = '/path/to/your/combined_outcome_df.parquet'
    
    # Process the data
    df_interpolated, filtered_time_series_columns = preprocess_data(parquet_path)
    
    # Print some information about the processed data
    print(f"Number of rows in processed data: {len(df_interpolated)}")
    print(f"Number of time series columns: {len(filtered_time_series_columns)}")
    print("\nTime series columns included:")
    print(filtered_time_series_columns)
    
    # Optional: verify interpolation for a specific row
    row_idx = 0
    row = df_interpolated.iloc[row_idx]
    if isinstance(row['Time'], np.ndarray) and row['Time'].size > 0:
        time_diffs = np.diff(row['Time'])
        print(f"\nTime differences between consecutive points for row {row_idx}:")
        print(f"Mean: {np.mean(time_diffs):.2f} seconds")
        print(f"Std: {np.std(time_diffs):.2f} seconds")