import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def feature_pruning(features: pd.DataFrame, time: pd.Series, time_interval:int = 5, freq_cut: float = 95/5, unique_cut: float = 0.1):
    start_at_zero = []
    interp_times = []

    feature_copy = features.copy()

    for idx, row in feature_copy.iterrows():

        x = np.array(time[idx]).astype(int)
        start_at_zero.append(True)
        if min(x) > 0:
            start_at_zero[-1] = False

        x_interp = np.arange(0, max(x), time_interval)
        interp_times.append(x_interp)

        for col in feature_copy.columns:
            cell = np.array(row[col])
            if len(list(filter(None, cell))) == 0:
                row[col] = []
            else:
                count_vals = pd.Series(cell).value_counts().to_list()
                if len(count_vals) <= 1:
                    to_drop = True
                elif ((count_vals[0] / count_vals[1]) < freq_cut):
                    to_drop = False
                elif (len(count_vals) / len(feature_copy[col]) > unique_cut):
                    to_drop = False
                else:
                    to_drop = True
                if to_drop:
                    row[col] = []
            try:
                y = cell.astype(float) 
                y_cubic = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=y[[0]])
                y_interp = y_cubic(x_interp)

                row[col] = y_interp

            except ValueError:
                continue  
    feature_copy['interpolated_time'] = interp_times
    feature_copy['time_start_at_zero'] = start_at_zero
    return feature_copy