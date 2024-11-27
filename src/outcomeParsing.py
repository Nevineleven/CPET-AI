import pandas as pd
import numpy as np

def preprocess(outcomes_data: pd.DataFrame):
    outcomes_data["Test Date"] = pd.to_datetime(outcomes_data["Test Date"], format='%m/%d/%Y')
    death_data = outcomes_data[["MRN", "Test Date", "Death", "death_date"]]
    death_data.dropna(subset=['Death']).sort_values(by=['MRN', 'death_date'], ascending=[True, False]).drop_duplicates(subset='MRN', keep='first')
    return death_data

def mergeDataOutcome(data_with_mrn: pd.DataFrame,
                     death_data: pd.DataFrame,
                     time_in_days: float):
    time_in_sec = time_in_days * 86400
    data_and_outcomes = pd.merge(data_with_mrn, death_data, on=['MRN'], how='left')
    data_and_outcomes['Death'] = ((data_and_outcomes['death_date'] - data_and_outcomes['time_of_test']).dt.total_seconds() <= time_in_sec).astype(int)
    return data_and_outcomes.drop_duplicates(keep='last')