import pandas as pd
import numpy as np

def preprocess(outcomes_data: pd.DataFrame):
    outcomes_data["Test Date"] = pd.to_datetime(outcomes_data["Test Date"], format='%m/%d/%Y')
    death_data = outcomes_data[["MRN", "Test Date", "Death"]]
    return death_data