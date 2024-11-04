import pandas as pd
import numpy as np

def testFunc():
    return True

def preprocess(raw_data: pd.DataFrame):
    
    preprocessed_data = raw_data.copy()
    #unpack the lists
    preprocessed_data.iloc[:,0:16] = preprocessed_data.iloc[:,0:16].apply(lambda x: x.str[0])
    #drop the MYEDOE test row
    preprocessed_data = preprocessed_data.drop(2, inplace=False)
    #conver the testdate and time to a single column as a datetime
    preprocessed_data["time_of_test"] = pd.to_datetime(preprocessed_data['testDate'] + ' ' + preprocessed_data['testTime'], format='%m/%d/%Y %I:%M %p')
    preprocessed_data["testDate"] = pd.to_datetime(preprocessed_data['testDate'], format='%m/%d/%Y')
    #preprocessed_data = preprocessed_data.drop(columns=["testDate", "testTime"])
    preprocessed_data = preprocessed_data.reset_index()
    
    return preprocessed_data



def validmrn(preprocessed_data: pd.DataFrame):

    #checks to see that the value is an MRN of form MRN-(#)########
    valid_mrn = preprocessed_data[(preprocessed_data['ID1'].str.contains('MRN')) & 
                             (preprocessed_data['ID1'].str.split('-').str.len() == 2) & 
                             (preprocessed_data['ID1'].str.split('-').str[1].str.len() > 7)]
    
    valid_mrn['MRN'] = pd.to_numeric(valid_mrn['ID1'].str.split('-').str[1], errors='coerce')

    return valid_mrn.dropna().reset_index()
