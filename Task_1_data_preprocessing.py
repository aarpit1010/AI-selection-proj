import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
import statsmodels

PATH='dataset/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv'

def data_preprocessing(csv_path): # handling csv data for stationary time series dataset
    
    df = pd.read_csv(csv_path)
    data_df = df.copy()
    
    # Converting Timestamps to Date and Time yyyy/mm/dd hr:min:sec
    data_df.Timestamp = data_df.Timestamp.map(lambda ts: 
                                              datetime.datetime.fromtimestamp(int(ts)))
    data_df.index = pd.to_datetime(data_df['Timestamp'], unit='s')
    
    # Dropping the unnecessary columns
    data_df.drop(['Timestamp', 'Volume_(Currency)', 'Weighted_Price'], axis=1, inplace=True)
    
    # Converting 1-min interval data to 15-min interval to reduce NaN and unknown values
    data_df=data_df.groupby(pd.Grouper(freq='15Min')).aggregate({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(BTC)': 'sum'
        })
    
#     data_df.dropna(inplace=True)
#     data_df.fillna(method='bfill').reset_index()

    columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']
    
    diff_df = data_df.copy()
    log_diff_df = data_df.copy()
    
    for column in columns:  # taking difference of continuous interval values
        diff_df[column] = diff_df[column] - \
            diff_df[column].shift(1)

    diff_df = diff_df.fillna(method='bfill')
    
    for column in columns:  # taking difference of log of continuous interval values for normalization
        log_diff_df.loc[data_df[column] == 0] = 1E-10  # since log(0) is not defined 
        log_diff_df[column] = log_diff_df[column] - \
            log_diff_df[column].shift(1)

    log_diff_df = log_diff_df.fillna(method='bfill')
    log_diff_df = log_diff_df.round(8)
    
    return log_diff_df


# btc_df=data_preprocessing(PATH)
# btc_df.to_csv('dataset/bitstamp_15-min.csv', index=False)


def data_preprocessing_15min(csv_path): # converting 1-min interval data to 15-min intervals and handling missing values
    
    df = pd.read_csv(csv_path)
    data_df = df.copy()
    
    # Converting Timestamps to Date and Time yyyy/mm/dd hr:min:sec
    data_df.Timestamp = data_df.Timestamp.map(lambda ts: 
                                              datetime.datetime.fromtimestamp(int(ts)))
    data_df.index = pd.to_datetime(data_df['Timestamp'], unit='s')
    
    # Dropping the unnecessary columns
    data_df.drop(['Timestamp','Volume_(Currency)', 'Weighted_Price'], axis=1, inplace=True)
    
    # Converting 1-min interval data to 15-min interval to reduce NaN and unknown values
    data_df=data_df.groupby(pd.Grouper(freq='15Min')).aggregate({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(BTC)': 'sum'
        })
    
#     data_df.dropna(inplace=True)
    data_df=data_df.fillna(method='ffill').reset_index()
    
    return data_df


# btc_df_15min=data_preprocessing_15min(PATH)
# btc_df_15min.to_csv('dataset/bitstamp_15-min_wo_nan.csv', index=False)