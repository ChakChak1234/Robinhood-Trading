import os
import pandas as pd
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

'''
Predicts daily adjusted closing price for a stock
'''
class Forecast:
    def __init__(self, data):
        self.data = data

    # prediction is set as the last observed value
    def last_value(self):
        df = self.data

        # drop volume col
        df.drop('Volume', axis=1, inplace=True)

        # prediction column is set to adj close shifted up by 1 unit
        df['Prediction'] = df['Adj Close'].shift(-1)

        # drop last row
        df = df[:-1]
        return df

    # prediction is set as the mean of the previous N values
    # hyper param N will need to be tuned
    def moving_average(self, n=2):
        df = self.data
        close = df[['Adj Close']]
        avgs = []
        for i in range(0,len(close)):
            avgs.append(close[i:i+n].mean())
        df['Prediction'] = np.array(avgs)
        return df
        
    # fit a linear regression model to the previous N values and use to predict the current adj close price
    def linear_regression(self,n):        
        df = self.data

'''
Common trading strategies & ML algorithms that generate buy/sell signals
'''
# Believe movement of a stock will continue in its current direction
class Signal:
    def __init__(self,data):
        self.data = data
    
    # return 1 if 50-day short-term average crosses over 200-day long-term average else 0
    def dual_moving_avg_cross(self):
        df = self.data
        fifty = moving_avg(df,50)
        twohun = moving_avg(df,200)
        res = -1
        # buy signal
        if fifty > twohun:
            res = 1
        # sell signal
        if fifty < twohun:
            res = 0
        print('Moving Avg Cross:',res)
        return res
    
    # buy on a 20-day high and sell on a 20-day low
    def turtle(self):
        df = self.data['Adj Close']
        curr = df[0]
        high = df[1:21].max()
        low = df[1:21].min()
        res = -1
        
        # buy signal
        if curr > high:
            res = 1
        # sell signal
        if curr < low:
            res = 0
        print('Turtle:',res)
        return res

    # believe stocks return to their mean & you can exploit when it deviates from the mean
    def mean_reversion(self):
        df = self.data
        ninety = moving_avg(df, 90) # 30 day moving avg
        thirty = moving_avg(df, 30) # 90 day moving avg
        res = -1
        # expect 30d avg to revert back to 90d avg price is too low and unlikely to increase, buy signal
        if thirty < ninety:
            res = 1
        
        # expect 30d avg to fall back to 90d avg curr price is too high, sell signal 
        if thirty > ninety:
            res = 0
        print('Mean Reversion:',res)
        return res
    

'''
Utility Functions
'''
# Calculates the daily percentage change in price of a stock
def daily_pct(df):
    close = df[['Adj Close']]
    change = close.pct_change()
    change.fillna(0,inplace=True)
    return change

# computes the moving average of a stock over the specified days
def moving_avg(df, days):
    df = df[:days]
    avg = df['Adj Close'].sum()/days
    return round(avg,2)

if __name__ == '__main__':
    data = 'historical_data/AAL.csv'
    data = pd.read_csv(data)
      
    # utilities
    #c = print(daily_pct(data))
    #avg = moving_avg(data,50)

    # price forecasting
    f = Forecast(data)
    f.last_value()
    f.moving_average(5)
    f.linear_regression(5)
    