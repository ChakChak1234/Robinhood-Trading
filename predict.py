import os
import pandas as pd
import numpy as np
import math
from datetime import date
from utilities import *
from LinearRegression import *
from DecisionTree import *
from robinhood import *
import matplotlib.pyplot as plt

client = Robinhood()

'''
Methods for stock price forecasting
'''
class Forecast:
    def __init__(self, data):
        self.data = data
        self.results = pd.DataFrame()
        
    # predicted price is set as previous days closing price
    def last_value(self):
        df = self.data

        # prediction column is set to adj close shifted up by 1 unit
        df['Prediction'] = df['Adj Close'].shift(1)
        df = df.rename(columns={'Adj Close': 'Actual'})
        
        # drop first row
        df.drop(df.iloc[0])
        
        return df

    # prediction is set as the mean of the previous n values
    def moving_average(self, n=2):
        df = self.data
        
        df['Prediction'] = exp_moving_average(df,n)
        df = df.rename(columns={'Adj Close': 'Actual'})
        df = df[['Actual','Prediction']]

        return df
    
    # performs multiple linear regression to forecast stock prices and up/down trends
    def linear_regression(self,new_feature=None):        
        pass
            

    # calculates rmse for each type of prediction strategy and writes results to 'LR Results/error.csv'
    def lr_eval(self):
        pass
        
'''
Common trading strategies & ML algorithms that generate buy signals and predict a stocks return 14days ahead
'''
class Signal:
    def __init__(self,data):
        self.data = data
        self.results = pd.DataFrame()
    
    # generates dataframe that can be used with decision tree
    def generate_data(self,stocks):
        temp = pd.DataFrame(index=stocks)
        for s in stocks:
            df = client.get_historicals(s,end='2020-04-14')
            df = df[:-10]
            fut_price = df['Adj Close'].iloc[-1]
            df = df[:-9]
            curr = df['Adj Close'].iloc[-1]
            temp.loc[s,'Price'] = curr
            temp.loc[s,'20day SMA'] = simple_moving_avg(df,20)[-1]
            temp.loc[s,'50day SMA'] = simple_moving_avg(df,50)[-1]
            temp.loc[s,'200day SMA'] = simple_moving_avg(df,200)[-1]
            temp.loc[s,'20day EMA'] = exp_moving_average(df,20)[-1]
            temp.loc[s,'50day EMA'] = exp_moving_average(df,50)[-1]
            temp.loc[s,'200day EMA'] = exp_moving_average(df,200)[-1]
            temp.loc[s,'Weekly Return'] = weekly_return(df)
            temp.loc[s,'Monthly Return'] = monthly_return(df)
            temp.loc[s,'14day Profit'] = fut_price > curr
        temp.to_csv('dt_test2.csv')
        return temp            
    
    '''
    generate training and testing data for watchlist and holdings
    training/testing used historical data until 3/31
    '''
    def decision_tree_prep(self):
        pass

    def decision_tree(self):
        pass

if __name__ == '__main__':
    stocks = ['DIS', 'MSFT', 'BAC', 'SNAP', 'UBER']
    df = client.get_historicals('DIS')
    
    # utilites
    #sma = simple_moving_avg(df,20)
    #ema = exp_moving_average(df,20)
    #mac = macd(df)
    #gc = golden_cross(df)
    #bb = boiler_bands(df)
    #rsi = RSI(df)
    #mfi = MFI(df)
    #turtle = turtle(df)