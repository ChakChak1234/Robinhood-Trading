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
        df['Predicted Price'] = df['Adj Close'].shift(1)
        df = df.rename(columns={'Adj Close': 'Actual Price'})
        
        # drop first row
        df.drop(df.iloc[0])
        
        return df

    # prediction is set as the mean of the previous n values
    def moving_average(self, n=2):
        df = self.data
        
        df['Predicted Price'] = exp_moving_average(df,n)
        df = df.rename(columns={'Adj Close': 'Actual Price'})

        return df
    
    # performs multiple linear regression to forecast stock prices and up/down trends
    def linear_regression(self): 
        df = self.data       
        df.drop('Close',inplace=True,axis=1)
        cols = list(df.columns)
        cols.remove('Adj Close')
        
        x = df[cols]
        y = df['Adj Close']
        model = LinearRegression()
        s = '2020-01-01'
        x_train,y_train,x_test,y_test = model.train_test_split(x,y,s)
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        
        res = pd.DataFrame(x_test,columns=cols)
        res['Actual Price'] = y_test
        res['Predicted Prices'] = preds
        res.index = df.loc[s:].index
        return res
        
    # calculates rmse for each type of prediction strategy and writes results to 'LR Results/error.csv'
    def lr_eval(self):
        pass
        
'''
Common trading strategies & ML algorithms that generate buy signals and predict a stocks return 14days ahead
'''
class Signal:
    def __init__(self):
        self.results = None           
    
    '''
    generate features for the decision tree
    '''
    def create_features(self,end=None):
        query = 'SELECT Ticker FROM stocks.collections WHERE Collection = "100-most-popular"'
        df = pd.read_sql(query,con=client.database.connection)
        df.set_index('Ticker',inplace=True)
        tickers = list(df.index)
        rsi,gc,turt = [],[],[]
        for t in tickers:
            temp = client.get_historicals(t)
            temp = temp.reset_index()
        
            rsi.append(relative_strength_index(temp))
            gc.append(golden_cross(temp))
            turt.append(turtle(temp))    
            
        
        print(len(rsi),len(gc),len(turt))
        df['RSI'] = rsi
        df['Golden Cross'] = gc
        df['Turtle'] = turt
        print(df)
    def decision_tree(self):
        pass

if __name__ == '__main__':
    q = 'SELECT * FROM stocks.collections WHERE Collection = "100-most-popular"'
    df = pd.read_sql(q,con=client.database.connection)
    tickers = list(df['Ticker'])
    
    df = client.get_historicals(tickers[0])
   
    # utilites
    '''
    dpc = daily_pct_change(df)
    print(dpc) # df
    
    wpc = weekly_pct_change(df)
    print(wpc) # float
    
    mpc = monthly_pct_change(df)
    print(mpc) # float
    
    sma = simple_moving_avg(df,20)
    print(sma) # df
    
    ema = exp_moving_average(df,20)
    print(ema) # df

    gc = golden_cross(df)
    print(gc) # bool
    
    s = '2020-04-01'
    df = df.loc[s:]
    mac = macd(df) # df
    
    bb = boiler_bands(df)
    print(bb)
    
    rsi = relative_strength_index(df)
    print(rsi)
    
    turtle = turtle(df)
    print(turtle)
    '''
    
    # Linear Regression
    '''
    f = Forecast(df)
    f.linear_regression()
    '''
    
    # Decision Tree
    s = Signal()
    s.create_features()