import pandas as pd
import numpy as np

'''
Utility Functions
'''
# Calculates the daily percentage change in price of a stock, returns df with daily % change cl
def daily_pct_change(df):
    close = df[['Adj Close']]
    change = close.pct_change()
    change.fillna(0,inplace=True)
    df['Daily % Change'] = change
    return change

# calculates weekly percentage change in a stocks price
def weekly_pct_change(df):
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    end = df['Adj Close'].iloc[-1]
    start = df['Adj Close'].iloc[-6]
    change = round((end-start)/start,3)
    return change

# calculates monthly percentage change in a stocks price
def monthly_pct_change(df):
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    df = df.set_index('Date')
    df = df['Adj Close'].resample('BM').last()
    start = df[-2]
    end = df[-1]
    change = round((end-start)/start,3)
    return change

# calculates daily return for a stock
def daily_return(df):
    ret = df['Close'] - df['Open']
    return ret

# calculates weekly return for a stock
def weekly_return(df):
    end = df['Adj Close'].iloc[-1]
    start = df['Adj Close'].iloc[-6]
    ret = round(end-start,2)
    return ret

# calculates monthly return for a stock
def monthly_return(df):
    #df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    #df = df.set_index('Date')
    df = df['Adj Close'].resample('BM').last()
    start = df[-2]
    end = df[-1]
    ret = round(end-start,3)
    return ret

# computes the simple moving average of a stock over the specified days
def simple_moving_avg(df, days):
    sma = df['Adj Close'].rolling(window=days).mean()
    sma.fillna(0,inplace=True)
    return sma

# computes the exponential moving average of a stock over the specified days
def exp_moving_average(df,days):
    ema = df['Adj Close'].ewm(span=days,adjust=False).mean()
    return ema