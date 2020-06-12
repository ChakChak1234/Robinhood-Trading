import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


############## INDICATORS/STRATEGIES ############## 
'''
Price crosses above 20-day SMA - buy
Price crosses below 20-day SMA - sell
Moving Average uses prior 19 days and current day - indicator is for next day

params: data = dataframe of historical data, current_day = day to get price for, window = days to compute MA
returns: 1 = buy, -1 = sell, 0 = hold
'''
def simple_moving_average(data,current_day,window=20):
    data = data.loc[:current_day,'Adj Close']
    price = data.iloc[-1]
    ma = round(data.iloc[-window:-1].mean(),2)
    
    if price > ma:
        return 1
    elif price < ma:
        return -1
    else:
        return 0

'''
Uses two averages of different window sizes
100 day MA (Slow)- takes longer to adjust to sudden price changes
20 day MA (Fast)- faster to account for sudden changes

Fast MA crosses above slow MA - buy
Slow MA crosses above fast MA - sell

params: data = dataframe of historical data, current_day = day to get price for, fast_window = days to compute fast MA, slow_window = days to compute slow MA
returns: 1 = buy, -1 = sell, 0 = hold
'''
def moving_average(data,current_day,fast_window=20,slow_window=100):
    data = data.loc[:current_day,'Adj Close']
    slow_ma = round(data.iloc[-slow_window:].mean(),2)
    fast_ma = round(data.iloc[-fast_window:].mean(),2)
    
    if fast_ma > slow_ma:
        return 1
    elif slow_ma < fast_ma:
        return -1
    else:
        return 0


'''
Moving Average Convergence/Divergence (MACD)
- indicator/oscillator for technical analysis

Composition
- MACD Series: difference between the fast and slow exponential moving averages (EMA)
- Signal: EMA on the MACD series
- Divergence: difference between MACD series and signal series

Logic
- MACD crosses above signal line -> buy
- MACD crosses below signal line -> sell

Returns: BUY signal if MACD line crosses above signal line and SELL signal if crosses below
'''
def macd(data,current_day,slow_ema=26,fast_ema=12):
    data = data.loc[:current_day,'Adj Close']
    
    slow = data.ewm(span=slow_ema,adjust=False).mean()
    fast = data.ewm(span=fast_ema,adjust=False).mean()

    macd = fast-slow
    signal = macd.ewm(span=9,adjust=False).mean()
    macd = macd.reset_index()
    signal = signal.reset_index()
    
    data = data.reset_index()
    data['MACD'] = macd['Adj Close']
    data['Signal'] = signal['Adj Close']
    
    if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
        return 1
    elif data['MACD'].iloc[-1] < data['Signal'].iloc[-1]:
        return -1
    else:
        return 0
    

'''
Relative Strength Index (RSI): Momentum oscillator that measures velocity and magnitude of directional price movements
- RSI crosses lower threshold -> buy
- RSI crosses upper threshold -> sell

returns: original dataframe with RSI column
'''
def relative_strength_index(data,current_day,lower_thresh=30,upper_thresh=70,period=14):
    data = data.loc[:current_day,'Adj Close']
    data = data.iloc[-period:]
    difference = data.diff()
    
    gain,loss = difference.copy(),difference.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    
    gain_ema = gain.ewm(span=period-1,adjust=False).mean()
    loss_ema = loss.ewm(span=period-1,adjust=False).mean().abs()
    
    rs = gain_ema/loss_ema
    rsi = 100-(100/(1+rs))
    rsi = rsi.reset_index()
    data = data.reset_index()
    data['RSI'] = rsi['Adj Close']
    
    if data['RSI'].iloc[-1] > lower_thresh:
        return 1
    elif data['RSI'].iloc[-1] > upper_thresh:
        return -1
    else:
        return 0