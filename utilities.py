import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    sma.dropna(inplace=True)
    sma = sma.round(2)
    return sma

# computes the exponential moving average of a stock over the specified days
def exp_moving_average(df,days):
    ema = df['Adj Close'].ewm(span=days,adjust=False).mean()
    ema = ema.round(2)
    return ema

############## INDICATORS ############## 
'''
moving average convergence divergence
- MACD line: Subtract the stocks 26day EMA from the 12day EMA
- Signal line: Calculate 9day EMA of the MACD line
Returns BUY signal if MACD line crosses above signal line and SELL signal if crosses below
'''
def macd(df):
    # 12 and 26 day exponential moving averages
    ema12 = exp_moving_average(df,12)
    ema26 = exp_moving_average(df,26)
     
    # create macd line
    mac = ema12-ema26
    mac.dropna(inplace=True)
    mac = mac.reset_index()
    
    # create signal line
    signal = mac['Adj Close'].ewm(span=9,adjust=False).mean()
    
    # plot signal and macd lines
    plt.plot(mac['Date'],mac['Adj Close'],label='MACD',color='r')
    plt.plot(mac['Date'],signal,label='Signal',color='b')
    plt.legend(loc='upper left')
    #plt.show()
    
    # put results in dataframe
    res = mac
    res['Signal'] = signal
    return res

'''
Returns buy signal if 50day MA crosses above 200day MA, false otherwise
'''
def golden_cross(df):
    # 50 and 200 day moving averages
    fifty_day = simple_moving_avg(df,50).iloc[-1]
    two_hun_day = simple_moving_avg(df,200).iloc[-1]

    # check for crossover
    if fifty_day <= two_hun_day:
        return False
    else:
        return True

# Closer to upperband means market is more overbought, closer to lower band means market more oversold
def boiler_bands(df):
    df = df.reset_index()
    df = df[['Date','Adj Close']]
    
    # calculate 20day sma and standard deviation
    sma = df.rolling(window=20).mean()
    sma.dropna(inplace=True)
    std = df.rolling(window=20).std()    
    std.dropna(inplace=True)
    
    # get upper and lower bands
    upper = sma['Adj Close'] + 2 * std['Adj Close']
    upper = upper.rename(columns={'Adj Close':'Upper'})
    lower = sma['Adj Close'] - 2 * std['Adj Close']
    lower = lower.rename(columns={'Adj Close':'Lower'})
    
    # add upper and lower bands to dataframe
    df['Upper'] = upper
    df['Lower'] = lower
    df.dropna(inplace=True)
    
    # plot
    plt.plot(df['Date'],df['Adj Close'],label='Adj Close')
    plt.plot(df['Date'],df['Upper'],label='Upper',color='g')
    plt.plot(df['Date'],df['Lower'],label='Lower',color='r')
    plt.fill_between(df['Date'],df['Lower'],df['Upper'],color='y')
    plt.legend(loc='upper left')
    #plt.show()
    return df

'''
relative strength index
- momentum indicator that measures the magnitude of recent price changes to evaluate overbought
or oversold stock price conditions
'''
def RSI(df,periods=14):
    df = df.reset_index()
    df = df.iloc[-periods-1:]
    df = df[['Date','Adj Close']]
    df['Prev'] = df['Adj Close'].shift(1)
    df.dropna(inplace=True)
    
    df['Diff'] = df['Adj Close'] - df['Prev']
    df['Gain'] = df.apply(lambda x: x['Diff'] if x['Diff'] > 0 else 0,axis=1)
    df['Loss'] = df.apply(lambda x: x['Diff']*-1 if x['Diff'] < 0 else 0,axis=1)
    
    avg_gain = df['Gain'].sum()/periods
    avg_loss = df['Loss'].sum()/periods
    
    rsi = round(100 - (100/(1 + (avg_gain/avg_loss))),3)
    return rsi

'''
money flow index
- oscillator (0...100) that uses price & volume features to indentify overbought or oversold signals for an asset
- used to identify divergences which indicate price trend changes
'''
'''def MFI(df):
    df = df.reset_index()
    df = df.iloc[-14:]
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close'])/3
    df['Money Flow Positive?'] = df['Typical Price'] >= df['Typical Price'].shift(1)
    df['Raw Money Flow'] = df.apply(lambda x: x['Typical Price']*x['Volume']*-1 if x['Money Flow Positive?'] is False else x['Typical Price']*x['Volume'],axis=1)    
    pos_flow,neg_flow = 0,0
    for i in list(df['Raw Money Flow']):
        if i < 0:
            neg_flow += i
        else:
            pos_flow += i
    
    flow_ratio = pos_flow/neg_flow
    
    mfi = 100 - (100/(1 + flow_ratio))
    print(mfi)
    return mfi'''

# buy on 20day high and sell on 20day low
def turtle(df):
    df = df.reset_index()
    df = df[['Date','Adj Close']]
    curr_price = df['Adj Close'].iloc[-1]
    
    df = df.iloc[:-1]
    df = df.iloc[-20:]
    
    high = df['Adj Close'].max()
    low = df['Adj Close'].min()
    
    if curr_price >= high:
        return 'Buy'
    elif curr_price <= low:
        return 'Sell'
    else:
        return None