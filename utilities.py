import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
params: dataframe of historical data for a stock
returns: dataframe of day-to-day profit/loss
'''
def daily_return(df):
    ret = df['Close'] - df['Open']
    return ret

'''
params: df = dataframe of historical data for a stock, days = window to compute the moving average
returns: dataframe of simple moving averages
'''
def simple_moving_avg(df,days):
    sma = df['Adj Close'].rolling(window=days).mean()
    sma.fillna(0)
    return sma.round(2)

'''
params: df = dataframe of historical data for a stock, days = window to compute the moving average
returns: dataframe of exponential moving averages
'''
def exponential_moving_average(df,days):
    exp = df['Adj Close'].ewm(span=days,adjust=False).mean()
    return exp.round(2)    


############## INDICATORS ############## 
'''
moving average convergence divergence
- MACD line: Subtract the stocks 26day EMA from the 12day EMA
- Signal line: Calculate 9day EMA of the MACD line

Params: dataframe of historical data for a stock
Returns: BUY signal if MACD line crosses above signal line and SELL signal if crosses below
'''
def macd(df):
    # 12 and 26 day exponential moving averages
    ema12 = exponential_moving_average(df,12)
    ema26 = exponential_moving_average(df,26)
     
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
    res['Signal Line'] = signal
    res['Indicator'] = res['Adj Close'] > res['Signal Line']
    return res

'''
Models the golden cross strategy which generates a buy signals if a stocks short-term moving average crosses above its long term moving-average
params: df = dataframe of historical data for a stock, short = window for short-term moving average, long = window for long-term moving average
returns: list of signals corresponding to each day
'''
def golden_cross(df,short,long):    
    # short and long term moving averages
    short_term = list(simple_moving_avg(df,short))
    long_term = list(simple_moving_avg(df,long))
   
    # set columns
    columns = [str(short) + 'Day MA', str(long) + 'Day MA']

    # resultant dataframe
    result = pd.DataFrame([short_term,long_term])
    result = result.T
    result.dropna(inplace=True)
    result.columns = columns
    
    # generate and return signals
    result['Signal'] = result[str(short) + 'Day MA'] > result[str(long) + 'Day MA']
    return list(result['Signal'])
    

'''
Models the Bollinger Bands indicator which provides information regarding price volatility. 
Consists of 3 bands:
- middle: 20day simple moving average
- upper/lower: 2 standard deviations away from the middle band

params: dataframe of historical data
returns: df containing the middle, upper and lower bands
'''
def boiler_bands(df):
    df = df.reset_index()
    
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
    
    df.set_index('Date',inplace=True)
    return df

'''
Momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold stock price conditions

params: df = dataframe of historical data for a stock, periods = number of days to compute the RSI
returns: (int) RSI for a stock
'''
def relative_strength_index(df,periods=14):
    df = df.iloc[-periods-1:]
    df['Prev'] = df['Adj Close'].shift(1)
    df.dropna(inplace=True)
    
    df['Diff'] = df['Prev']-df['Adj Close']
    df['Gain'] = df['Diff']
    df['Loss'] = df['Diff']
    df['Gain'][df['Gain'] < 0] = 0
    df['Loss'][df['Loss'] > 0] = 0
    df['Loss'] = df['Loss'] * -1
    
    avg_gain = df['Gain'].sum()/periods
    avg_loss = df['Loss'].sum()/periods
    
    rsi = round(100 - (100/(1 + (avg_gain/avg_loss))),3)
    return rsi