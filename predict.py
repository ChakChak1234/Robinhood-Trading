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
Common trading strategies & ML algorithms that generate buy signals and predict a stocks return 14days ahead
'''
class Signal:
    def __init__(self,data):
        self.portfolio = data
        self.results = None           
    
    '''
    Generate features using indicators from utilities.py
    '''
    def create_features(self,end_date='2020-05-29',future_date='2020-06-05'):
        df = self.portfolio
        
        for t in list(df.index):
            # historical data
            hist = client.get_historicals(t)
            
            # past and future prices
            try:
                past_price = hist.loc[end_date,'Adj Close']
                future_price = hist.loc[future_date,'Adj Close']
            except KeyError:
                continue
                
            # signals
            sma = simple_moving_average(hist,end_date) # simple MA 
            ma = moving_average(hist,end_date) # MA
            mac = macd(hist,end_date) # MACD
            rsi = relative_strength_index(hist,end_date) # RSI
            
            # add features for each stock
            df.loc[t,'Simple MA'] = sma
            df.loc[t,'MA'] = ma
            df.loc[t,'MACD'] = mac
            df.loc[t,'RSI'] = rsi
            df.loc[t,'7-Day Profit']  = future_price > past_price

        df.dropna(inplace=True,axis=0)
        return df
    
    '''
    Decision Tree implementation
    '''
    def decision_tree(self):        
        pass

    '''
    Multivariate Linear Regression implementation
    '''
    def linear_regression(self):
        pass

        
if __name__ == '__main__':
    df = pd.read_csv('100-most-popular.csv',index_col='Ticker')
    df.drop(['Name','Price','Collection'],axis=1,inplace=True)
    s = Signal(df)
    df1 = pd.read_csv('train_04-30-2020.csv')
    df2 = pd.read_csv('train_05-29-2020.csv')
    df3 = pd.read_csv('train_06-01-2020.csv')
    
    print(df1,df2,df3)
        