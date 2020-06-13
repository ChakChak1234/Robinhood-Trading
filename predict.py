import os
import pandas as pd
import numpy as np
import math
from datetime import date

from utilities import *
from robinhood import *
from perceptron import *

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

    def mlp_network(self,df):
        df.set_index('Ticker',inplace=True)
        columns = list(df.columns)
        
        # set target classifications
        target = df[str(columns[-1])]#['7-Day Profit']
        df.drop(str(columns[-1]),axis=1,inplace=True)
        
        # set input data and add col for bias
        inputs = df
        inputs['Bias'] = 1
        
        # initialize mlp model and train
        model = MLP(df,inputs,target)
        model.train()
        
        # generate outputs
        out = [model.predict(p) for p in model.inputs]
        df['Target'] = target
        df['Target'] = df['Target'].astype(int)
        df['Output'] = out
        
        df.drop('Bias',axis=1,inplace=True)
        return df,model
    
    def mlp_evaluation(self,train,test):     
        cols = list(test.columns)
        
        y = test[str(cols[-1])]
        test.drop(str(cols[-1]),axis=1,inplace=True)
        x = test
        x.set_index('Ticker',inplace=True)
        x = x.to_numpy()
        
        train_results,model = self.mlp_network(train)
        
        
if __name__ == '__main__':
    df1 = pd.read_csv('train_04-30-2020.csv')
    df2 = pd.read_csv('train_05-29-2020.csv')
    df3 = pd.read_csv('train_06-01-2020.csv')
    s = Signal(df1)
    s.mlp_evaluation(df1,df2)
        