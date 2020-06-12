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
    def __init__(self):
        self.results = None           
    
    '''
    generate features
    '''
    def create_features(self,end=None):
        # pull stocks from db
        pass
        
    '''
    Decision Tree implementation
    '''
    def decision_tree(self):        
        #print('Train Accuracy: %.2f' % (sum(train['Actual Signal'] == train['Prediction'])/len(train)))
        #print('Test Accuracy: %.2f' % (sum(test['Actual Signal'] == test['Prediction'])/len(test)))
        pass

    '''
    Multivariate Linear Regression implementation
    '''
    def linear_regression(self):
        pass

        
if __name__ == '__main__':
    df = pd.read_csv('100-most-popular.csv')
    print(df)
    
    #tickers = list(df['Ticker'])
    #end = '2020-06-05'
    #df = client.get_historicals(tickers[0])
    #df = df.loc[:end,:]
    
    # utilites
    #ret = daily_return(df)
    #print(ret)
    
    #sma = simple_moving_avg(df,20)
    #print(sma) # df
    
    #ema = exponential_moving_average(df,20)
    #print(ema) # df

    #gc = golden_cross(df,50,200)
    #print(gc) 
    
    #s = '2020-04-01'
    #df = df.loc[s:]
    #mac = macd(df) # df
    
    #bb = boiler_bands(df)
    #print(bb)
    
    #rsi = relative_strength_index(df)
    #print(rsi)
    