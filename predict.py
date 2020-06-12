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
        self.portfolio = None
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
    
    tickers = list(df['Ticker'])
    df = client.get_historicals(tickers[0])
    
    x = macd(df,'2020-05-29')
    x1 = relative_strength_index(df,'2020-05-29')
    print(x,x1)
    