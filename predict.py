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
    generate features
    '''
    def create_features(self,end=None):
        # pull stocks from db
        query = 'SELECT Ticker FROM stocks.collections WHERE Collection = "100-most-popular"'
        df = pd.read_sql(query,con=client.database.connection)
        
        # set ticker as df index
        df.set_index('Ticker',inplace=True)
        tickers = list(df.index)
        
        # 
        future_date = '2020-05-08'
        
        rsi,gc,start_prices,future_prices = [],[],[],[]
        for t in tickers:
            temp = client.get_historicals(t,end=end)       
            future_prices.append(temp.loc[future_date,'Adj Close'])
            
            temp = temp.loc[:end,:]
            start_prices.append(temp.loc[end,'Adj Close'])
            
            rsi.append(relative_strength_index(temp))
            gc.append(golden_cross(temp))
            df.loc[t,'Weekly % Change'] = weekly_pct_change(temp)
            df.loc[t,'Monthly % Change'] = monthly_pct_change(temp)
            
        df['Start Price'] = start_prices
        df['End Price'] = future_prices
        df['RSI'] = rsi
        df['Golden Cross'] = gc
        df['Actual Signal'] = df['End Price'] > df['Start Price']
        print(df)
        df.drop(['Start Price','End Price'],axis=1,inplace=True)
        #df.to_csv('decision-tree-data/test1.csv')
        return df
    
    def decision_tree(self):
        max_depth = 10
        min_size = 1
        
        df = pd.read_csv('train.csv',index_col='Ticker')
        df.drop('Golden Cross',axis=1,inplace=True)
        
        train = df
       
        model = DecisionTree(max_depth)
        tree = model.build_tree(train,min_size)        
        
        for t in list(train.index):
            pred = model.predict(tree,train.loc[t,:])
            train.loc[t,'Prediction'] = pred
            
        for t in list(test.index):
            pred = model.predict(tree,test.loc[t,:])
            test.loc[t,'Prediction'] = pred
        
        print('Train Accuracy: %.2f' % (sum(train['Actual Signal'] == train['Prediction'])/len(train)))
        print('Test Accuracy: %.2f' % (sum(test['Actual Signal'] == test['Prediction'])/len(test)))
        
        
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
    #s.decision_tree()
    #res = pd.read_csv('decision-tree-data/train_results.csv',index_col='Ticker')
    #print(sum(res['Actual Signal'] == res['Prediction'])/len(res))