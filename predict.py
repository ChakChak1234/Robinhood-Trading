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
Predicts daily adjusted closing price for a stock
'''
class Forecast:
    def __init__(self, data):
        self.data = data
        
    # prediction is set as the last observed value
    def last_value(self):
        #df = self.data
        test1 = pd.read_csv('historical_data/AAL.csv')
        test2 = pd.read_csv('historical_data/NKE.csv')
    
        df = test2
        # drop volume col
        df.drop(['Open','High','Low','Close','Volume'], axis=1, inplace=True)

        # prediction column is set to adj close shifted up by 1 unit
        df['Prediction'] = df['Adj Close'].shift(1)
        df = df.rename(columns={'Adj Close': 'Actual'})
        
        # drop first row
        df = df[306:]
        
        #df.to_csv('Forecast Results/NKE/last_val.csv')
        return df

    # prediction is set as the mean of the previous n values
    def moving_average(self, n=2):
        test1 = pd.read_csv('historical_data/AAL.csv')
        test2 = pd.read_csv('historical_data/NKE.csv')
        df = test2
        df['Prediction'] = exp_moving_average(df,n)
        df = df.rename(columns={'Adj Close': 'Actual'})
        df = df[['Actual','Prediction']]
        df = df[306:].round(2)
    
        #df.to_csv('Forecast Results/NKE/5day EMA.csv')
        return df
    
    # performs multiple linear regression to forecast stock prices and up/down trends
    def linear_regression(self,new_feature=None):        
        test1 = pd.read_csv('historical_data/AAL.csv')
        test2 = pd.read_csv('historical_data/NKE.csv')
        test1['Daily % Change'] = daily_pct_change(test1)
        test1['5day SMA'] = simple_moving_avg(test1,5)
        test1['5day EMA'] = exp_moving_average(test1,5)
        test2['Daily % Change'] = daily_pct_change(test2)
        test2['5day SMA'] = simple_moving_avg(test2,5)
        test2['5day EMA'] = exp_moving_average(test2,5)
        
        # drop close column and set date as index
        test1.drop('Close',axis=1,inplace=True)
        test1.set_index('Date',inplace=True)
        test2.drop('Close',axis=1,inplace=True)
        test2.set_index('Date',inplace=True)

        # drop last 9 rows
        test1 = test1.iloc[:-9]
        test2 = test2.iloc[:-9]

        # create linear regression model
        model = LinearRegression(test2)
        
        # fit training data, make predictions, and drop a column each iteration
        for i in range(7,0,-1):
            # split into 60% train 40% test
            x_train,y_train,x_test,y_test = model.train_test_split()
    
            # fit training data to model
            model.fit(x_train,y_train)

            # predict based on training data
            pred = model.predict(x_test,y_test)
            #pred.to_csv('Forecast Results/NKE/LR_' + str(i) + 'feat.csv')
            print(pred)

            model.x = model.x.iloc[:,:-1]
            model.cols.pop()
            

    # calculates rmse for each type of prediction strategy and writes results to 'LR Results/error.csv'
    def lr_eval(self):
        # index set as strategy and col is error for that strategy
        df = pd.DataFrame(columns=['RMSE'])
        e = []
        se = []
        rows = []
        
        for f in os.listdir('Forecast Results/AAL/'):
            temp = pd.read_csv('Forecast Results/AAL/'+ f)
            rows.append(os.path.splitext(f)[0])
            
            # standard error
            

            #root mean square error
            rmse = round(np.sqrt(np.mean( np.power((np.array(temp['Actual Price'])-np.array(temp['Prediction'])),2))),3)
            e.append(rmse)
            
        df['RMSE'] = e
        df.index = rows
        #df.to_csv('Forecast Results/AAL/error.csv',index_label='Strategy')
       
        
'''
Common trading strategies & ML algorithms that generate buy signals and predict a stocks return 14days ahead
'''
class Signal:
    def __init__(self,data):
        self.data = data
        self.results = pd.DataFrame()

    # return 1 if 50-day short-term simple moving average crosses over 200-day long-term simple moving average else 0
    def moving_avg_crossSMA(self):
        data = self.data

        self.results['SMA Cross'] = data['50day SMA'] >= data['200day SMA']
    
    # return 1 if 50-day short-term exponential moving average crosses over 200-day long-term moving average else 0
    def moving_avg_crossEMA(self):
        data = self.data
    
        self.results['EMA Cross'] = data['50day EMA'] >= data['200day EMA']
    
    # buy on a 20-day high and sell on a 20-day low
    def turtle(self):
        data = self.data
        temp = []
        for t,p in zip(list(data['Ticker']), list(data['Price'])):
            df = client.get_historicals(t,end='2020-04-14')#pd.read_csv('historical_data/' + t + '.csv')
            df = df.iloc[:-9] #drop last 9 rows
            
            high = df['Adj Close'].tail(20).max()
            curr = p
            #print(curr,high)
            if curr >= high:
                temp.append(True)
            else:
                temp.append(False)           

        self.results['Turtle'] = temp
    
    # buy if 30day MA crosses below 90day MA, otherwise sell
    def mean_reversion(self):
        data = self.data
        res = []

        for t in list(data['Ticker']):
            df = client.get_historicals(t,end='2020-04-14')#pd.read_csv('historical_data/' + t + '.csv')
            df = df.iloc[:-9]
            thirty = round(simple_moving_avg(df,30).iloc[-1],2)
            ninety = round(simple_moving_avg(df,90).iloc[-1],2)
            #print(t,thirty,ninety)
            if thirty < ninety:
                res.append(True)
            else:
                res.append(False)
        self.results['Mean Reversion'] = res
    
    # generates dataframe that can be used with decision tree
    def generate_data(self,stocks):
        temp = pd.DataFrame(index=stocks)
        for s in stocks:
            df = client.get_historicals(s,end='2020-04-14')
            df = df[:-10]
            fut_price = df['Adj Close'].iloc[-1]
            df = df[:-9]
            curr = df['Adj Close'].iloc[-1]
            temp.loc[s,'Price'] = curr
            temp.loc[s,'20day SMA'] = simple_moving_avg(df,20)[-1]
            temp.loc[s,'50day SMA'] = simple_moving_avg(df,50)[-1]
            temp.loc[s,'200day SMA'] = simple_moving_avg(df,200)[-1]
            temp.loc[s,'20day EMA'] = exp_moving_average(df,20)[-1]
            temp.loc[s,'50day EMA'] = exp_moving_average(df,50)[-1]
            temp.loc[s,'200day EMA'] = exp_moving_average(df,200)[-1]
            temp.loc[s,'Weekly Return'] = weekly_return(df)
            temp.loc[s,'Monthly Return'] = monthly_return(df)
            temp.loc[s,'14day Profit'] = fut_price > curr
        temp.to_csv('dt_test2.csv')
        return temp            
    
    '''
    generate training and testing data for watchlist and holdings
    training/testing used historical data until 3/31
    '''
    def decision_tree_prep(self):
        df = pd.read_csv('dt_test3.csv')
        df.set_index('Ticker',inplace=True)

        profit = []
        for t in df.index:
            temp = pd.read_csv('historical_data/' + t + '.csv')
            temp = temp.iloc[:-9]
            future = temp['Adj Close'].iloc[-1] # closing price on 4/14/20
            curr = df.loc[t,'Price'] # closing price on 3/31/20
            if future > curr:
                profit.append(True)
            else:
                profit.append(False)
        df['14day Profit'] = profit
        df.to_csv('dt_test3.csv')

    def decision_tree(self):
        test1 = pd.read_csv('dt_test1.csv',index_col='Ticker')
        y_test1 = test1['14day Profit'] # target values
        test1.drop(['14day Profit'],axis=1,inplace=True) # drop target vals from testing data
        
        
        test2 = pd.read_csv('dt_test2.csv',index_col='Ticker')
        y_test2 = test2['14day Profit'] # target values
        test2.drop(['14day Profit'],axis=1,inplace=True) # drop target vals from testing data
        
        train = pd.read_csv('dt_train.csv',index_col='Ticker') # training data

        # create decision tree model and build the tree
        model = DecisionTree(max_depth=5)
        root = model.build_tree(train,model.max_depth,1)
        model.print(root)

        # train
        y_train = train['14day Profit'] # target values
        train.drop(['14day Profit'],axis=1,inplace=True)
        preds = []
        for t in train.index:
            pred = model.predict(root,train.loc[t,])
            preds.append(pred)
        train['14day Profit'] = y_train
        train['Prediction'] = preds
        print(train)
        print('Accuracy:',model.accuracy_metric(train))
        #test1.to_csv('dt_train_results.csv')

        # test1
        preds = []
        for t in test1.index:
            pred = model.predict(root,test1.loc[t,])
            preds.append(pred)
        test1['14day Profit'] = y_test1
        test1['Prediction'] = preds
        print(test1)
        print('Accuracy:',model.accuracy_metric(test1))
        #test1.to_csv('dt_test1_results.csv')

        # test2
        preds = []
        for t in test2.index:
            pred = model.predict(root,test2.loc[t,])
            preds.append(pred)
        test2['14day Profit'] = y_test2
        test2['Prediction'] = preds
        print(test2)
        print('Accuracy:',model.accuracy_metric(test2))
        #test2.to_csv('dt_test2_results.csv')

if __name__ == '__main__':
    data = 'dt_train.csv'
    #data = 'historical_data/AAL.csv'
    data = pd.read_csv(data)
    
    # utilities
    '''
    c = daily_pct_change(data)
    d = daily_return(data)
    sma = simple_moving_avg(data,20)
    ema = exp_moving_average(data,20)
    print(c)
    print(d)
    print(sma)
    print(ema)
    weekly_pct_change(data)
    monthly_pct_change(data)
    weekly_return(data)
    monthly_return(data)
    '''

    # price forecasting
    #f = Forecast(data)
    #f.last_value()
    #f.moving_average(5)
    #f.linear_regression()
    #f.lr_eval()
    
    
    # signal generation
    
    data = pd.read_csv('dt_test2.csv')
    
    s = Signal(data)
    #s.moving_avg_crossSMA()
    #s.moving_avg_crossEMA()
    #s.turtle()
    #s.mean_reversion()
    #res = s.results
    #res['14day Profit'] = data['14day Profit']
    #res['SMA Cross Accuracy'] = round(sum(res['SMA Cross'] == res['14day Profit'])/len(res) * 100,2)
    #res['EMA Cross Accuracy'] = round(sum(res['EMA Cross'] == res['14day Profit'])/len(res) * 100,2)
    #res['Turtle Accuracy'] = round(sum(res['Turtle'] == res['14day Profit'])/len(res) * 100,2)
    #res['Mean Reversion Accuracy'] = round(sum(res['Mean Reversion'] == res['14day Profit'])/len(res) * 100,2)
    #res.index = data['Ticker']
    #print(res)
    #s.decision_tree_prep()
    s.decision_tree()
    #st = ['CVX','NVAX','PCG','IMGN','PSX','HON','DENN','CLX','JCI','SEDG','NBL','QEP','KR','MRVL','PEP','VLO','LMT','HMC']
    #s.generate_data(st)
    

    # base strategy accuracy
    '''
    s.results['14day Profit'] = data['14day Profit']
    res = s.results
    res['SMA Cross Accuracy'] = round(sum(res['SMA Cross'] == res['14day Profit'])/len(res) * 100,2)
    res['EMA Cross Accuracy'] = round(sum(res['EMA Cross'] == res['14day Profit'])/len(res) * 100,2)
    res['Turtle Accuracy'] = round(sum(res['Turtle'] == res['14day Profit'])/len(res) * 100,2)
    res['Mean Reversion Accuracy'] = round(sum(res['Mean Reversion'] == res['14day Profit'])/len(res) * 100,2)
    res.index = data['Ticker']
    print(res)
    res.to_csv('base_strategy_results.csv')
    '''
