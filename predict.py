import os
import pandas as pd
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

'''
                        Historical Data
        Feature1    Feature2    Feature3 ...    Target(y)
T                                                   y1
T+1                                                 y2
T+2                                                 y3

x -> ML Model -> Y

            Predicted Y
        x1      x2      x3      ....
T                                                   
T+1                                                 
T+2                                                 
'''

class LR:
    def __init__(self, data):
        self.data = data
        
    def predict(self):        
        # get close price
        df = self.data
        df.iloc[:] = df.iloc[::-1].values
        
        
        plt.plot(df['Date'], df['Adj Close'])
        #plt.show()

# classifies input data under optimal trading strategy
class KNN:
    def __init__(self,data):
        self.data = data
        self.momentum = Momentum(data)
        self.reversion = Reversion(data)

    def standardize(self):
        df = self.data[['Open','Adj Close']]

        # set moving avg cross col
        close = df[['Adj Close']]
        avgs = []
        for i in range(0,len(close)):
            avgs.append(moving_avg(close[i:],20))
        df['20day MA'] = avgs
        df = df.iloc[:len(df)-20]

        # set turtle col
        
        # set mean reversion col
        

        print(df)
        self.data = df
    
    def classify(self, neighbors=1):
        df = self.data

# Multilayer Perceptron for computing buy/sell signals      
class MLP:
    def __init__(self,data):
        self.data = data

'''
Believe movement of a stock will continue in its current direction
'''
class Momentum:
    def __init__(self,data):
        self.data = data
    
    # return 1 if 50-day short-term average crosses over 200-day long-term average else 0
    def dual_moving_avg_cross(self):
        df = self.data
        fifty = moving_avg(df,50)
        twohun = moving_avg(df,200)
        res = -1
        # buy signal
        if fifty > twohun:
            res = 1
        # sell signal
        if fifty < twohun:
            res = 0
        print('Moving Avg Cross:',res)
        return res
    
    # buy on a 20-day high and sell on a 20-day low
    def turtle(self):
        df = self.data['Adj Close']
        curr = df[0]
        high = df[1:21].max()
        low = df[1:21].min()
        res = -1
        
        # buy signal
        if curr > high:
            res = 1
        # sell signal
        if curr < low:
            res = 0
        print('Turtle:',res)
        return res
            
'''
Follows belief that movement of a stock will eventually reverse
'''
class Reversion:
    def __init__(self,data):
        self.data = data

    # believe stocks return to their mean & you can exploit when it deviates from the mean
    def mean_reversion(self):
        df = self.data
        ninety = moving_avg(df, 90) # 30 day moving avg
        thirty = moving_avg(df, 30) # 90 day moving avg
        res = -1
        # expect 30d avg to revert back to 90d avg price is too low and unlikely to increase, buy signal
        if thirty < ninety:
            res = 1
        
        # expect 30d avg to fall back to 90d avg curr price is too high, sell signal 
        if thirty > ninety:
            res = 0
        print('Mean Reversion:',res)
        return res

'''
Utility Functions
'''
# Calculates the daily percentage change in price of a stock
def daily_pct(df):
    close = df[['Adj Close']]
    change = close.pct_change()
    change.fillna(0,inplace=True)
    return change

# computes the moving average of a stock over the specified days
def moving_avg(df, days):
    df = df[:days]
    avg = df['Adj Close'].sum()/days
    return round(avg,2)

if __name__ == '__main__':
    data = 'historical_data/AAL.csv'
    data = pd.read_csv(data)
    #print(data)
    
    # utilities
    #c = print(daily_pct(data))
    #avg = moving_avg(data,50)

    # momentum
    m = Momentum(data)
    #m.dual_moving_avg_cross()
    #m.turtle()
    
    # reversion
    r = Reversion(data)
    #r.mean_reversion()

    # linear regression
    lr = LR(data)
    lr.predict()
    
    # support vector machine
    #svm = SVM(data)
    #svm.predict(1)
    
    # k nearest neighbors
    knn = KNN(data)
    #knn.standardize()
    
    '''# decision tree
    dt = DecisionTree(data)
    
    # random forest
    rf = RandomForest(data)
    
    # neural network
    nn = NeuralNetwork(data)'''
    