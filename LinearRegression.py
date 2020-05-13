import pandas as pd
import numpy as np
from utilities import *

# y = B0 + B1*x1 + ... + Bn*xn
class LinearRegression:
    def __init__(self,data):
        self.data = data #data for model
        self.coef = [] # beta values
        self.x = None # independent vars
        self.y = None # dependent vars
        self.cols = list(data.columns)
        self.prep()

    def prep(self):
        df = self.data
        df = df.iloc[5:]
        self.cols.remove('Adj Close')
        self.x = df[self.cols] # set x as predictors
        self.y = df['Adj Close'] # set y as target col
    
    def info(self):
        print('x:\n',self.x)
        print('y:\n',self.y)
        print('coeficients:\n',self.coef)

    # split data into 60% training 40% testing
    def train_test_split(self):
        x = self.x
        y = self.y

        x_train = x.iloc[:300].to_numpy()
        y_train = y.iloc[:300].to_numpy()
        x_test = x.iloc[300:].to_numpy()
        y_test = y.iloc[300:].to_numpy()

        return x_train,y_train,x_test,y_test

    def ones(self,x):
        ones = np.ones(shape=x.shape[0]).reshape(-1,1)
        return np.concatenate((ones,x),1)

    # b = (x_T dot x)^-1 dot x_T dot y
    def fit(self,x,y):
        # add col of ones to x matrix
        x = self.ones(x)
        # generate coeficients 
        coef = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
        self.coef = coef

    def predict(self,x_test,y_test):
        predictions = []

        b0 = self.coef[0] # first coeficient
        b = self.coef[1:] # other coeficients

        # loop through each row in x matrics and predict
        for row in x_test:
            pred = b0
            for xi,bi in zip(row,b):
                pred += (xi*bi)
            predictions.append(round(pred,2))

        x_test = pd.DataFrame(x_test,columns=self.cols)
        x_test['Actual Price'] = y_test
        x_test['Prediction'] = predictions
        x_test.index = self.x.index[300:]
        #print(x_test)
        return x_test
        
if __name__ == '__main__':
    data = 'historical_data/AAL.csv'
    df = pd.read_csv(data)
    df['Daily % Change'] = daily_pct_change(df)
    df['5day SMA'] = simple_moving_avg(df,5)
    df['5day EMA'] = exp_moving_average(df,5)
    df.drop(['Close'],axis=1,inplace=True)
    df.set_index('Date',inplace=True)

    lr  = LinearRegression(df)
    x_train,y_train,x_test,y_test = lr.train_test_split()
    lr.fit(x_train,y_train)
    lr.predict(x_test,y_test)