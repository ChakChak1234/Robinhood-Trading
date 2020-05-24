import pandas as pd
import numpy as np
from utilities import *
from robinhood import *

client = Robinhood()

# y = B0 + B1*x1 + ... + Bn*xn
class LinearRegression:
    def __init__(self):
        self.coef = [] # beta values

    # split data into 60% training 40% testing
    def train_test_split(self,x,y,start,end=None):
        x_train = x.loc[:start].to_numpy()
        y_train = y.loc[:start].to_numpy()
        x_test = x.loc[start:].to_numpy()
        y_test = y.loc[start:].to_numpy()

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
        
    def cost_function(self,x,y,beta):
        n = len(y)
        preds = x.dot(beta)
        j = (1/2*n) * np.sum(np.square(preds-y))
        return j
    
    def predict(self,x_test):
        predictions = []

        b0 = self.coef[0] # first coeficient
        b = self.coef[1:] # other coeficients

        # loop through each row in x matrics and predict
        for row in x_test:
            pred = b0
            for xi,bi in zip(row,b):
                pred += (xi*bi)
            predictions.append(round(pred,2))

        return predictions
        
if __name__ == '__main__':
    stocks = ['DIS', 'MSFT', 'BAC', 'SNAP', 'UBER']
    df = client.get_historicals('DIS')
    
    model = LinearRegression()