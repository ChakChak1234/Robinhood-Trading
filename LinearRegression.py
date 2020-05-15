import pandas as pd
import numpy as np
from utilities import *
from robinhood import *

client = Robinhood()

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
        self.cols.remove('Adj Close')
        self.cols.remove('Close')
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

        x_train = x.iloc[:-14].to_numpy()
        y_train = y.iloc[:-14].to_numpy()
        x_test = x.iloc[-14:].to_numpy()
        y_test = y.iloc[-14:].to_numpy()

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
        
    def gradient_descent(self,x,y,beta,rate=0.01,iterations=1000):
        n = len(y)
        cost_hist = np.zeros(iterations)
        beta_hist = np.zeros((iterations,2))
        for i in range(iterations):
            pred = np.dot(x,beta)
            beta -= (1/n) * rate * (x.transpose().dot((pred-y)))
            beta_hist[i,:] = beta.transpose()
            cost_hist[i] = self.cost_function(x,y,beta)
        return beta,cost_hist,beta_hist
    
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
        x_test.index = self.x.index[-14:]
        print(x_test)
        return x_test
        
if __name__ == '__main__':
    stocks = ['DIS', 'MSFT', 'BAC', 'SNAP', 'UBER']
    df = client.get_historicals('DIS')
    
    model = LinearRegression(df)
    x_train,y_train,x_test,y_test = model.train_test_split()
    
    betas = np.random.randn(2,1)
    x_b = np.c_[np.ones((len(x_train),1)),x_train]
    b,chist,bhist = model.gradient_descent(x_b,y_train,betas)
    print(b)