import stock
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
import time
import matplotlib as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from pandas_datareader import data    

# predict the price of a stock n days
def predict_stock(ticker, days):
    df = stock.get_data(ticker)
    df.index
    
    # new data is adjusted close price
    df = df[['Adj Close']]

    # add our dependent var to the data set and shift up
    df['Prediction'] = df[['Adj Close']].shift(-days)

    # Independent dataset
    # convert data from to array and remove last ndays rows
    X = np.array(df.drop(['Prediction'], 1))
    X = X[:-days]

    # Dependent dataset
    # convert to array, get all values except last ndays rows
    Y = np.array(df['Prediction'])
    Y = Y[:-days]

    # split 80% train 20% test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    # create & train linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test) # coefficient of determination R^2 of prediction
    print("lr confidence: ", lr_confidence)

    # last ndays rows of the original dataset from adj close
    forecast = np.array(df.drop(['Prediction'],1))[-days:]
    print(forecast)

    # linear regression model predictions for next n days
    lr_prediction = lr.predict(forecast)
    print(lr_prediction)
    
    return lr_prediction, round(lr_confidence,3)
    
def main():
    predict_stock('AAPL', 7)
    
if __name__ == '__main__':
    main()