import requests
import random
import robin_stocks as rs
import yfinance as yf
import pandas as pd
import numpy as np
import db
from datetime import datetime

import pymysql
from sqlalchemy import create_engine

class Robinhood:
    username = None
    password = None
    auth_token = None
    refresh_token = None
    
    # database
    client_id = "c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS"
    sql_engine = create_engine('mysql+pymysql://root:zach@localhost/stocks')
    connection = sql_engine.connect()
    db = db.Database('stocks')
    
    def __init__(self):
        self.device_token = self.GenerateDeviceToken() # generate device token upon initialization
        
    # generates a device token for the user
    # FIND SOURCE FOR THIS 
    def GenerateDeviceToken(self):
        rands = []
        for i in range(0,16):
            r = random.random()
            rand = 4294967296.0 * r
            rands.append((int(rand) >> ((3 & i) << 3)) & 255)

        hexa = []
        for i in range(0,256):
            hexa.append(str(hex(i+256)).lstrip("0x").rstrip("L")[1:])

        id = ""
        for i in range(0,16):
            id += hexa[rands[i]]
            if (i == 3) or (i == 5) or (i == 7) or (i == 9):
                id += "-"

        device_token = id
        return device_token

    '''
    Logs user into their robinhood account
    '''
    def login(self, username, password):
        self.username = username
        self.password = password
        
        # ensure that a device token has been generated
        if self.device_token == "":
                self.GenerateDeviceToken()

        # login via the robinhood api and update global authentication/refresh tokens
        login = rs.login(username, password)
        self.auth_token = login.get('access_token')
        self.refresh_token = login.get('refresh_token')
        return login
    
    # logs user out 
    def logout(self):
        logout = rs.logout()
        self.auth_token = None
        return logout
    
    # prints details of dataframe
    def get_details(self, data):
        print(data.columns)
        print(data.shape)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum()) # get cols with null vals
    
    '''
    Takes list of tickers and returns list of latest prices for input stocks
    '''
    def get_prices(self, stocks):
        prices = rs.get_latest_price(stocks)
        for i in range(len(prices)):
            prices[i] = round(float(prices[i]),2)
        print(prices)
        return prices
    
    '''
    Takes list of ticker symbols and returns a list of stock names
    '''
    def get_names(self, stocks):
        names = []
        for s in stocks:
            names.append(rs.get_name_by_symbol(s))
        print(names)
        return names
    
    '''
    Takes a ticker and writes historical data to csv file named after the stock
    returns a dataframe of historical data
    '''
    def get_historicals(self, stock):
        # read dict of historical data into df
        df = pd.DataFrame.from_dict(rs.get_historicals(stock, span='year'))
        
        # reverse indicies so dates are in decsending order
        df.iloc[:] = df.iloc[::-1].values
        
        # drop worthless cols and rename others
        df.drop(['interpolated', 'session'], axis=1,inplace=True)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        
        # set type of date,open,close,high,low cols
        df = df.astype({'Date':'datetime64[ns]', 'Open':'float64', 'Close':'float64', 'High':'float64', 'Low':'float64', 'Volume':'int64'})
        
        # write df to csv file
        df.to_csv('historical_data/'+stock[0]+'.csv',index=False,)
        return df
    
    '''
    Updates holdings database table and returns users portfolio as dataframe
    '''
    def get_holdings(self):
        df = pd.DataFrame.from_dict(rs.build_holdings(),'index') # read dict consisting of users portfolio into a df

        # drops useless cols
        df.drop(['average_buy_price', 'equity', 'equity_change', 'type', 'id', 'pe_ratio'], axis=1, inplace=True)

        # Set column names
        df.columns = ['Price', 'Quantity', 'Percent Change', 'Name','Portfolio Percentage']

        # reorder cols
        df = df[['Name', 'Price', 'Percent Change', 'Quantity', 'Portfolio Percentage']]
        
        # convert ticker indicies to column
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Ticker'}, inplace=True)
        
        # set type of date,open,close,high,low cols
        df = df.astype({'Price':'float64', 'Percent Change':'float64', 'Quantity':'float64'})
        df['Quantity'] = df['Quantity'].astype('int64')

        # write df to holdings table in MySQL db
        df.to_sql('holdings', con=self.connection, if_exists='replace')
        return df
    
    '''
    Updates watchlist database table and returns users watchlist as dataframe
    '''
    def get_watchlist(self):
        df = {} # dictionary to create watchlist dataframe
        tickers = []
        
        # return none if watchlist is empty
        if rs.get_watchlist_by_name() is None:
            return None
        
        for d in rs.get_watchlist_by_name():
            url = d.get('instrument')
            ticker = rs.request_get(url).get('symbol')
            tickers.append(ticker)
        
        # create watchlist dataframe
        df = {'Ticker':tickers, 'Name':self.get_names(tickers), 'Price':self.get_prices(tickers)}
        df = pd.DataFrame(df)
        
        # change price col to float
        df = df.astype({'Price':'float64'})
        
        # write df to watchlist table in MySQL db
        df.to_sql('watchlist', con=self.connection, if_exists='replace')
        return df
                
def r_main():
    client = Robinhood()
    client.login('zloeffler22@gmail.com', 'Zach4268!12')
    s = ['AAL']
    #client.get_prices(s)
    #client.get_names(s)
    #client.get_historicals(s)
    #client.get_holdings()
    #client.get_watchlist()
    
if __name__ == '__main__':
    r_main()