'''
robinhood.py: interacts with the Robinhood API to allow users to login, logout, and use any robinhood functionality
that requires authentication
'''

import requests
import random
import robin_stocks as r
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

    # allows users to log into their robinhood account
    def login(self, username, password):
        self.username = username
        self.password = password
        
        # ensure that a device token has been generated
        if self.device_token == "":
                self.GenerateDeviceToken()

        # login via the robinhood api and update global authentication/refresh tokens
        login = r.login(username, password)
        self.auth_token = login.get('access_token')
        self.refresh_token = login.get('refresh_token')
        return login
    
    # logs user out 
    def logout(self):
        logout = r.logout()
        self.auth_token = None
        return logout
    
    '''
    returns a list of current prices for input stocks
    '''
    def price(self, stocks):
        p = [round(float(i),2) for i in r.get_latest_price(stocks)]
        return p
    
    '''
    input: list of ticker symbols
    out: list of names corresponding to the input list
    '''
    def names(self, stocks):
        temp = r.get_instruments_by_symbols(stocks)
        names = []
        for t in temp:
            if t is not None:
                names.append(t.get('name'))
        return names
    
    
    '''
    returns the n day moving averages for input stocks
    '''
    def get_nday_moving_avg(self, ticker, days):
        ticker = ticker.upper()
        df = yf.download(ticker, period=str(days)+'d')
        avg = (round(df['Adj Close'].sum()/days,2))
        return avg
    
    '''
    updates holdings database table and returns users portfolio as dataframe
    '''
    def get_holdings(self):
        portfolio = r.build_holdings() # dictionary consisting of users portfolio
        tickers = list(portfolio.keys()) # extract ticker symbols from portfolio
        data = {} # dictionary to be converted into dataframe
        
        # if nothing in holdings, clear db table and global var is set to None
        if len(tickers) == 0:
            self.db.run_query('TRUNCATE TABLE holdings')
            return None
        
        quantities = [] # quantities of stocks owned
        percentages = [] # percentage of portfolio each stock makes
        # extract quantity and percentage  
        for k in portfolio.keys():
            info = portfolio.get(k)
            quantities.append(info.get('quantity'))
            percentages.append(info.get('percentage'))
        
        # populate dictionary and convert to dataframe
        data = {'Ticker':tickers, 'Name':self.names(tickers), 'Price':self.price(tickers), 'Quantity':quantities, 'Percentage':percentages, 'Sell':False}
        data = pd.DataFrame(data)

        # write dataframe to holdings table in MySql db
        data.to_sql('holdings', con=self.connection, if_exists='replace')
        return data
    
    '''
    updates watchlist database table and returns users watchlist as dataframe
    '''
    def get_watchlist(self):
        data = {} # dictionary to create watchlist dataframe
        tickers = [] # ticker symbols in watchlist
        
        # return none if watchlist is empty
        if r.get_watchlist_by_name() is None:
            return None
        
        for d in r.get_watchlist_by_name():
            url = d.get('instrument')
            ticker = r.request_get(url).get('symbol')
            tickers.append(ticker)
        
        # create watchlist dataframe and write it to db table
        data = {'Ticker':tickers, 'Name':self.names(tickers), 'Price':self.price(tickers), 'Buy': False}
        data = pd.DataFrame(data)
        
        data.to_sql('watchlist', con=self.connection, if_exists='replace')
        return data        
    
    '''
    adds symbols to users watchlist
    '''
    def add_to_watchlist(self, tickers):
        # add ticker to watchlist and update db table
        r.post_symbols_to_watchlist(tickers)
        self.get_watchlist()
    
    '''
    removes symbols from users watchlist
    '''
    def remove_from_watchlist(self, tickers):
        try:
            # remove ticker from watchlist and update db table
            r.delete_symbols_from_watchlist(tickers)
            self.get_watchlist()
        except UnboundLocalError as error: # handles some kind of strange error associated with the delete function
            pass
    
    '''
    scans portfolio and watchlist for stocks to buy/sell
    and updates db table with buy/sell signals
    '''
    def scan_stocks(self):
        w = self.get_watchlist()
        h = self.get_holdings()
        signals = []
        
        # update watchlist dataframe with buy signals based on 50 and 200 day moving average
        # signal is set to false if stock is already owned
        for s in list(w['Ticker']):
            fifty_day = self.get_nday_moving_avg(s, 50)
            two_hun_day = self.get_nday_moving_avg(s, 200)
            #print(s,fifty_day, two_hun_day)
            if fifty_day > two_hun_day and s not in list(h['Ticker']):
                signals.append(True)
            else:
                signals.append(False)
        w['Buy'] = signals
        
        # update holdings dataframe with sell signals based on 50 and 200 day moving average
        signals = []
        for s in list(h['Ticker']):
            fifty_day = self.get_nday_moving_avg(s, 50)
            two_hun_day = self.get_nday_moving_avg(s, 200)
            #print(s,fifty_day, two_hun_day)
            if fifty_day < two_hun_day:
                signals.append(True)
            else:
                signals.append(False)
        h['Sell'] = signals
        
        # update holdings and watchlist db tables
        w.to_sql('watchlist', con=self.connection, if_exists='replace')
        h.to_sql('holdings', con=self.connection, if_exists='replace')
    
    '''
    Finds stocks in watchlist and portfolio that have buy/sell signals and executes the order
    Also logs each transaction in db table
    '''
    def run_trade_bot(self):
        # update db table with buy/sell signals for current price and then get the data
        self.scan_stocks()
        holdings = pd.read_sql('SELECT * FROM stocks.holdings WHERE Sell = 1', self.connection)
        watchlist = pd.read_sql('SELECT * FROM stocks.watchlist WHERE Buy = 1', self.connection)
        #cash = float(r.load_portfolio_profile().get('withdrawable_amount'))
        
        '''
        If there's a stock in users portfolio with a sell signal execute sell order, create dataframe 
        for the transaction, and write it to db table
        '''
        if holdings is not None:
            for t in list(holdings['Ticker']): 
                #r.order_sell_market(t, 1)
                data = {'Date': datetime.now().strftime('%d/%m/%Y %H:%M'), 'Ticker': t, 'Name': self.names(t), 
                        'Quantity': 1, 'Price': self.price(t), 'Order': 'Sell'}
                data = pd.DataFrame(data)
                data.to_sql('history', con=self.connection, if_exists='append')
        
        '''
        If there's a stock in users watchlist with a buy signal execute buy order, create dataframe 
        for the transaction, and write it to db table
        '''
        if watchlist is not None:
            for t in list(watchlist['Ticker']):
                #r.order_buy_market(t, 1)
                data = {'Date': datetime.now().strftime('%d/%m/%Y %H:%M'), 'Ticker': t, 'Name': self.names(t), 
                        'Quantity': 1, 'Price': self.price(t), 'Order': 'Buy'}
                data = pd.DataFrame(data)
                data.to_sql('history', con=self.connection, if_exists='append')
                
def r_main():
    client = Robinhood()
    client.login('email', 'passwd')
    client.run_trade_bot()
if __name__ == '__main__':
    r_main()