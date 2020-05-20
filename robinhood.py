import os
import requests
import random
import robin_stocks as rs
import yfinance as yf
import pandas as pd
import numpy as np
import db

from bs4 import BeautifulSoup


class Robinhood:
    username = None
    password = None
    auth_token = None
    refresh_token = None
    database = db.Database('stocks')
    
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
    
    '''
    Takes a ticker and writes historical data to csv file named after the stock
    returns a dataframe of historical data
    '''
    def get_historicals(self, stock, period='2y', end=None):
        if end:
            df = yf.download(stock, period=period,end=end)
        else:
            df = yf.download(stock, period=period)

        # remove null values
        df.dropna(inplace=True)
        
        # set type of date,open,close,high,low cols
        df = df.astype({'Open':'float64', 'Close':'float64', 'High':'float64', 'Low':'float64', 'Adj Close':'float64', 'Volume':'int64'})
        
        # round values to 3 decimal places
        df = df.round(3)
        
        # write df to csv file
        #df.to_csv('historical_data/' + stock + '.csv',index=True)

        return df    
    
    # scrapes Robinhood and returns the tickers associated with the specified collection
    def get_collection(self, url):
        res = requests.get(url)
        data = res.text
        soup = BeautifulSoup(data)
        soup = soup.find('tbody')
        
        tickers = []
        for row in soup.findAll('tr'):
            i = 0
            for item in row.findAll('td'):
                if i > 1:
                    break
                temp = item.findAll('span')
                i += 1   
            tickers.append(temp[0].text)

        print(tickers)
        return tickers
        
if __name__ == '__main__':
    client = Robinhood()
    #client.login('', '')
    client.get_collection('https://robinhood.com/collections/100-most-popular')
    
