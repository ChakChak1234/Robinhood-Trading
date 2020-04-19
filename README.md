# Robinhood Bot
Web application that interacts with a user's Robin hood account to analyze historical data, predict future prices, and execute buy/sell orders via algorithmic trading. 

## Inspiration
My inspiration for this project began with my interest in Machine Learning and Artificial Intelligence. Having minimal experience in the stock market, I thought this project would be an exciting opportunity to learn more about the stock market, improve my web-development skills, and apply my knowledge of machine learning and artificial intellegence to something that could potentially repay me in cold, hard cash. 

## Development
The application is written using Python 3, MySql, HTML5, CSS, and utilizes several libraries for its functionality
* robin-stocks - interacts with the Robinhood API allows users to get account data, get realtime information, and execute buy/sell orders
* Flask - framework that drives the web-application, chosen because of its simplicity and scalability 
* pandas/numpy - used for data analysis and cleaning
* pymysql/sqlalchemy/mysql.connector - used to connect to MySql db and execute read/write operations with pandas dataframes 
* sklearn - used for the prediction algorithm

## Current Features
* View user Portfolio and Watchlist data with buy/sell recommendations
* Add/Remove stocks from watchlist
* Analyze metrics for a particular stock and view daily, monthly, and quartarly changes
* Predict the price of a stock over a number of days
* More to come soon

## Components
#### robinhood.py
Class that handles any functionality that requires interaction with the Robinhood API. Also contains utility functions for calculating
stock metrics.

#### app.py
Functions for the web-application's main structure and functionality

#### db.py
Allows connection to MySql database

#### predict.py
Code for stock prediction using Linear Regression

#### templates
Directory containing HTML pages for the main application
