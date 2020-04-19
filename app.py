'''
app.py: drives the Flask web application, utilizes a robinhood object to give user access when
necessary
'''

from flask import Flask, render_template, redirect, url_for, request
from stock import *
import robinhood
import predict
import random
#import StringIO
import matplotlib.pyplot as plt

app = Flask('Robinhood') # initalize flask app
user = robinhood.Robinhood() # user object

'''
Login page for users, if login is successful they are redirected to the home page and an auth token
is created/stored
'''
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        try:
            user.login(request.form['Email'], request.form['Password'])
            return redirect(url_for('home'))
        except:
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

''' Logs a user out and redirects them to the login page '''
@app.route('/logout')
def logout():
    user.logout()
    return redirect('/')

''' 
Homepage displaying a users current holdings, watchlist, and buy/sell functionality
Watchlist and portfolio also includes buy/sell recomendations for each stock
'''
@app.route('/home', methods=['GET', 'POST'])
def home():
    user.get_holdings()
    user.get_watchlist()
    holdings = pd.read_sql('SELECT * FROM stocks.holdings', user.connection)
    watchlist = pd.read_sql('SELECT * FROM stocks.watchlist', user.connection)
    
    if holdings is None and watchlist is not None:
        return render_template('home.html', watchlist=watchlist.to_html(table_id='watchlist'))
    if holdings is not None and watchlist is None:
        return render_template('home.html', holdings=holdings.to_html(table_id='holdings'))
    
    # Add a stock to watchlist
    if request.form.get('add') is not None and request.method == 'POST':
        new = request.form.get('add').upper()
        user.add_to_watchlist([new])
    
    # Remove a stock from watchlist
    if request.form.get('remove') is not None and request.method == 'POST':
        remove = request.form.get('remove').upper()
        user.remove_from_watchlist([remove])
    
    return render_template('home.html', holdings=holdings.to_html(table_id='holdings'), 
                                        watchlist=watchlist.to_html(table_id='watchlist'))

''' 
 User can enter a ticker for a particular stock to see helpful analytics such as:
 - open, high, low, close prices and day-to-day difference
 - daily & monthly return
 - daily, monthly, & quartarly changes
'''
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    data = None
    if request.method == 'POST':        
        if request.form.get('stock') is not None:
            data = get_data(request.form['stock'].upper())
            data1 = pd.DataFrame(difference(request.form['stock'].upper()))
            data2 = pd.DataFrame(pct_change(request.form['stock'].upper(), 'd'))
            data3 = pd.DataFrame(pct_change(request.form['stock'].upper(), 'm'))
            data4 = pd.DataFrame(pct_change(request.form['stock'].upper(), 'q'))
            data5 = pd.DataFrame(daily_return(request.form['stock'].upper()))
            data6 = pd.DataFrame(monthly_return(request.form['stock'].upper()))
            data = pd.concat([data, data1, data5, data6, data2, data3, data4], axis=1)   
            data.columns = ['Open', 'High', 'Low','Adj Close', 'Close', 'Volume', 'Difference',
                            'Daily Return', 'Monthly Return', 'Daily % Change', 'Monthly % Change',
                            'Quarterly % Change']
            data = data.iloc[::-1]
            data = data.drop(['Close', 'Volume', 'High', 'Low'], axis=1)
            
        if data is not None:
            return render_template('analyze.html', data=data.to_html(table_id='results'))                           
    return render_template('analyze.html')


'''
Predicts the price of a stock for a specified number of days
Utilizes both Linear Regression and Support Vector Machine prediction algorithms and displays the calculated
confidence for each prediction
Users can enter the ticker for a particular stock and number of days
'''
@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        stock = request.form.get('stock')
        days = request.form.get('days')        
        if request.form.get('stock') is not None and request.form.get('days') is not None:
            data = pd.DataFrame(predict.predict_stock(stock.upper(), int(days))[0])
            data.columns = ['Linear Regression']            
            lr_conf = predict.predict_stock(stock.upper(), int(days))[1]
                                            
            return render_template('predict.html', data=data.to_html(table_id='prediction'), lr=lr_conf) 
    return render_template('predict.html')

@app.route('/trade', methods=['GET', 'POST'])
def trade():
    pass

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
    