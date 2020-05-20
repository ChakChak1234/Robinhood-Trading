import pymysql
from sqlalchemy import create_engine

'''
Class used to connect to MySql db
'''
class Database:    
    client_id = "c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS"
    sql_engine = create_engine('mysql+pymysql://root:zach@localhost/stocks')
   
    def __init__(self, db=None): 
        self.connection = self.sql_engine.connect()
    
if __name__ == '__main__':
    db = Database('stocks')
    print(db)