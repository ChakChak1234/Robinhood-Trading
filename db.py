import mysql.connector as mysql


'''
Class used to connect to MySql db
'''
class Database:
    connection = None
    cursor = None
    
    def __init__(self, db=None):
        self.connection = mysql.connect(
            host='localhost',
            user='root',
            passwd='zach',
            db = db
        )
        self.cursor = self.connection.cursor()

    def create_db(self, name):
        self.cursor.execute('CREATE DATABASE ' + name)

    def run_query(self, query):
        self.cursor.execute(query)
        
def db_main():
    db = Database('stocks')
    print(db.connection)
    
    
    
if __name__ == '__main__':
    db_main()