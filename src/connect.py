# Module Imports
import mariadb
import sys

connection = None
cursor = None

def connect_imdb():
    """Connect to IMDb dataset

    Returns:
        Cursor: Cursor for MariaDB
    """
    return connect('imdb_ijs')

def connect_cora():
    """Connect to CORA dataset

    Returns:
        Cursor: Cursor for MariaDB
    """
    return connect('CORA')

def connect_citeseer():
    """Connect to CiteSeer dataset

    Returns:
        Cursor: Cursor for MariaDB
    """
    return connect('CiteSeer')

# Connect to MariaDB Platform
def connect(db_name):
    global connection
    global cursor
    close()

    try:
        conn = mariadb.connect(
            user="guest",
            password="relational",
            host="relational.fit.cvut.cz",
            port=3306,
            database=db_name

        )
        connection=conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return None

    # Get Cursor
    cur = conn.cursor()
    cursor=cur

    return cur

def close():
    global connection
    global cursor

    if cursor:
        cursor.close()
        cursor=None
    if connection:
        connection.close()
        connection=None