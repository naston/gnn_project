# Module Imports
import mariadb
import sys

# Connect to MariaDB Platform
def connect():
    """Connect to CORA dataset

    Returns:
        Cursor: Cursor for MariaDB
    """
    try:
        conn = mariadb.connect(
            user="guest",
            password="relational",
            host="relational.fit.cvut.cz",
            port=3306,
            database="CORA"

        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return None

    # Get Cursor
    cur = conn.cursor()

    return cur