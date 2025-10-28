# database.py
import sqlite3
import pandas as pd
from datetime import datetime

def init_database():
    conn = sqlite3.connect('creditcard_transactions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL,
            v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL, v6 REAL, v7 REAL, v8 REAL, v9 REAL,
            v10 REAL, v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL,
            v20 REAL, v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,
            amount REAL,
            prediction INTEGER,
            actual_class INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(transaction_data, prediction, actual_class=None):
    conn = sqlite3.connect('creditcard_transactions.db')
    cursor = conn.cursor()
    
    columns = ['time'] + [f'v{i}' for i in range(1, 29)] + ['amount']
    placeholders = ','.join(['?'] * len(columns))
    
    cursor.execute(f'''
        INSERT INTO transactions ({','.join(columns)}, prediction, actual_class)
        VALUES ({placeholders}, ?, ?)
    ''', list(transaction_data) + [prediction, actual_class])
    
    conn.commit()
    conn.close()

def get_all_transactions():
    conn = sqlite3.connect('creditcard_transactions.db')
    df = pd.read_sql('SELECT * FROM transactions ORDER BY timestamp DESC', conn)
    conn.close()
    return df