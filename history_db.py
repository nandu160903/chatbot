import sqlite3

DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id, user_id, role, message):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (session_id, user_id, role, message)
        VALUES (?, ?, ?, ?)
    ''', (session_id, user_id, role, message))
    conn.commit()
    conn.close()

def get_chat_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT role, message, timestamp FROM chat_history
        WHERE user_id = ?
        ORDER BY timestamp ASC
    ''', (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows
