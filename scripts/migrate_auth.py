import sqlite3
import os

DB_PATH = 'backend/database.db'

def update_db():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Add phone column to users (Simpler version)
    try:
        # SQLite limitation: Cannot ADD COLUMN with UNIQUE constraint directly if table populated
        # We will add it as normal text, application will enforce uniqueness
        c.execute('ALTER TABLE users ADD COLUMN phone TEXT')
        print("Added 'phone' column to 'users' table.")
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e).lower():
            print("'phone' column already exists in 'users' table.")
        else:
            print(f"Error adding phone column: {e}")

    # 2. Create OTPs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            otp TEXT NOT NULL,
            type TEXT NOT NULL, 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            used BOOLEAN DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    print("Created/Verified 'otps' table.")

    conn.commit()
    conn.close()

if __name__ == '__main__':
    update_db()
