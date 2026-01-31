import sqlite3

def inspect():
    db_path = 'database.db'
    print(f"Connecting to {db_path}...")
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        print("\n=== USERS TABLE ===")
        print(f"{'ID':<5} {'Name':<20} {'Email':<30}")
        print("-" * 60)
        for row in c.execute('SELECT id, name, email FROM users'):
            print(f"{row[0]:<5} {row[1]:<20} {row[2]:<30}")

        print("\n=== PREDICTIONS TABLE ===")
        print(f"{'ID':<5} {'User ID':<10} {'Label':<20} {'Conf':<10} {'Timestamp'}")
        print("-" * 70)
        for row in c.execute('SELECT id, user_id, label, confidence, timestamp FROM predictions'):
            print(f"{row[0]:<5} {row[1]:<10} {row[2]:<20} {row[3]:<10} {row[4]}")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
