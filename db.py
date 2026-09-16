import psycopg2
from werkzeug.security import generate_password_hash

def setup_database():
    try:
        # Connect to your PostgreSQL instance
        conn = psycopg2.connect(
            dbname="Chat",
            user="postgres",
            password="Bel123",
            host="localhost"
        )
        cur = conn.cursor()

        # 1. Create the table as defined in your schema
        create_table_query = """
        CREATE TABLE IF NOT EXISTS login (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            reset_code VARCHAR(6),
            code_expiry TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        print("✅ Table 'login' checked/created successfully.")

        # 2. Generate a secure hash for the password 'test123'
        hashed_password = generate_password_hash("test123")

        # 3. Insert the initial admin user
        insert_query = """
        INSERT INTO login (username, email, password_hash)
        VALUES (%s, %s, %s)
        ON CONFLICT (username) DO NOTHING;
        """
        cur.execute(insert_query, ("admin", "admin@smartcare.com", hashed_password))
        
        conn.commit()
        print("✅ Admin user 'admin' created with password 'test123'.")

    except Exception as e:
        print(f"❌ Error during setup: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    setup_database()