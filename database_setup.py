# This script is simpler now, relying on the Flask CLI command.
# You can run 'flask init-db' from your terminal in the Attendify root directory
# (after setting FLASK_APP=backend.app).

# Or, if you want a standalone script:
import sqlite3
import os

DATABASE_NAME = 'attendify.db'
SCHEMA_NAME = 'backend/schema.sql' # Path relative to Attendify root

def setup_database():
    """Initializes the database using the schema file."""
    db_path = DATABASE_NAME
    schema_path = SCHEMA_NAME

    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found at '{schema_path}'")
        return

    # Optional: Remove existing DB if you want a clean setup each time
    # if os.path.exists(db_path):
    #     print(f"Removing existing database '{db_path}'...")
    #     os.remove(db_path)

    print(f"Creating/Updating database '{db_path}' using '{schema_path}'...")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            conn.executescript(schema_sql)
        conn.commit()
        print("Database setup complete.")
    except sqlite3.Error as e:
        print(f"Database error during setup: {e}")
    except FileNotFoundError:
        print(f"Error: Could not read schema file '{schema_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Ensure the script is run from the Attendify root directory
    if not os.path.exists('backend'):
         print("Error: Please run this script from the main 'Attendify' project directory.")
    else:
        setup_database()