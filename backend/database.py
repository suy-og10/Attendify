# backend/database.py
import mysql.connector
import click
from flask import current_app, g
from flask.cli import with_appcontext
import os

def get_db():
    """Connects to the MySQL database specified in config."""
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=current_app.config['MYSQL_HOST'],
                user=current_app.config['MYSQL_USER'],
                password=current_app.config['MYSQL_PASSWORD'],
                database=current_app.config['MYSQL_DB']
            )
            current_app.logger.info(f"MySQL connection opened to {current_app.config['MYSQL_DB']}@{current_app.config['MYSQL_HOST']}")
        except mysql.connector.Error as err:
            current_app.logger.error(f"MySQL connection error: {err}")
            raise
    return g.db

def close_db(e=None):
    """Closes the database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()
        current_app.logger.info("MySQL connection closed.")

def init_db():
    """Applies the schema_mysql.sql to the database."""
    db = get_db()
    cursor = db.cursor() # Get a cursor
    schema_path = os.path.join(current_app.root_path, 'schema_mysql.sql')
    try:
        with current_app.open_resource(schema_path, mode='r') as f:
            sql_script = f.read()
            # Execute with multi=True to handle multiple statements properly
            for result in cursor.execute(sql_script, multi=True):
                pass # Consume results to ensure all statements execute
        db.commit()
        current_app.logger.info("MySQL database schema applied.")
    except FileNotFoundError:
        current_app.logger.error(f"Schema file not found at {schema_path}")
    except mysql.connector.Error as e:
        current_app.logger.error(f"Error applying MySQL schema: {e}")
        db.rollback()
    except Exception as e:
         current_app.logger.error(f"Unexpected error initializing DB: {e}")
         db.rollback()
    finally:
        cursor.close() # Always close the cursor

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear existing data (optional) and create new tables."""
    click.echo('Initializing the MySQL database...')
    init_db()
    click.echo('Initialized the MySQL database.')

def init_app(app):
    """Register database functions with the Flask app."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

# --- UPDATED Helper functions for queries ---
def query_db(query, args=(), one=False):
    """Executes a SELECT query using a cursor and returns results."""
    db = get_db()
    cursor = db.cursor(dictionary=True) # Get rows as dictionaries
    try:
        cursor.execute(query, args)
        results = cursor.fetchall()
        return (results[0] if results else None) if one else results
    except mysql.connector.Error as err:
        current_app.logger.error(f"Query failed: {query} \nArgs: {args}\nError: {err}")
        return None # Or raise an exception
    finally:
        cursor.close() # Close the cursor

def execute_db(query, args=()):
    """Executes an INSERT/UPDATE/DELETE query using a cursor."""
    db = get_db()
    cursor = db.cursor() # No dictionary needed for execute
    last_id = None
    try:
        cursor.execute(query, args)
        db.commit() # Commit changes
        last_id = cursor.lastrowid
    except mysql.connector.Error as err:
        current_app.logger.error(f"Execute failed: {query} \nArgs: {args}\nError: {err}")
        db.rollback() # Rollback on error
        raise # Re-raise the error
    finally:
        cursor.close() # Close the cursor
    return last_id