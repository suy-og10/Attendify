# This file now just creates the app using the factory

from backend import create_app

app = create_app()

if __name__ == '__main__':
    # Check if DB exists before running, guide user if not
    import os
    from backend.config import Config
    if not os.path.exists(Config.DATABASE):
        print(f"Database file not found at {Config.DATABASE}.")
        print("Run 'flask init-db' command first from the Attendify root directory to create the database.")
        print("(Ensure FLASK_APP=backend.app is set in your environment or use 'python -m flask --app backend.app init-db')")
    else:
        # debug=True is for development only! Remove for production.
        # Use host='0.0.0.0' to make it accessible on your network
        app.run(debug=True, host='0.0.0.0')