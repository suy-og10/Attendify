# This file now just creates the app using the factory

from backend import create_app

app = create_app()

if __name__ == '__main__':
    # Check if DB exists before running, guide user if not
    import os
    from backend.config import Config
    # debug=True is for development only! Remove for production.
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0')