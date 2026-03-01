# backend/config.py
import os
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')

    ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = ENV == 'development'

    # --- MySQL Configuration ---
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost') # e.g., 'localhost' or '127.0.0.1'
    MYSQL_DB = os.environ.get('MYSQL_DB', 'attendify') # e.g., 'attendify_db'

    # Connection string for SQLAlchemy (recommended) or direct use
    # Using mysql-connector-python driver
    SQLALCHEMY_DATABASE_URI = f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}'
    # OR if using PyMySQL driver:
    # SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}'

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Session/Security Cookie Settings ---
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', '1' if not DEBUG else '0') == '1'

    # --- Other Configs ---
    UPLOAD_FOLDER = os.path.join(rootdir, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    FACE_SIMILARITY_THRESHOLD = 0.5
    FACE_CONFIDENCE_THRESHOLD = 0.65
    EMBEDDINGS_FILE = os.path.join(rootdir, 'test', 'known_embeddings.pkl') # Keep for now if needed

    # os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Optional
