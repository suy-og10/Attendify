from functools import wraps
from flask import session, redirect, url_for, flash, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import base64
import io


# --- Password Hashing ---
def hash_password(password):
    return generate_password_hash(password)

def verify_password(stored_hash, provided_password):
    return check_password_hash(stored_hash, provided_password)
# (This file was missing, and your routes.py imports it. This is the correct content.)


# --- Image Utilities ---

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    """Checks if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_image_from_base64(base64_string):
    """
    Decodes a Base64 string (from data URL) into a BGR NumPy array.

    Args:
        base64_string (str): The Base64 encoded image string (e.g., "data:image/jpeg;base64,...")

    Returns:
        tuple: (img_np, error_message)
               img_np is a BGR NumPy array if successful, else None.
               error_message is None if successful, else a string.
    """
    if base64_string is None or not base64_string:
        return None, "No image data provided."
        
    try:
        # Check if the string has the data URL prefix and remove it
        if "base64," in base64_string:
            header, base64_data = base64_string.split("base64,", 1)
        else:
            # Assume it's just the data
            base64_data = base64_string

        # Decode the Base64 data
        image_bytes = base64.b64decode(base64_data)
        
        # Convert bytes to NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode the NumPy array into an image (BGR format by default)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_np is None:
            return None, "Could not decode image data. Invalid format."
            
        # img_np is now in BGR format, ready for InsightFace
        return img_np, None

    except base64.binascii.Error as e:
        current_app.logger.error(f"Base64 decoding error: {e}")
        return None, f"Invalid Base64 data: {e}"
    except Exception as e:
        current_app.logger.error(f"Image decoding error: {e}", exc_info=True)
        return None, f"Error processing image: {e}"


# --- Decorators ---

def login_required(f):
    """
    Decorate routes to require login.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or g.user is None:
            flash("You need to be logged in to access this page.", "warning")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role_name):
    """
    Decorate routes to require a specific user role.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not g.user:
                flash("You must be logged in.", "warning")
                return redirect(url_for('auth.login'))
            
            # Handle single role or list of roles
            if isinstance(role_name, list):
                if g.user.get('role') not in role_name:
                    flash(f"You do not have permission ({g.user.get('role')}) to access this page.", "error")
                    return redirect(url_for('teacher.dashboard')) # Redirect to a safe page
            elif g.user.get('role') != role_name:
                flash(f"You must be a {role_name} to access this page.", "error")
                return redirect(url_for('teacher.dashboard')) # Redirect to a safe page
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator