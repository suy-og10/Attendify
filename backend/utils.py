from functools import wraps
from flask import session, redirect, url_for, flash, current_app, g, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import base64


# --- Password Hashing ---
def hash_password(password):
    return generate_password_hash(password)


def verify_password(stored_hash, provided_password):
    return check_password_hash(stored_hash, provided_password)


# --- Image Utilities ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """Checks if a filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image_from_base64(base64_string):
    """
    Decodes a Base64 string (from data URL) into a BGR NumPy array.

    Returns:
        tuple: (img_np, error_message)
    """
    if base64_string is None or not base64_string:
        return None, "No image data provided."

    try:
        if "base64," in base64_string:
            _, base64_data = base64_string.split("base64,", 1)
        else:
            base64_data = base64_string

        image_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_np is None:
            return None, "Could not decode image data. Invalid format."

        return img_np, None

    except base64.binascii.Error as e:
        current_app.logger.error(f"Base64 decoding error: {e}")
        return None, f"Invalid Base64 data: {e}"
    except Exception as e:
        current_app.logger.error(f"Image decoding error: {e}", exc_info=True)
        return None, f"Error processing image: {e}"


# --- Decorators ---
def login_required(f):
    """Decorate routes to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or g.user is None:
            flash("You need to be logged in to access this page.", "warning")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)

    return decorated_function


def role_required(role_name):
    """
    Decorate routes to require one or more roles.

    - For API endpoints, respond with JSON 401/403.
    - For browser routes, redirect to role-appropriate dashboard/home.
    """

    def _is_api_request():
        endpoint = (request.endpoint or '').lower()
        path = (request.path or '').lower()
        return path.startswith('/api/') or '.api_' in endpoint or endpoint.startswith('api_')

    def _redirect_for_role(role):
        role_to_endpoint = {
            'Admin': 'admin.dashboard',
            'HOD': 'hod.dashboard',
            'Teacher': 'teacher.dashboard',
            'Student': 'student.dashboard',
        }
        endpoint = role_to_endpoint.get(role)
        if endpoint:
            return redirect(url_for(endpoint))
        return redirect(url_for('auth.home'))

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not g.user:
                if _is_api_request():
                    return jsonify({"error": "Authentication required"}), 401
                flash("You must be logged in.", "warning")
                return redirect(url_for('auth.login'))

            allowed_roles = role_name if isinstance(role_name, (list, tuple, set)) else [role_name]
            current_role = g.user.get('role')
            if current_role not in allowed_roles:
                if _is_api_request():
                    return jsonify({"error": "Forbidden", "required_roles": list(allowed_roles)}), 403
                flash("You do not have permission to access this page.", "error")
                return _redirect_for_role(current_role)

            return f(*args, **kwargs)

        return decorated_function

    return decorator
