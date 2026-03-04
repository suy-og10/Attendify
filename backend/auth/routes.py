# backend/auth/routes.py
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session, g, current_app
)
from werkzeug.security import check_password_hash, generate_password_hash
# Import the UPDATED helper functions
from backend.database import get_db, query_db, execute_db
from backend.utils import login_required, role_required

auth_bp = Blueprint('auth', __name__, template_folder='../../frontend/templates')

@auth_bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        # Use query_db helper
        g.user = query_db(
            'SELECT user_id, username, role, full_name, dept_id FROM users WHERE user_id = %s', (user_id,), one=True
        )

@auth_bp.route('/')
def landing():
    if g.user:
        if g.user['role'] == 'Admin': return redirect(url_for('admin.dashboard'))
        if g.user['role'] == 'HOD': return redirect(url_for('hod.dashboard'))
        if g.user['role'] == 'Teacher': return redirect(url_for('teacher.dashboard'))
        if g.user['role'] == 'Student': return redirect(url_for('student.dashboard'))
        return redirect(url_for('auth.home'))
    return render_template('index.html')

@auth_bp.route('/about')
def about():
    return render_template('about.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if g.user: # Redirect if already logged in
        if g.user['role'] == 'Admin': return redirect(url_for('admin.dashboard'))
        if g.user['role'] == 'HOD': return redirect(url_for('hod.dashboard'))
        if g.user['role'] == 'Teacher': return redirect(url_for('teacher.dashboard'))
        if g.user['role'] == 'Student': return redirect(url_for('student.dashboard'))
        return redirect(url_for('auth.home')) # Use blueprint name

    role = request.args.get('role')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Extract role from form if it was passed via hidden input (optional, keeps heading during error)
        if 'role' in request.form:
            role = request.form['role']
            
        error = None
        # Use query_db helper
        user = query_db(
            'SELECT * FROM users WHERE username = %s', (username,), one=True
        )

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password_hash'], password):
            error = 'Incorrect password.'
        elif not user['is_active']:
             error = 'Account is inactive. Please contact the administrator.'

        if error is None:
            session.clear()
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            if user['dept_id']:
                session['dept_id'] = user['dept_id']

            current_app.logger.info(f"User {user['username']} logged in successfully.")
            if user['role'] == 'Admin':
                return redirect(url_for('admin.dashboard'))
            elif user['role'] == 'HOD':
                return redirect(url_for('hod.dashboard'))
            elif user['role'] == 'Teacher':
                return redirect(url_for('teacher.dashboard'))
            elif user['role'] == 'Student':
                return redirect(url_for('student.dashboard'))
            else:
                return redirect(url_for('auth.home')) # Use blueprint name
        else:
            flash(error, 'error')
            current_app.logger.warning(f"Failed login attempt for username: {username}")

    return render_template('login.html', role=role)

@auth_bp.route('/logout')
def logout():
    # (This function doesn't use the db directly, so it's fine)
    username = session.get('username', 'Unknown')
    session.clear()
    flash('You have been logged out successfully.', 'success')
    current_app.logger.info(f"User {username} logged out.")
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    departments = query_db("SELECT dept_id, dept_name FROM departments ORDER BY dept_name")
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        role = request.form.get('role', 'HOD') # Default to HOD if missing
        dept_id = request.form.get('dept_id', type=int)
        
        # HODs are active by default (requiring Admin approve ideally, but keeping it simple or HODs are active, let's keep HOD active for now or we make them inactive too. The prompt said: "The Admin currently approves HODs. This logic should be mirrored for Teachers. Change: Update auth_bp.register so that when a 'Teacher' registers, they are is_active = False by default.")
        is_active = True if role != 'Teacher' else False
        error = None

        # --- Validation ---
        if not username: error = 'Username is required.'
        elif not password: error = 'Password is required.'
        elif password != confirm_password: error = 'Passwords do not match.'
        elif not full_name: error = 'Full name is required.'
        elif role not in ['HOD', 'Teacher']: error = 'Invalid role selected.'
        elif not dept_id and role == 'Teacher': error = 'Department is required for Teachers.'
        else:
            # Use query_db to check existence
            if query_db('SELECT user_id FROM users WHERE username = %s', (username,), one=True) is not None:
                error = f"Username '{username}' is already registered."
            elif email and query_db('SELECT user_id FROM users WHERE email = %s', (email,), one=True) is not None:
                 error = f"Email '{email}' is already in use."

        if error is None:
            try:
                # Use execute_db helper
                execute_db(
                    "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (username, generate_password_hash(password), full_name, email, role, dept_id, is_active),
                )
                current_app.logger.info(f"New user registration: {username} ({role}), is_active: {is_active}.")
                if not is_active:
                    flash('Registration successful! Your account needs HOD approval.', 'success')
                else:
                    flash('Registration successful! You can now log in.', 'success')
                return redirect(url_for("auth.login"))
            except Exception as e: # Catch errors raised by execute_db
                 error = f"An error occurred during registration: {e}"
                 current_app.logger.error(f"Registration Exception for {username}: {e}", exc_info=True)

        flash(error, 'error')

    return render_template('register.html', departments=departments)

@auth_bp.route('/home')
@login_required
def home():
    return render_template('home.html', user=g.user) # g.user is loaded by before_app_request