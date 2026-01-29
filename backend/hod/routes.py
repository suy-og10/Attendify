# backend/hod/routes.py
print("--- EXECUTING hod/routes.py ---")
from flask import Blueprint, render_template, request, flash, redirect, url_for, g, session, current_app, send_file # Added current_app
from backend.database import get_db, query_db, execute_db
from backend.utils import login_required, role_required
from werkzeug.security import generate_password_hash
# Import the teacher's reporting logic
from backend.teacher.reporting_logic import generate_attendance_excel

hod_bp = Blueprint('hod', __name__, template_folder='../../frontend/templates/hod', url_prefix='/hod')
print("--- hod_bp Blueprint CREATED ---")
# In: backend/hod/routes.py

# In: backend/hod/routes.py

@hod_bp.route('/dashboard')
@login_required
@role_required('HOD')
def dashboard():
    print("--- ENTERING HOD dashboard function ---")
    hod_dept_id = g.user.get('dept_id')
    
    # --- FIX: Added queries for department_name and teachers ---
    
    # Get department name
    dept_info = query_db("SELECT dept_name FROM departments WHERE dept_id = %s", (hod_dept_id,), one=True)
    department_name = dept_info['dept_name'] if dept_info else "Unknown Department"

    # Get teacher list
    teachers = query_db("""
        SELECT user_id, full_name, username, email, is_active 
        FROM users 
        WHERE dept_id = %s AND role = 'Teacher'
        ORDER BY full_name
    """, (hod_dept_id,))

    # Get stats
    stats = query_db("""
        SELECT 
            (SELECT COUNT(*) FROM users WHERE dept_id = %s AND role = 'Teacher' AND is_active = TRUE) as teacher_count,
            (SELECT COUNT(*) FROM students WHERE dept_id = %s AND is_active = TRUE) as student_count,
            (SELECT COUNT(*) FROM subjects WHERE dept_id = %s AND is_active = TRUE) as subject_count
    """, (hod_dept_id, hod_dept_id, hod_dept_id), one=True)
    
    # --- END OF FIX ---
    
    print("--- Rendering hod/dashboard.html template ---")
    # Pass all required variables to the template
    return render_template(
        'hod/dashboard.html', 
        user=g.user, 
        stats=stats, 
        teachers=teachers, 
        department_name=department_name
    )

@hod_bp.route('/add_teacher', methods=['GET', 'POST'])
@login_required
@role_required('HOD')
def add_teacher():
    hod_dept_id = session.get('dept_id')
    if not hod_dept_id:
        flash("Cannot add teacher: Your department ID is not set.", "error")
        return redirect(url_for('.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower() or None
        role = 'Teacher'
        is_active = True
        error = None

        # --- Validation ---
        if not username: error = 'Username is required.'
        elif not password: error = 'Password is required.'
        elif password != confirm_password: error = 'Passwords do not match.'
        elif not full_name: error = 'Full name is required.'
        else:
            if query_db('SELECT user_id FROM users WHERE username = %s', (username,), one=True):
                error = f"Username '{username}' already exists."
            elif email and query_db('SELECT user_id FROM users WHERE email = %s', (email,), one=True):
                 error = f"Email '{email}' is already in use."

        if error is None:
            try:
                execute_db(
                    "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (username, generate_password_hash(password), full_name, email, role, hod_dept_id, is_active)
                )
                flash(f"Teacher '{full_name}' added successfully.", "success")
                return redirect(url_for('.dashboard'))
            except Exception as e:
                error = f"Database error adding teacher: {e}"
                current_app.logger.error(f"Error adding teacher {username} by HOD {g.user['username']}: {e}", exc_info=True)

        flash(error, 'error')
        return render_template('add_teacher.html', user=g.user, form_data=request.form)

    # --- GET Request ---
    return render_template('add_teacher.html', user=g.user)


# --- Subject Management ---

@hod_bp.route('/subjects')
@login_required
@role_required('HOD')
def manage_subjects():
    hod_dept_id = session.get('dept_id')
    if not hod_dept_id:
        flash("Your department ID is not set.", "error")
        return redirect(url_for('.dashboard')) # Or appropriate error page

    subjects = query_db("""
        SELECT subject_id, subject_code, subject_name, semester, is_active
        FROM subjects
        WHERE dept_id = %s
        ORDER BY semester, subject_code
    """, (hod_dept_id,))

    return render_template('manage_subjects.html', subjects=subjects if subjects else [])

@hod_bp.route('/add_subject', methods=['GET', 'POST'])
@login_required
@role_required('HOD')
def add_subject():
    hod_dept_id = session.get('dept_id')
    if not hod_dept_id:
        flash("Your department ID is not set.", "error")
        return redirect(url_for('.dashboard'))

    if request.method == 'POST':
        code = request.form.get('subject_code', '').strip().upper()
        name = request.form.get('subject_name', '').strip()
        semester = request.form.get('semester', type=int)
        is_active = request.form.get('is_active') == 'on' # Checkbox value
        error = None

        if not code: error = "Subject code is required."
        elif not name: error = "Subject name is required."
        elif semester is None or semester < 1: error = "Valid semester is required."
        else:
            # Check if code already exists in this dept (or globally depending on rules)
            existing = query_db("SELECT subject_id FROM subjects WHERE subject_code = %s AND dept_id = %s", (code, hod_dept_id), one=True)
            if existing:
                error = f"Subject code '{code}' already exists in your department."

        if error is None:
            try:
                execute_db(
                    "INSERT INTO subjects (subject_code, subject_name, dept_id, semester, is_active) VALUES (%s, %s, %s, %s, %s)",
                    (code, name, hod_dept_id, semester, is_active)
                )
                flash(f"Subject '{name}' added successfully.", "success")
                return redirect(url_for('.manage_subjects'))
            except Exception as e:
                error = f"Database error adding subject: {e}"
                current_app.logger.error(f"Error adding subject {code} by HOD {g.user['username']}: {e}", exc_info=True)

        flash(error, 'error')
        # Re-render form with errors
        return render_template('add_subject.html', form_data=request.form)

    # GET Request
    return render_template('add_subject.html')


# --- Schedule Management ---

@hod_bp.route('/schedules')
@login_required
@role_required('HOD')
def manage_schedules():
    hod_dept_id = session.get('dept_id')
    if not hod_dept_id:
        flash("Your department ID is not set.", "error")
        return redirect(url_for('.dashboard'))

    schedules = query_db("""
        SELECT cs.*, s.subject_name, s.subject_code, u.full_name AS teacher_name
        FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        JOIN users u ON cs.teacher_id = u.user_id
        WHERE s.dept_id = %s
        ORDER BY cs.day_of_week, cs.start_time, cs.division
    """, (hod_dept_id,))

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    schedules_with_day_names = []
    if schedules:
        for schedule in schedules:
            schedule_dict = dict(schedule)
            if schedule_dict['day_of_week'] is not None and 0 <= schedule_dict['day_of_week'] < len(days):
                schedule_dict['day_name'] = days[schedule_dict['day_of_week']]
            else:
                 schedule_dict['day_name'] = 'Invalid Day'
            schedules_with_day_names.append(schedule_dict)

    return render_template('manage_schedules.html', schedules=schedules_with_day_names)


@hod_bp.route('/add_schedule', methods=['GET', 'POST'])
@login_required
@role_required('HOD')
def add_schedule():
    hod_dept_id = session.get('dept_id')
    if not hod_dept_id:
        flash("Your department ID is not set.", "error")
        return redirect(url_for('.dashboard'))

    # Fetch subjects and teachers from the HOD's department for dropdowns
    subjects = query_db("SELECT subject_id, subject_name, subject_code FROM subjects WHERE dept_id = %s AND is_active = TRUE ORDER BY subject_code", (hod_dept_id,))
    teachers = query_db("SELECT user_id, full_name FROM users WHERE dept_id = %s AND role = 'Teacher' AND is_active = TRUE ORDER BY full_name", (hod_dept_id,))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # For display

    if request.method == 'POST':
        subject_id = request.form.get('subject_id', type=int)
        teacher_id = request.form.get('teacher_id', type=int)
        division = request.form.get('division', '').strip().upper()
        day_of_week = request.form.get('day_of_week', type=int)
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')
        academic_year = request.form.get('academic_year', '').strip()
        classroom = request.form.get('classroom', '').strip() or None
        is_active = True # Default new schedules to active
        error = None

        # --- Basic Validation ---
        if not all([subject_id, teacher_id, division, academic_year]) or day_of_week is None or not start_time or not end_time:
            error = "All fields except Classroom are required."
        elif day_of_week < 0 or day_of_week > 6:
             error = "Invalid day selected."
        elif start_time >= end_time:
             error = "Start time must be before end time."
        else:
            # TODO: Add more validation (e.g., check for schedule conflicts)
            pass

        if error is None:
            try:
                execute_db("""
                    INSERT INTO class_schedules (subject_id, teacher_id, division, day_of_week, start_time, end_time, academic_year, classroom, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (subject_id, teacher_id, division, day_of_week, start_time, end_time, academic_year, classroom, is_active))
                flash("Class schedule added successfully.", "success")
                return redirect(url_for('.manage_schedules'))
            except Exception as e:
                 error = f"Database error adding schedule: {e}"
                 current_app.logger.error(f"Error adding schedule by HOD {g.user['username']}: {e}", exc_info=True)

        flash(error, 'error')
        # Re-render form with errors and existing data
        return render_template('add_schedule.html', subjects=subjects or [], teachers=teachers or [], days=days, form_data=request.form)

    # --- GET Request ---
    return render_template('add_schedule.html', subjects=subjects or [], teachers=teachers or [], days=days)




@hod_bp.route('/reports/department', methods=['GET', 'POST'])
@login_required
@role_required('HOD')
def department_report():
    hod_dept_id = g.user.get('dept_id')
    
    # Fetch filters based on HOD's department
    subjects = query_db("""
        SELECT subject_id, subject_code, subject_name
        FROM subjects
        WHERE dept_id = %s AND is_active = TRUE
        ORDER BY subject_code
    """, (hod_dept_id,))
    
    divisions = query_db("""
        SELECT DISTINCT division FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        WHERE s.dept_id = %s
    """, (hod_dept_id,))
    
    academic_years = query_db("""
        SELECT DISTINCT academic_year FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        WHERE s.dept_id = %s
    """, (hod_dept_id,))

    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        division = request.form.get('division')
        academic_year = request.form.get('academic_year') # Added this filter

        subject_id = int(subject_id) if subject_id and subject_id != 'all' else None
        division = division if division and division != 'all' else None
        academic_year = academic_year if academic_year and academic_year != 'all' else None

        if not start_date or not end_date:
            flash("Start date and end date are required.", "error")
        else:
            try:
                # We can reuse the same report generator!
                # We pass 'teacher_id=None' to get all teachers for the subject
                excel_data = generate_attendance_excel(
                    subject_id=subject_id,
                    start_date=start_date,
                    end_date=end_date,
                    division=division,
                    teacher_id=None  # HOD gets data for all teachers
                )
                
                # Create a dynamic filename
                filename_parts = ["dept_report"]
                if subject_id: filename_parts.append(f"subj{subject_id}")
                if division: filename_parts.append(f"div{division}")
                filename_parts.append(start_date)
                filename_parts.append(end_date)
                filename = "_".join(filename_parts) + ".xlsx"

                return send_file(
                    excel_data,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True, download_name=filename
                )
            except ValueError as ve:
                flash(f"{ve}", "warning") # Catches "No records found"
            except Exception as e:
                flash(f"Error generating report: {e}", "error")
                current_app.logger.error(f"HOD Report generation error: {e}", exc_info=True)

    # On GET request or if POST fails
    return render_template(
        'hod/department_report.html',
        subjects=subjects,
        divisions=divisions,
        academic_years=academic_years
    )