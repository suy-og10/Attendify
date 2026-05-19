from flask import Blueprint, render_template, request, flash, redirect, url_for, g, session, current_app
from werkzeug.security import generate_password_hash
import mysql.connector

from backend.utils import login_required, role_required
from backend.database import query_db, execute_db


admin_bp = Blueprint('admin', __name__, template_folder='../../frontend/templates/admin', url_prefix='/admin')


def _admin_id():
    return session.get('user_id')


def _log_admin_action(action, entity_type=None, entity_id=None, details=None):
    """Best-effort audit logging for sensitive admin actions."""
    try:
        execute_db(
            """
            INSERT INTO audit_logs (actor_user_id, action, entity_type, entity_id, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (_admin_id(), action, entity_type, str(entity_id) if entity_id is not None else None, details),
        )
    except Exception as exc:
        current_app.logger.warning("Audit logging failed for %s: %s", action, exc)


def _departments():
    return query_db("SELECT dept_id, dept_name, dept_code FROM departments ORDER BY dept_name") or []


def _setting_value(key, default=None):
    row = query_db("SELECT setting_value FROM system_settings WHERE setting_key = %s", (key,), one=True)
    return row['setting_value'] if row else default


@admin_bp.route('/dashboard')
@login_required
@role_required('Admin')
def dashboard():
    pending_hods = query_db(
        "SELECT user_id, username, full_name, email FROM users WHERE role = %s AND is_active = FALSE",
        ('HOD',),
    )
    stats = query_db(
        """
        SELECT
            (SELECT COUNT(*) FROM departments) AS departments,
            (SELECT COUNT(*) FROM users WHERE role = 'HOD') AS hods,
            (SELECT COUNT(*) FROM users WHERE role = 'Teacher') AS teachers,
            (SELECT COUNT(*) FROM students) AS students,
            (SELECT COUNT(*) FROM class_sessions WHERE status = 'ONGOING') AS live_sessions,
            (SELECT COUNT(*) FROM attendance_records) AS attendance_records
        """,
        one=True,
    )
    recent_actions = query_db(
        """
        SELECT al.*, u.full_name AS actor_name
        FROM audit_logs al
        LEFT JOIN users u ON al.actor_user_id = u.user_id
        ORDER BY al.created_at DESC
        LIMIT 8
        """
    ) or []
    return render_template(
        'dashboard.html',
        user=g.user,
        pending_hods=pending_hods or [],
        stats=stats or {},
        recent_actions=recent_actions,
    )


@admin_bp.route('/approve_hod/<int:user_id>', methods=['POST'])
@login_required
@role_required('Admin')
def approve_hod(user_id):
    hod = query_db(
        "SELECT user_id, username FROM users WHERE user_id = %s AND role = %s AND is_active = FALSE",
        (user_id, 'HOD'),
        one=True,
    )
    if hod:
        try:
            execute_db("UPDATE users SET is_active = TRUE WHERE user_id = %s", (user_id,))
            _log_admin_action('APPROVE_HOD', 'users', user_id, f"Approved HOD {hod['username']}")
            flash(f'HOD account (ID: {user_id}) approved successfully.', 'success')
        except Exception as e:
            flash(f'Error approving HOD: {e}', 'error')
            current_app.logger.error("Error approving HOD ID %s: %s", user_id, e, exc_info=True)
    else:
        flash('HOD not found or already active.', 'warning')
    return redirect(url_for('admin.dashboard'))


@admin_bp.route('/reject_hod/<int:user_id>', methods=['POST'])
@login_required
@role_required('Admin')
def reject_hod(user_id):
    hod = query_db("SELECT username FROM users WHERE user_id = %s AND role = 'HOD' AND is_active = FALSE", (user_id,), one=True)
    if hod:
        try:
            execute_db("DELETE FROM users WHERE user_id = %s", (user_id,))
            _log_admin_action('REJECT_HOD', 'users', user_id, f"Rejected HOD {hod['username']}")
            flash(f"HOD registration for '{hod['username']}' rejected and removed.", 'success')
        except Exception as e:
            flash(f'Error rejecting HOD: {e}', 'error')
            current_app.logger.error("Error rejecting HOD ID %s: %s", user_id, e, exc_info=True)
    else:
        flash('HOD not found or already active/rejected.', 'warning')
    return redirect(url_for('admin.dashboard'))


@admin_bp.route('/departments', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def departments():
    if request.method == 'POST':
        dept_name = request.form.get('dept_name', '').strip()
        dept_code = request.form.get('dept_code', '').strip().upper()
        if not dept_name or not dept_code:
            flash('Department name and code are required.', 'error')
        else:
            try:
                dept_id = execute_db(
                    "INSERT INTO departments (dept_name, dept_code) VALUES (%s, %s)",
                    (dept_name, dept_code),
                )
                _log_admin_action('CREATE_DEPARTMENT', 'departments', dept_id, f"{dept_code} - {dept_name}")
                flash('Department created successfully.', 'success')
                return redirect(url_for('admin.departments'))
            except mysql.connector.IntegrityError:
                flash('Department name or code already exists.', 'error')
            except Exception as e:
                flash(f'Error creating department: {e}', 'error')

    departments_list = query_db(
        """
        SELECT d.*,
               (SELECT COUNT(*) FROM users u WHERE u.dept_id = d.dept_id AND u.role = 'HOD') AS hod_count,
               (SELECT COUNT(*) FROM users u WHERE u.dept_id = d.dept_id AND u.role = 'Teacher') AS teacher_count,
               (SELECT COUNT(*) FROM students s WHERE s.dept_id = d.dept_id) AS student_count
        FROM departments d
        ORDER BY d.created_at DESC
        """
    )
    return render_template('departments.html', user=g.user, departments=departments_list or [])


@admin_bp.route('/departments/<int:dept_id>/edit', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def edit_department(dept_id):
    dept = query_db("SELECT * FROM departments WHERE dept_id = %s", (dept_id,), one=True)
    if not dept:
        flash('Department not found.', 'error')
        return redirect(url_for('admin.departments'))

    if request.method == 'POST':
        dept_name = request.form.get('dept_name', '').strip()
        dept_code = request.form.get('dept_code', '').strip().upper()
        if not dept_name or not dept_code:
            flash('Department name and code are required.', 'error')
        else:
            try:
                execute_db(
                    "UPDATE departments SET dept_name = %s, dept_code = %s WHERE dept_id = %s",
                    (dept_name, dept_code, dept_id),
                )
                _log_admin_action('UPDATE_DEPARTMENT', 'departments', dept_id, f"{dept_code} - {dept_name}")
                flash('Department updated successfully.', 'success')
                return redirect(url_for('admin.departments'))
            except mysql.connector.IntegrityError:
                flash('Department name or code already exists.', 'error')
            except Exception as e:
                flash(f'Error updating department: {e}', 'error')

    return render_template('department_form.html', dept=dept)


@admin_bp.route('/departments/<int:dept_id>/delete', methods=['POST'])
@login_required
@role_required('Admin')
def delete_department(dept_id):
    refs = query_db(
        """
        SELECT
            (SELECT COUNT(*) FROM users WHERE dept_id = %s) AS users_count,
            (SELECT COUNT(*) FROM students WHERE dept_id = %s) AS students_count,
            (SELECT COUNT(*) FROM subjects WHERE dept_id = %s) AS subjects_count
        """,
        (dept_id, dept_id, dept_id),
        one=True,
    )
    if refs and (refs['users_count'] or refs['students_count'] or refs['subjects_count']):
        flash('Department has users, students, or subjects. Transfer/archive them before deleting.', 'error')
        return redirect(url_for('admin.departments'))

    try:
        execute_db("DELETE FROM departments WHERE dept_id = %s", (dept_id,))
        _log_admin_action('DELETE_DEPARTMENT', 'departments', dept_id, 'Deleted empty department')
        flash('Department deleted.', 'success')
    except Exception as e:
        flash(f'Error deleting department: {e}', 'error')
    return redirect(url_for('admin.departments'))


@admin_bp.route('/users')
@login_required
@role_required('Admin')
def users():
    role_filter = request.args.get('role', 'all')
    status_filter = request.args.get('status', 'all')
    search = request.args.get('search', '').strip()
    params = []
    where = ["1=1"]

    if role_filter != 'all':
        where.append("u.role = %s")
        params.append(role_filter)
    if status_filter == 'active':
        where.append("u.is_active = TRUE")
    elif status_filter == 'inactive':
        where.append("u.is_active = FALSE")
    if search:
        where.append("(u.full_name LIKE %s OR u.username LIKE %s OR u.email LIKE %s)")
        q = f"%{search}%"
        params.extend([q, q, q])

    users_list = query_db(
        f"""
        SELECT u.*, d.dept_name
        FROM users u
        LEFT JOIN departments d ON u.dept_id = d.dept_id
        WHERE {' AND '.join(where)}
        ORDER BY u.role, u.full_name
        """,
        params,
    ) or []
    students = query_db(
        """
        SELECT s.*, d.dept_name, u.username, u.is_active AS account_active, u.user_id
        FROM students s
        LEFT JOIN departments d ON s.dept_id = d.dept_id
        LEFT JOIN users u ON s.user_id = u.user_id
        ORDER BY s.created_at DESC
        """
    ) or []
    return render_template(
        'users.html',
        users=users_list,
        students=students,
        departments=_departments(),
        filters={'role': role_filter, 'status': status_filter, 'search': search},
    )


@admin_bp.route('/users/create', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def create_user():
    if request.method == 'POST':
        role = request.form.get('role')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower() or None
        dept_id = request.form.get('dept_id', type=int)
        is_active = request.form.get('is_active') == 'on'
        student_payload = None

        if role == 'Student':
            student_payload = {
                'prn': request.form.get('prn', '').strip(),
                'division': request.form.get('division', '').strip().upper(),
                'academic_year': request.form.get('academic_year', '').strip(),
                'roll_no': request.form.get('roll_no', '').strip() or None,
                'phone': request.form.get('phone', '').strip() or None,
            }

        if role not in ['Admin', 'HOD', 'Teacher', 'Student']:
            flash('Invalid role selected.', 'error')
        elif not username or not password or not full_name:
            flash('Username, password, and full name are required.', 'error')
        elif role in ['HOD', 'Teacher', 'Student'] and not dept_id:
            flash('Department is required for HOD, Teacher, and Student accounts.', 'error')
        elif role == 'Student' and not all([student_payload['prn'], student_payload['division'], student_payload['academic_year']]):
            flash('PRN, division, and academic year are required for students.', 'error')
        else:
            user_id = None
            try:
                user_id = execute_db(
                    """
                    INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (username, generate_password_hash(password), full_name, email, role, dept_id, is_active),
                )
                if role == 'Student':
                    try:
                        execute_db(
                            """
                            INSERT INTO students (prn, student_name, roll_no, division, dept_id, academic_year, email, phone, user_id, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                student_payload['prn'],
                                full_name,
                                student_payload['roll_no'],
                                student_payload['division'],
                                dept_id,
                                student_payload['academic_year'],
                                email,
                                student_payload['phone'],
                                user_id,
                                is_active,
                            ),
                        )
                    except Exception:
                        execute_db("DELETE FROM users WHERE user_id = %s", (user_id,))
                        raise
                _log_admin_action('CREATE_USER', 'users', user_id, f"Created {role} {username}")
                flash('User created successfully.', 'success')
                return redirect(url_for('admin.users'))
            except mysql.connector.IntegrityError as e:
                flash(f'Username, email, or PRN already exists. {e}', 'error')
            except Exception as e:
                flash(f'Error creating user: {e}', 'error')

    return render_template('user_form.html', departments=_departments(), user=None)


@admin_bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def edit_user(user_id):
    user = query_db("SELECT * FROM users WHERE user_id = %s", (user_id,), one=True)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin.users'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower() or None
        role = request.form.get('role')
        dept_id = request.form.get('dept_id', type=int)
        is_active = request.form.get('is_active') == 'on'
        password = request.form.get('password', '')

        if role not in ['Admin', 'HOD', 'Teacher', 'Student']:
            flash('Invalid role selected.', 'error')
        elif not full_name:
            flash('Full name is required.', 'error')
        else:
            try:
                if password:
                    execute_db(
                        """
                        UPDATE users SET full_name = %s, email = %s, role = %s, dept_id = %s,
                            is_active = %s, password_hash = %s
                        WHERE user_id = %s
                        """,
                        (full_name, email, role, dept_id, is_active, generate_password_hash(password), user_id),
                    )
                else:
                    execute_db(
                        """
                        UPDATE users SET full_name = %s, email = %s, role = %s, dept_id = %s, is_active = %s
                        WHERE user_id = %s
                        """,
                        (full_name, email, role, dept_id, is_active, user_id),
                    )
                execute_db(
                    "UPDATE students SET student_name = %s, email = %s, dept_id = COALESCE(%s, dept_id), is_active = %s WHERE user_id = %s",
                    (full_name, email, dept_id, is_active, user_id),
                )
                _log_admin_action('UPDATE_USER', 'users', user_id, f"Updated {user['username']}")
                flash('User updated successfully.', 'success')
                return redirect(url_for('admin.users'))
            except mysql.connector.IntegrityError:
                flash('Email is already in use.', 'error')
            except Exception as e:
                flash(f'Error updating user: {e}', 'error')

    return render_template('user_form.html', departments=_departments(), user=user)


@admin_bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@login_required
@role_required('Admin')
def toggle_user(user_id):
    user = query_db("SELECT username, is_active FROM users WHERE user_id = %s", (user_id,), one=True)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin.users'))
    new_status = not bool(user['is_active'])
    execute_db("UPDATE users SET is_active = %s WHERE user_id = %s", (new_status, user_id))
    execute_db("UPDATE students SET is_active = %s WHERE user_id = %s", (new_status, user_id))
    _log_admin_action('TOGGLE_USER_STATUS', 'users', user_id, f"{user['username']} active={new_status}")
    flash('User status updated.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/reset-password', methods=['POST'])
@login_required
@role_required('Admin')
def reset_user_password(user_id):
    password = request.form.get('password', '')
    if len(password) < 6:
        flash('New password must be at least 6 characters.', 'error')
        return redirect(url_for('admin.users'))
    execute_db("UPDATE users SET password_hash = %s WHERE user_id = %s", (generate_password_hash(password), user_id))
    _log_admin_action('RESET_PASSWORD', 'users', user_id, 'Password reset by admin')
    flash('Password reset successfully.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/subjects')
@login_required
@role_required('Admin')
def subjects():
    subjects_list = query_db(
        """
        SELECT s.*, d.dept_name
        FROM subjects s
        JOIN departments d ON s.dept_id = d.dept_id
        ORDER BY d.dept_name, s.semester, s.subject_code
        """
    ) or []
    return render_template('subjects.html', subjects=subjects_list)


@admin_bp.route('/subjects/<int:subject_id>/toggle', methods=['POST'])
@login_required
@role_required('Admin')
def toggle_subject(subject_id):
    subject = query_db("SELECT subject_code, is_active FROM subjects WHERE subject_id = %s", (subject_id,), one=True)
    if not subject:
        flash('Subject not found.', 'error')
    else:
        new_status = not bool(subject['is_active'])
        execute_db("UPDATE subjects SET is_active = %s WHERE subject_id = %s", (new_status, subject_id))
        _log_admin_action('TOGGLE_SUBJECT_STATUS', 'subjects', subject_id, f"{subject['subject_code']} active={new_status}")
        flash('Subject status updated.', 'success')
    return redirect(url_for('admin.subjects'))


@admin_bp.route('/schedules')
@login_required
@role_required('Admin')
def schedules():
    schedules_list = query_db(
        """
        SELECT cs.*, s.subject_code, s.subject_name, d.dept_name, u.full_name AS teacher_name
        FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        JOIN departments d ON s.dept_id = d.dept_id
        JOIN users u ON cs.teacher_id = u.user_id
        ORDER BY d.dept_name, cs.day_of_week, cs.start_time
        """
    ) or []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for row in schedules_list:
        row['day_name'] = days[row['day_of_week']] if row['day_of_week'] is not None and 0 <= row['day_of_week'] < 7 else 'Invalid'
    return render_template('schedules.html', schedules=schedules_list)


@admin_bp.route('/sessions')
@login_required
@role_required('Admin')
def sessions_view():
    status = request.args.get('status', 'all')
    params = []
    where = []
    if status != 'all':
        where.append("csess.status = %s")
        params.append(status)
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    sessions = query_db(
        f"""
        SELECT csess.*, sub.subject_code, sub.subject_name, d.dept_name, u.full_name AS teacher_name,
               cs.division, cs.academic_year
        FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        JOIN subjects sub ON cs.subject_id = sub.subject_id
        JOIN departments d ON sub.dept_id = d.dept_id
        JOIN users u ON cs.teacher_id = u.user_id
        {where_sql}
        ORDER BY csess.session_date DESC, cs.start_time DESC
        LIMIT 200
        """,
        params,
    ) or []
    return render_template('sessions.html', sessions=sessions, selected_status=status)


@admin_bp.route('/sessions/<int:session_id>/status', methods=['POST'])
@login_required
@role_required('Admin')
def update_session_status(session_id):
    status = request.form.get('status')
    if status not in ['SCHEDULED', 'ONGOING', 'COMPLETED', 'CANCELLED']:
        flash('Invalid session status.', 'error')
    else:
        execute_db("UPDATE class_sessions SET status = %s WHERE session_id = %s", (status, session_id))
        _log_admin_action('UPDATE_SESSION_STATUS', 'class_sessions', session_id, f"status={status}")
        flash('Session status updated.', 'success')
    return redirect(url_for('admin.sessions_view'))


@admin_bp.route('/attendance')
@login_required
@role_required('Admin')
def attendance():
    records = query_db(
        """
        SELECT ar.*, st.student_name, st.prn, sub.subject_code, sub.subject_name, d.dept_name,
               csess.session_date, u.full_name AS marked_by_name
        FROM attendance_records ar
        JOIN students st ON ar.student_id = st.student_id
        JOIN class_sessions csess ON ar.session_id = csess.session_id
        JOIN class_schedules sch ON csess.schedule_id = sch.schedule_id
        JOIN subjects sub ON sch.subject_id = sub.subject_id
        JOIN departments d ON sub.dept_id = d.dept_id
        LEFT JOIN users u ON ar.marked_by = u.user_id
        ORDER BY ar.marked_time DESC
        LIMIT 250
        """
    ) or []
    return render_template('attendance.html', records=records)


@admin_bp.route('/attendance/<int:attendance_id>/update', methods=['POST'])
@login_required
@role_required('Admin')
def update_attendance(attendance_id):
    status = request.form.get('status')
    reason = request.form.get('reason', '').strip()
    if status not in ['Present', 'Absent', 'Late']:
        flash('Invalid attendance status.', 'error')
    elif not reason:
        flash('A reason is required for admin attendance edits.', 'error')
    else:
        execute_db(
            """
            UPDATE attendance_records
            SET status = %s, verification_method = 'MANUAL', marked_by = %s, notes = %s, marked_time = CURRENT_TIMESTAMP
            WHERE attendance_id = %s
            """,
            (status, _admin_id(), reason, attendance_id),
        )
        _log_admin_action('UPDATE_ATTENDANCE', 'attendance_records', attendance_id, f"status={status}; reason={reason}")
        flash('Attendance updated and audited.', 'success')
    return redirect(url_for('admin.attendance'))


@admin_bp.route('/settings', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def settings():
    defaults = {
        'attendance_threshold': '75',
        'late_after_minutes': '10',
        'auto_close_minutes': '60',
        'face_similarity_threshold': '0.50',
        'biometric_consent_required': '1',
    }
    if request.method == 'POST':
        for key in defaults:
            value = request.form.get(key, defaults[key]).strip()
            execute_db(
                """
                INSERT INTO system_settings (setting_key, setting_value, updated_by)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value), updated_by = VALUES(updated_by), updated_at = CURRENT_TIMESTAMP
                """,
                (key, value, _admin_id()),
            )
        _log_admin_action('UPDATE_SETTINGS', 'system_settings', None, 'Updated admin system settings')
        flash('Settings saved.', 'success')
        return redirect(url_for('admin.settings'))

    settings_data = {key: _setting_value(key, value) for key, value in defaults.items()}
    return render_template('settings.html', settings=settings_data)


@admin_bp.route('/audit-logs')
@login_required
@role_required('Admin')
def audit_logs():
    logs = query_db(
        """
        SELECT al.*, u.full_name AS actor_name, u.username AS actor_username
        FROM audit_logs al
        LEFT JOIN users u ON al.actor_user_id = u.user_id
        ORDER BY al.created_at DESC
        LIMIT 300
        """
    ) or []
    return render_template('audit_logs.html', logs=logs)
