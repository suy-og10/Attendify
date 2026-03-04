from flask import Blueprint, render_template, session, redirect, url_for, flash
from backend.utils import login_required, role_required
from backend.database import query_db

student_bp = Blueprint('student', __name__, template_folder='../../frontend/templates')

@student_bp.route('/dashboard')
@login_required
@role_required('Student')
def dashboard():
    user_id = session.get('user_id')
    
    # Get student info
    student = query_db(
        "SELECT * FROM students WHERE user_id = %s", (user_id,), one=True
    )
    
    if not student:
        flash("Student profile not found. Please contact the administrator.", "error")
        return redirect(url_for('auth.home'))
        
    # Query to get all subjects for the student's division, academic_year, and dept_id
    # Calculate total COMPLETED sessions and total Present records per subject
    stats_query = """
        SELECT 
            s.subject_id,
            s.subject_name,
            s.subject_code,
            u.full_name as teacher_name,
            COUNT(DISTINCT cs.session_id) as total_sessions,
            SUM(CASE WHEN ar.status = 'Present' THEN 1 ELSE 0 END) as present_count
        FROM subjects s
        JOIN class_schedules sch ON s.subject_id = sch.subject_id
        JOIN users u ON sch.teacher_id = u.user_id
        LEFT JOIN class_sessions cs ON sch.schedule_id = cs.schedule_id AND cs.status = 'COMPLETED'
        LEFT JOIN attendance_records ar ON cs.session_id = ar.session_id AND ar.student_id = %s
        WHERE sch.division = %s 
          AND sch.academic_year = %s 
          AND s.dept_id = %s
          AND sch.is_active = TRUE
        GROUP BY s.subject_id, s.subject_name, s.subject_code, u.full_name
    """
    
    attendance_stats = query_db(
        stats_query, 
        (student['student_id'], student['division'], student['academic_year'], student['dept_id'])
    )
    
    # Calculate percentage
    for stat in attendance_stats:
        if stat['total_sessions'] > 0:
            stat['attendance_percentage'] = round((stat['present_count'] / stat['total_sessions']) * 100, 2)
        else:
            stat['attendance_percentage'] = 0.0

    return render_template('student/dashboard.html', student=student, attendance_stats=attendance_stats)

@student_bp.route('/subject/<int:subject_id>/details')
@login_required
@role_required('Student')
def view_details(subject_id):
    user_id = session.get('user_id')
    
    student = query_db("SELECT student_id, division, academic_year FROM students WHERE user_id = %s", (user_id,), one=True)
    if not student:
        flash("Student profile not found.", "error")
        return redirect(url_for('auth.home'))
        
    subject = query_db("SELECT * FROM subjects WHERE subject_id = %s", (subject_id,), one=True)
    if not subject:
        flash("Subject not found.", "error")
        return redirect(url_for('student.dashboard'))
        
    details_query = """
        SELECT 
            cs.session_date,
            cs.actual_start_time,
            cs.actual_end_time,
            u.full_name as teacher_name,
            COALESCE(ar.status, 'Absent') as status
        FROM class_sessions cs
        JOIN class_schedules sch ON cs.schedule_id = sch.schedule_id
        JOIN users u ON sch.teacher_id = u.user_id
        LEFT JOIN attendance_records ar ON cs.session_id = ar.session_id AND ar.student_id = %s
        WHERE sch.subject_id = %s 
          AND cs.status = 'COMPLETED'
          AND sch.division = %s
          AND sch.academic_year = %s
        ORDER BY cs.session_date DESC, cs.actual_start_time DESC
    """
    
    records = query_db(
        details_query, 
        (student['student_id'], subject_id, student['division'], student['academic_year'])
    )
    
    return render_template('student/view_details.html', subject=subject, records=records)
