from flask import Blueprint, render_template, session, redirect, url_for, flash, current_app, send_file, request, jsonify
import os
from datetime import date
from backend.utils import login_required, role_required
from backend.database import query_db
from backend.database import execute_db

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

    # Get face embedding count
    embedding_count = query_db(
        "SELECT COUNT(*) as count FROM face_embeddings WHERE student_id = %s AND is_active = TRUE",
        (student['student_id'],), one=True
    )
    
    # Get selected date schedule
    target_date = request.args.get('date')
    if not target_date:
        target_date = date.today().isoformat()
        
    todays_classes = query_db("""
        SELECT csess.session_id, csess.status, csess.session_date, cs.start_time, cs.end_time, 
               sub.subject_name, sub.subject_code, u.full_name as teacher_name,
               ar.status as attendance_status
        FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        JOIN subjects sub ON cs.subject_id = sub.subject_id
        JOIN users u ON cs.teacher_id = u.user_id
        LEFT JOIN attendance_records ar ON csess.session_id = ar.session_id AND ar.student_id = %s
        WHERE csess.session_date = %s 
          AND cs.division = %s 
          AND cs.academic_year = %s
        ORDER BY cs.start_time ASC
    """, (student['student_id'], target_date, student['division'], student['academic_year']))

    # Get recent course materials
    recent_materials = query_db("""
        SELECT DISTINCT cm.material_id, cm.title, cm.file_name, cm.created_at, sub.subject_code
        FROM course_materials cm
        JOIN class_schedules cs ON cm.subject_id = cs.subject_id
        JOIN subjects sub ON cm.subject_id = sub.subject_id
        WHERE cs.division = %s AND cs.academic_year = %s AND cm.is_published = TRUE
        ORDER BY cm.created_at DESC
        LIMIT 5
    """, (student['division'], student['academic_year']))

    return render_template('student/dashboard.html', 
                           student=student, 
                           attendance_stats=attendance_stats,
                           embedding_count=embedding_count['count'] if embedding_count else 0,
                           todays_classes=todays_classes or [],
                           recent_materials=recent_materials or [],
                           selected_date=target_date)


@student_bp.route('/api/corrections/submit', methods=['POST'])
@login_required
@role_required('Student')
def api_submit_student_correction():
    user_id = session.get('user_id')
    # map to student_id
    student = query_db("SELECT student_id FROM students WHERE user_id = %s", (user_id,), one=True)
    if not student:
        return jsonify({"error": "Student profile not found"}), 403

    data = request.json or {}
    attendance_id = data.get('attendance_id')
    requested_status = data.get('requested_status')
    reason = (data.get('reason') or '').strip()

    if not attendance_id or not requested_status or not reason:
        return jsonify({"error": "attendance_id, requested_status, and reason are required"}), 400

    if requested_status not in ('Present', 'Absent', 'Late'):
        return jsonify({"error": "Invalid requested_status"}), 400

    # Verify this attendance record belongs to this student
    ownership = query_db("""
        SELECT ar.attendance_id, csess.status AS session_status
        FROM attendance_records ar
        JOIN class_sessions csess ON ar.session_id = csess.session_id
        WHERE ar.attendance_id = %s AND ar.student_id = %s
    """, (attendance_id, student['student_id']), one=True)

    if not ownership:
        return jsonify({"error": "Attendance record not found or access denied"}), 403

    if ownership['session_status'] != 'COMPLETED':
        return jsonify({"error": "Corrections can only be requested for completed sessions"}), 400

    existing = query_db(
        "SELECT request_id FROM attendance_correction_requests WHERE attendance_id = %s AND status = 'PENDING'",
        (attendance_id,), one=True
    )
    if existing:
        return jsonify({"error": "A pending correction request already exists for this record"}), 409

    try:
        request_id = execute_db("""
            INSERT INTO attendance_correction_requests (attendance_id, requested_status, reason, requested_by)
            VALUES (%s, %s, %s, %s)
        """, (attendance_id, requested_status, reason, user_id))
        return jsonify({"success": True, "request_id": request_id}), 201
    except Exception as e:
        current_app.logger.error(f"Error submitting student correction: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {e}"}), 500

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
            cs.session_id,
            cs.session_date,
            cs.actual_start_time,
            cs.actual_end_time,
            u.full_name as teacher_name,
            ar.attendance_id,
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
    
    # Fetch notes for these sessions
    if records:
        session_ids = [str(r['session_id']) for r in records]
        placeholders = ','.join(['%s'] * len(session_ids))
        notes = query_db(f"""
            SELECT sn.session_id, sn.note_text, u.full_name as author, sn.created_at
            FROM session_notes sn
            JOIN users u ON sn.teacher_id = u.user_id
            WHERE sn.session_id IN ({placeholders})
            ORDER BY sn.created_at ASC
        """, tuple(session_ids))
        
        # Attach notes to records
        notes_by_session = {}
        for n in (notes or []):
            if n['session_id'] not in notes_by_session:
                notes_by_session[n['session_id']] = []
            notes_by_session[n['session_id']].append(n)
            
        for r in records:
            r['notes'] = notes_by_session.get(r['session_id'], [])
    
    return render_template('student/view_details.html', subject=subject, records=records)

@student_bp.route('/materials')
@login_required
@role_required('Student')
def course_materials():
    user_id = session.get('user_id')
    student = query_db("SELECT division, academic_year FROM students WHERE user_id = %s", (user_id,), one=True)
    if not student:
        flash("Student profile not found.", "error")
        return redirect(url_for('auth.home'))
        
    subject_filter = request.args.get('subject_id', type=int)
    
    # Get subjects for filter
    subjects = query_db("""
        SELECT DISTINCT sub.subject_id, sub.subject_code, sub.subject_name
        FROM class_schedules cs
        JOIN subjects sub ON cs.subject_id = sub.subject_id
        WHERE cs.division = %s AND cs.academic_year = %s
        ORDER BY sub.subject_code
    """, (student['division'], student['academic_year']))
    
    # Build materials query
    query = """
        SELECT DISTINCT cm.material_id, cm.title, cm.description, cm.file_name, cm.file_size, cm.file_path, cm.is_published, cm.created_at, cm.subject_id, cm.teacher_id, sub.subject_name, sub.subject_code, u.full_name as teacher_name
        FROM course_materials cm
        JOIN class_schedules cs ON cm.subject_id = cs.subject_id AND cm.teacher_id = cs.teacher_id
        JOIN subjects sub ON cm.subject_id = sub.subject_id
        JOIN users u ON cm.teacher_id = u.user_id
        WHERE cs.division = %s AND cs.academic_year = %s AND cm.is_published = TRUE
    """
    params = [student['division'], student['academic_year']]
    
    if subject_filter:
        query += " AND cm.subject_id = %s"
        params.append(subject_filter)
        
    query += " ORDER BY cm.created_at DESC"
    
    materials = query_db(query, params)
    
    return render_template('student/materials.html', 
                           materials=materials or [], 
                           subjects=subjects or [],
                           subject_filter=subject_filter)

@student_bp.route('/materials/download/<int:material_id>')
@login_required
@role_required('Student')
def download_material(material_id):
    user_id = session.get('user_id')
    student = query_db("SELECT division, academic_year FROM students WHERE user_id = %s", (user_id,), one=True)
    
    # Verify student is enrolled in the subject of this material
    material = query_db("""
        SELECT cm.file_path, cm.file_name 
        FROM course_materials cm
        JOIN class_schedules cs ON cm.subject_id = cs.subject_id AND cm.teacher_id = cs.teacher_id
        WHERE cm.material_id = %s 
          AND cs.division = %s 
          AND cs.academic_year = %s
          AND cm.is_published = TRUE
    """, (material_id, student['division'], student['academic_year']), one=True)
    
    if not material or not os.path.exists(material['file_path']):
        flash("File not found or you do not have permission to access it.", "error")
        return redirect(url_for('student.course_materials'))
        
    return send_file(material['file_path'], as_attachment=True, download_name=material['file_name'])
