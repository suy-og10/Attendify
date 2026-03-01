# backend/teacher/routes.py
from flask import (
    Blueprint, render_template, request, jsonify, flash, redirect, url_for, g, session,
    current_app, send_file
)
# Make sure helpers are imported
from .student_logic import add_single_student, add_students_from_sheet, process_and_save_embeddings
from backend.database import get_db, query_db, execute_db
# Import decorators and helpers
from backend.utils import login_required, role_required, decode_image_from_base64, allowed_file
from werkzeug.utils import secure_filename
import os
import io
import pandas as pd
import cv2 # Make sure OpenCV is imported if needed directly here
import numpy as np # Needed for image processing
from datetime import date

from werkzeug.utils import secure_filename
# Import the new logic function we will create
from .student_logic import process_and_save_embeddings

# Import logic functions from sibling modules
from .attendance_logic import recognize_faces_in_image ,recognize_faces_in_image, generate_embedding_for_student# Import the core recognition function
from .reporting_logic import generate_attendance_excel

teacher_bp = Blueprint('teacher', __name__, template_folder='../../frontend/templates/teacher', url_prefix='/teacher')
  
# --- Dashboard ---
@teacher_bp.route('/dashboard')
@login_required
@role_required('Teacher')
def dashboard():
    teacher_id = session.get('user_id')
    schedules = query_db("""
        SELECT cs.schedule_id, s.subject_name, s.subject_code, cs.division, cs.day_of_week, cs.start_time, cs.end_time, cs.academic_year
        FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        WHERE cs.teacher_id = %s AND cs.is_active = TRUE
        ORDER BY cs.day_of_week, cs.start_time
    """, (teacher_id,))

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

    return render_template('teacher/dashboard.html', user=g.user, schedules=schedules_with_day_names)

# --- Student Management ---
@teacher_bp.route('/students')
@login_required
@role_required('Teacher')
def list_students():
    teacher_dept_id = g.user.get('dept_id') if g.user else None
    students_query = """
        SELECT s.student_id, s.prn, s.student_name, s.roll_no, s.division, s.academic_year, d.dept_code,
               (SELECT COUNT(embedding_id) FROM face_embeddings WHERE student_id = s.student_id AND is_active = TRUE) as embedding_count
        FROM students s
        JOIN departments d ON s.dept_id = d.dept_id
        WHERE s.is_active = TRUE
    """
    params = []
    if teacher_dept_id:
        students_query += " AND s.dept_id = %s" 
        params.append(teacher_dept_id)

    students_query += " ORDER BY s.academic_year, s.division, s.roll_no, s.student_name"

    students = query_db(students_query, params)
    return render_template('student_list.html', students=students)


@teacher_bp.route('/students/add', methods=['GET', 'POST'])
@login_required
@role_required('Teacher')
def add_student():
    departments = query_db("SELECT dept_id, dept_name FROM departments ORDER BY dept_name")
    teacher_dept_id = g.user.get('dept_id')

    if request.method == 'POST':
        form_dept_id = request.form.get('dept_id', type=int)
        image_files = request.files.getlist('student_images') 
        sheet_file = request.files.get('student_sheet')

        if not form_dept_id:
            flash("Department selection is required.", "error")
            return render_template('add_student.html', departments=departments, teacher_dept_id=teacher_dept_id)

        # --- Handle Sheet Upload (logic assumed fixed in student_logic.py) ---
        if sheet_file and sheet_file.filename != '':
            if not (sheet_file.filename.endswith('.csv') or sheet_file.filename.endswith(('.xls', '.xlsx'))):
                 flash("Invalid sheet file type. Please use CSV or Excel.", "error")
            else:
                added_count, skipped_info, error = add_students_from_sheet(sheet_file, form_dept_id)
                
                # Check for *any* failure status
                if error or skipped_info:
                    flash_category = 'warning' if error and added_count > 0 else 'error'
                    
                    if error:
                        flash(f"Error processing sheet: {error}", flash_category)
                    if skipped_info:
                        flash(f"Skipped {len(skipped_info)} students/rows during sheet import. Details: {'; '.join(skipped_info[:3])}{'...' if len(skipped_info) > 3 else ''}", "warning") 
                
                if added_count > 0:
                    flash(f"Successfully added {added_count} students from sheet.", "success")
                    # Redirect to list view on success
                    return redirect(url_for('teacher.list_students')) 
                elif error:
                    # If it failed entirely, re-render the form
                    return render_template('add_student.html', departments=departments, teacher_dept_id=teacher_dept_id, form_data=request.form)
                else:
                    flash("File processed but no new valid students were added.", "info")
                    return redirect(url_for('teacher.list_students'))

        # --- Handle Single Student Add ---
        elif request.form.get('prn'):
            prn = request.form.get('prn').strip()
            name = request.form.get('student_name').strip()
            
            if not prn or not name or not request.form.get('division') or not request.form.get('academic_year'):
                 flash("Name, PRN, Division, and Academic Year are required for single student add.", "error")
                 return render_template('add_student.html', departments=departments, teacher_dept_id=teacher_dept_id, form_data=request.form)

            roll_no = request.form.get('roll_no', '').strip() or None
            division = request.form.get('division').strip().upper()
            academic_year = request.form.get('academic_year').strip()
            email = request.form.get('email', '').strip() or None
            phone = request.form.get('phone', '').strip() or None

            image_files_to_process = []
            file_errors = []
            if image_files:
                for file in image_files:
                    if file and file.filename != '' and allowed_file(file.filename):
                        image_files_to_process.append(file)
                    elif file and file.filename != '':
                        file_errors.append(f"Skipped invalid file type: {file.filename}")
            
            if file_errors:
                 flash("Some files were skipped due to invalid type (Allowed: png, jpg, jpeg).", "warning")

            # Call logic function with the plural 'image_files' argument
            student_id, error = add_single_student(
                prn, name, roll_no, division, form_dept_id, academic_year, email, phone,
                image_files=image_files_to_process # Use 'image_files' (plural)
            )

            if error:
                flash(f"Error adding student: {error}", "error")
                return render_template('add_student.html', departments=departments, teacher_dept_id=teacher_dept_id, form_data=request.form)
            else:
                flash(f"Student '{name}' added successfully. You can now add webcam data.", "success")
                return redirect(url_for('teacher.edit_student', student_id=student_id))
        else:
             flash("No student data or sheet provided for adding.", "warning")

    return render_template('add_student.html', departments=departments, teacher_dept_id=teacher_dept_id, form_data={})

# --- Attendance Taking (Routes assumed fixed or relying on query_db/execute_db) ---

@teacher_bp.route('/attendance/select_session', methods=['GET'])
@login_required
@role_required('Teacher')
def select_session():
    teacher_id = session['user_id']
    today_weekday = date.today().weekday()
    today_date_iso = date.today().isoformat()

    schedules = query_db("""
        SELECT cs.schedule_id, s.subject_name, s.subject_code, cs.division, cs.start_time, cs.end_time, cs.academic_year
        FROM class_schedules cs
        JOIN subjects s ON cs.subject_id = s.subject_id
        WHERE cs.teacher_id = %s AND cs.day_of_week = %s AND cs.is_active = TRUE
        ORDER BY cs.start_time
    """, (teacher_id, today_weekday))

    sessions_data = query_db(
        "SELECT schedule_id, session_id, status FROM class_sessions WHERE session_date = %s",
        (today_date_iso,)
    )
    sessions_today = {}
    if schedules and sessions_data: 
        schedule_ids = {s['schedule_id'] for s in schedules}
        sessions_today = {
            row['schedule_id']: {'session_id': row['session_id'], 'status': row['status']}
            for row in sessions_data
            if row['schedule_id'] in schedule_ids 
        }

    return render_template('session_select.html', schedules=schedules, sessions_today=sessions_today, today_date=today_date_iso)


@teacher_bp.route('/attendance/start_session', methods=['POST'])
@login_required
@role_required('Teacher')
def start_session():
    schedule_id = request.form.get('schedule_id', type=int)
    session_date_str = request.form.get('session_date')
    db = get_db() 

    valid_schedule = query_db(
        "SELECT schedule_id FROM class_schedules WHERE schedule_id = %s AND teacher_id = %s",
        (schedule_id, session['user_id']), one=True
    )

    if not valid_schedule:
        flash("Invalid schedule selected.", "error")
        return redirect(url_for('.select_session'))

    session_id = None
    try:
        execute_db(
             "INSERT IGNORE INTO class_sessions (schedule_id, session_date, status) VALUES (%s, %s, %s)",
             (schedule_id, session_date_str, 'ONGOING')
         )

        session_row = query_db(
            "SELECT session_id FROM class_sessions WHERE schedule_id = %s AND session_date = %s",
            (schedule_id, session_date_str), one=True
        )
        session_id = session_row['session_id'] if session_row else None

        if session_id:
            execute_db("UPDATE class_sessions SET status = 'ONGOING' WHERE session_id = %s AND status = 'SCHEDULED'", (session_id,))
            current_app.logger.info(f"Ensured session {session_id} is ONGOING for schedule {schedule_id} on {session_date_str}")
        else:
             raise ValueError("Session could not be created or found after insert attempt.")

    except Exception as e:
        flash(f"Error starting session: {e}", "error")
        current_app.logger.error(f"Error starting session for schedule {schedule_id}: {e}", exc_info=True)
        return redirect(url_for('.select_session'))

    return redirect(url_for('.take_attendance_page', session_id=session_id))


@teacher_bp.route('/attendance/take/<int:session_id>', methods=['GET'])
@login_required
@role_required('Teacher')
def take_attendance_page(session_id):
    teacher_id = session['user_id']

    session_info = query_db("""
        SELECT csess.*, cs.*, s.subject_name, s.subject_code, u.full_name as teacher_name
        FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        JOIN subjects s ON cs.subject_id = s.subject_id
        JOIN users u ON cs.teacher_id = u.user_id
        WHERE csess.session_id = %s AND cs.teacher_id = %s
    """, (session_id, teacher_id), one=True)

    if not session_info:
        flash("Session not found or access denied.", "error")
        return redirect(url_for('.select_session'))

    students = query_db("""
        SELECT s.student_id, s.prn, s.student_name, s.roll_no,
               COALESCE(ar.status, 'Absent') as status,
               ar.attendance_id, ar.recognition_confidence, ar.verification_method
        FROM students s
        LEFT JOIN attendance_records ar ON s.student_id = ar.student_id AND ar.session_id = %s
        JOIN class_schedules cs ON cs.schedule_id = %s
        JOIN subjects sub ON cs.subject_id = sub.subject_id
        WHERE s.division = cs.division
          AND s.academic_year = cs.academic_year
          AND s.dept_id = sub.dept_id
          AND s.is_active = TRUE
        ORDER BY s.roll_no, s.student_name
    """, (session_id, session_info['schedule_id']))

    return render_template('teacher/take_attendance.html', session_info=dict(session_info), students=students)


@teacher_bp.route('/api/attendance/recognize', methods=['POST'])
@login_required
@role_required('Teacher')
def api_recognize_attendance():
    session_id_raw = request.json.get('session_id')
    session_id = int(session_id_raw) if session_id_raw is not None else None
    base64_image = request.json.get('image_data')
    verification_method = request.json.get('method', 'FACE_LIVE')
    db = get_db() 

    if not session_id or not base64_image:
        return jsonify({"error": "Missing session_id or image_data"}), 400

    # 1. Verify Session Ownership and Enrollment Filters (using query_db)
    session_owner = query_db("""
        SELECT cs.teacher_id, cs.division, cs.academic_year, s.dept_id, cs.subject_id
        FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        JOIN subjects s ON cs.subject_id = s.subject_id
        WHERE csess.session_id = %s
    """, (session_id,), one=True)

    if not session_owner or session_owner['teacher_id'] != session['user_id']:
         return jsonify({"error": "Permission denied for this session"}), 403

    # 2. Decode Image
    img_np, decode_error = decode_image_from_base64(base64_image)
    if decode_error:
        return jsonify({"error": f"Invalid image data: {decode_error}"}), 400

    # 3. Perform Recognition (Calls initialize_models_and_data internally)
    try:
        recognized_students_raw, unknown_count = recognize_faces_in_image(img_np)
    except RuntimeError as re:
        return jsonify({"error": f"Recognition System Error: {re}"}), 500
    except Exception as e:
         current_app.logger.error(f"Unexpected error in recognition: {e}")
         return jsonify({"error": "Unexpected recognition failure."}), 500


    recognized_and_enrolled = []
    marked_count = 0
    db_errors = []

    # 4. Filter Results by Class Roster & Update DB
    if recognized_students_raw:
        recognized_ids = [s['student_id'] for s in recognized_students_raw]
        placeholders = ','.join(['%s'] * len(recognized_ids))

        # Fetch details ONLY for students who are supposed to be in THIS class
        enrolled_students = query_db(f"""
            SELECT student_id, student_name, prn
            FROM students
            WHERE student_id IN ({placeholders})
              AND division = %s AND academic_year = %s AND dept_id = %s
              AND is_active = TRUE
        """, (*recognized_ids, session_owner['division'], session_owner['academic_year'], session_owner['dept_id']))

        enrolled_map = {row['student_id']: row for row in enrolled_students}

        for student_result in recognized_students_raw:
            student_id = student_result['student_id']
            confidence = student_result['confidence']

            if student_id in enrolled_map:
                # Student is enrolled and recognized, mark them present
                student_details = enrolled_map[student_id]
                recognized_and_enrolled.append({
                    "student_id": student_id,
                    "name": student_details['student_name'],
                    "prn": student_details['prn'],
                    "confidence": confidence,
                    "bbox": student_result['bbox']
                })

                # Use execute_db for UPSERT
                try:
                    execute_db("""
                        INSERT INTO attendance_records (session_id, student_id, status, recognition_confidence, verification_method, marked_by, marked_time)
                        VALUES (%s, %s, 'Present', %s, %s, %s, CURRENT_TIMESTAMP)
                        ON DUPLICATE KEY UPDATE
                           status = VALUES(status),
                           recognition_confidence = VALUES(recognition_confidence),
                           verification_method = VALUES(verification_method),
                           marked_by = VALUES(marked_by)
                    """, (session_id, student_id, confidence, verification_method, session['user_id']))
                    marked_count += 1
                except Exception as e:
                    db_errors.append(f"DB Error marking attendance for {student_id}: {e}")
                    current_app.logger.error(f"DB marking error for {student_id}: {e}")
            else:
                 # Recognized face is not in the current class roster
                 unknown_count += 1 

    # 5. Return Results
    return jsonify({
        "recognized": recognized_and_enrolled,
        "unknown_count": unknown_count,
        "marked_count": marked_count,
        "db_errors": db_errors
    }), 200



@teacher_bp.route('/api/attendance/mark_manual', methods=['POST'])
@login_required
@role_required('Teacher')
def api_mark_manual():
    session_id_raw = request.json.get('session_id')
    session_id = int(session_id_raw) if session_id_raw is not None else None

    # FIX 2: Corrected dict.get() usage for student_id
    student_id_raw = request.json.get('student_id')
    student_id = int(student_id_raw) if student_id_raw is not None else None

    status = request.json.get('status')

    if not all([session_id, student_id, status]) or status not in ['Present', 'Absent', 'Late']:
        return jsonify({"error": "Invalid input"}), 400

    # Use query_db, %s, one=True
    session_owner = query_db("SELECT cs.teacher_id FROM class_sessions csess JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id WHERE csess.session_id = %s", (session_id,), one=True)
    if not session_owner or session_owner['teacher_id'] != session['user_id']:
         return jsonify({"error": "Permission denied"}), 403

    student_in_roster = query_db(
        """
        SELECT s.student_id
        FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        JOIN subjects sub ON cs.subject_id = sub.subject_id
        JOIN students s
          ON s.division = cs.division
         AND s.academic_year = cs.academic_year
         AND s.dept_id = sub.dept_id
        WHERE csess.session_id = %s AND s.student_id = %s AND s.is_active = TRUE
        """,
        (session_id, student_id),
        one=True
    )
    if not student_in_roster:
        return jsonify({"error": "Student is not part of this session roster"}), 403

    try:
        # Use execute_db with %s and MySQL's ON DUPLICATE KEY UPDATE
        execute_db("""
            INSERT INTO attendance_records (session_id, student_id, status, verification_method, marked_by, marked_time, recognition_confidence)
            VALUES (%s, %s, %s, 'MANUAL', %s, CURRENT_TIMESTAMP, NULL)
            ON DUPLICATE KEY UPDATE
               status = VALUES(status),
               verification_method = VALUES(verification_method),
               marked_time = VALUES(marked_time),
               marked_by = VALUES(marked_by),
               recognition_confidence = NULL
        """, (session_id, student_id, status, session['user_id']))
        current_app.logger.info(f"Manually marked student {student_id} as {status} for session {session_id} by user {session['user_id']}")
        return jsonify({"success": True, "student_id": student_id, "status": status}), 200
    except Exception as e:
         current_app.logger.error(f"Error manual marking attendance for {student_id} in session {session_id}: {e}")
         return jsonify({"error": f"Database error: {e}"}), 500
    

@teacher_bp.route('/attendance/view/<int:session_id>', methods=['GET'])
@login_required
@role_required('Teacher')
def view_attendance(session_id):
     # Use query_db with %s and one=True
     session_info = query_db("""
         SELECT csess.*, cs.*, s.subject_name, s.subject_code, u.full_name as teacher_name
         FROM class_sessions csess
         JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
         JOIN subjects s ON cs.subject_id = s.subject_id
         JOIN users u ON cs.teacher_id = u.user_id
         WHERE csess.session_id = %s AND cs.teacher_id = %s
     """, (session_id, session['user_id']), one=True)

     if not session_info:
          flash("Session not found or access denied.", "error")
          return redirect(url_for('.select_session')) # Corrected redirect target

     # Use query_db with %s
     attendance_records = query_db("""
         SELECT
             s.student_id, s.prn, s.student_name, s.roll_no,
             COALESCE(ar.status, 'Absent') as status,
             ar.marked_time, ar.verification_method, ar.recognition_confidence
         FROM students s
         LEFT JOIN attendance_records ar ON s.student_id = ar.student_id AND ar.session_id = %s
         JOIN class_schedules cs ON cs.schedule_id = %s -- Use schedule_id from session_info
         JOIN subjects sub ON cs.subject_id = sub.subject_id
         WHERE s.division = cs.division
           AND s.academic_year = cs.academic_year
           AND s.dept_id = sub.dept_id
           AND s.is_active = TRUE
         ORDER BY s.roll_no, s.student_name
     """, (session_id, session_info['schedule_id']))

     return render_template('view_attendance.html', session_info=dict(session_info), records=attendance_records)

# --- Reporting ---
@teacher_bp.route('/attendance/report', methods=['GET', 'POST'])
@login_required
@role_required('Teacher')
def attendance_report():
    # Use query_db with %s
    subjects = query_db("""
        SELECT DISTINCT s.subject_id, s.subject_code, s.subject_name
        FROM subjects s JOIN class_schedules cs ON s.subject_id = cs.subject_id
        WHERE cs.teacher_id = %s AND s.is_active = TRUE
        ORDER BY s.subject_code
    """, (session['user_id'],))
    divisions = query_db("SELECT DISTINCT division FROM class_schedules WHERE teacher_id = %s", (session['user_id'],))
    academic_years = query_db("SELECT DISTINCT academic_year FROM class_schedules WHERE teacher_id = %s", (session['user_id'],))

    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        division = request.form.get('division')

        subject_id = int(subject_id) if subject_id and subject_id != 'all' else None
        division = division if division and division != 'all' else None

        if not start_date or not end_date:
            flash("Start date and end date are required.", "error")
        else:
            try:
                excel_data = generate_attendance_excel(
                    subject_id=subject_id, start_date=start_date, end_date=end_date,
                    division=division, teacher_id=session['user_id']
                )
                filename_parts = ["attendance_report"]
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
                flash(f"{ve}", "warning")
            except Exception as e:
                flash(f"Error generating report: {e}", "error")
                current_app.logger.error(f"Report generation error: {e}", exc_info=True)

    return render_template('attendance_report.html', subjects=subjects, divisions=divisions, academic_years=academic_years)

@teacher_bp.route('/students/edit/<int:student_id>', methods=['GET', 'POST'])
@login_required
@role_required('Teacher')
def edit_student(student_id):
    db = get_db()
    # Fetch student data (ensure teacher has permission, e.g., belongs to dept)
    # Use query_db with %s and one=True
    student = query_db("SELECT * FROM students WHERE student_id = %s", (student_id,), one=True)

    if not student:
        flash("Student not found.", "error")
        return redirect(url_for('.list_students'))

    # Optional: Check if teacher's dept matches student's dept
    teacher_dept_id = g.user.get('dept_id')
    if teacher_dept_id and student['dept_id'] != teacher_dept_id:
         flash("You do not have permission to edit this student.", "error")
         return redirect(url_for('.list_students'))

    departments = query_db("SELECT dept_id, dept_name FROM departments ORDER BY dept_name")

    if request.method == 'POST':
        # --- Handle Form Submission (Update Student Details) ---
        # Get updated details from form (similar to add_student, but use UPDATE query)
        name = request.form.get('student_name', '').strip()
        prn = request.form.get('prn', '').strip() # Usually PRN shouldn't change, but allow correction maybe?
        roll_no = request.form.get('roll_no', '').strip() or None
        division = request.form.get('division', '').strip().upper() or None
        academic_year = request.form.get('academic_year', '').strip()
        form_dept_id = request.form.get('dept_id', type=int)
        email = request.form.get('email', '').strip() or None
        phone = request.form.get('phone', '').strip() or None
        is_active = request.form.get('is_active') == 'on'
        error = None

        # --- Validation ---
        if not name or not prn or not form_dept_id or not academic_year:
            error = "Name, PRN, Department, and Academic Year are required."
        elif len(prn) != 10 or not prn.isdigit():
             error = "PRN must be 10 digits."
        else:
            # Check if updated PRN conflicts with ANOTHER student
            existing_prn = query_db("SELECT student_id FROM students WHERE prn = %s AND student_id != %s", (prn, student_id), one=True)
            if existing_prn:
                error = f"Another student already has PRN {prn}."
            # Check for email conflict
            elif email and query_db("SELECT student_id FROM students WHERE email = %s AND student_id != %s", (email, student_id), one=True):
                 error = f"Another student already has email {email}."

        if error is None:
            try:
                execute_db("""
                    UPDATE students SET
                    student_name = %s, prn = %s, roll_no = %s, division = %s, dept_id = %s,
                    academic_year = %s, email = %s, phone = %s, is_active = %s
                    WHERE student_id = %s
                """, (name, prn, roll_no, division, form_dept_id, academic_year, email, phone, is_active, student_id))
                flash("Student details updated successfully.", "success")
                # Stay on edit page or redirect? Redirecting for now.
                # return redirect(url_for('.edit_student', student_id=student_id))
            except Exception as e:
                error = f"Database error updating student: {e}"
                current_app.logger.error(f"Error updating student {student_id}: {e}", exc_info=True)

        if error:
            flash(error, 'error')
            # Re-render form with errors and existing student data (as dict) + form data
            return render_template('edit_student.html', student=dict(student), departments=departments, form_data=request.form)

        # --- Handle Multiple Image Upload for Embeddings ---
        uploaded_files = request.files.getlist('student_images') # Use getlist for multiple files
        image_files_to_process = []
        file_errors = []
        if uploaded_files:
            for file in uploaded_files:
                if file and file.filename != '' and allowed_file(file.filename):
                    image_files_to_process.append(file)
                elif file and file.filename != '':
                    file_errors.append(f"Skipped invalid file type: {file.filename}")

            if file_errors:
                 flash("Some files were skipped due to invalid type (Allowed: png, jpg, jpeg).", "warning")

            if image_files_to_process:
                processed_count, embedding_errors = process_and_save_embeddings(student_id, image_files_to_process)
                if embedding_errors:
                     flash(f"Error processing some images: {'; '.join(embedding_errors)}", "error")
                if processed_count > 0:
                     flash(f"Successfully processed {processed_count} images and saved embeddings.", "success")

        # Redirect after processing everything
        return redirect(url_for('.edit_student', student_id=student_id))


    # --- GET Request ---
    # Fetch current embeddings count for display
    embedding_count = query_db("SELECT COUNT(*) as count FROM face_embeddings WHERE student_id = %s AND is_active = TRUE", (student_id,), one=True)['count']
    return render_template('edit_student.html', student=dict(student), departments=departments, embedding_count=embedding_count or 0)


# In: backend/teacher/routes.py

@teacher_bp.route('/api/students/capture_embedding', methods=['POST'])
@login_required
@role_required('Teacher')
def api_capture_embedding():
    
    # --- FIX ---
    # request.json is a standard dict and doesn't take 'type=int'
    student_id_raw = request.json.get('student_id')
    student_id = int(student_id_raw) if student_id_raw is not None else None
    # --- END FIX ---

    base64_image = request.json.get('image_data')

    if not student_id or not base64_image:
        return jsonify({"error": "Missing student_id or image_data"}), 400

    teacher_dept_id = g.user.get('dept_id')
    allowed_student = query_db(
        "SELECT student_id FROM students WHERE student_id = %s AND is_active = TRUE AND dept_id = %s",
        (student_id, teacher_dept_id),
        one=True
    )
    if not allowed_student:
        return jsonify({"error": "Permission denied for this student"}), 403

    img_np, decode_error = decode_image_from_base64(base64_image)
    if decode_error:
        return jsonify({"error": f"Invalid image data: {decode_error}"}), 400

    # Process this single image
    processed_count, embedding_errors = process_and_save_embeddings(student_id, [img_np], deactivate_existing=False) 

    if embedding_errors:
         return jsonify({"error": f"Error processing image: {'; '.join(embedding_errors)}"}), 500
    elif processed_count > 0:
         # Fetch updated count
         new_count = query_db("SELECT COUNT(*) as count FROM face_embeddings WHERE student_id = %s AND is_active = TRUE", (student_id,), one=True)['count']
         return jsonify({"success": True, "message": "Embedding saved successfully.", "new_embedding_count": new_count or 0}), 200
    else:
         # This case might happen if generate_embedding_for_student failed
         return jsonify({"error": "Image processed but no embedding was saved (e.g., no face detected)."}), 400

@teacher_bp.route('/students/delete/<int:student_id>', methods=['POST'])
@login_required
@role_required('Teacher')
def delete_student(student_id):
    # This route handles the deletion request from the form
    
    # Optional: We can check the hidden _method field, but since the form is POSTing 
    # directly to a 'delete' URL, it's already secured enough for simple Flask apps.
    
    # 1. Fetch the student to verify existence and permissions
    student = query_db("SELECT prn, dept_id FROM students WHERE student_id = %s", (student_id,), one=True)

    if not student:
        flash(f"Student ID {student_id} not found.", "error")
        return redirect(url_for('.list_students'))

    # 2. Check Permission (Student must belong to the teacher's department)
    teacher_dept_id = g.user.get('dept_id')
    if teacher_dept_id and student['dept_id'] != teacher_dept_id:
        flash("Permission denied. You can only delete students within your department.", "error")
        current_app.logger.warning(f"User {session['username']} attempted unauthorized deletion of student ID {student_id}.")
        return redirect(url_for('.list_students'))

    try:
        # The schema uses FOREIGN KEY...ON DELETE CASCADE for face_embeddings and attendance_records,
        # so deleting the student record should automatically delete associated data.
        
        # 3. Delete the student record
        execute_db("DELETE FROM students WHERE student_id = %s", (student_id,))
        
        flash(f"Student PRN {student['prn']} successfully deleted (including all associated data).", "success")
        current_app.logger.info(f"Student ID {student_id} ({student['prn']}) deleted by Teacher {session['username']}.")
        
    except Exception as e:
        flash(f"A database error occurred during deletion: {e}", "error")
        current_app.logger.error(f"Error deleting student ID {student_id}: {e}", exc_info=True)

    return redirect(url_for('.list_students'))

# In: backend/teacher/routes.py

@teacher_bp.route('/api/attendance/end_session', methods=['POST'])
@login_required
@role_required('Teacher')
def api_end_session():
    session_id_raw = request.json.get('session_id')
    session_id = int(session_id_raw) if session_id_raw is not None else None
    
    if not session_id:
        return jsonify({"error": "Invalid session_id"}), 400

    # 1. Verify teacher owns this session
    session_owner = query_db("""
        SELECT cs.teacher_id FROM class_sessions csess
        JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
        WHERE csess.session_id = %s
    """, (session_id,), one=True)

    if not session_owner or session_owner['teacher_id'] != session['user_id']:
        return jsonify({"error": "Permission denied"}), 403

    try:
        # 2. Update the session status to 'COMPLETED'
        execute_db(
            "UPDATE class_sessions SET status = 'COMPLETED', actual_end_time = CURRENT_TIMESTAMP WHERE session_id = %s AND status = 'ONGOING'",
            (session_id,)
        )
        
        # 3. Mark all remaining 'Absent' students
        schedule_info = query_db("""
            SELECT cs.division, cs.academic_year, s.dept_id
            FROM class_sessions csess
            JOIN class_schedules cs ON csess.schedule_id = cs.schedule_id
            JOIN subjects s ON cs.subject_id = s.subject_id
            WHERE csess.session_id = %s
        """, (session_id,), one=True)
        
        if schedule_info:
            # This query inserts an 'Absent' record ONLY IF one doesn't already exist
            execute_db(f"""
                INSERT INTO attendance_records (session_id, student_id, status, verification_method, marked_by)
                SELECT %s, s.student_id, 'Absent', 'MANUAL', %s
                FROM students s
                WHERE s.division = %s
                  AND s.academic_year = %s
                  AND s.dept_id = %s
                  AND s.is_active = TRUE
                  AND s.student_id NOT IN (
                      SELECT ar.student_id FROM attendance_records ar WHERE ar.session_id = %s
                  )
            """, (
                session_id, session['user_id'], 
                schedule_info['division'], schedule_info['academic_year'], schedule_info['dept_id'],
                session_id
            ))
            current_app.logger.info(f"Session {session_id} finalized. Marked remaining students as 'Absent'.")

        return jsonify({"success": True, "message": "Session finalized successfully."}), 200

    except Exception as e:
        current_app.logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {e}"}), 500