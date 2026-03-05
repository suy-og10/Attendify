from backend.database import get_db, query_db, execute_db
from flask import current_app, flash
import numpy as np
import pandas as pd
import cv2
import io
import os
from .attendance_logic import generate_embedding_for_student, clear_embedding_cache
import mysql.connector
from werkzeug.utils import secure_filename  # You use this in process_and_save_embeddings
from werkzeug.security import generate_password_hash
def add_single_student(prn, name, roll_no, division, dept_id, academic_year, email, phone, username, password, image_files=None):
    """Adds a single student, creates their user login account, and optionally saves face embeddings."""
    db = get_db()
    error = None
    student_id = None

    # 1. Validate inputs (basic)
    if not prn or not name or not dept_id or not academic_year or not division or not username or not password:
        return None, "Missing required student details (PRN, Name, Division, Dept, Year, Username, Password)."

    # 2. Check if PRN already exists
    existing_prn = query_db("SELECT student_id FROM students WHERE prn = %s", (prn,), one=True)
    if existing_prn:
        return None, f"Student with PRN {prn} already exists."
    # Check for email conflict
    if email and query_db("SELECT student_id FROM users WHERE email = %s", (email,), one=True):
         return None, f"Email '{email}' is already in use."


    # 3. Check if Username already exists in users table
    existing_username = query_db("SELECT user_id FROM users WHERE username = %s", (username,), one=True)
    if existing_username:
         return None, f"Username '{username}' is already taken."

    # 4. Insert User and Student into Database
    try:
        # First, create the User login account
        hashed_password = generate_password_hash(password)
        new_user_id = execute_db(
            "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
            (username, hashed_password, name, email, 'Student', dept_id)
        )
        
        if not new_user_id:
             raise Exception("Database did not return a new user ID after insert.")

        # Then, create the Student record linked to this user_id
        student_id = execute_db(
            """INSERT INTO students (prn, student_name, roll_no, division, dept_id, academic_year, email, phone, user_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (prn, name, roll_no, division, dept_id, academic_year, email, phone, new_user_id)
        )
        # execute_db (if modified as discussed) should return the lastrowid
        if not student_id:
             raise Exception("Database did not return a new student ID after insert.")
        
        current_app.logger.info(f"Added student {name} (ID: {student_id})")

        # 4. Process and Save Embeddings (if files provided)
        if image_files and student_id:
            processed_count, embedding_errors = process_and_save_embeddings(student_id, image_files)
            if embedding_errors:
                 # Attach errors as a warning, but student creation was successful
                 flash(f"Student created, but image processing failed: {'; '.join(embedding_errors)}", "warning")
            if processed_count > 0:
                 flash(f"Successfully processed {processed_count} images for new student.", "info")


    except mysql.connector.IntegrityError as e:
        error = f"Database error: Could PRN {prn} already exist? Details: {e}"
        current_app.logger.error(error)
    except Exception as e:
        error = f"An unexpected error occurred: {e}"
        current_app.logger.error(error, exc_info=True)
        # If student was created but embedding failed, student_id might exist
        # The transaction logic in execute_db should handle rollback if insert fails
        student_id = None # Ensure we don't return a partial success ID if error occurred

    return student_id, error # Return the ID and any error

def add_students_from_sheet(sheet_file, dept_id):
    """Adds multiple students from an uploaded Excel or CSV file."""
    error = None
    added_count = 0
    skipped_info = [] # Use a list to store reasons for skipping

    try:
        # Determine file type and read using pandas
        if sheet_file.filename.endswith('.csv'):
            df = pd.read_csv(sheet_file, dtype={'prn': str, 'roll_no': str, 'phone': str}) # Read key IDs/numbers as strings
        elif sheet_file.filename.endswith(('.xls', '.xlsx')):
            # Make sure engine is specified if needed, read IDs as strings
            df = pd.read_excel(sheet_file, dtype={'prn': str, 'roll_no': str, 'phone': str}, engine='openpyxl')
        else:
            return 0, [], "Unsupported file type. Please upload CSV or Excel."

        df.fillna('', inplace=True) # Replace NaN with empty strings for easier handling

        # Define expected columns (adjust case/names as needed)
        required_cols = ['student_name', 'prn', 'division', 'academic_year']
        optional_cols = ['roll_no', 'email', 'phone']
        df.columns = df.columns.str.lower().str.strip() # Normalize column names

        # Validate columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return 0, [], f"Missing required columns in sheet: {', '.join(missing_cols)}"

        # Get DB connection once
        db = get_db()

        for index, row in df.iterrows():
            prn = str(row.get('prn', '')).strip()
            name = str(row.get('student_name', '')).strip()
            division = str(row.get('division', '')).strip().upper() # Standardize case
            academic_year = str(row.get('academic_year', '')).strip()

            # Optional fields
            roll_no = str(row.get('roll_no', '')).strip() or None # Use None if empty
            email = str(row.get('email', '')).strip() or None
            phone = str(row.get('phone', '')).strip() or None

            # Basic validation
            if not prn or not name or not division or not academic_year:
                skipped_info.append(f"Row {index+2}: Missing required data (PRN, Name, Division, Year)")
                continue
            if len(prn) != 10 or not prn.isdigit(): # Example PRN validation
                skipped_info.append(f"Row {index+2}: Invalid PRN format for {prn}")
                continue


            # --- FIX 1: Use query_db ---
            # Check if PRN already exists
            existing = query_db("SELECT student_id FROM students WHERE prn = %s", (prn,), one=True)
            if existing:
                skipped_info.append(f"Row {index+2}: PRN {prn} already exists")
                continue

            # --- FIX 2: Use execute_db ---
            # Insert student
            try:
                execute_db(
                    """INSERT INTO students (prn, student_name, roll_no, division, dept_id, academic_year, email, phone)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (prn, name, roll_no, division, dept_id, academic_year, email, phone)
                )
                added_count += 1
            except mysql.connector.IntegrityError as cie: # Explicitly catch DB constraint violations
                 skipped_info.append(f"Row {index+2}: PRN {prn} conflict. IntegrityError: {cie.msg}")
            except Exception as insert_err:
                 # Catch generic errors and log them fully
                 error = f"Unhandled database error during insert loop: {insert_err}"
                 current_app.logger.error(f"Sheet insert error row {index+2}, PRN {prn}: {insert_err}", exc_info=True)
                 return 0, skipped_info, error


        # No need for db.commit() here if execute_db handles it

        if skipped_info:
            error = "Some students were skipped during import."

    except pd.errors.EmptyDataError:
        error = "The uploaded file is empty."
    except ImportError as ie: # Catch openpyxl error specifically if it happens again
         error = f"Error processing sheet: {ie}. Ensure 'openpyxl' is installed (`pip install openpyxl`)."
         current_app.logger.error(f"Sheet processing ImportError: {ie}")
    except Exception as e:
        error = f"Error processing sheet: {e}"
        current_app.logger.error(f"Sheet processing error: {e}", exc_info=True)

    return added_count, skipped_info, error

def process_and_save_embeddings(student_id, image_sources, deactivate_existing=True):
    """
    Processes multiple image files or numpy arrays to generate and save embeddings.
    Deactivates previous embeddings for the student.

    Args:
        student_id (int): The ID of the student.
        image_sources (list): A list containing either file storage objects (from request.files)
                              or NumPy arrays (BGR format, from webcam capture).

    Returns:
        tuple: (processed_count, list_of_errors)
    """
    db = get_db()
    processed_count = 0
    errors = []

    if not image_sources:
        return 0, ["No image sources provided."]

    if deactivate_existing:
        try:
            execute_db("UPDATE face_embeddings SET is_active = FALSE WHERE student_id = %s", (student_id,))
            current_app.logger.info(f"Deactivated previous embeddings for student ID {student_id}")
        except Exception as e:
            errors.append(f"Could not deactivate old embeddings: {e}")
            current_app.logger.error(f"Error deactivating embeddings for {student_id}: {e}")

    # --- Process New Images ---
    for idx, source in enumerate(image_sources):
        img_np = None
        source_name = f"Image {idx+1}"
        try:
            if hasattr(source, 'filename'): # Check if it's a FileStorage object
                source_name = secure_filename(source.filename) # Clean filename
                in_memory_file = io.BytesIO()
                source.save(in_memory_file)
                in_memory_file.seek(0)
                file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
                img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_np is None:
                    raise ValueError("Could not decode image file.")
            elif isinstance(source, np.ndarray): # Check if it's already a NumPy array
                img_np = source
                source_name = "Webcam Capture"
            else:
                raise TypeError(f"Unsupported image source type: {type(source)}")

            # Generate embedding
            embedding_bytes, emb_error = generate_embedding_for_student(img_np) # This is from attendance_logic

            if emb_error:
                errors.append(f"{source_name}: {emb_error}")
                continue

            if embedding_bytes:
                # Save the new embedding as active
                execute_db(
                    "INSERT INTO face_embeddings (student_id, embedding_vector, is_active) VALUES (%s, %s, TRUE)",
                    (student_id, embedding_bytes)
                )
                processed_count += 1
                current_app.logger.info(f"Saved new embedding for student {student_id} from {source_name}")
            else:
                 errors.append(f"{source_name}: Embedding could not be generated (no face detected?).")

        except Exception as e:
            error_msg = f"Error processing {source_name}: {e}"
            errors.append(error_msg)
            current_app.logger.error(error_msg, exc_info=True)

    # --- THIS IS THE CACHING FIX ---
    # If we successfully added any new embeddings, clear the global cache
    # to force a reload from the DB on the next recognition attempt.
    if processed_count > 0:
        clear_embedding_cache()
    # --- END OF CACHING FIX ---

    return processed_count, errors

