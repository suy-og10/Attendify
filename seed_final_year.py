import os
import sys
import json
import cv2
import traceback
import numpy as np
from werkzeug.security import generate_password_hash

# Ensure root folder is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Disable background thread start during script initialization
os.environ['AUTO_ATTENDANCE_ENABLED'] = '0'

from backend import create_app
from backend.database import get_db, execute_db, query_db
from backend.teacher.attendance_logic import initialize_models_and_data

def main():
    app = create_app()
    with app.app_context():
        # Ensure model is initialized
        initialize_models_and_data()
        from backend.teacher.attendance_logic import _embedder

        db = get_db()
        cursor = db.cursor()

        print("=== Step 1: Clearing existing data ===")
        try:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
            tables = [
                "attendance_correction_requests",
                "attendance_records",
                "class_sessions",
                "class_schedules",
                "face_embeddings",
                "students",
                "subjects",
                "users",
                "departments"
            ]
            for t in tables:
                print(f"Clearing table {t}...")
                cursor.execute(f"DELETE FROM {t};")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
            db.commit()
            print("Database cleared successfully.")
        except Exception as e:
            db.rollback()
            print(f"Error clearing database: {e}")
            traceback.print_exc()
            sys.exit(1)

        print("\n=== Step 2: Creating Department ===")
        try:
            cursor.execute(
                "INSERT INTO departments (dept_name, dept_code) VALUES (%s, %s)",
                ("Computer Science & Engineering", "CSE")
            )
            dept_id = cursor.lastrowid
            db.commit()
            print(f"Created department CSE (ID: {dept_id})")
        except Exception as e:
            db.rollback()
            print(f"Error creating department: {e}")
            sys.exit(1)

        print("\n=== Step 3: Creating Admin and ERP Coordinator accounts ===")
        credentials = []
        try:
            admin_pwd = "AdminPassword@123"
            coord_pwd = "CoordinatorPassword@123"

            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, NULL, TRUE)",
                ("admin", generate_password_hash(admin_pwd), "System Administrator", "admin@attendify.com", "Admin")
            )
            credentials.append({"Name": "System Administrator", "Role": "Admin", "Username": "admin", "Password": admin_pwd})

            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
                ("coordinator", generate_password_hash(coord_pwd), "ERP Coordinator", "coordinator@attendify.com", "HOD", dept_id)
            )
            credentials.append({"Name": "ERP Coordinator", "Role": "ERP Coordinator", "Username": "coordinator", "Password": coord_pwd})

            db.commit()
            print("Admin and ERP Coordinator accounts created.")
        except Exception as e:
            db.rollback()
            print(f"Error creating administrative accounts: {e}")
            sys.exit(1)

        print("\n=== Step 4: Creating Teacher accounts ===")
        teachers_data = [
            {"name": "Ms. Saloni", "username": "saloni", "password": "SaloniPassword@123"},
            {"name": "Mr. Sandeep Shetke", "username": "sandeep", "password": "SandeepPassword@123"},
            {"name": "Mr. J. S. Pujari", "username": "jspujari", "password": "JspujariPassword@123"},
            {"name": "Mr. Samrat Killedar", "username": "samrat", "password": "SamratPassword@123"}
        ]
        
        teacher_id_map = {}
        try:
            for t in teachers_data:
                cursor.execute(
                    "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
                    (t["username"], generate_password_hash(t["password"]), t["name"], f"{t['username']}@attendify.com", "Teacher", dept_id)
                )
                t_id = cursor.lastrowid
                teacher_id_map[t["name"]] = t_id
                credentials.append({"Name": t["name"], "Role": "Teacher", "Username": t["username"], "Password": t["password"]})
            db.commit()
            print("Teacher accounts created.")
        except Exception as e:
            db.rollback()
            print(f"Error creating Teacher accounts: {e}")
            sys.exit(1)

        print("\n=== Step 5: Creating Subjects ===")
        subjects_data = [
            {"code": "DL-PE-III", "name": "Deep Learning(DL) PE-III", "teacher": "Ms. Saloni"},
            {"code": "IoT", "name": "Internet of Things(IoT)", "teacher": "Mr. Sandeep Shetke"},
            {"code": "NLP", "name": "Natural Language Processing", "teacher": "Mr. J. S. Pujari"},
            {"code": "MAD", "name": "Mobile Application Development", "teacher": "Mr. Samrat Killedar"}
        ]

        subject_id_map = {}
        try:
            for s in subjects_data:
                cursor.execute(
                    "INSERT INTO subjects (subject_code, subject_name, dept_id, semester, is_active) VALUES (%s, %s, %s, 7, TRUE)",
                    (s["code"], s["name"], dept_id)
                )
                sub_id = cursor.lastrowid
                subject_id_map[s["name"]] = sub_id
                
                teacher_uid = teacher_id_map[s["teacher"]]
                for day in range(5): # Monday to Friday
                    cursor.execute("""
                        INSERT INTO class_schedules (subject_id, teacher_id, division, day_of_week, start_time, end_time, academic_year, classroom)
                        VALUES (%s, %s, 'A', %s, '10:00:00', '11:00:00', 'BE', 'Room 301')
                    """, (sub_id, teacher_uid, day))
            db.commit()
            print("Subjects and schedules created successfully.")
        except Exception as e:
            db.rollback()
            print(f"Error creating subjects or schedules: {e}")
            sys.exit(1)

        print("\n=== Step 6: Importing Students from Dataset ===")
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{dataset_path}' does not exist.")
            sys.exit(1)

        student_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found {len(student_dirs)} student folders to process.")

        processed_students = 0
        failed_students = 0

        for d in sorted(student_dirs):
            dir_path = os.path.join(dataset_path, d)
            info_file = os.path.join(dir_path, "person_info.json")

            if not os.path.exists(info_file):
                print(f"Skipping folder {d} (person_info.json not found)")
                continue

            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                raw_name = info.get("name", d)
                prn = info.get("PRN")
                if not prn:
                    print(f"Skipping folder {d} (PRN missing in json)")
                    continue

                # Format name nicely
                clean_name = raw_name.replace("_", " ").title()
                roll_no = d.split("_")[-1] if "_" in d else prn[-4:]
                
                first_name = raw_name.split("_")[0].title()
                username = f"{raw_name.lower()}"
                password = f"{first_name}Password@123"

                email = f"{username}@attendify.com"
                phone = "9876543210"

                # 1. Create student user login account
                cursor.execute(
                    "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
                    (username, generate_password_hash(password), clean_name, email, "Student", dept_id)
                )
                user_id = cursor.lastrowid

                # 2. Create student record
                cursor.execute(
                    """INSERT INTO students (prn, student_name, roll_no, division, dept_id, academic_year, email, phone, user_id, is_active)
                       VALUES (%s, %s, %s, 'A', %s, 'BE', %s, %s, %s, TRUE)""",
                    (prn, clean_name, roll_no, dept_id, email, phone, user_id)
                )
                student_id = cursor.lastrowid
                
                credentials.append({"Name": clean_name, "Role": "Student (BE - Div A)", "Username": username, "Password": password, "PRN": prn})

                # 3. Generate embeddings for the first 5 images in the directory
                image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f != "person_info.json"]
                image_files = sorted(image_files)[:5]

                embedding_count = 0
                for img_name in image_files:
                    img_path = os.path.join(dir_path, img_name)
                    img_np = cv2.imread(img_path)
                    
                    if img_np is None:
                        print(f"  Warning: Could not read image {img_name}")
                        continue
                    
                    # Generate embedding vector using custom largest-face logic
                    if _embedder is None:
                        print("  Error: _embedder is None")
                        continue

                    faces = _embedder.get(img_np)
                    if not faces:
                        print(f"  Warning: No face detected in {img_name}")
                        continue
                    
                    # If multiple faces are detected, sort by bounding box area and select the largest face
                    if len(faces) > 1:
                        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                        print(f"  Info: Multiple faces ({len(faces)}) detected in {img_name}. Selected largest face (area: {(faces[0].bbox[2]-faces[0].bbox[0])*(faces[0].bbox[3]-faces[0].bbox[1])}).")

                    embedding = faces[0].embedding
                    embedding_bytes = embedding.tobytes()

                    if embedding_bytes:
                        cursor.execute(
                            "INSERT INTO face_embeddings (student_id, embedding_vector, reference_image_path, is_active) VALUES (%s, %s, %s, TRUE)",
                            (student_id, embedding_bytes, img_path)
                        )
                        embedding_count += 1

                db.commit()
                processed_students += 1
                print(f"Successfully processed {clean_name} (PRN: {prn}): Registered and generated {embedding_count}/5 active face embeddings.")

            except Exception as se:
                db.rollback()
                failed_students += 1
                print(f"Error processing folder {d}: {se}")
                traceback.print_exc()

        print("\n=== Seeding Completion Summary ===")
        print(f"Total students successfully registered: {processed_students}")
        print(f"Total students failed: {failed_students}")

        credentials_file = "credentials_summary.json"
        with open(credentials_file, 'w') as cf:
            json.dump(credentials, cf, indent=4)
        print(f"Credentials summary saved to: {credentials_file}")

if __name__ == "__main__":
    main()
