import os
import datetime
from werkzeug.security import generate_password_hash
from backend import create_app
from backend.database import get_db

app = create_app()

def seed_database():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        try:
            print("Clearing existing data...")
            # Disable FK checks to clear tables
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
            tables = ['attendance_records', 'class_sessions', 'class_schedules', 'face_embeddings', 'subjects', 'students', 'users', 'departments']
            for table in tables:
                cursor.execute(f"TRUNCATE TABLE {table};")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
            db.commit()

            print("Seeding dummy data...")

            # 1. Departments
            print("Creating departments...")
            cursor.execute("INSERT INTO departments (dept_name, dept_code) VALUES (%s, %s)", ('Computer Science', 'CS'))
            cs_id = cursor.lastrowid
            cursor.execute("INSERT INTO departments (dept_name, dept_code) VALUES (%s, %s)", ('Information Technology', 'IT'))
            it_id = cursor.lastrowid

            # 2. Users (Admin, HOD, Teacher)
            print("Creating users...")
            admin_hash = generate_password_hash('admin123')
            hod_hash = generate_password_hash('hod123')
            teacher_hash = generate_password_hash('teacher123')

            users = [
                ('admin', admin_hash, 'System Admin', 'admin@attendify.local', 'Admin', None, True),
                ('hod_cs', hod_hash, 'Dr. Rajesh Sharma', 'hod_cs@attendify.local', 'HOD', cs_id, True),
                ('teacher1', teacher_hash, 'Prof. Sneha Kulkarni', 'sneha@attendify.local', 'Teacher', cs_id, True),
                ('teacher2', teacher_hash, 'Prof. Vikram Mehta', 'vikram@attendify.local', 'Teacher', cs_id, True)
            ]
            
            teacher_ids = []
            for u in users:
                cursor.execute(
                    "INSERT INTO users (username, password_hash, full_name, email, role, dept_id, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)", 
                    u
                )
                if u[4] == 'Teacher':
                    teacher_ids.append(cursor.lastrowid)

            # 3. Students
            print("Creating students...")
            students = [
                ('CS2101', 'Aarav Patil', 'Roll-01', 'A', cs_id, 'TE', 'aarav@student.local', '9876543210'),
                ('CS2102', 'Ishani Deshmukh', 'Roll-02', 'A', cs_id, 'TE', 'ishani@student.local', '9876543211'),
                ('CS2103', 'Rohan Joshi', 'Roll-03', 'A', cs_id, 'SE', 'rohan@student.local', '9876543212'),
                ('CS2104', 'Ananya Iyer', 'Roll-04', 'A', cs_id, 'SE', 'ananya@student.local', '9876543213'),
                ('CS2105', 'Sahil More', 'Roll-05', 'B', cs_id, 'FE', 'sahil@student.local', '9876543214'),
            ]
            for s in students:
                cursor.execute(
                    "INSERT INTO students (prn, student_name, roll_no, division, dept_id, academic_year, email, phone) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    s
                )

            # 4. Subjects
            print("Creating subjects...")
            subjects = [
                ('CS-DB101', 'Database Management', cs_id, 3),
                ('CS-OS201', 'Operating Systems', cs_id, 4),
                ('CS-AI301', 'Artificial Intelligence', cs_id, 5)
            ]
            
            subject_ids = []
            for s in subjects:
                cursor.execute(
                    "INSERT INTO subjects (subject_code, subject_name, dept_id, semester) VALUES (%s, %s, %s, %s)",
                    s
                )
                subject_ids.append(cursor.lastrowid)

            # 5. Class Schedules
            print("Creating class schedules...")
            today = datetime.datetime.today()
            today_day_of_week = today.weekday() # 0 = Monday, ... 6 = Sunday

            # We'll create one schedule for today for teacher1
            schedules = [
                # subject_id, teacher_id, division, day_of_week, start_time, end_time, academic_year, classroom
                (subject_ids[0], teacher_ids[0], 'A', today_day_of_week, '09:00:00', '10:30:00', 'TE', 'Room 101'), # DB
                (subject_ids[1], teacher_ids[0], 'A', today_day_of_week, '11:00:00', '12:30:00', 'TE', 'Room 102'), # OS
                (subject_ids[2], teacher_ids[1], 'B', (today_day_of_week + 1) % 7, '10:00:00', '11:30:00', 'FE', 'Room 201') # AI tomorrow
            ]
            
            for sched in schedules:
                cursor.execute(
                    "INSERT INTO class_schedules (subject_id, teacher_id, division, day_of_week, start_time, end_time, academic_year, classroom) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    sched
                )

            db.commit()
            print("Successfully seeded the database with dummy data!")
            print("="*40)
            print("Credentials to test:")
            print("Admin   - admin / admin123")
            print("HOD     - hod_cs / hod123")
            print("Teacher - teacher1 / teacher123")
            print("Teacher - teacher2 / teacher123")
            print("="*40)

        except Exception as e:
            db.rollback()
            print(f"Error seeding database: {e}")
        finally:
            cursor.close()

if __name__ == "__main__":
    seed_database()
