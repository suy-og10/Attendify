-- Attendify Database Schema (SQLite compatible)

PRAGMA foreign_keys = ON; -- Enforce foreign key constraints in SQLite

-- 1. Departments Table
CREATE TABLE IF NOT EXISTS departments (
    dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dept_name TEXT NOT NULL UNIQUE,
    dept_code TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Users Table (For Admins, HODs, Teachers)
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL, -- Store hashed passwords!
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    role TEXT NOT NULL CHECK(role IN ('Admin', 'HOD', 'Teacher')),
    dept_id INTEGER, -- HODs/Teachers belong to a department
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

-- 3. Students Table
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prn TEXT NOT NULL UNIQUE,
    student_name TEXT NOT NULL,
    roll_no TEXT,
    division TEXT,
    dept_id INTEGER NOT NULL,
    academic_year TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
CREATE INDEX IF NOT EXISTS idx_student_prn ON students(prn);
CREATE INDEX IF NOT EXISTS idx_student_roll_division ON students(roll_no, division);

-- 4. Subjects Table
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_code TEXT NOT NULL UNIQUE,
    subject_name TEXT NOT NULL,
    dept_id INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
CREATE INDEX IF NOT EXISTS idx_subject_dept_semester ON subjects(dept_id, semester);

-- 5. Face Embeddings Table
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    embedding_vector BLOB NOT NULL, -- Stores the face embedding blob
    reference_image_path TEXT, -- Optional: Path to the image used for this embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_embedding_student ON face_embeddings(student_id);

-- 6. Class Schedules Table (Timetable)
CREATE TABLE IF NOT EXISTS class_schedules (
    schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL, -- Link to the user ID of the teacher
    division TEXT NOT NULL,
    day_of_week INTEGER NOT NULL CHECK(day_of_week BETWEEN 0 AND 6), -- 0=Mon, 6=Sun
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    academic_year TEXT NOT NULL,
    classroom TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    FOREIGN KEY (teacher_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_schedule_subject_div_day ON class_schedules(subject_id, division, day_of_week);

-- 7. Class Sessions Table (Instances of scheduled classes)
CREATE TABLE IF NOT EXISTS class_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL,
    session_date DATE NOT NULL,
    actual_start_time TIME,
    actual_end_time TIME,
    status TEXT CHECK(status IN ('SCHEDULED', 'ONGOING', 'COMPLETED', 'CANCELLED')) DEFAULT 'SCHEDULED',
    attendance_marked_by INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (schedule_id) REFERENCES class_schedules(schedule_id),
    FOREIGN KEY (attendance_marked_by) REFERENCES users(user_id),
    UNIQUE (schedule_id, session_date)
);
CREATE INDEX IF NOT EXISTS idx_session_date_status ON class_sessions(session_date, status);

-- 8. Attendance Records Table
CREATE TABLE IF NOT EXISTS attendance_records (
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Present', 'Absent', 'Late')),
    marked_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recognition_confidence REAL,
    verification_method TEXT CHECK(verification_method IN ('FACE_LIVE', 'FACE_UPLOAD', 'MANUAL', 'SHEET_UPLOAD')) DEFAULT 'MANUAL',
    marked_by INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES class_sessions(session_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (marked_by) REFERENCES users(user_id),
    UNIQUE (session_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_attendance_session_status ON attendance_records(session_id, status);
CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance_records(student_id, marked_time);

-- Add triggers for updated_at if needed in SQLite
CREATE TRIGGER IF NOT EXISTS update_student_timestamp AFTER UPDATE ON students FOR EACH ROW BEGIN
    UPDATE students SET updated_at = CURRENT_TIMESTAMP WHERE student_id = OLD.student_id;
END;