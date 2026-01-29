-- Attendify Database Schema (MySQL compatible)

SET FOREIGN_KEY_CHECKS=0; -- Disable FK checks temporarily for setup

-- 1. Departments Table
DROP TABLE IF EXISTS departments;
CREATE TABLE departments (
    dept_id INT AUTO_INCREMENT PRIMARY KEY,
    dept_name VARCHAR(100) NOT NULL UNIQUE,
    dept_code VARCHAR(10) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. Users Table
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Increased length for hash
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(120) UNIQUE,
    role ENUM('Admin', 'HOD', 'Teacher') NOT NULL,
    dept_id INT ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- 3. Students Table
DROP TABLE IF EXISTS students;
CREATE TABLE students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    prn VARCHAR(10) NOT NULL UNIQUE,
    student_name VARCHAR(100) NOT NULL,
    roll_no VARCHAR(20),
    division VARCHAR(10),
    dept_id INT NOT NULL,
    academic_year VARCHAR(10) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(15),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
    INDEX idx_prn (prn),
    INDEX idx_roll_division (roll_no, division)
) ENGINE=InnoDB;

-- 4. Subjects Table
DROP TABLE IF EXISTS subjects;
CREATE TABLE subjects (
    subject_id INT AUTO_INCREMENT PRIMARY KEY,
    subject_code VARCHAR(20) NOT NULL UNIQUE,
    subject_name VARCHAR(100) NOT NULL,
    dept_id INT NOT NULL,
    semester INT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
    INDEX idx_dept_semester (dept_id, semester)
) ENGINE=InnoDB;

-- 5. Face Embeddings Table
DROP TABLE IF EXISTS face_embeddings;
CREATE TABLE face_embeddings (
    embedding_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    embedding_vector LONGBLOB NOT NULL, -- Stores the face embedding blob
    reference_image_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    INDEX idx_embedding_student (student_id)
) ENGINE=InnoDB;

-- 6. Class Schedules Table
DROP TABLE IF EXISTS class_schedules;
CREATE TABLE class_schedules (
    schedule_id INT AUTO_INCREMENT PRIMARY KEY,
    subject_id INT NOT NULL,
    teacher_id INT NOT NULL,
    division VARCHAR(10) NOT NULL,
    day_of_week INT NOT NULL, -- 0=Mon, 6=Sun (Consistent with Python weekday)
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    academic_year VARCHAR(10) NOT NULL,
    classroom VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    FOREIGN KEY (teacher_id) REFERENCES users(user_id),
    INDEX idx_schedule_subject_div_day (subject_id, division, day_of_week)
) ENGINE=InnoDB;

-- 7. Class Sessions Table
DROP TABLE IF EXISTS class_sessions;
CREATE TABLE class_sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    schedule_id INT NOT NULL,
    session_date DATE NOT NULL,
    actual_start_time TIME NULL,
    actual_end_time TIME NULL,
    status ENUM('SCHEDULED', 'ONGOING', 'COMPLETED', 'CANCELLED') DEFAULT 'SCHEDULED',
    attendance_marked_by INT NULL,
    notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (schedule_id) REFERENCES class_schedules(schedule_id),
    FOREIGN KEY (attendance_marked_by) REFERENCES users(user_id),
    UNIQUE KEY unique_session (schedule_id, session_date),
    INDEX idx_session_date_status (session_date, status)
) ENGINE=InnoDB;

-- 8. Attendance Records Table
DROP TABLE IF EXISTS attendance_records;
CREATE TABLE attendance_records (
    attendance_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    student_id INT NOT NULL,
    status ENUM('Present', 'Absent', 'Late') NOT NULL,
    marked_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recognition_confidence FLOAT NULL, -- Use FLOAT or DECIMAL
    verification_method ENUM('FACE_LIVE', 'FACE_UPLOAD', 'MANUAL', 'SHEET_UPLOAD') DEFAULT 'MANUAL',
    marked_by INT NULL,
    notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- ON UPDATE CURRENT_TIMESTAMP is less common here, handled by application logic
    FOREIGN KEY (session_id) REFERENCES class_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (marked_by) REFERENCES users(user_id),
    UNIQUE KEY unique_attendance (session_id, student_id),
    INDEX idx_attendance_session_status (session_id, status),
    INDEX idx_attendance_student_time (student_id, marked_time)
) ENGINE=InnoDB;

SET FOREIGN_KEY_CHECKS=1; -- Re-enable FK checks