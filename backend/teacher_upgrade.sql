-- Teacher Portal Upgrade Migration
-- Run this against the attendify MySQL database

-- 1. Session Notes / Timeline
CREATE TABLE IF NOT EXISTS session_notes (
    note_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    teacher_id INT NOT NULL,
    note_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES class_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES users(user_id),
    INDEX idx_session_notes_session (session_id)
) ENGINE=InnoDB;

-- 2. Course Materials
CREATE TABLE IF NOT EXISTS course_materials (
    material_id INT AUTO_INCREMENT PRIMARY KEY,
    subject_id INT NOT NULL,
    teacher_id INT NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    file_path VARCHAR(512) NOT NULL,
    file_name VARCHAR(200) NOT NULL,
    file_size INT,
    is_published BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    FOREIGN KEY (teacher_id) REFERENCES users(user_id),
    INDEX idx_materials_subject (subject_id),
    INDEX idx_materials_teacher (teacher_id)
) ENGINE=InnoDB;
