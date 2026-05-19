-- Non-destructive upgrade for expanded ERP Coordinator/HOD authorities.
-- Run after admin_portal_upgrade.sql, or ensure audit_logs exists first.

CREATE TABLE IF NOT EXISTS attendance_correction_requests (
    request_id INT AUTO_INCREMENT PRIMARY KEY,
    attendance_id INT NOT NULL,
    requested_status ENUM('Present', 'Absent', 'Late') NOT NULL,
    reason TEXT NOT NULL,
    requested_by INT NOT NULL,
    reviewed_by INT NULL,
    status ENUM('PENDING', 'APPROVED', 'REJECTED') DEFAULT 'PENDING',
    reviewed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (attendance_id) REFERENCES attendance_records(attendance_id) ON DELETE CASCADE,
    FOREIGN KEY (requested_by) REFERENCES users(user_id),
    FOREIGN KEY (reviewed_by) REFERENCES users(user_id),
    INDEX idx_correction_status (status, created_at)
) ENGINE=InnoDB;
