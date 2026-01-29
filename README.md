
````markdown
# Attendify Web Application 🎓 Attendance Management System

This document outlines the complete, start-to-finish process for setting up and running the Attendify web application, which utilizes **MySQL** for data persistence and **Face Recognition** for attendance tracking.

---

## I. Project Status Summary

* **Database:** Switched to **MySQL**. Tables are expected to be created.
* **Backend:** All major logic files are present.
* **Frontend:** HTML is ready. **JavaScript has been moved inline** into `take_attendance.html` to bypass static file loading issues.

---

## II. Phase 1: Setup and Initialization (Run Once)

This phase prepares your environment and database connection.

### 1. Environment Setup & Dependency Installation

Because the project path may have changed, you must create a new **virtual environment** (`venv`).

1.  **Create New Venv** (Run from the project root, e.g., `C:\...\Attendify_project\attendify`):
    ```powershell
    python -m venv venv
    ```
2.  **Activate & Install Dependencies:**
    ```powershell
    .\venv\Scripts\Activate.ps1
    # Install required Python modules
    pip install Flask openpyxl Werkzeug pandas numpy opencv-python scikit-learn insightface onnxruntime mysql-connector-python
    ```

### 2. Configure Environment Variables

You must set these in your PowerShell session every time you open a new terminal:

```powershell
$env:FLASK_APP = "backend.app"
$env:FLASK_ENV = "development"
````

### 3\. Database Initialization

Ensure your MySQL server is running and the credentials in `backend/config.py` are correct.

```powershell
# Initialize database schema (Run this once after environment setup)
flask init-db
```

-----

## III. Phase 2: Running the Application

1.  **Start the Server:** (Ensure environment variables from Step II.2 are set first)
    ```powershell
    flask run --host=0.0.0.0
    ```
2.  **Access the App:** Open your browser and navigate to:
    $$\text{http://localhost:5000}$$

-----

## IV. Phase 3: Post-Launch Workflow (Initial Data Setup)

The system requires initial setup by the Admin, HOD, and Teachers before attendance can be taken via face recognition.

### 1\. Initial Admin Login & Setup

1.  **Create Admin User:** First, generate a **hashed password** using Python:
    ```python
    from werkzeug.security import generate_password_hash
    # !! Replace 'your_secret_admin_password' with the actual password you want !!
    hashed_pw = generate_password_hash('your_secret_admin_password')
    print(hashed_pw)
    exit()
    ```
    Then, insert the admin user into MySQL:
    ```sql
    INSERT INTO users (username, password_hash, full_name, role, is_active, email)
    VALUES ('admin', 'paste_the_hashed_password_here', 'Administrator', 'Admin', 1, 'admin@example.com');
    ```
2.  **Log in** as the `admin` user.
3.  **Approve HOD:** Approve any pending **Head of Department (HOD)** registrations on the Admin Dashboard.

### 2\. HOD Tasks (Department & Subject Configuration)

Log in as the **HOD**:

  * **Assign Department:** Ensure the HOD's `dept_id` is correctly set in the `users` table.
    *Example (assuming Dept ID 1 is 'CSBS'):*
    ```sql
    UPDATE users
    SET dept_id = 1
    WHERE user_id = [HOD_USER_ID]; -- Use the correct HOD user_id
    ```
  * **Add Teachers:** Use the HOD Dashboard to add teacher accounts.
  * **Add Subjects:** Configure subjects for the department (e.g., COA (CS01)).
  * **Add Schedules:** Create class schedules, linking a **Subject**, **Teacher**, **Division**, and **Day/Time**.

### 3\. Teacher Tasks (Face Data & Attendance)

Log in as the **Teacher**:

#### A. Add Student Face Data

1.  Navigate to **Manage Students** $\rightarrow$ **Student List**.
2.  Click **Edit** next to a student.
3.  Use **Option 2: Webcam Capture** to click **"Capture & Save Embedding"**. This registers the student's face for recognition.

#### B. Take Attendance

1.  Click **Take Attendance**.
2.  Select a relevant scheduled session and click **"Continue Session"**.
3.  Use one of the methods on the attendance page:
      * **Capture & Recognize Frame** (Webcam input)
      * **Upload & Recognize Image(s)** (File input)
      * **Present/Absent Buttons** (Manual override)

<!-- end list -->

```
```