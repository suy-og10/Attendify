# Attendify - Smart Attendance Management System

Attendify is an AI-powered face recognition attendance management system built for modern educational institutions. It provides a comprehensive web-based platform with distinct portals for administrators, heads of departments (HOD), teachers, and students.

## Key Features

- **Automated Face Recognition Attendance**: Teachers can mark attendance using live webcam feeds or uploaded classroom photos via the web interface.
- **Role-Based Access Control**:
  - **Admin**: Manage the entire system, approve new users.
  - **HOD**: Manage departments, subjects, students, and class schedules.
  - **Teacher**: Take attendance via face recognition or manual entry, and view class-specific attendance records.
  - **Student**: View personalized attendance dashboards and detailed subject-wise attendance records.
- **Smart Reports & Analytics**: Instantly generate detailed attendance reports per subject, division, or date range.
- **Timetable Management**: HODs can assign teachers to subjects and configure weekly class schedules.
- **Face Dataset Management**: Utilities to capture and register student faces for the recognition engine.

## Technology Stack

- **Backend**: Python, Flask, MySQL
- **Frontend**: HTML5, Vanilla CSS, TailwindCSS (for utility styling via JS or CDN), JavaScript
- **Computer Vision**: OpenCV, InsightFace / RetinaFace (for face detection and embeddings)

## Requirements

- Python 3.8+
- MySQL Server
- A webcam (for live face capture and attendance)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suy-og10/Attendify.git
   cd Attendify
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Configuration:**
   - Ensure MySQL is running locally.
   - Set up the environment variables or update `backend/config.py` with your MySQL connection details.
   - Run the database initialization command (if available) or manually execute the schemas in `backend/schema_mysql.sql`.

5. **Run the Web Application:**
   ```bash
   # From the project root
   python backend/app.py
   ```
   Access the system at `http://127.0.0.1:5000/`.

## Registration & Usage

- **User Accounts**: By default, new accounts register as unapproved users. They require approval from an existing 'Admin' account in the database.
- **Students**: Students can log in to view their attendance percentage and detailed daily logs per subject.
- **Image Capture Utility**: Use `capture_images.py` in the root folder for bulk face dataset collection on headless/local setups.

## Project Structure

```
AMS/
├── backend/            # Flask server, routes, database schemas, and AI logic
│   ├── auth/           # Authentication routes
│   ├── admin/, hod/, teacher/, student/ # Role-specific blueprints
│   ├── app.py          # Application entry point
│   └── schema_mysql.sql# MySQL Database schema definition
├── frontend/           # Web interface files
│   ├── static/         # CSS, JS, and Images
│   └── templates/      # Jinja2 HTML templates
├── dataset/            # Storage for raw captured face datasets
├── test/               # Face detection & recognition sandbox utilities
├── capture_images.py   # Bulk image capture script
└── requirements.txt    # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
