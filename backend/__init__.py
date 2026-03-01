import datetime
import threading
import time
import cv2
import numpy as np
from flask import Flask
import os

def create_app(config_class=None):
    app = Flask(__name__,
                template_folder=os.path.abspath('../frontend/templates'),
                static_folder=os.path.abspath('../frontend/static'))
    if config_class is None:
        from .config import Config
        app.config.from_object(Config)
    else:
        app.config.from_object(config_class)

    if not app.config.get('SECRET_KEY'):
        if app.config.get('DEBUG'):
            app.config['SECRET_KEY'] = 'dev-only-secret-key'
            app.logger.warning('SECRET_KEY not configured; using development fallback key.')
        else:
            raise RuntimeError('SECRET_KEY is required. Set SECRET_KEY environment variable.')

    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    from . import database
    database.init_app(app)

    from .auth.routes import auth_bp
    from .admin.routes import admin_bp

    from .hod.routes import hod_bp
    from .teacher.routes import teacher_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(hod_bp, url_prefix='/hod')
    app.register_blueprint(teacher_bp, url_prefix='/teacher')

    @app.context_processor
    def inject_now():
        return {'now': datetime.datetime.utcnow}

    @app.route('/')
    def index():
        from flask import redirect, url_for
        return redirect(url_for('auth.login'))

    def _ensure_session_and_get_id(db_query, db_exec, schedule_id, today_iso):
        db_exec(
            "INSERT IGNORE INTO class_sessions (schedule_id, session_date, status) VALUES (%s, %s, 'ONGOING')",
            (schedule_id, today_iso)
        )
        row = db_query(
            "SELECT session_id FROM class_sessions WHERE schedule_id = %s AND session_date = %s",
            (schedule_id, today_iso), one=True
        )
        if row:
            db_exec("UPDATE class_sessions SET status='ONGOING', actual_start_time = COALESCE(actual_start_time, CURRENT_TIME) WHERE session_id=%s", (row['session_id'],))
            return row['session_id']
        return None

    def _mark_recognized_for_session(app_ctx_db, session_meta, recognized):
        if not recognized:
            return 0
        recognized_ids = [s['student_id'] for s in recognized]
        placeholders = ','.join(['%s'] * len(recognized_ids))
        enrolled = app_ctx_db['query'](
            f"""
            SELECT student_id, student_name, prn
            FROM students
            WHERE student_id IN ({placeholders})
              AND division = %s AND academic_year = %s AND dept_id = %s
              AND is_active = TRUE
            """,
            (*recognized_ids, session_meta['division'], session_meta['academic_year'], session_meta['dept_id'])
        )
        enrolled_map = {r['student_id']: r for r in enrolled}
        marked = 0
        for s in recognized:
            sid = s['student_id']
            if sid in enrolled_map:
                app_ctx_db['exec'](
                    """
                    INSERT INTO attendance_records (session_id, student_id, status, recognition_confidence, verification_method, marked_by, marked_time)
                    VALUES (%s, %s, 'Present', %s, 'FACE_LIVE', %s, CURRENT_TIMESTAMP)
                    ON DUPLICATE KEY UPDATE status=VALUES(status), recognition_confidence=VALUES(recognition_confidence), verification_method=VALUES(verification_method), marked_by=VALUES(marked_by)
                    """,
                    (session_meta['session_id'], sid, s['confidence'], session_meta['teacher_id'])
                )
                marked += 1
        return marked

    def _capture_for_first_5_minutes(app):
        from .database import get_db, query_db as _q, execute_db as _x
        from .teacher.attendance_logic import recognize_faces_in_image
        running = set()
        poll_interval = int(app.config.get('AUTO_ATTENDANCE_POLL_SECONDS', 30))
        camera_index = int(app.config.get('CAMERA_INDEX', 0))
        while True:
            try:
                with app.app_context():
                    now = datetime.datetime.now()
                    today_iso = now.date().isoformat()
                    weekday = now.weekday()
                    schedules = _q(
                        """
                        SELECT cs.schedule_id, cs.start_time, cs.end_time, cs.division, cs.academic_year, cs.teacher_id, s.dept_id
                        FROM class_schedules cs
                        JOIN subjects s ON cs.subject_id = s.subject_id
                        WHERE cs.is_active = TRUE AND cs.day_of_week = %s
                        """,
                        (weekday,)
                    )
                    for sch in schedules or []:
                        start_dt = datetime.datetime.combine(now.date(), (datetime.datetime.min + sch['start_time']).time()) if isinstance(sch['start_time'], datetime.timedelta) else datetime.datetime.combine(now.date(), sch['start_time'])
                        window_start = start_dt
                        window_end = start_dt + datetime.timedelta(minutes=5)
                        if now < window_start or now > window_end:
                            continue
                        session_id = _ensure_session_and_get_id(_q, _x, sch['schedule_id'], today_iso)
                        if not session_id:
                            continue
                        key = f"{session_id}"
                        if key in running:
                            continue
                        running.add(key)
                        def _worker(sess_id, meta):
                            with app.app_context():
                                cap = cv2.VideoCapture(camera_index)
                                try:
                                    end_ts = window_end
                                    while datetime.datetime.now() <= end_ts:
                                        ok, frame = cap.read()
                                        if not ok or frame is None:
                                            time.sleep(1)
                                            continue
                                        recognized, _ = recognize_faces_in_image(frame)
                                        _mark_recognized_for_session(
                                            {'query': _q, 'exec': _x},
                                            {**meta, 'session_id': sess_id},
                                            recognized
                                        )
                                        time.sleep(int(app.config.get('AUTO_ATTENDANCE_FRAME_INTERVAL_SECONDS', 3)))
                                finally:
                                    try:
                                        cap.release()
                                    except Exception:
                                        pass
                                    running.discard(key)
                        meta = {
                            'division': sch['division'],
                            'academic_year': sch['academic_year'],
                            'dept_id': sch['dept_id'],
                            'teacher_id': sch['teacher_id']
                        }
                        t = threading.Thread(target=_worker, args=(session_id, meta), daemon=True)
                        t.start()
            except Exception as e:
                app.logger.error(f"Auto-attendance scheduler loop error: {e}", exc_info=True)
            time.sleep(poll_interval)

    def start_auto_attendance_scheduler():
        enabled = bool(os.environ.get('AUTO_ATTENDANCE_ENABLED', '1') != '0')
        if not enabled:
            app.logger.info("Auto-attendance scheduler disabled via AUTO_ATTENDANCE_ENABLED=0")
            return
        app.logger.info("Starting auto-attendance scheduler thread")
        threading.Thread(target=_capture_for_first_5_minutes, args=(app,), daemon=True).start()

    start_auto_attendance_scheduler()

    return app