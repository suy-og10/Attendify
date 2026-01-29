from flask import Blueprint, render_template, request, flash, redirect, url_for, g,  session, current_app
from backend.database import get_db
from backend.utils import login_required, role_required
from backend.database import get_db, query_db, execute_db


admin_bp = Blueprint('admin', __name__, template_folder='../../frontend/templates/admin', url_prefix='/admin')

@admin_bp.route('/dashboard')
@login_required
@role_required('Admin')
def dashboard():
    db = get_db()
    # Fetch data needed for admin dashboard (e.g., pending HODs, user counts)
    pending_hods = query_db("SELECT user_id, username, full_name, email FROM users WHERE role = %s AND is_active = FALSE", ('HOD',))
    return render_template('dashboard.html', user=g.user, pending_hods=pending_hods)

@admin_bp.route('/approve_hod/<int:user_id>', methods=['POST'])
@login_required
@role_required('Admin')
def approve_hod(user_id):
    db = get_db()
    # Check if user exists and is an inactive HOD
    hod = query_db("SELECT user_id FROM users WHERE user_id = %s AND role = %s AND is_active = FALSE", (user_id, 'HOD'), one=True)
    if hod:
        try:
            execute_db("UPDATE users SET is_active = TRUE WHERE user_id = %s", (user_id,))
            db.commit()
            flash(f'HOD account (ID: {user_id}) approved successfully.', 'success')
            current_app.logger.info(f"Admin {session.get('username')} approved HOD ID {user_id}")
        except Exception as e:
            db.rollback()
            flash(f'Error approving HOD: {e}', 'error')
            current_app.logger.error(f"Error approving HOD ID {user_id}: {e}", exc_info=True)
    else:
        flash('HOD not found or already active.', 'warning')
    return redirect(url_for('admin.dashboard'))



@admin_bp.route('/reject_hod/<int:user_id>', methods=['POST'])
@login_required
@role_required('Admin')
def reject_hod(user_id):
    db = get_db()
    # Check if user exists and is an inactive HOD
    hod = query_db("SELECT username FROM users WHERE user_id = %s AND role = 'HOD' AND is_active = FALSE", (user_id,), one=True)
    if hod:
        try:
            # Delete the user record
            execute_db("DELETE FROM users WHERE user_id = %s", (user_id,))
            flash(f"HOD registration for '{hod['username']}' rejected and removed.", 'success')
            current_app.logger.info(f"Admin {session.get('username')} rejected HOD ID {user_id} ({hod['username']})")
        except Exception as e:
            flash(f'Error rejecting HOD: {e}', 'error')
            current_app.logger.error(f"Error rejecting HOD ID {user_id}: {e}", exc_info=True)
    else:
        flash('HOD not found or already active/rejected.', 'warning')
    return redirect(url_for('admin.dashboard'))

# --- Add other admin routes below ---