import pandas as pd
import io
from backend.database import get_db, query_db # We need query_db now
from flask import current_app
from datetime import date # To add the report generation date
from openpyxl.styles import Font # To make the header labels bold

def generate_attendance_excel(subject_id=None, start_date=None, end_date=None, division=None, teacher_id=None):
    """
    Generates an Excel attendance report with headers based on filters.
    """
    db = get_db() # Gets the raw DB connection
    
    # --- NEW: Fetch Header Information ---
    teacher_name = "N/A"
    if teacher_id:
        teacher_info = query_db("SELECT full_name FROM users WHERE user_id = %s", (teacher_id,), one=True)
        if teacher_info:
            teacher_name = teacher_info['full_name']
    
    subject_name = "All Subjects"
    if subject_id:
        subject_info = query_db("SELECT subject_name, subject_code FROM subjects WHERE subject_id = %s", (subject_id,), one=True)
        if subject_info:
            subject_name = f"{subject_info['subject_name']} ({subject_info['subject_code']})"
    
    report_date = date.today().isoformat()
    
    header_data = {
        "Faculty Name": teacher_name,
        "Subject": subject_name,
        "Report Date": report_date,
        "Filters": f"From {start_date} to {end_date}" + (f", Division: {division}" if division else "")
    }
    # --- END NEW HEADER LOGIC ---

    query = """
    SELECT
        s.prn AS PRN,
        s.student_name AS Name,
        s.roll_no AS RollNo,
        s.division AS Division,
        sub.subject_code AS SubjectCode,
        sess.session_date AS SessionDate,
        cs.start_time AS StartTime,
        ar.status AS Status,
        ar.recognition_confidence AS Confidence,
        ar.verification_method AS Method,
        ar.marked_time AS MarkedTime
    FROM attendance_records ar
    JOIN students s ON ar.student_id = s.student_id
    JOIN class_sessions sess ON ar.session_id = sess.session_id
    JOIN class_schedules cs ON sess.schedule_id = cs.schedule_id
    JOIN subjects sub ON cs.subject_id = sub.subject_id
    WHERE 1=1
    """
    params = []

    if subject_id:
        query += " AND cs.subject_id = %s"
        params.append(subject_id)
    if division:
        query += " AND cs.division = %s"
        params.append(division)
    if start_date:
        query += " AND sess.session_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND sess.session_date <= %s"
        params.append(end_date)
    if teacher_id:
        query += " AND cs.teacher_id = %s"
        params.append(teacher_id)

    query += " ORDER BY sess.session_date, cs.start_time, s.roll_no, s.student_name;"

    try:
        df = pd.read_sql_query(query, db, params=params)

        if df.empty:
            raise ValueError("No attendance records found for the selected filters.")
        
        # --- Create Excel ---
        excel_buffer = io.BytesIO()
        # NOTE: We are no longer using datetime_format/date_format here,
        # openpyxl will handle it.
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            
            # --- Helper function to write headers ---
            def write_header_to_sheet(worksheet, data):
                worksheet.insert_rows(1, amount=5) # Insert 5 blank rows at the top
                bold_font = Font(bold=True)
                
                worksheet['A1'] = "Faculty Name:"
                worksheet['B1'] = data["Faculty Name"]
                worksheet['A2'] = "Subject:"
                worksheet['B2'] = data["Subject"]
                worksheet['A3'] = "Report Date:"
                worksheet['B3'] = data["Report Date"]
                worksheet['A4'] = "Filters:"
                worksheet['B4'] = data["Filters"]
                
                # Style the labels
                for row in range(1, 5):
                    worksheet[f'A{row}'].font = bold_font

            
            # 1. Main Data Sheet
            # Write the main data first
            df.to_excel(writer, sheet_name='Attendance_Data', index=False)
            # Get the worksheet object
            data_ws = writer.sheets['Attendance_Data']
            # Write headers to it
            write_header_to_sheet(data_ws, header_data)
            
            # 2. Summary Sheet (Pivot Table)
            if not df.empty:
                try:
                    # Create a pivot table: Student Name vs SessionDate
                    summary_pivot = pd.pivot_table(
                        df,
                        values='Status',
                        index=['PRN', 'Name', 'RollNo'],
                        columns=['SessionDate'],
                        aggfunc=lambda x: 'Present' if 'Present' in x.values else ('Late' if 'Late' in x.values else 'Absent')
                    )
                    
                    # Calculate counts
                    status_counts = df.groupby(['PRN', 'Name', 'RollNo'])['Status'].value_counts().unstack(fill_value=0)
                    if 'Present' not in status_counts.columns: status_counts['Present'] = 0
                    if 'Absent' not in status_counts.columns: status_counts['Absent'] = 0
                    if 'Late' not in status_counts.columns: status_counts['Late'] = 0
                        
                    total_sessions = status_counts['Present'] + status_counts['Absent'] + status_counts['Late']
                    
                    # Calculate Percentage
                    attendance_percentage = (status_counts['Present'] + status_counts['Late']) / total_sessions * 100
                    
                    # Combine summary data
                    summary_df = status_counts
                    summary_df['Total_Sessions'] = total_sessions
                    summary_df['Attendance_%'] = attendance_percentage.round(2)
                    
                    # Merge pivot with summary stats
                    final_summary = summary_pivot.join(summary_df).reset_index()
                    
                    # Write the summary data
                    final_summary.to_excel(writer, sheet_name='Summary_Sheet', index=False)
                    # Get the summary worksheet
                    summary_ws = writer.sheets['Summary_Sheet']
                    # Write headers to it
                    write_header_to_sheet(summary_ws, header_data)
                    
                    # Auto-adjust column widths for summary
                    for col in summary_ws.columns:
                        max_length = 0
                        column = col[0].column_letter 
                        for cell in summary_ws[column]:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(cell.value)
                            except:
                                pass
                        adjusted_width = (max_length + 2)
                        summary_ws.column_dimensions[column].width = adjusted_width

                except Exception as pivot_e:
                    current_app.logger.error(f"Error creating pivot table: {pivot_e}")
                    # If pivot fails, just write a note
                    df_note = pd.DataFrame([{"note": "Could not generate summary pivot."}])
                    df_note.to_excel(writer, sheet_name='Summary_Sheet', index=False)
                    # Still add header to the error sheet
                    write_header_to_sheet(writer.sheets['Summary_Sheet'], header_data)


            # Auto-adjust column widths for data
            for col in data_ws.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in data_ws[column]:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                data_ws.column_dimensions[column].width = adjusted_width

        excel_buffer.seek(0)
        
    except Exception as e:
        # Log the specific pandas/db error
        current_app.logger.error(f"Pandas DB error during report generation: {e}")
        raise e # Re-raise the exception to be caught by the route

    return excel_buffer