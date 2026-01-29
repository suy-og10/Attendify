# Attendify-main/test/generate_embeddings.py
import os
import cv2
import numpy as np
import pickle
import insightface
import sqlite3

print("Starting embedding generation script...")

# --- Database Configuration ---
DATABASE_NAME = '../attendify.db' # Path relative to this script in test/

def connect_db():
    """Connects to the SQLite database. Returns the connection object."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        print(f"Successfully connected to database '{DATABASE_NAME}'")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_students_from_db(conn):
    """Retrieves student records (id, name, roll_number, photo_folder) from the database."""
    if not conn:
        print("Database connection not available.")
        return []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, roll_number, photo_folder FROM students WHERE photo_folder IS NOT NULL")
        students = cursor.fetchall()
        print(f"Found {len(students)} student records with photo folders in the database.")
        return students
    except sqlite3.Error as e:
        print(f"Error fetching students from database: {e}")
        return []

# --- Model Initialization ---
print("Initializing model...")
try:
    embedder = insightface.app.FaceAnalysis(name='buffalo_l')
    embedder.prepare(ctx_id=0) # Try GPU
except Exception:
    print("Falling back to CPU (ctx_id=-1)")
    embedder = insightface.app.FaceAnalysis(name='buffalo_l')
    embedder.prepare(ctx_id=-1) # Force CPU
print("Model initialized successfully")

# --- Load Student Data from Database ---
print("\nConnecting to database to get student list...")
db_conn = connect_db()
if not db_conn:
    print("Exiting.")
    exit(1)

students_data = get_students_from_db(db_conn)
db_conn.close() # Close connection after fetching data

if not students_data:
    print("No student data found in the database. Make sure students have been added via capture_images.py.")
    exit(1)

# --- Process Images and Generate Embeddings ---
known_embeddings = []
known_names = [] # We'll store name_rollnumber for identification

print("\nProcessing student images...")

script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of this script (test/)
root_dir = os.path.abspath(os.path.join(script_dir, '..')) # Project root directory

for student in students_data:
    student_id = student['id']
    student_name = student['name']
    roll_number = student['roll_number']
    photo_folder_relative = student['photo_folder'] # Path stored in DB (e.g., dataset/Suyog_0455)

    # Construct the absolute path to the photo folder
    photo_folder_abs = os.path.join(root_dir, photo_folder_relative)

    # Use a unique identifier (like Name_RollNumber) for matching later
    # Using just ID might be slightly better, but Name_Roll is more readable
    identifier = f"{student_name}_{roll_number}"
    print(f"Processing: {identifier} (ID: {student_id}) from folder: {photo_folder_relative}")

    if not os.path.isdir(photo_folder_abs):
        print(f"  Warning: Photo folder not found at {photo_folder_abs}")
        continue

    image_files = [f for f in os.listdir(photo_folder_abs) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print(f"  No images found in {photo_folder_abs}")
        continue

    processed_count = 0
    sample_size = min(10, len(image_files)) # Process up to 10 images per student
    for file in image_files[:sample_size]:
        try:
            img_path = os.path.join(photo_folder_abs, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Could not read image: {file}")
                continue

            faces = embedder.get(img)
            if faces: # Check if any faces were detected
                # Assuming the first detected face is the correct one for dataset images
                face = faces[0]
                embedding = face.embedding
                known_embeddings.append(embedding)
                known_names.append(identifier) # Store the combined identifier
                processed_count += 1
            # else: print(f"  No face detected in: {file}") # Optional: uncomment for debugging

        except Exception as e:
            print(f"  Error processing {file}: {str(e)}")

    print(f"  Successfully processed {processed_count} faces for {identifier}")

# --- Save Embeddings ---
embeddings_file_path = os.path.join(script_dir, 'known_embeddings.pkl')
if known_embeddings:
    with open(embeddings_file_path, 'wb') as f:
        pickle.dump({'embeddings': known_embeddings, 'names': known_names}, f)
    print(f"\nEmbedding file saved successfully with {len(known_embeddings)} embeddings at: {os.path.abspath(embeddings_file_path)}")
else:
    print("\nWarning: No embeddings were generated. Embedding file was not created/updated.")