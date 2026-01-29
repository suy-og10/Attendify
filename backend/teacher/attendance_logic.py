import cv2
import numpy as np
import pickle
import insightface
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import current_app
from backend.database import get_db, query_db
import mysql.connector
# from .attendance_logic import generate_embedding_for_student, clear_embedding_cache
# Global variable to hold the embedder model (initialize once)
_embedder = None
_known_embeddings = None
_known_names = None
_student_id_map = {} # Map known_names (folder names) to student_ids

def initialize_models_and_data():
    """Loads embeddings and initializes the InsightFace model."""
    global _embedder, _known_embeddings, _known_names, _student_id_map

    # --- 1. Initialize Embedder Model (buffalo_l) ---
    if _embedder is None:
        try:
            current_app.logger.info("Initializing face analysis model...")
            _embedder = insightface.app.FaceAnalysis(name='buffalo_l')
            ctx_id = 0 
            try:
                _embedder.prepare(ctx_id=ctx_id)
            except Exception as gpu_err:
                 # FIX: Changed 'err' to 'gpu_err' for clarity, though logic remains same
                 current_app.logger.warning(f"GPU context {ctx_id} failed: {gpu_err}. Trying CPU (ctx_id=-1)...")
                 ctx_id = -1
                 _embedder.prepare(ctx_id=ctx_id)
            current_app.logger.info(f"Model initialized on context {ctx_id}.")
        except Exception as e:
            current_app.logger.error(f"Fatal error initializing InsightFace model: {e}")
            raise RuntimeError(f"Could not initialize InsightFace: {e}")

    # --- 2. Load Embeddings from Database ---
    if _known_embeddings is None or len(_known_embeddings) == 0:
        try:
            # FIX: Use query_db instead of db.execute (THIS WAS THE MISSING CURSOR)
            rows = query_db("""
                SELECT s.prn, s.student_id, fe.embedding_vector
                FROM face_embeddings fe
                JOIN students s ON fe.student_id = s.student_id
                WHERE fe.is_active = TRUE AND s.is_active = TRUE
            """)

            if not rows:
                 current_app.logger.warning("No active embeddings found in the database.")
                 _known_embeddings = np.array([])
                 _known_names = []
                 _student_id_map = {}
                 return

            temp_embeddings = []
            temp_names = [] 
            temp_id_map = {}
            for row in rows:
                try:
                    # Deserialize the embedding vector (stored as BLOB/bytes)
                    embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                    temp_embeddings.append(embedding)
                    prn = row['prn']
                    temp_names.append(prn)
                    temp_id_map[prn] = row['student_id']
                except Exception as deser_err:
                    current_app.logger.error(f"Error deserializing embedding for student PRN {row.get('prn', 'N/A')}: {deser_err}")

            _known_embeddings = np.array(temp_embeddings)
            _known_names = temp_names
            _student_id_map = temp_id_map
            current_app.logger.info(f"Loaded {len(_known_embeddings)} known embeddings.")

        except Exception as db_err:
            current_app.logger.error(f"Error loading embeddings from database: {db_err}")
            # This is the line that generates the error message seen in the trace
            raise RuntimeError(f"Could not load embeddings from DB: {db_err}")


# In: backend/teacher/attendance_logic.py

def recognize_faces_in_image(image_np):
    """
    Recognizes faces in an image and matches them against known embeddings.
    (FIXED: Simplified threshold logic)
    """
    initialize_models_and_data() # Ensure models/data are loaded

    if _embedder is None or _known_embeddings is None or len(_known_embeddings) == 0:
        current_app.logger.warning("Embedder not initialized or no known embeddings. Cannot recognize.")
        return [], 0 # Return empty list and 0 unknown

    results = []
    unknown_count = 0

    try:
        faces = _embedder.get(image_np) # Expects BGR
        current_app.logger.debug(f"Detected {len(faces)} faces in the image.")

        if not faces:
            return [], 0

        # --- THIS IS THE FIX ---
        # We will use the standard similarity threshold directly.
        similarity_threshold = current_app.config.get('FACE_SIMILARITY_THRESHOLD', 0.5)
        # --- END OF FIX ---

        for face in faces:
            if face.embedding is None:
                current_app.logger.warning("Detected face has no embedding.")
                unknown_count += 1
                continue

            test_emb = face.embedding.reshape(1, -1)
            sims = cosine_similarity(test_emb, _known_embeddings)

            best_match_idx = int(np.argmax(sims))
            best_similarity = float(sims[0][best_match_idx])

            # --- MODIFIED LOGIC ---
            # Check if the best similarity meets the threshold
            if best_similarity >= similarity_threshold:
                matched_prn = _known_names[best_match_idx]
                student_id = _student_id_map.get(matched_prn)

                if student_id is not None:
                    # High confidence match
                    results.append({
                        "student_id": student_id,
                        "prn": matched_prn,
                        "confidence": round(best_similarity, 2), # Report the actual similarity
                        "bbox": face.bbox.astype(int).tolist()
                    })
                    current_app.logger.info(f"Recognized PRN: {matched_prn} (ID: {student_id}) with similarity {best_similarity:.2f}")
                else:
                    # Matched a PRN not in the map (should not happen)
                    unknown_count += 1
                    current_app.logger.warning(f"Match for PRN {matched_prn} but no student_id found.")
            else:
                # Below similarity threshold
                unknown_count += 1
                current_app.logger.info(f"No match found (best score {best_similarity:.2f} < threshold {similarity_threshold}). Marked as Unknown.")
            # --- END OF MODIFIED LOGIC ---

    except Exception as e:
        current_app.logger.error(f"Error during face recognition: {e}", exc_info=True)
        # Estimate unknown count if detection worked but matching failed
        unknown_count = len(faces) - len(results) if 'faces' in locals() and faces else 0
        
    return results, unknown_count

def generate_embedding_for_student(image_np):
    """Generates a face embedding from a single image."""
    initialize_models_and_data() # Ensure model is loaded

    if _embedder is None:
        raise RuntimeError("Face analysis model not initialized.")

    try:
        faces = _embedder.get(image_np)
        if not faces:
            return None, "No face detected in the provided image."
        if len(faces) > 1:
            return None, "Multiple faces detected. Please provide an image with only one face."

        embedding = faces[0].embedding
        # Convert numpy array to bytes for storing in BLOB
        embedding_bytes = embedding.tobytes()
        return embedding_bytes, None # Return bytes and no error
    except Exception as e:
        current_app.logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None, f"Error during embedding generation: {e}"
    
def match_embedding_to_known_faces(test_emb_np):
    """Matches a single test embedding against the known dataset."""
    initialize_models_and_data()

    if _known_embeddings is None or len(_known_embeddings) == 0:
         return None, 0, None, "No known faces loaded for comparison."

    similarity_threshold = current_app.config.get('FACE_SIMILARITY_THRESHOLD', 0.5)
    confidence_threshold = current_app.config.get('FACE_CONFIDENCE_THRESHOLD', 0.65)
    
    test_emb = test_emb_np.reshape(1, -1)
    
    # Cosine Similarity check
    sims = cosine_similarity(test_emb, _known_embeddings)
    best_match_idx = int(np.argmax(sims))
    best_similarity = float(sims[0][best_match_idx])

    if best_similarity < similarity_threshold:
        return None, best_similarity, None, "No match above similarity threshold."
        
    matched_prn = _known_names[best_match_idx]
    student_id = _student_id_map.get(matched_prn)

    # Calculate confidence based on threshold distance
    confidence = min(1.0, (best_similarity - similarity_threshold) / (1 - similarity_threshold))
    
    if confidence < confidence_threshold:
        return student_id, best_similarity, round(confidence, 2), "Low confidence match."
    else:
        return student_id, best_similarity, round(confidence, 2), "High confidence match."
    
def clear_embedding_cache():
    """
    Clears the globally cached embeddings, forcing a reload on the next
    recognition attempt.
    """
    global _known_embeddings, _known_names, _student_id_map
    
    current_app.logger.info("Clearing face embedding cache. Will reload from DB on next recognition.")
    _known_embeddings = None
    _known_names = None
    _student_id_map = {}