# face_recognizer.py (updated version)

import os
import cv2
import numpy as np
import pickle
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from retina_face_detector import RetinaFaceDetector

print("Starting face recognition script...")

# Initialize detector and embedding model
print("Initializing models...")
detector = RetinaFaceDetector()
embedder = insightface.app.FaceAnalysis(name='buffalo_l')
embedder.prepare(ctx_id=0)
print("Models initialized successfully")

# Load pre-generated dataset embeddings
print("\nLoading pre-generated embeddings...")
script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_file = os.path.join(script_dir, 'known_embeddings.pkl')

if not os.path.exists(embeddings_file):
    print("Error: Embeddings file not found. Please run generate_embeddings.py first.")
    exit(1)

with open(embeddings_file, 'rb') as f:
    data = pickle.load(f)
    known_embeddings = data['embeddings']
    known_names = data['names']

print(f"Loaded {len(known_embeddings)} known faces.")

print("\nProcessing test image...")
# Test image
test_img_path = os.path.join(script_dir, 'class.jpg')
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"Test image not found at {test_img_path}")

test_img = cv2.imread(test_img_path)
print(f"Test image loaded: {test_img.shape[1]}x{test_img.shape[0]}")

print("Detecting faces in test image...")
faces = embedder.get(test_img)
print(f"Found {len(faces)} faces in test image")

if not known_embeddings:
    print("\nError: No faces were processed from the dataset")
    exit(1)

# Face matching with improved thresholding
SIMILARITY_THRESHOLD = 0.5  # Lower threshold to be more inclusive
CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for final decision

for i, face in enumerate(faces, 1):
    try:
        print(f"\nProcessing face {i}/{len(faces)}")
        test_emb = face.embedding.reshape(1, -1)
        
        # Calculate similarities with all known faces
        sims = cosine_similarity(test_emb, known_embeddings)
        best_match_idx = np.argmax(sims)
        best_similarity = sims[0][best_match_idx]
        
        # Get all matches above similarity threshold
        valid_matches = [(i, s) for i, s in enumerate(sims[0]) if s >= SIMILARITY_THRESHOLD]
        
        if valid_matches:
            valid_matches.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score = valid_matches[0]
            name = known_names[best_idx]
            
            confidence = min(1.0, (best_score - SIMILARITY_THRESHOLD) / (1 - SIMILARITY_THRESHOLD))
            
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"  Match found: {name} (confidence: {confidence:.2f})")
            else:
                print(f"  Low confidence match: {name} (confidence: {confidence:.2f})")
                name = f"Possible {name.split('_')[0]}"
        else:
            name = "Unknown"
            confidence = 0.0
            print(f"  No match found (best similarity: {best_similarity:.2f})")
        
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        label = f"{name} ({confidence:.2f})"
        cv2.putText(test_img, label, 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
    except Exception as e:
        print(f"  Error processing face {i}: {str(e)}")

output_path = "recognized_class4.jpg"
cv2.imwrite(output_path, test_img)
print(f"\nOutput saved as {os.path.abspath(output_path)}")