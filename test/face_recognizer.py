import os
import cv2
import numpy as np
import pickle
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from retina_face_detector import RetinaFaceDetector
from datetime import datetime

def get_user_input_image_path():
    """Prompts the user for the path to the image to be processed."""
    return input("Please enter the path to the image you want to recognize faces in: ").strip()

def create_output_directory(output_dir='recognized_faces'):
    """Creates an output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

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

# Get input image path from user
test_img_path = get_user_input_image_path()

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
SIMILARITY_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.65

for i, face in enumerate(faces, 1):
    try:
        print(f"\nProcessing face {i}/{len(faces)}")
        test_emb = face.embedding.reshape(1, -1)
        
        sims = cosine_similarity(test_emb, known_embeddings)
        best_match_idx = np.argmax(sims)
        best_similarity = sims[0][best_match_idx]
        
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

# Create output directory and generate a unique output filename
output_dir = create_output_directory()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
input_filename = os.path.splitext(os.path.basename(test_img_path))[0]
output_path = os.path.join(output_dir, f"recognized_{input_filename}_{timestamp}.jpg")

cv2.imwrite(output_path, test_img)
print(f"\nOutput saved as {os.path.abspath(output_path)}")