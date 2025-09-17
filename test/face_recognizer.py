import os
import cv2
import numpy as np
from retina_face_detector import RetinaFaceDetector
from sklearn.metrics.pairwise import cosine_similarity
import insightface

print("Starting face recognition script...")

# Initialize detector and embedding model
print("Initializing models...")
detector = RetinaFaceDetector()
embedder = insightface.app.FaceAnalysis(name='buffalo_l')
embedder.prepare(ctx_id=0)
print("Models initialized successfully")

# Load dataset embeddings
print("\nLoading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))
print(f"Looking for dataset in: {dataset_dir}")
print(f"Directory exists: {os.path.exists(dataset_dir)}")

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

known_embeddings = []
known_names = []

# Process all person directories
person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
if not person_dirs:
    raise FileNotFoundError(f"No person directories found in {dataset_dir}")

print(f"\nFound {len(person_dirs)} person directories")
print("Processing all persons' images...")

# Process each person's directory
for person in person_dirs:
    person_dir = os.path.join(dataset_dir, person)
    print(f"\nProcessing: {person}")
    
    # Get all images for this person
    image_files = [f for f in os.listdir(person_dir) 
                  if f.lower().endswith((".jpg", ".png", ".jpeg")) 
                  and not f.lower().endswith("person_info.json")]  # Skip JSON files
    
    if not image_files:
        print(f"  No images found for {person}")
        continue
        
    # Process up to 10 images per person for better accuracy
    sample_size = min(10, len(image_files))
    print(f"  Processing {sample_size} out of {len(image_files)} images")
    
    processed_count = 0
    for file in image_files[:sample_size]:
        try:
            img_path = os.path.join(person_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Could not read image: {file}")
                continue
                
            # Detect faces
            faces = embedder.get(img)
            if faces:
                # Use the first face found
                face = faces[0]
                embedding = face.embedding
                known_embeddings.append(embedding)
                known_names.append(person)
                processed_count += 1
                
        except Exception as e:
            print(f"  Error processing {file}: {str(e)}")
    
    print(f"  Successfully processed {processed_count} faces for {person}")

print("\nProcessing test image...")
# Test image
test_img_path = os.path.join(script_dir, 'cctv2.jpg')
print(f"Looking for test image at: {test_img_path}")
print(f"Test image exists: {os.path.exists(test_img_path)}")

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
            # Sort by similarity score
            valid_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Get the best match
            best_idx, best_score = valid_matches[0]
            name = known_names[best_idx]
            
            # Calculate confidence (normalize score between thresholds)
            confidence = min(1.0, (best_score - SIMILARITY_THRESHOLD) / (1 - SIMILARITY_THRESHOLD))
            
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"  Match found: {name} (confidence: {confidence:.2f})")
            else:
                print(f"  Low confidence match: {name} (confidence: {confidence:.2f})")
                name = f"Possible {name.split('_')[0]}"  # Use just first name if low confidence
        else:
            name = "Unknown"
            confidence = 0.0
            print(f"  No match found (best similarity: {best_similarity:.2f})")
        
        # Draw on image
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # Display name and confidence
        label = f"{name} ({confidence:.2f})"
        cv2.putText(test_img, label, 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
    except Exception as e:
        print(f"  Error processing face {i}: {str(e)}")

output_path = "recognized_class3.jpg"
cv2.imwrite(output_path, test_img)
print(f"\nOutput saved as {os.path.abspath(output_path)}")
