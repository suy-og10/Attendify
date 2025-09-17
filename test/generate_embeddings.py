# generate_embeddings.py

import os
import cv2
import numpy as np
import pickle
import insightface

print("Starting embedding generation script...")

# Initialize embedding model
print("Initializing model...")
embedder = insightface.app.FaceAnalysis(name='buffalo_l')
embedder.prepare(ctx_id=0)
print("Model initialized successfully")

# Load dataset images
print("\nLoading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

known_embeddings = []
known_names = []

person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
if not person_dirs:
    raise FileNotFoundError(f"No person directories found in {dataset_dir}")

print(f"Found {len(person_dirs)} person directories")
print("Processing all persons' images...")

for person in person_dirs:
    person_dir = os.path.join(dataset_dir, person)
    print(f"Processing: {person}")
    
    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if not image_files:
        print(f"  No images found for {person}")
        continue
    
    processed_count = 0
    # You can still use a sample size to make the embedding process faster
    sample_size = min(10, len(image_files)) 
    for file in image_files[:sample_size]:
        try:
            img_path = os.path.join(person_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Could not read image: {file}")
                continue
            
            faces = embedder.get(img)
            if faces:
                face = faces[0]
                embedding = face.embedding
                known_embeddings.append(embedding)
                known_names.append(person)
                processed_count += 1
        except Exception as e:
            print(f"  Error processing {file}: {str(e)}")
    
    print(f"  Successfully processed {processed_count} faces for {person}")

# Save the embeddings and names
embeddings_file = os.path.join(script_dir, 'known_embeddings.pkl')
with open(embeddings_file, 'wb') as f:
    pickle.dump({'embeddings': known_embeddings, 'names': known_names}, f)

print(f"\nEmbedding file saved successfully at: {os.path.abspath(embeddings_file)}")