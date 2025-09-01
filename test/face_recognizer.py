import os
import cv2
import numpy as np
from retina_face_detector import RetinaFaceDetector
from sklearn.metrics.pairwise import cosine_similarity
import insightface

# Initialize detector and embedding model
detector = RetinaFaceDetector()
embedder = insightface.app.FaceAnalysis(name='buffalo_l')
embedder.prepare(ctx_id=0)

# Load dataset embeddings
dataset_dir = "../dataset"
known_embeddings = []
known_names = []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        for file in os.listdir(person_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path)
                faces = embedder.get(img)
                if faces:
                    known_embeddings.append(faces[0].embedding)
                    known_names.append(person)

# Test image
test_img_path = "class.jpg"
test_img = cv2.imread(test_img_path)
faces = embedder.get(test_img)

for face in faces:
    test_emb = face.embedding.reshape(1, -1)
    sims = cosine_similarity(test_emb, known_embeddings)
    best_match_idx = np.argmax(sims)
    name = known_names[best_match_idx]

    # Draw on image
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(test_img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(test_img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imwrite("recognized_class.jpg", test_img)
print("✅ Output saved as recognized_class.jpg")
