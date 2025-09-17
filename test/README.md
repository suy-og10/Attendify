# Face Recognition System

This directory contains Python scripts for face detection and recognition:
1. `retina_face_detector.py` - Face detection using RetinaFace with facial landmarks
2. `face_recognizer.py` - Face recognition using InsightFace for face detection + embeddings and cosine similarity matching
3. `generate_embeddings.py` - Precomputes embeddings for your dataset and stores them in `known_embeddings.pkl`

## Prerequisites

Install the required packages from the root directory:
```bash
pip install -r ../requirements.txt
```

Ensure you have a dataset prepared under `../dataset/` (see structure below).

## 1. Face Detection with RetinaFace

Detects faces in an image using RetinaFace, providing bounding boxes, confidence scores, and facial landmarks.

### Usage
```bash
python retina_face_detector.py --image path/to/image.jpg [--threshold 0.7] [--output output.jpg]
```

### Parameters
- `--image`: Path to the input image (required)
- `--threshold`: Confidence threshold (0-1, default: 0.7)
- `--output`: Output file path (default: `output.jpg`)

### Example
```bash
python retina_face_detector.py --image class.jpg --threshold 0.7 --output detected_faces.jpg
```

## 2. Face Recognition Workflow

The recognition pipeline compares faces in a test image against a dataset of known faces using InsightFace embeddings and cosine similarity.

### Dataset Structure
Place your dataset in the `../dataset` directory with the following structure:
```
dataset/
├── Person1_Name_ID/
│   ├── Person1_0001.jpg
│   ├── Person1_0002.jpg
│   └── ...
└── Person2_Name_ID/
    ├── Person2_0001.jpg
    └── ...
```

### Step 1: Generate embeddings
Precompute embeddings for all persons in the dataset. This creates `known_embeddings.pkl` inside this `test/` directory.
```bash
python generate_embeddings.py
```
- Output: `test/known_embeddings.pkl`

### Step 2: Run face recognition
Run the recognizer and provide the test image path when prompted. The script will detect faces using InsightFace and match them against the known embeddings.
```bash
python face_recognizer.py
```
- You will be prompted: "Please enter the path to the image you want to recognize faces in:"
- Output image is saved under `test/recognized_faces/recognized_<input-name>_<timestamp>.jpg`

### Thresholds and configuration
`face_recognizer.py` uses internal thresholds:
- Similarity threshold: `SIMILARITY_THRESHOLD = 0.5`
- Confidence threshold: `CONFIDENCE_THRESHOLD = 0.65`

You can adjust these values inside `face_recognizer.py` to tune matching behavior.

### Notes
- The system uses InsightFace's `buffalo_l` model for both detection and embeddings in `face_recognizer.py`.
- `retina_face_detector.py` is a standalone detector utility; it is not required to run `face_recognizer.py`.
- GPU vs CPU: Both `generate_embeddings.py` and `face_recognizer.py` call `embedder.prepare(ctx_id=0)`.
  - Use `ctx_id=0` for GPU (if available)
  - Set `ctx_id=-1` to force CPU execution
- Ensure images are clear, well-lit, and mostly frontal for best results.
