# Face Recognition System

This directory contains Python scripts for face detection and recognition:
1. `retina_face_detector.py` - Face detection using RetinaFace with facial landmarks
2. `face_recognizer.py` - Face recognition system using InsightFace for face embedding and matching

## Prerequisites

Install the required packages from the root directory:
```bash
pip install -r ../requirements.txt
```

## 1. Face Detection with RetinaFace

Detects faces in an image using RetinaFace, providing bounding boxes, confidence scores, and facial landmarks.

### Usage
```bash
python retina_face_detector.py --image path/to/image.jpg [--threshold 0.7] [--output output.jpg]
```

### Parameters
- `--image`: Path to the input image (required)
- `--threshold`: Confidence threshold (0-1, default: 0.7)
- `--output`: Output file path (default: 'detected_faces.jpg')

### Example
```bash
python retina_face_detector.py --image class.jpg --threshold 0.7 --output detected_faces.jpg
```

## 2. Face Recognition System

Recognizes faces by comparing them against a dataset of known faces using InsightFace's deep learning models.

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

### Usage
```bash
python face_recognizer.py --image path/to/test_image.jpg [--threshold 0.5] [--output output.jpg]
```

### Parameters
- `--image`: Path to the test image (required)
- `--threshold`: Similarity threshold for face matching (0-1, default: 0.5)
- `--output`: Output file path (default: 'recognized_faces.jpg')

### Example
```bash
python face_recognizer.py --image test_image.jpg --threshold 0.6 --output recognized.jpg
```

## Notes
- The system uses the 'buffalo_l' model from InsightFace by default
- Face detection confidence threshold affects both detection and recognition accuracy
- For best results, ensure good lighting and frontal face images in the dataset
