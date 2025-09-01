# Computer Vision Detection Tools

This directory contains two Python scripts for different computer vision tasks:
1. `retinaface.py` - Face detection using RetinaFace
2. `people_detector.py` - People detection using YOLOv8

## Prerequisites

Install the required packages:
```bash
pip install -r requirements.txt
```

## 1. Face Detection with RetinaFace

Detects multiple faces in an image using RetinaFace, with bounding boxes, confidence scores, and facial landmarks.

### Usage
```bash
python retinaface.py --image path/to/image.jpg [--threshold 0.7] [--output output.jpg]
```

### Parameters
- `--image`: Path to the input image (required)
- `--threshold`: Confidence threshold (0-1, default: 0.5)
- `--output`: Output file path (default: 'output.jpg')

### Example
```bash
python retinaface.py --image group_photo.jpg --threshold 0.7 --output detected_faces.jpg
```

## Requirements
All required packages are listed in the root `requirements.txt` file.


