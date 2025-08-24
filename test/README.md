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

## 2. People Detection with YOLOv8

Detects people in images using YOLOv8 model, providing bounding boxes and confidence scores.

### Usage
```bash
python people_detector.py --image path/to/image.jpg [--threshold 0.5] [--output output.jpg]
```

### Parameters
- `--image`: Path to the input image (required)
- `--threshold`: Confidence threshold (0-1, default: 0.5)
- `--output`: Output file path (default: 'detected_people.jpg')

### Example
```bash
python people_detector.py --image crowd.jpg --threshold 0.6 --output crowd_detected.jpg
```

## Notes
- Both scripts will automatically use GPU if CUDA is available
- The first run will download the required model files automatically
- For best results, use clear, well-lit images
- Lower threshold values will detect more objects but might include false positives
- Higher threshold values will be more strict about detections but might miss some objects

## Requirements
All required packages are listed in the root `requirements.txt` file.


