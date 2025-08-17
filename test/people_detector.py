import cv2
import torch
from ultralytics import YOLO
import numpy as np

def detect_people_yolo(image_path, confidence_threshold=0.5):
    # Initialize YOLOv8 model (will auto-download on first run)
    model = YOLO('yolov8n.pt')  # Using nano version for speed, can use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Run YOLOv8 inference
    results = model(image, conf=confidence_threshold, classes=[0])  # class 0 is for person
    
    # Process results
    person_count = 0
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        
        for box, score in zip(boxes, scores):
            if score >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence score
                label = f'Person: {score:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                person_count += 1
    
    # Add person count to the image
    cv2.putText(image, f'Total People: {person_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the result
    output_path = 'detected_people_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, image)
    print(f"Detection complete! Saved as: {output_path}")
    
    # Display the result
    cv2.imshow('People Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return person_count

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import ultralytics
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "torchvision"])
    
    image_path = input("Enter the path to your image: ").strip('"\' ')
    detect_people_yolo(image_path)