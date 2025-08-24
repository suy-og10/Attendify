import cv2
import numpy as np
from PIL import Image
import torch
from retinaface import RetinaFace
import matplotlib.pyplot as plt

class RetinaFaceDetector:
    def __init__(self):
        """Initialize the RetinaFace detector."""
        # The model will be loaded on first detection
        pass
        
    def detect_faces(self, image_path, threshold=0.7):
        """
        Detect faces in an image using RetinaFace.
        
        Args:
            image_path (str): Path to the input image
            threshold (float): Confidence threshold for face detection
            
        Returns:
            tuple: (image with bounding boxes, list of face detections)
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = RetinaFace.detect_faces(img_rgb, threshold=threshold)
        
        # Convert detections to list format
        faces = []
        if isinstance(detections, dict):
            for face_id, face_data in detections.items():
                faces.append({
                    'x1': face_data['facial_area'][0],
                    'y1': face_data['facial_area'][1],
                    'x2': face_data['facial_area'][2],
                    'y2': face_data['facial_area'][3],
                    'confidence': face_data['score'],
                    'landmarks': {
                        'left_eye': face_data['landmarks']['left_eye'],
                        'right_eye': face_data['landmarks']['right_eye'],
                        'nose': face_data['landmarks']['nose'],
                        'mouth_left': face_data['landmarks']['mouth_left'],
                        'mouth_right': face_data['landmarks']['mouth_right']
                    }
                })
        
        return img_rgb, faces
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and landmarks on the image.
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            detections (list): List of face detections
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        img_with_boxes = image.copy()
        
        for face in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"{confidence:.2f}"
            cv2.putText(img_with_boxes, label, (int(x1), int(y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw facial landmarks if available
            if 'landmarks' in face:
                for point in face['landmarks'].values():
                    cv2.circle(img_with_boxes, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        
        return img_with_boxes

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection using RetinaFace')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for detection')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save the output image')
    args = parser.parse_args()
    
    # Initialize the detector
    detector = RetinaFaceDetector()
    
    try:
        # Detect faces
        image, detections = detector.detect_faces(args.image, args.threshold)
        
        # Draw detections
        result_image = detector.draw_detections(image, detections)
        
        # Save the result
        cv2.imwrite(args.output, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"Detected {len(detections)} faces. Result saved to {args.output}")
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
