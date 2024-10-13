import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import mss
import pynput
import time

# Initialize YOLO model (you can choose different model sizes like 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8s.pt')  # Ensure the model file is downloaded or specify the correct path

def detect_people(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(img_rgb)
    
    # Initialize list to store detection data (we'll convert this to a DataFrame later)
    detection_data = []
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = model.names[int(box.cls[0])]
            if cls != 'person':
                continue  # Only interested in people
            confidence = box.conf[0]
            # Box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            box_coords = (int(x1), int(y1), int(x2), int(y2))
            
            # Add detection data to the list
            detection_data.append({
                'Class': cls,
                'Confidence': float(confidence),
                'Box': box_coords
            })
            
            # Draw bounding box on the image
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_rgb, f"{cls} {confidence:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert list to DataFrame
    detections_df = pd.DataFrame(detection_data)
    
    # Display the image with detections
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('People Detection')
    plt.show()
    
    return detections_df


if __name__ == "__main__":
    image_path = 'image.png'  # Replace with your image path
    detections = detect_people(image_path)
    print(detections)