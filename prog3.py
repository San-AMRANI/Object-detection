import cv2
import pandas as pd
from ultralytics import YOLO
import time
import os

# Initialize YOLO model (you can choose different model sizes like 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8s.pt')  # Ensure the model file is downloaded or specify the correct path

# Create a directory to store saved faces
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

# Dictionary to keep track of saved people and detection times
detected_people = {}  # Key: bounding box (x1, y1, x2, y2), Value: [timestamp of first detection, is_saved]

# Tolerance for recognizing "same" person
TOLERANCE = 90  # Pixels

def is_same_person(box1, box2):
    """Checks if two boxes are close enough to be considered the same person."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return abs(x1_1 - x1_2) < TOLERANCE and abs(y1_1 - y1_2) < TOLERANCE and abs(x2_1 - x2_2) < TOLERANCE and abs(y2_1 - y2_2) < TOLERANCE

def save_face(img, box, person_id):
    """Saves the detected face as an image."""
    x1, y1, x2, y2 = box
    face = img[y1:y2, x1:x2]
    face_path = f"saved_faces/person_{person_id}.png"
    cv2.imwrite(face_path, face)
    print(f"Saved face as {face_path}")

def detect_people_live():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

    person_id_counter = 1  # To assign a unique ID for each new person

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        # Convert BGR to RGB for the YOLO model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection using the YOLO model
        results = model(img_rgb)

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = model.names[int(box.cls[0])]
                if cls != 'person':
                    continue  # We only want to detect people

                confidence = box.conf[0]
                # Box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = (x1, y1, x2, y2)
                
                # Check if this person (bounding box) has been detected before
                same_person_detected = False
                for saved_box, (timestamp, is_saved) in detected_people.items():
                    if is_same_person(saved_box, current_box):
                        # Same person detected, check if already saved
                        same_person_detected = True
                        if is_saved:
                            # Draw green rectangle for saved person
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{cls} {confidence:.2f} (Saved)", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            # If 3 seconds passed since first detection, save the face
                            current_time = time.time()
                            if current_time - timestamp >= 3:  # 3-second delay
                                save_face(frame, current_box, person_id_counter)
                                person_id_counter += 1
                                detected_people[saved_box] = (timestamp, True)  # Mark as saved
                            # Draw red rectangle before saving
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{cls} {confidence:.2f} (New)", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        break
                
                if not same_person_detected:
                    # New person detected, add to tracking list
                    detected_people[current_box] = (time.time(), False)
                    # Draw red rectangle for new person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{cls} {confidence:.2f} (New)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame with detections
        cv2.imshow('Live People Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_people_live()
