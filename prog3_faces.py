import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time

# Create directory to save detected faces
if not os.path.exists('saved_faces'):
    os.makedirs('saved_faces')

# Initialize face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to store previously detected faces' locations
detected_faces = {}

# Tolerance for recognizing "same" face
TOLERANCE = 90  # Pixel tolerance for face comparison

def is_same_face(new_face, existing_face, tolerance=TOLERANCE):
    """Check if the new detected face is similar to the existing one within tolerance."""
    x1, y1, w1, h1 = new_face
    x2, y2, w2, h2 = existing_face
    return (abs(x1 - x2) < tolerance and abs(y1 - y2) < tolerance and 
            abs(w1 - w2) < tolerance and abs(h1 - h2) < tolerance)

def save_face(face_img, face_id):
    """Save face image to the 'saved_faces' folder."""
    file_name = f"saved_faces/face_{face_id}.png"
    cv2.imwrite(file_name, face_img)
    print(f"Saved face: {file_name}")

def detect_and_track_faces():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    face_id = 0  # ID for saving new faces
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract the face region
            face_img = frame[y:y+h, x:x+w]
            
            # Check if the detected face is a "new" face or already detected
            same_face_found = False
            for saved_face_id, face_coords in detected_faces.items():
                if is_same_face((x, y, w, h), face_coords):
                    same_face_found = True
                    # Draw green rectangle for already detected face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    break

            if not same_face_found:
                # Draw red rectangle for a newly detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                detected_faces[face_id] = (x, y, w, h)
                
                # Save face image after 3 seconds (simulating delay)
                time.sleep(3)
                save_face(face_img, face_id)
                
                # Increment face ID
                face_id += 1

        # Display the video with rectangles
        cv2.imshow("Face Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_track_faces()
