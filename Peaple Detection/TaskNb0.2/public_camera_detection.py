import numpy as np
import cv2
from time import sleep

# Define the client class
class Client:
    def __init__(self, camera_url="https://csea-me-webcam.cse.umn.edu/mjpg/video.mjpg"):
        self.camera_url = camera_url
        self.zoom_level = 1.0  # Default zoom level (no zoom)
        self.paused = False  # Flag for pause functionality
        # Initialize HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def start(self):
        # Open the video capture from the public camera
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print(f"Error opening video stream from {self.camera_url}")
            return

        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                    # Resize the frame for faster processing (optional)
                    frame = cv2.resize(frame, (640, 480))

                    # Detect people in the frame
                    boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
                    people_count = len(boxes)
                    print(f"[INFO] People Count: {people_count}")
                    
                    # Draw bounding boxes around detected people
                    for (x, y, w, h) in boxes:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the frame with detections
                    cv2.imshow("Camera Feed", frame)

                    # Exit on pressing 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera feed closed.")

# Start the client
if __name__ == "__main__":
    client = Client()
    client.start()
