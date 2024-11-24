import cv2
import pandas as pd
from ultralytics import YOLO

# Initialize YOLO model (you can choose different model sizes like 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8s.pt')  # Ensure the model file is downloaded or specify the correct path

def detect_people_live():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

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
                x1, y1, x2, y2 = box.xyxy[0]
                box_coords = (int(x1), int(y1), int(x2), int(y2))

                # Draw bounding box on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{cls} {confidence:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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
