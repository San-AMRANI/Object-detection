import socket
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import time

def start_server(host='127.0.0.1', port=65432):
    """Starts a TCP server to receive images and log detections."""
    # Initialize detection statistics
    detections_per_minute = {}
    start_time = time.time()

    # Prepare the CSV file to log detections
    csv_file = "detected_people.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Face_ID", "Timestamp"])  # Write CSV headers

    # Prepare for live plotting with a stepped line style
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of People Detected")
    ax.set_title("People Detected Per Minute")
    x_data, y_data = [], []

    def update_chart():
        """Updates the chart with current detection statistics."""
        current_time = time.time()
        elapsed_minutes = int((current_time - start_time) / 60)

        # Only update at the end of each minute to create the stepped effect
        if x_data and x_data[-1] == elapsed_minutes - 1:
            x_data.append(elapsed_minutes)
            y_data.append(detections_per_minute.get(elapsed_minutes, y_data[-1]))
        else:
            x_data.append(elapsed_minutes)
            y_data.append(detections_per_minute.get(elapsed_minutes, 0))

        ax.clear()
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Number of People Detected")
        ax.set_title("People Detected Per Minute")
        ax.step(x_data, y_data, where='post', color='blue', linewidth=2)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.draw()
        plt.pause(0.1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # Receive the size of the incoming image first
                img_size_data = conn.recv(8)  # Expecting 8 bytes for size
                if not img_size_data:
                    print("Connection closed.")
                    break

                # Unpack the size of the image
                img_size = int.from_bytes(img_size_data, byteorder='big')

                # Validate image size
                if img_size <= 0 or img_size > 10 * 1024 * 1024:  # Limit to 10MB for safety
                    print(f"Received invalid image size: {img_size}. Closing connection.")
                    break

                print(f"Expecting an image of size: {img_size} bytes")

                # Receive the actual image data
                img_data = bytearray()
                while len(img_data) < img_size:
                    packet = conn.recv(4096)
                    if not packet:
                        print("Connection closed unexpectedly.")
                        break
                    img_data.extend(packet)

                # Convert the byte array back to an image
                img_np = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                # Ensure the image is valid before saving
                if img is not None and img.size > 0:
                    # Save the received image
                    face_id = len(os.listdir("received_faces")) + 1  # Unique ID for each received face
                    face_path = f"received_faces/person_{face_id}.png"
                    cv2.imwrite(face_path, img)
                    print(f"Received and saved face as {face_path}")

                    # Log the detection with timestamp in CSV file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([face_id, timestamp])

                    # Update detection statistics
                    elapsed_minutes = int((time.time() - start_time) / 60)
                    if elapsed_minutes not in detections_per_minute:
                        detections_per_minute[elapsed_minutes] = 0
                    detections_per_minute[elapsed_minutes] += 1

                    # Update the live chart at the end of each minute
                    if elapsed_minutes % 1 == 0:
                        update_chart()
                else:
                    print("Failed to decode image. Image may be empty.")

    # Final save of the chart
    plt.savefig("detection_statistics1.png")
    print("Chart saved as detection_statistics.png")

if __name__ == "__main__":
    os.makedirs("received_faces", exist_ok=True)  # Create the directory if it doesn't exist
    start_server()
