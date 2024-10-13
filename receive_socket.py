import socket
import cv2
import numpy as np
import os

def start_server(host='192.168.11.142', port=65432):
    """Starts a TCP server to receive images."""
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
                else:
                    print("Failed to decode image. Image may be empty.")

if __name__ == "__main__":
    os.makedirs("received_faces", exist_ok=True)  # Create the directory if it doesn't exist
    start_server()
