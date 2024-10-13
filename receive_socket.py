import socket
import os
import cv2
import numpy as np

# Create a directory to store received faces
if not os.path.exists("received_faces"):
    os.makedirs("received_faces")

def start_server(host='127.0.0.1', port=65432):
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
                img_size_data = conn.recv(8)  # Assuming image size will be sent as a fixed-size header
                if not img_size_data:
                    break
                
                # Unpack the size of the image
                img_size = int.from_bytes(img_size_data, byteorder='big')
                print(f"Expecting an image of size: {img_size} bytes")

                # Receive the actual image data
                img_data = bytearray()
                while len(img_data) < img_size:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    img_data.extend(packet)

                # Convert the byte array back to an image
                img_np = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                # Save the received image
                face_id = len(os.listdir("received_faces")) + 1  # Unique ID for each received face
                face_path = f"received_faces/person_{face_id}.png"
                cv2.imwrite(face_path, img)
                print(f"Received and saved face as {face_path}")

if __name__ == "__main__":
    start_server()
