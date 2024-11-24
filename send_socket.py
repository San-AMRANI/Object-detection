import socket
import numpy as np
import cv2
from mss import mss
from pynput.mouse import Listener
import pickle
import time

# Define the client class
class Client:
    def __init__(self, host='localhost', port=65432):
        self.host = host
        self.port = port
        self.cont = 0
        self.pos = {"x": [], "y": []}

    def is_clicked(self, x, y, button, pressed):
        if pressed:
            print('Clicked!')
            self.pos["x"].append(x)
            self.pos["y"].append(y)
            self.cont += 1
            if self.cont == 2:
                return False  # stop the listener after two clicks
            
    def start(self):
        # Start the client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print("Connected to the server.")

        with Listener(on_click=self.is_clicked) as listener:
            listener.join()
            
        try:
            bounding_box = {
                'top': self.pos["y"][0],
                'left': self.pos["x"][0],
                'width': self.pos["x"][1] - self.pos["x"][0],
                'height': self.pos["y"][1] - self.pos["y"][0]
            }

            sct = mss()
            last_saved_time = time.time()
            save_interval = 5  # seconds

            while True:
                # Capture the screen
                sct_img = sct.grab(bounding_box)
                screen_np = np.array(sct_img)
                screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

                # Display the screen capture
                cv2.imshow('screen', screen_np)

                # Check if it's time to save an image
                current_time = time.time()
                if current_time - last_saved_time >= save_interval:
                    print(f"[INFO] Sending Data...")
                    result = pickle.dumps(screen_np)

                    # Send the size of the data first
                    size = len(result)
                    client_socket.sendall(size.to_bytes(4, byteorder='big'))  # Send size as 4 bytes

                    # Send the image data
                    client_socket.sendall(result)

                    # Wait for acknowledgment from the server
                    data = client_socket.recv(5)
                    people_counter = int(data.decode('utf8'))
                    last_saved_time = current_time
                    print("[INFO] Data Received...")
                    print(f"[INFO] People: {people_counter}")
                    print("*" * 80)

                # Exit when 'q' is pressed
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    cv2.destroyAllWindows()
                    break

        finally:
            client_socket.close()
            print("Connection closed.")

# Start the client
if __name__ == "__main__":
    client = Client()
    client.start()
