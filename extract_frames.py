import numpy as np
import cv2
from mss import mss
from pynput.mouse import Listener
import os

cont = 0
pos = {"x": [], "y": []}

def is_clicked(x, y, button, pressed):
    global cont, pos
    if pressed:
        print('Clicked!')
        pos["x"].append(x)
        pos["y"].append(y)
        cont += 1
        if cont == 2:
            return False  # Stop the listener after 2 clicks

# Start the listener for mouse clicks
with Listener(on_click=is_clicked) as listener:
    listener.join()

# Check if we have enough clicks to create a bounding box
if len(pos["x"]) < 2 or len(pos["y"]) < 2:
    print("Not enough clicks to define a bounding box.")
    exit()

# Define the bounding box based on the mouse clicks
bounding_box = {
    'top': min(pos["y"]),
    'left': min(pos["x"]),
    'width': abs(pos["x"][1] - pos["x"][0]),
    'height': abs(pos["y"][1] - pos["y"][0])
}

# Validate bounding box dimensions
if bounding_box['width'] <= 0 or bounding_box['height'] <= 0:
    print("Invalid bounding box dimensions.")
    exit()

# Get screen dimensions
sct = mss()
screen = sct.monitors[1]  # Assuming the first monitor is at index 1
screen_width = screen['width']
screen_height = screen['height']

# Check if bounding box exceeds screen dimensions
if (bounding_box['left'] + bounding_box['width'] > screen_width) or (bounding_box['top'] + bounding_box['height'] > screen_height):
    print("Bounding box exceeds screen dimensions.")
    exit()

try:
    # Check if DISPLAY is set
    if 'DISPLAY' not in os.environ:
        print("DISPLAY environment variable is not set.")
        exit()

    # Draw the bounding box on a blank screen for feedback
    img_feedback = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.rectangle(img_feedback, (bounding_box['left'], bounding_box['top']),
                  (bounding_box['left'] + bounding_box['width'], bounding_box['top'] + bounding_box['height']),
                  (0, 255, 0), 2)
    cv2.imshow('Bounding Box Feedback', img_feedback)
    cv2.waitKey(2000)  # Show feedback for 2 seconds
    cv2.destroyWindow('Bounding Box Feedback')

    while True:
        # Capture the screen
        sct_img = sct.grab(bounding_box)
        screen_np = np.array(sct_img)

        # Convert BGRA to BGR
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2BGR)

        # Display the screen capture
        cv2.imshow('Screen Capture', screen_np)

        # Exit when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    print("Exited gracefully.")
