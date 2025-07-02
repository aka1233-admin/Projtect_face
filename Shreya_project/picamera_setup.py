# camera_setup.py

import cv2
from picamera2 import Picamera2
import time

# Initialize Picamera2
picam2 = Picamera2()

# Set resolution and format
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

# Start camera
picam2.start()
time.sleep(1)  # Warm-up time

print("📸 Press 'q' to quit the camera preview window")

while True:
    frame = picam2.capture_array()

    # Optional: flip if needed (horizontal/vertical)
    # frame = cv2.flip(frame, 1)  # Horizontal flip (mirror)

    cv2.imshow("Smart Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
cv2.destroyAllWindows()
