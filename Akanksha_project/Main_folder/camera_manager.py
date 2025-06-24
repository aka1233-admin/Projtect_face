# camera_manager.py
"""
Camera initialization and management
"""

import cv2
from config import CAMERA_CONFIG


class CameraManager:
    def __init__(self):
        self.cap = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize camera with configuration settings"""
        # Set up camera with DirectShow backend (Windows)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])

        if not self.cap.isOpened():
            raise RuntimeError("[ERROR] Camera not accessible.")

        print(
            f"[INFO] Camera initialized: {CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']} @ {CAMERA_CONFIG['fps']}fps")

    def read_frame(self):
        """
        Read and process frame from camera
        Returns: (success, flipped_frame)
        """
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
        return ret, frame

    def is_opened(self):
        """Check if camera is successfully opened"""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Camera released")

    def get_frame_dimensions(self):
        """Get current frame dimensions"""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return None, None