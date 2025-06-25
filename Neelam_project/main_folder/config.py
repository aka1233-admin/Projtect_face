"""
Configuration settings for Face Recognition System
"""
import os

# === Environment Configuration ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# === Camera Settings ===
CAMERA_INDEX = 0
FRAME_WIDTH = 800
FRAME_HEIGHT = 500
FRAME_FPS = 30

# === Recognition Settings ===
FACE_RECOGNITION_THRESHOLD = 0.5
MIN_FACE_SIZE = 60
UPSAMPLE_TIMES = 1
RECOGNITION_MODEL = "hog"

# === MediaPipe Settings ===
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# === Voice Recognition Settings ===
VOICE_ENERGY_THRESHOLD = 4000
VOICE_DYNAMIC_THRESHOLD = True
VOICE_TIMEOUT = 1
VOICE_PHRASE_TIME_LIMIT = 3
VOICE_LANGUAGE = "en-IN"

# === Verification System Settings ===
MAX_VERIFICATION_ATTEMPTS = 3
VERIFICATION_COOLDOWN = 30  # seconds
ATTEMPT_INTERVAL = 3.0  # seconds between attempts

# === System Settings ===
NO_FACE_RESET_THRESHOLD = 30  # frames
FACE_MOVEMENT_THRESHOLD = 50  # pixels
IMAGES_DIRECTORY = "images"

# === Display Settings ===
FONT = cv2.FONT_HERSHEY_SIMPLEX if 'cv2' in globals() else None
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# === Colors (BGR format) ===
COLOR_UNKNOWN = (0, 0, 255)  # Red
COLOR_KNOWN = (0, 255, 0)    # Green
COLOR_INFO = (255, 255, 0)   # Yellow
COLOR_TEXT = (0, 255, 255)   # Cyan
COLOR_WHITE = (255, 255, 255)