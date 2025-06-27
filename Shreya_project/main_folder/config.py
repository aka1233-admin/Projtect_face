# Configuration file for Smart Camera System

# Face Recognition Settings
FACE_MATCH_THRESHOLD = 0.6  # Lower values = stricter matching
MANAGER_IMAGE_PATH = "Shreya.jpg"

# System Timing Settings
NO_PERSON_TIMEOUT = 10  # seconds before reset if no person detected
RECOGNITION_RESET_DELAY = 3  # seconds before allowing retry after failed recognition

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
DISPLAY_WIDTH = 800  # Width for display frame (processing optimization)

# Processing Frame Size (for faster processing)
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 360
RECOGNITION_WIDTH = 320  # Even smaller for face recognition
RECOGNITION_HEIGHT = 180

# Gaze Detection Settings
GAZE_CENTER_MIN = 0.4
GAZE_CENTER_MAX = 0.6

# Voice Recognition Settings
AMBIENT_NOISE_DURATION = 0.3
VOICE_TIMEOUT = 1  # seconds to wait for voice input
PHRASE_TIME_LIMIT = 5  # maximum seconds for a single phrase
WHATSAPP_WAIT_TIME = 15  # seconds to wait before closing WhatsApp tab

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_NUM_FACES = 1
REFINE_LANDMARKS = True

# File Paths
CONTACTS_FILE = "contacts.json"

# Display Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GREEN = (0, 200, 0)
COLOR_MAGENTA = (255, 0, 255)

# Window Settings
WINDOW_NAME = "Smart Camera"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

# Counter Thresholds
MAX_NO_FACE_COUNTER = 150  # frames without face before reset