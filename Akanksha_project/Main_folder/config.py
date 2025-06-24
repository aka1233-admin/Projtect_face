# config.py
"""
Configuration settings for the face recognition system
"""

import os

# === Directory Configuration ===
BASE_DIR = os.path.dirname(__file__)
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "Faces")

# === Camera Configuration ===
CAMERA_CONFIG = {
    'width': 1280,
    'height': 720,
    'fps': 30,
    'backend': 'cv2.CAP_DSHOW'
}

# === Recognition Configuration ===
RECOGNITION_CONFIG = {
    'resize_factor': 0.5,
    'min_face_size': 60,
    'base_threshold': 0.5,
    'brightness_adjustment': 0.2,
    'upsample_times': 1,
    'model': 'hog'
}

# === Gaze Detection Configuration ===
GAZE_CONFIG = {
    'threshold': 0.3,
    'eye_line_threshold': 0.3,
    'symmetry_threshold': 0.4
}

# === Voice Recognition Configuration ===
VOICE_CONFIG = {
    'energy_threshold': 4000,
    'dynamic_energy_threshold': True,
    'timeout': 5,
    'phrase_time_limit': 5,
    'language': 'en-IN'
}

# === Verification System Configuration ===
VERIFICATION_CONFIG = {
    'max_attempts': 3,
    'cooldown_seconds': 30,
    'attempt_delay': 2.0,
    'authorized_name': 'Akanksha'
}

# === System Configuration ===
SYSTEM_CONFIG = {
    'no_face_reset_frames': 45,  # frames before reset (roughly 1.5 seconds at 30 FPS)
    'tf_log_level': '3',
    'cv2_threads': 1
}

# === MediaPipe Landmarks ===
LANDMARKS = {
    'LEFT_EYE_INNER': 133,
    'LEFT_EYE_OUTER': 33,
    'RIGHT_EYE_INNER': 362,
    'RIGHT_EYE_OUTER': 263,
    'LEFT_IRIS_CENTER': 468,
    'RIGHT_IRIS_CENTER': 473,
    'NOSE_TIP': 1,
    'CHIN': 175
}