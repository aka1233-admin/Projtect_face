# face_recognition_module.py
"""
Face recognition and loading functionality
"""

import os
import cv2
import face_recognition
import numpy as np
from config import KNOWN_FACES_DIR, RECOGNITION_CONFIG


class FaceRecognitionManager:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the faces directory"""
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        print("[INFO] Loading known faces...")
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(path)
                name = os.path.splitext(filename)[0].split('_')[0]

                locations = face_recognition.face_locations(
                    image,
                    number_of_times_to_upsample=2
                )
                if not locations:
                    locations = face_recognition.face_locations(image, model="cnn")

                if locations:
                    encodings = face_recognition.face_encodings(image, locations)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"[SUCCESS] Loaded {filename} as '{name}'")
                    else:
                        print(f"[WARN] Encoding failed for {filename}")
                else:
                    print(f"[WARN] No face detected in {filename}")

    def recognize_faces(self, frame):
        """
        Recognize faces in the given frame
        Returns list of detection dictionaries
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0),
                           fx=RECOGNITION_CONFIG['resize_factor'],
                           fy=RECOGNITION_CONFIG['resize_factor'])

        locations = face_recognition.face_locations(
            small,
            number_of_times_to_upsample=RECOGNITION_CONFIG['upsample_times'],
            model=RECOGNITION_CONFIG['model']
        )

        detections = []
        if locations:
            encodings = face_recognition.face_encodings(small, locations)
            brightness = np.mean(frame)
            threshold = (RECOGNITION_CONFIG['base_threshold'] +
                         RECOGNITION_CONFIG['brightness_adjustment'] *
                         (128 - brightness) / 128)

            for enc, loc in zip(encodings, locations):
                # Scale back up locations
                scale = 1 / RECOGNITION_CONFIG['resize_factor']
                top, right, bottom, left = [int(v * scale) for v in loc]

                # Filter out small faces
                if (right - left < RECOGNITION_CONFIG['min_face_size'] or
                        bottom - top < RECOGNITION_CONFIG['min_face_size']):
                    continue

                # Find best match
                distances = face_recognition.face_distance(self.known_face_encodings, enc)

                if len(distances) > 0:
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]

                    name, color = "Unknown Face", (0, 0, 255)
                    confidence = 0

                    if best_dist < threshold:
                        name = self.known_face_names[best_idx]
                        color = (0, 255, 0)
                        confidence = 1 - best_dist

                    detections.append({
                        "name": name,
                        "location": (top, right, bottom, left),
                        "color": color,
                        "distance": best_dist,
                        "confidence": confidence
                    })

        return detections