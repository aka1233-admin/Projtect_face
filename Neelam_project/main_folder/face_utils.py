"""
Face recognition utility functions
"""
import cv2
import os
import numpy as np
import face_recognition
from config import *

def load_known_faces():
    """Load all known face encodings from images directory"""
    encodings, names = [], []
    print("Loading known faces...")
    
    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIRECTORY, exist_ok=True)
    
    for file in os.listdir(IMAGES_DIRECTORY):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMAGES_DIRECTORY, file)
            img = face_recognition.load_image_file(path)
            faces = face_recognition.face_encodings(img)
            
            if faces:
                encodings.append(faces[0])
                names.append(os.path.splitext(file)[0])
                print(f"  [OK] {file}")
            else:
                print(f"  [NO FACE FOUND] {file}")
    
    print(f"Total known faces: {len(encodings)}")
    return encodings, names

def calculate_face_center(face_box):
    """Calculate the center point of a face bounding box"""
    top, right, bottom, left = face_box
    return ((left + right) // 2, (top + bottom) // 2)

def has_face_moved_significantly(old_center, new_center, threshold=FACE_MOVEMENT_THRESHOLD):
    """Check if face has moved significantly based on center positions"""
    if old_center is None or new_center is None:
        return True
    distance = ((old_center[0] - new_center[0]) ** 2 + (old_center[1] - new_center[1]) ** 2) ** 0.5
    return distance > threshold

def recognize_faces(rgb_small, known_encodings, known_names, brightness):
    """Recognize faces in the given frame"""
    locations = face_recognition.face_locations(rgb_small, 
                                              number_of_times_to_upsample=UPSAMPLE_TIMES, 
                                              model=RECOGNITION_MODEL)
    
    if not locations:
        return []
    
    face_boxes = [[v * 2 for v in loc] for loc in locations]
    encodings = face_recognition.face_encodings(rgb_small, locations)
    
    # Adjust threshold based on brightness
    threshold = FACE_RECOGNITION_THRESHOLD + 0.2 * (128 - brightness) / 128
    
    results = []
    for idx, (enc, box) in enumerate(zip(encodings, face_boxes)):
        top, right, bottom, left = box
        
        # Skip very small faces
        if right - left < MIN_FACE_SIZE or bottom - top < MIN_FACE_SIZE:
            continue
        
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = np.argmin(distances) if len(distances) > 0 else -1
        best_dist = distances[best_idx] if len(distances) > 0 else 1.0
        
        name, color = "Unknown Face", COLOR_UNKNOWN
        confidence = 0
        
        if best_dist < threshold:
            name = known_names[best_idx]
            color = COLOR_KNOWN
            confidence = 1 - best_dist
        
        results.append({
            "name": name,
            "location": box,
            "color": color,
            "distance": best_dist,
            "confidence": confidence,
            "gaze": False
        })
    
    return results

def save_unknown_face(frame, face_location, name):
    """Save an unknown face to the images directory"""
    top, right, bottom, left = face_location
    unknown_face_img = frame[top:bottom, left:right]
    
    if unknown_face_img.size == 0 or unknown_face_img.shape[0] < 40 or unknown_face_img.shape[1] < 40:
        print("[ERROR] Cropped face is too small or empty! Not saving.")
        return False
    
    # Show cropped face for confirmation
    cv2.imshow("Cropped Face", unknown_face_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Cropped Face")
    
    if name.strip():
        save_path = os.path.join(IMAGES_DIRECTORY, f"{name}.jpg")
        cv2.imwrite(save_path, unknown_face_img)
        print(f"[SAVED] {save_path}")
        return True
    else:
        print("[INFO] Name cannot be empty.")
        return False