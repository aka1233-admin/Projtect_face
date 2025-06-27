
import cv2
import os
import numpy as np
import face_recognition
import threading
import time
import speech_recognition as sr
import webbrowser
import mediapipe as mp


cv2.setNumThreads(1)  # Prevent OpenCV multithread lag

# === Suppress warnings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Speech Recognizer Configuration ===
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000
recognizer.dynamic_energy_threshold = True

# === MediaPipe Face Mesh for Gaze Detection ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

def is_gazing_directly(frame, landmarks):
    if landmarks is None:
        return False
    ih, iw, _ = frame.shape
    try:
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        left_x = int(left_iris.x * iw)
        right_x = int(right_iris.x * iw)
        center_x = iw // 2
        if abs(left_x - center_x) < iw * 0.1 and abs(right_x - center_x) < iw * 0.1:
            nose_x = int(landmarks[4].x * iw)
            if abs(nose_x - center_x) < iw * 0.15:
                return True
    except:
        pass
    return False

def is_gazing_directly_face_relative(frame, landmarks, face_box):
    top, right, bottom, left = face_box
    face_center_x = (left + right) // 2
    face_center_y = (top + bottom) // 2
    ih, iw, _ = frame.shape
    try:
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        left_x = int(left_iris.x * iw)
        right_x = int(right_iris.x * iw)
        # Compare with face center, not frame center
        face_width = right - left
        if abs(left_x - face_center_x) < face_width * 0.1 and abs(right_x - face_center_x) < face_width * 0.1:
            nose_x = int(landmarks[4].x * iw)
            if abs(nose_x - face_center_x) < face_width * 0.15:
                return True
    except:
        pass
    return False

def calculate_face_center(face_box):
    """Calculate the center point of a face bounding box"""
    top, right, bottom, left = face_box
    return ((left + right) // 2, (top + bottom) // 2)

def has_face_moved_significantly(old_center, new_center, threshold=50):
    """Check if face has moved significantly based on center positions"""
    if old_center is None or new_center is None:
        return True
    distance = ((old_center[0] - new_center[0]) ** 2 + (old_center[1] - new_center[1]) ** 2) ** 0.5
    return distance > threshold

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

# === Load all known face encodings from 'images' folder ===
def load_known_faces():
    encodings, names = [], []
    print("Loading known faces...")
    for file in os.listdir("images"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("images", file)
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

# === Smooth Webcam Class using background thread ===
class VideoCaptureThreaded:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)  
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# === Start ===
known_encodings, known_names = load_known_faces()
cap = VideoCaptureThreaded()
print("Camera started. Press 'q' to quit, 'n' to save unknown face.")

# === State Variables ===
debug_mode = False
recognition_done = False
last_voice_input = ""
last_detections = []
processed_faces = set()
gaze_detected = False
unknown_person_detected = False
no_face_counter = 0
unknown_face_img = None

# Add a new flag to track if currently listening
is_listening = False

# Add a flag to lock recognition after first detection
recognition_locked = False

# === 3-Attempt Verification Variables ===
unknown_attempt_count = 0
max_attempts = 3
attempt_start_time = None
verification_cooldown = 30  # seconds between verification cycles
last_verification_time = 0
verification_in_progress = False
verification_message = ""

def reset_verification_system():
    """Reset the 3-attempt verification system"""
    global unknown_attempt_count, attempt_start_time, verification_in_progress, verification_message
    unknown_attempt_count = 0
    attempt_start_time = None
    verification_in_progress = False
    verification_message = ""
    print("[VERIFICATION] System reset")

def handle_unknown_person_verification():
    """Handle the 3-attempt verification process for unknown persons"""
    global unknown_attempt_count, attempt_start_time, verification_in_progress, verification_message
    global last_verification_time, recognition_done, last_detections, recognition_locked

    current_time = time.time()

    # Check if we're in cooldown period
    if current_time - last_verification_time < verification_cooldown:
        remaining_cooldown = verification_cooldown - (current_time - last_verification_time)
        verification_message = f"Access denied. Try again in {int(remaining_cooldown)}s"
        return "cooldown"

    # Start new verification cycle if not already in progress
    if not verification_in_progress:
        verification_in_progress = True
        unknown_attempt_count = 1
        attempt_start_time = current_time
        verification_message = f"Unknown person detected. Attempt {unknown_attempt_count}/{max_attempts}"
        print(f"[VERIFICATION] Starting attempt {unknown_attempt_count}/{max_attempts}")
        return "first_attempt"

    # Continue existing verification cycle
    else:
        # Check if enough time has passed for next attempt (3 seconds between attempts)
        if current_time - attempt_start_time >= 3.0:
            unknown_attempt_count += 1
            attempt_start_time = current_time

            if unknown_attempt_count <= max_attempts:
                verification_message = f"Verification failed. Attempt {unknown_attempt_count}/{max_attempts}"
                print(f"[VERIFICATION] Attempt {unknown_attempt_count}/{max_attempts}")

                # Reset recognition to try again
                recognition_done = False
                recognition_locked = False
                last_detections.clear()
                processed_faces.clear()

                return "retry_attempt"
            else:
                # All attempts exhausted
                verification_message = "Access denied. Maximum attempts reached."
                last_verification_time = current_time
                reset_verification_system()
                recognition_locked = True  # Lock recognition after max attempts
                print("[VERIFICATION] All attempts exhausted. Access denied.")
                return "access_denied"
        else:
            # Still within the same attempt window
            return "waiting"

    return "unknown"

def check_for_known_person():
    """Check if a known person is now detected and reset verification if so"""
    global verification_in_progress

    if verification_in_progress and last_detections:
        for detection in last_detections:
            if detection['name'] != "Unknown Face":
                print(f"[VERIFICATION] Known person '{detection['name']}' detected. Resetting verification.")
                reset_verification_system()
                return True
    return False

def is_known_face_present():
    for d in last_detections:
        if d["name"] != "Unknown Face":
            return True
    return False

def continuous_voice_listener():
    global last_voice_input, gaze_detected, unknown_person_detected, last_detections, is_listening, verification_in_progress
    while True:
        try:
            if is_known_face_present() and not unknown_person_detected and not verification_in_progress:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    while is_known_face_present() and not unknown_person_detected and not verification_in_progress:
                        try:
                            # Start listening
                            is_listening = True
                            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                            word = recognizer.recognize_google(audio, language="en-IN")
                            print(f"[WORD] {word}")
                            last_voice_input = f"You said: {word}"
                            if "google" in word.lower():
                                webbrowser.open("https://www.google.com")
                            elif "time" in word.lower():
                                print(f"[TIME] {time.strftime('%I:%M %p')}")
                        except sr.WaitTimeoutError:
                            # No speech detected, stop listening message
                            is_listening = False
                            continue
                        except:
                            is_listening = False
                            continue
                is_listening = False
            else:
                is_listening = False
                time.sleep(0.5)
        except:
            is_listening = False
            time.sleep(0.5)

threading.Thread(target=continuous_voice_listener, daemon=True).start()

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    # === Gaze Detection ===
    frame_small = cv2.resize(frame, (640, 360))
    rgb_small_gaze = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_small_gaze)

    gaze_detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            gaze_detected = is_gazing_directly(frame, face_landmarks.landmark)
            if gaze_detected:
                break

    # === Face Recognition Logic (only if not locked) ===
    if not recognition_locked:
        if not recognition_done:
            locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=1, model="hog")
            face_boxes = [[v * 2 for v in loc] for loc in locations]
            current = []
            if locations:
                encodings = face_recognition.face_encodings(rgb_small, locations)
                brightness = np.mean(frame)
                threshold = 0.5 + 0.2 * (128 - brightness) / 128
                for idx, (enc, box) in enumerate(zip(encodings, face_boxes)):
                    top, right, bottom, left = box
                    if right - left < 60 or bottom - top < 60:
                        continue
                    distances = face_recognition.face_distance(known_encodings, enc)
                    best_idx = np.argmin(distances) if len(distances) > 0 else -1
                    best_dist = distances[best_idx] if len(distances) > 0 else 1.0
                    name, color = "Unknown Face", (0, 0, 255)
                    confidence = 0
                    if best_dist < threshold:
                        name = known_names[best_idx]
                        color = (0, 255, 0)
                        confidence = 1 - best_dist
                    current.append({
                        "name": name,
                        "location": box,
                        "color": color,
                        "distance": best_dist,
                        "confidence": confidence,
                        "gaze": False
                    })
                last_detections = current
                no_face_counter = 0
                recognition_done = True

                # Check if known person detected during verification
                check_for_known_person()

                # For known face, lock immediately
                if any(d["name"] != "Unknown Face" for d in last_detections):
                    recognition_locked = True
                    reset_verification_system()  # Reset verification when known person is found

            else:
                no_face_counter += 1
        else:
            if not results.multi_face_landmarks:
                no_face_counter += 1
            else:
                no_face_counter = 0

    # === Reset recognition if no face detected for a while ===
    if no_face_counter > 30:
        last_detections.clear()
        processed_faces.clear()
        recognition_done = False
        last_voice_input = ""
        no_face_counter = 0
        recognition_locked = False
        reset_verification_system()  # Reset verification when no face
        print("[INFO] No face detected - system reset")

    # === Handle Unknown Person Detection and Verification ===
    if last_detections:
        unknown_person_detected = any(d['name'] == "Unknown Face" for d in last_detections)

        if unknown_person_detected:
            verification_status = handle_unknown_person_verification()
        else:
            # A known face detected — reset verification if needed
            check_for_known_person()
    else:
        # No detections — face has moved away
        if verification_in_progress:
            print("[INFO] Face disappeared during verification — resetting system")
            reset_verification_system()
        unknown_person_detected = False

    # === Voice/Mic Display Feedback ===
    if unknown_person_detected or verification_in_progress:
        if verification_message:
            last_voice_input = verification_message
        cv2.putText(frame, "Microphone Disabled - Unknown Person", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif not last_detections:
        cv2.putText(frame, "Microphone Disabled - Looking for face", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif is_listening:
        cv2.putText(frame, "Listening to your command...", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # No message when not speaking
        pass

    # === Draw rectangles and labels ===
    face_found = False
    unknown_detected = False
    known_detected = False

    for d in last_detections:
        top, right, bottom, left = d["location"]
        name = d["name"]
        color = d["color"]
        confidence = d.get("confidence", 0.0)
        if name == "Unknown Face":
            unknown_detected = True
            label = "Unknown Person"
        else:
            known_detected = True
            label = f"{name} Verified" if not debug_mode else f"{name} ({confidence:.2f})"
        face_found = True
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Store unknown face for saving
        if name == "Unknown Face":
            unknown_face_img = frame[top:bottom, left:right]

    # === Display status message at top ===
    if not face_found:
        cv2.putText(frame, "Looking for someone...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    elif unknown_detected:
        if verification_in_progress:
            message = f"Verification in progress - Attempt {unknown_attempt_count}/{max_attempts}"
        elif verification_message and "Access denied" in verification_message:
            message = "Access Denied - Max attempts reached"
        else:
            message = "Unknown Person Detected!"
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif known_detected:
        cv2.putText(frame, "Face Verified - Voice Commands Ready", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Voice input display ===
    if last_voice_input:
        cv2.putText(frame, last_voice_input, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    # === Debug mode toggle ===
    if debug_mode:
        cv2.putText(frame, f"Verification: {'YES' if verification_in_progress else 'NO'}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Attempts: {unknown_attempt_count}/{max_attempts}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, "Press 'd' to toggle debug mode, 'r' to reset", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # === Show frame ===
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit on 'q'
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"[DEBUG] Debug mode: {'ON' if debug_mode else 'OFF'}")
    elif key == ord('r'):  # Manual reset key
        reset_verification_system()
        last_detections.clear()
        processed_faces.clear()
        recognition_done = False
        recognition_locked = False
        print("[INFO] Manual system reset")

    # Save unknown face on 'n'
    if key == ord('n'):
        for d in last_detections:
            if d["name"] == "Unknown Face":
                top, right, bottom, left = d["location"]
                unknown_face_img = frame[top:bottom, left:right]
                if unknown_face_img.size == 0 or unknown_face_img.shape[0] < 40 or unknown_face_img.shape[1] < 40:
                    print("[ERROR] Cropped face is too small or empty! Not saving.")
                    continue
                cv2.imshow("Cropped Face", unknown_face_img)
                cv2.waitKey(0)
                cv2.destroyWindow("Cropped Face")
                voice_name = input("Enter name for New face: ").strip()
                if voice_name:
                    save_path = f"images/{voice_name}.jpg"
                    cv2.imwrite(save_path, unknown_face_img)
                    print(f"[SAVED] {save_path}")
                    print("[INFO] Reloading known faces...")
                    known_encodings, known_names = load_known_faces()
                    # Reset system after adding new face
                    reset_verification_system()
                    last_detections.clear()
                    processed_faces.clear()
                    recognition_done = False
                    recognition_locked = False
                else:
                    print("[INFO] Name cannot be empty.")

cap.release()
cv2.destroyAllWindows()