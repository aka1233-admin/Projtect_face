# === Suppress warnings and logs from TensorFlow and MediaPipe ===
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# === Imports ===
import cv2
import face_recognition
import numpy as np
import mediapipe as mp
import time
import speech_recognition as sr
import threading
import webbrowser

cv2.setNumThreads(1)

# === Speech Recognizer Configuration ===
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000
recognizer.dynamic_energy_threshold = True

# === Paths and Known Faces ===
base_dir = os.path.dirname(__file__)
known_faces_dir = os.path.join(base_dir, "Faces")
os.makedirs(known_faces_dir, exist_ok=True)

known_face_encodings = []
known_face_names = []

print("[INFO] Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(path)
        name = os.path.splitext(filename)[0].split('_')[0]

        locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)
        if not locations:
            locations = face_recognition.face_locations(image, model="cnn")

        if locations:
            encodings = face_recognition.face_encodings(image, locations)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[SUCCESS] Loaded {filename} as '{name}'")
            else:
                print(f"[WARN] Encoding failed for {filename}")
        else:
            print(f"[WARN] No face detected in {filename}")

# === MediaPipe Face Mesh for Gaze Detection ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Key landmarks for better gaze detection
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
NOSE_TIP = 1
CHIN = 175


def is_person_looking_at_camera(frame, landmarks):
    """
    Improved gaze detection that works when approaching from left/right
    """
    if landmarks is None:
        return False

    ih, iw, _ = frame.shape

    try:
        # Get eye landmarks
        left_eye_inner = landmarks[LEFT_EYE_INNER]
        left_eye_outer = landmarks[LEFT_EYE_OUTER]
        right_eye_inner = landmarks[RIGHT_EYE_INNER]
        right_eye_outer = landmarks[RIGHT_EYE_OUTER]

        # Get iris centers
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]

        # Convert to pixel coordinates
        left_eye_inner_px = (int(left_eye_inner.x * iw), int(left_eye_inner.y * ih))
        left_eye_outer_px = (int(left_eye_outer.x * iw), int(left_eye_outer.y * ih))
        right_eye_inner_px = (int(right_eye_inner.x * iw), int(right_eye_inner.y * ih))
        right_eye_outer_px = (int(right_eye_outer.x * iw), int(right_eye_outer.y * ih))

        left_iris_px = (int(left_iris.x * iw), int(left_iris.y * ih))
        right_iris_px = (int(right_iris.x * iw), int(right_iris.y * ih))

        # Calculate eye centers
        left_eye_center = (
            (left_eye_inner_px[0] + left_eye_outer_px[0]) // 2,
            (left_eye_inner_px[1] + left_eye_outer_px[1]) // 2
        )
        right_eye_center = (
            (right_eye_inner_px[0] + right_eye_outer_px[0]) // 2,
            (right_eye_inner_px[1] + right_eye_outer_px[1]) // 2
        )

        # Calculate eye widths for relative positioning
        left_eye_width = abs(left_eye_outer_px[0] - left_eye_inner_px[0])
        right_eye_width = abs(right_eye_outer_px[0] - right_eye_inner_px[0])

        # Check if iris is relatively centered within each eye
        # This works regardless of face angle
        left_iris_ratio = abs(left_iris_px[0] - left_eye_center[0]) / max(left_eye_width / 2, 1)
        right_iris_ratio = abs(right_iris_px[0] - right_eye_center[0]) / max(right_eye_width / 2, 1)

        # More lenient thresholds for gaze detection
        gaze_threshold = 0.3  # Allow more deviation (was very strict before)

        # Check if both eyes are looking relatively forward
        if left_iris_ratio < gaze_threshold and right_iris_ratio < gaze_threshold:
            # Additional check: ensure eyes are roughly horizontal (person is upright)
            eye_line_angle = abs(left_eye_center[1] - right_eye_center[1]) / max(
                abs(left_eye_center[0] - right_eye_center[0]), 1)
            if eye_line_angle < 0.3:  # Eyes are roughly on same horizontal line
                return True

    except Exception as e:
        if debug_mode:
            print(f"[DEBUG] Gaze detection error: {e}")
        pass

    return False


def has_clear_face_view(landmarks):
    """
    Check if we have a clear view of the face (not too much profile)
    """
    if landmarks is None:
        return False

    try:
        # Check if we can see both eyes clearly
        left_eye_inner = landmarks[LEFT_EYE_INNER]
        right_eye_inner = landmarks[RIGHT_EYE_INNER]
        nose_tip = landmarks[NOSE_TIP]

        # Calculate face symmetry - if too asymmetric, it's too much profile
        left_dist = abs(nose_tip.x - left_eye_inner.x)
        right_dist = abs(nose_tip.x - right_eye_inner.x)

        symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)

        # Allow more profile views - reduced threshold
        return symmetry_ratio > 0.4  # Was likely higher before

    except Exception:
        return True  # If we can't calculate, assume it's fine


# === Camera Setup ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] Camera not accessible.")
    exit()

# === State Variables ===
debug_mode = False
recognition_done = False
recognition_history = []
last_voice_input = ""
last_detections = []
processed_faces = set()
gaze_detected = False
unknown_person_detected = False
no_face_counter = 0

# === 3-Attempt Verification Variables ===
unknown_attempt_count = 0
max_attempts = 3
attempt_start_time = None
verification_cooldown = 30  # seconds between verification cycles
last_verification_time = 0
verification_in_progress = False
verification_message = ""

authorized_name = "Akanksha"  # Define your voice identity


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
    global last_verification_time, recognition_done, last_detections

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
        # Check if enough time has passed for next attempt (2 seconds between attempts)
        if current_time - attempt_start_time >= 2.0:
            unknown_attempt_count += 1
            attempt_start_time = current_time

            if unknown_attempt_count <= max_attempts:
                verification_message = f"Verification failed. Attempt {unknown_attempt_count}/{max_attempts}"
                print(f"[VERIFICATION] Attempt {unknown_attempt_count}/{max_attempts}")

                # Reset recognition to try again
                recognition_done = False
                last_detections.clear()
                processed_faces.clear()

                return "retry_attempt"
            else:
                # All attempts exhausted
                verification_message = "Access denied. Maximum attempts reached."
                last_verification_time = current_time
                reset_verification_system()
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


# === Modified continuous voice listener ===
def continuous_voice_listener():
    global last_voice_input, gaze_detected, unknown_person_detected

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[VOICE] Microphone calibrated")

    while True:
        try:
            if (
                    gaze_detected
                    and last_detections
                    and not unknown_person_detected
                    and not verification_in_progress
                    and any(d['name'] == authorized_name for d in last_detections)
            ):
                with mic as source:
                    print("[VOICE] Listening...")
                    try:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        print("[VOICE] Processing...")
                        word = recognizer.recognize_google(audio, language="en-IN")
                        print(f"[WORD] {word}")
                        last_voice_input = f"You said: {word}"

                        if "google" in word.lower():
                            webbrowser.open("https://www.google.com")
                        elif "time" in word.lower():
                            current_time = time.strftime('%I:%M %p')
                            print(f"[TIME] {current_time}")
                            last_voice_input = f"Time is {current_time}"
                        else:
                            print("[VOICE] No recognized command")

                    except sr.UnknownValueError:
                        print("[ERROR] Could not understand audio")
                        last_voice_input = "Sorry, I didn't catch that."
                    except sr.RequestError as e:
                        print(f"[ERROR] API error: {e}")
                        last_voice_input = "Voice recognition failed."
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] Voice thread exception: {e}")
            time.sleep(1)


threading.Thread(target=continuous_voice_listener, daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame, (640, 360))
        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_small)

        gaze_detected = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Check both gaze and face view
                if (is_person_looking_at_camera(frame, face_landmarks.landmark) and
                        has_clear_face_view(face_landmarks.landmark)):
                    gaze_detected = True
                    if debug_mode:
                        print("[DEBUG] Gaze and clear face view detected")
                    break

        # === Face Recognition Logic (only once per gaze session) ===
        if not recognition_done and gaze_detected:
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small, number_of_times_to_upsample=1, model="hog")
            current = []
            if locations:
                encodings = face_recognition.face_encodings(small, locations)
                brightness = np.mean(frame)
                threshold = 0.5 + 0.2 * (128 - brightness) / 128

                for enc, loc in zip(encodings, locations):
                    top, right, bottom, left = [v * 2 for v in loc]
                    if right - left < 60 or bottom - top < 60:
                        continue
                    distances = face_recognition.face_distance(known_face_encodings, enc)
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]

                    name, color = "Unknown Face", (0, 0, 255)
                    confidence = 0
                    if best_dist < threshold:
                        name = known_face_names[best_idx]
                        color = (0, 255, 0)
                        confidence = 1 - best_dist

                    current.append({
                        "name": name, "location": (top, right, bottom, left),
                        "color": color, "distance": best_dist, "confidence": confidence
                    })
                    processed_faces.add(name)

                last_detections = current
                recognition_done = True
                no_face_counter = 0
                print(f"[INFO] Recognition completed. Detected: {[d['name'] for d in current]}")

                # Check if known person detected during verification
                check_for_known_person()

            else:
                no_face_counter += 1
        else:
            # === If no face landmarks at all ===
            if not results.multi_face_landmarks:
                no_face_counter += 1
            else:
                no_face_counter = 0

        # === Reset logic if no face detected for some frames ===
        if no_face_counter > 45:  # roughly 1.5 seconds if 30 FPS - increased time
            last_detections.clear()
            processed_faces.clear()
            recognition_done = False
            last_voice_input = ""
            no_face_counter = 0
            reset_verification_system()  # Reset verification when no face
            print("[INFO] No face detected for extended time - system reset")

        # === Handle Unknown Person Detection and Verification ===
        if last_detections:
            unknown_person_detected = any(d['name'] == "Unknown Face" for d in last_detections)

            if unknown_person_detected:
                verification_status = handle_unknown_person_verification()

                if verification_status == "access_denied":
                    # Optional: Add additional security measures here
                    # For example: save screenshot, send alert, etc.
                    pass

        else:
            unknown_person_detected = False

        # === Voice/Mic Display Feedback ===
        if unknown_person_detected or verification_in_progress:
            if verification_message:
                last_voice_input = verification_message
            cv2.putText(frame, "Microphone OFF - Verification Required", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif not gaze_detected:
            last_voice_input = "Please look at the camera to activate"
            cv2.putText(frame, "Microphone OFF - Look at camera", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Microphone ON - Listening", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === Draw detections and message ===
        message = "Looking for faces..."
        if last_detections:
            for d in last_detections:
                top, right, bottom, left = d["location"]
                cv2.rectangle(frame, (left, top), (right, bottom), d["color"], 2)
                label = f"{d['name']} {d['confidence']:.2f}" if debug_mode else d["name"]
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, d["color"], 2)

            if any(d['name'] == authorized_name for d in last_detections) and not verification_in_progress:
                message = f"{authorized_name} verified - Ready for commands"
                cv2.putText(frame, "Listening for voice commands", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                            2)
            elif verification_in_progress:
                message = f"Verification in progress - Attempt {unknown_attempt_count}/{max_attempts}"
            elif unknown_person_detected:
                current_time = time.time()
                if current_time - last_verification_time < verification_cooldown:
                    remaining_cooldown = verification_cooldown - (current_time - last_verification_time)
                    message = f"Access denied - Cooldown: {int(remaining_cooldown)}s"
                else:
                    message = "Unknown person detected - Starting verification"

        cv2.putText(frame, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if last_voice_input:
            cv2.putText(frame, last_voice_input, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        # Debug info
        if debug_mode:
            cv2.putText(frame, f"Gaze: {'YES' if gaze_detected else 'NO'}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Recognition: {'DONE' if recognition_done else 'PENDING'}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Verification: {'YES' if verification_in_progress else 'NO'}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Attempts: {unknown_attempt_count}/{max_attempts}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, "Press 'd' to toggle debug mode", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('d'):
            debug_mode = not debug_mode
            print(f"[INFO] Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key & 0xFF == ord('r'):  # Manual reset key
            reset_verification_system()
            last_detections.clear()
            processed_faces.clear()
            recognition_done = False
            print("[INFO] Manual system reset")

except KeyboardInterrupt:
    print("\n[INFO] Shutting down gracefully...")
except Exception as e:
    print(f"[ERROR] {str(e)}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released")