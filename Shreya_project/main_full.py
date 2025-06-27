import threading
import cv2
import face_recognition
import imutils
import os
import pyttsx3
import speech_recognition as sr
import mediapipe as mp
import time
import pywhatkit
import json

# Initialize components
engine = pyttsx3.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Constants
CONTACTS_FILE = "contacts.json"
FACE_MATCH_THRESHOLD = 0.48  # Fixed threshold for proper face matching
NO_PERSON_TIMEOUT = 10  # seconds before reset if no person detected

# Global variables
manager_verified = False
system_active = True
manager_encoding = None
recognition_completed = False  # Flag to track if recognition was done
both_eyes_gaze_detected = False
last_person_detected_time = time.time()
no_face_counter = 0
previous_face_encoding = None
verification_attempts = 0
MAX_ATTEMPTS = 3
ATTEMPT_COOLDOWN = 3
last_attempt_time = 0

# Thread control
face_recognition_running = False
continuous_listening_active = False
voice_thread = None

# Speech function with error handling
def speak(text):
    try:
        print("SPEAKING:", text)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

# Gaze detection for both eyes separately
def get_both_eyes_gaze_direction(landmarks):
    try:
        if len(landmarks) <= 473:
            return {"left_eye": "unknown", "right_eye": "unknown", "both_center": False}
        
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        left_eye_left_corner = landmarks[33]
        left_eye_right_corner = landmarks[133]
        right_eye_left_corner = landmarks[362]
        right_eye_right_corner = landmarks[263]
        
        left_eye_width = abs(left_eye_right_corner.x - left_eye_left_corner.x)
        right_eye_width = abs(right_eye_right_corner.x - right_eye_left_corner.x)
        
        left_iris_relative_x = (left_iris.x - left_eye_left_corner.x) / left_eye_width if left_eye_width > 0 else 0.5
        right_iris_relative_x = (right_iris.x - right_eye_left_corner.x) / right_eye_width if right_eye_width > 0 else 0.5
        
        def get_eye_direction(relative_x):
            if 0.35 < relative_x < 0.65:
                return "center"
            elif relative_x <= 0.35:
                return "left"
            else:
                return "right"
        
        left_eye_direction = get_eye_direction(left_iris_relative_x)
        right_eye_direction = get_eye_direction(right_iris_relative_x)
        
        both_center = (left_eye_direction == "center" and right_eye_direction == "center")
        
        return {
            "left_eye": left_eye_direction,
            "right_eye": right_eye_direction,
            "both_center": both_center,
            "left_relative_x": left_iris_relative_x,
            "right_relative_x": right_iris_relative_x
        }
        
    except Exception as e:
        print(f"Gaze detection error: {e}")
        return {"left_eye": "unknown", "right_eye": "unknown", "both_center": False}

# One-time face recognition function
def run_face_recognition_once(frame_rgb):
    global face_recognition_running, manager_verified, recognition_completed
    global verification_attempts, last_attempt_time, previous_face_encoding

    face_recognition_running = True

    try:
        current_time = time.time()
        if current_time - last_attempt_time < ATTEMPT_COOLDOWN:
            print("Cooldown active. Waiting before next attempt.")
            return
        last_attempt_time = current_time

        print(f"Running face recognition... (Attempt {verification_attempts + 1})")

        frame_for_recognition = cv2.resize(frame_rgb, (320, 180))
        gray = cv2.cvtColor(frame_for_recognition, cv2.COLOR_RGB2GRAY)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

        if clarity < 40:
            speak("Face too blurry. Please come closer or adjust lighting.")
            print("Blurry face detected. Clarity:", clarity)
            return

        face_locations = face_recognition.face_locations(frame_for_recognition, model="hog")

        if not face_locations:
            speak("No face detected. Try again.")
            return

        scaled_locations = [(top * 2, right * 2, bottom * 2, left * 2)
                            for (top, right, bottom, left) in face_locations]
        face_encodings = face_recognition.face_encodings(frame_rgb, scaled_locations, num_jitters=1, model="small")

        for face_encoding in face_encodings:
            # Reset attempts if it's a different face than before
            if previous_face_encoding is not None:
                distance_from_previous = face_recognition.face_distance([previous_face_encoding], face_encoding)[0]
                if distance_from_previous > 0.4:  # consider it a new face
                    print("New face detected. Resetting attempts.")
                    verification_attempts = 0

            previous_face_encoding = face_encoding

            # Compare to manager's encoding
            face_distance = face_recognition.face_distance([manager_encoding], face_encoding)[0]
            print(f"Face distance from manager: {face_distance}")

            if face_distance < FACE_MATCH_THRESHOLD:
                manager_verified = True
                recognition_completed = True
                speak("Manager verified. Ready for commands.")
                print(" Manager verification successful")
                start_continuous_listening()
                return

            verification_attempts += 1
            print(f" Face not matched - Attempt {verification_attempts}")

            if verification_attempts >= MAX_ATTEMPTS:
                speak("Access denied after three failed attempts.")
                recognition_completed = True
                print("Access locked for this person.") 
            else:
                speak(f"Face not recognized. Attempt {verification_attempts} of {MAX_ATTEMPTS}. Try again.")

    except Exception as e:
        print(f"Error during face recognition: {e}")
        speak("An error occurred during face recognition.")
        recognition_completed = True

    finally:
        face_recognition_running = False


# Continuous voice listener in background thread
def continuous_voice_listener():
    global continuous_listening_active, manager_verified, system_active
    
    print("Starting continuous voice listening...")
    
    while continuous_listening_active and manager_verified and system_active:
        try:
            with mic as source:
                print("Ready for voice command...")
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            
            if "send message" in command or "notify" in command:
                print("Processing WhatsApp command...")
                send_whatsapp_message(command)
            elif "reset" in command:
                reset_system()
                break
            elif "quit" in command or "exit" in command:
                speak("Shutting down system")
                system_active = False
                break
            elif "stop listening" in command or "pause" in command:
                speak("Voice commands paused")
                continuous_listening_active = False
                break
            else:
                print(f"Unknown command: {command}")
                speak("Command not recognized. Try 'send message', 'reset', or 'quit'")
                
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            print("Could not understand audio")
            continue
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Unexpected error in voice listener: {e}")
            time.sleep(1)
            continue
    
    print("Continuous voice listening stopped")

# Start continuous listening thread
def start_continuous_listening():
    global continuous_listening_active, voice_thread
    
    if not continuous_listening_active:
        continuous_listening_active = True
        voice_thread = threading.Thread(target=continuous_voice_listener, daemon=True)
        voice_thread.start()
        speak("Voice commands now active. I'm listening continuously.")

# Stop continuous listening
def stop_continuous_listening():
    global continuous_listening_active
    continuous_listening_active = False
    print("Stopping continuous voice listening...")

# Reset system flags and stop listening
def reset_system():
    global manager_verified, recognition_completed, both_eyes_gaze_detected, continuous_listening_active
    global continuous_listening_active, verification_attempts, last_attempt_time, previous_face_encoding
    stop_continuous_listening()
    manager_verified = False
    recognition_completed = False
    both_eyes_gaze_detected = False
    verification_attempts = 0
    last_attempt_time = 0
    previous_face_encoding = None
    speak("System reset")
    print("System manually reset - recognition will be performed again")

# Send WhatsApp message thread with error handling
# Send WhatsApp message thread with error handling
def send_whatsapp_message(command):
    def send_message():
        global continuous_listening_active, voice_thread

        try:
            # Temporarily pause voice listening
            if continuous_listening_active:
                print("Pausing voice command listening during message sending.")
                continuous_listening_active = False
                voice_thread.join(timeout=5)  # Wait for current voice thread to stop

            if not os.path.exists(CONTACTS_FILE):
                speak("Contacts file not found.")
                return

            with open(CONTACTS_FILE, "r") as file:
                contacts = json.load(file)

            command_parts = command.split("to")
            if len(command_parts) < 2:
                speak("Please specify who to send the message to.")
                return

            name_part = command_parts[-1].strip().lower()

            name = None
            for contact in contacts:
                if contact.lower() in name_part or name_part in contact.lower():
                    name = contact
                    break

            if name and name in contacts:
                number = contacts[name]
                message = f"Hello {name.title()}, the manager wants to see you."
                print(f"Sending message to {name} at {number}")
                pywhatkit.sendwhatmsg_instantly(number, message, wait_time=15, tab_close=True)
                speak(f"Message sent to {name}")
            else:
                speak("Contact not found. Please check the name.")
                print(f"Available contacts: {list(contacts.keys())}")

        except json.JSONDecodeError:
            speak("Contacts file format error.")
        except Exception as e:
            print(f"Error sending message: {e}")
            speak("Failed to send message due to an error.")
        finally:
            # Resume voice command listening
            print("Resuming voice command listening.")
            start_continuous_listening()

    threading.Thread(target=send_message, daemon=True).start()


def main():
    global manager_encoding, system_active, manager_verified, recognition_completed, both_eyes_gaze_detected
    global last_person_detected_time, no_face_counter, face_recognition_running, continuous_listening_active
    
    # Load manager image and encoding
    try:
        if not os.path.exists("C:\\Users\\DeLL\\smart_camera_project\\mains\\Shreya.jpg"):
            print("Error: Shreya.jpg not found")
            speak("Manager image file not found.")
            return
        
        manager_image = face_recognition.load_image_file("Shreya.jpg")
        encodings = face_recognition.face_encodings(manager_image)
        if not encodings:
            print("No face detected in Shreya.jpg")
            speak("No face detected in manager image.")
            return
        
        manager_encoding = encodings[0]
        print("Manager face encoding loaded successfully.")
    except Exception as e:
        print(f"Error loading manager image: {e}")
        speak("Could not load manager's face image.")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        speak("Camera not accessible.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow("Smart Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Camera", 1200, 900)
    
    speak("System ready. Looking for manager...")
    
    try:
        while system_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            frame = cv2.flip(frame, 1)
            frame_small = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(frame_rgb)
            
            both_eyes_gaze_detected = False
            person_detected = False
            gaze_info = {"left_eye": "unknown", "right_eye": "unknown", "both_center": False}
            
            if results.multi_face_landmarks:
                person_detected = True
                last_person_detected_time = time.time()
                no_face_counter = 0
                
                for landmarks_obj in results.multi_face_landmarks:
                    gaze_info = get_both_eyes_gaze_direction(landmarks_obj.landmark)
                    if gaze_info["both_center"]:
                        both_eyes_gaze_detected = True
                        break
                
                if not recognition_completed and both_eyes_gaze_detected and not face_recognition_running:
                    print("Both eyes centered - starting one-time face recognition")
                    threading.Thread(target=run_face_recognition_once, args=(frame_rgb.copy(),), daemon=True).start()
            else:
                no_face_counter += 1
            
            if (time.time() - last_person_detected_time) > NO_PERSON_TIMEOUT or no_face_counter > 150:
                if manager_verified or recognition_completed:
                    print("No person detected - resetting system")
                    stop_continuous_listening()
                    reset_system()
                    speak("System reset due to no person detected")
                last_person_detected_time = time.time()
                no_face_counter = 0
            
            display_frame = imutils.resize(frame, width=800)
            
            if manager_verified and continuous_listening_active:
                status = "Verified & Active"
                color = (0, 255, 0)
            elif manager_verified:
                status = "Verified (Idle)"
                color = (0, 255, 255)
            elif recognition_completed:
                status = "Verification Failed"
                color = (0, 0, 255)
            else:
                status = "Waiting for Recognition"
                color = (255, 255, 0)
            
            cv2.putText(display_frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_frame, f"Both Eyes Centered: {both_eyes_gaze_detected}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Left Eye Gaze: {gaze_info['left_eye']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Right Eye Gaze: {gaze_info['right_eye']}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Smart Camera", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                speak("System shutting down")
                break
            elif key == ord('r'):
                reset_system()
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_continuous_listening()
        cap.release()
        cv2.destroyAllWindows()
        print("System closed")

if __name__ == "__main__":
    main()
