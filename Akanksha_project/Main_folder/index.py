# main.py
"""
Main application file for the Face Recognition System
"""

# === Suppress warnings and logs from TensorFlow and MediaPipe ===
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# === Imports ===
import cv2
import time
from config import SYSTEM_CONFIG, VERIFICATION_CONFIG
from camera_manager import CameraManager
from face_recognition_module import FaceRecognitionManager
from gaze_detection import GazeDetector
from voice_recognition import VoiceRecognitionManager
from verification_system import VerificationSystem
from ui_manager import UIManager

# Set OpenCV threads
cv2.setNumThreads(SYSTEM_CONFIG['cv2_threads'])


class FaceRecognitionApp:
    def __init__(self):
        print("[INFO] Initializing Face Recognition System...")

        # Initialize all components
        self.camera_manager = CameraManager()
        self.face_recognition_manager = FaceRecognitionManager()
        self.gaze_detector = GazeDetector()
        self.voice_manager = VoiceRecognitionManager()
        self.verification_system = VerificationSystem()
        self.ui_manager = UIManager()

        # State variables
        self.debug_mode = False
        self.recognition_done = False
        self.last_detections = []
        self.processed_faces = set()
        self.gaze_detected = False
        self.unknown_person_detected = False
        self.no_face_counter = 0

        print("[INFO] Face Recognition System initialized successfully!")

    def reset_system_state(self):
        """Reset the system state when no face is detected"""
        self.last_detections.clear()
        self.processed_faces.clear()
        self.recognition_done = False
        self.voice_manager.clear_last_input()
        self.no_face_counter = 0
        self.verification_system.reset_verification_system()
        print("[INFO] No face detected for extended time - system reset")

    def process_frame(self, frame):
        """Process a single frame for gaze detection and face recognition"""
        # Detect gaze and face landmarks
        self.gaze_detected, has_landmarks = self.gaze_detector.detect_gaze_and_face_view(frame)

        if self.debug_mode and self.gaze_detected:
            print("[DEBUG] Gaze and clear face view detected")

        # Face recognition logic (only once per gaze session)
        if not self.recognition_done and self.gaze_detected:
            detections = self.face_recognition_manager.recognize_faces(frame)

            if detections:
                self.last_detections = detections
                self.recognition_done = True
                self.no_face_counter = 0

                # Update processed faces
                for detection in detections:
                    self.processed_faces.add(detection['name'])

                print(f"[INFO] Recognition completed. Detected: {[d['name'] for d in detections]}")

                # Check if known person detected during verification
                self.verification_system.check_for_known_person(self.last_detections)
            else:
                self.no_face_counter += 1
        else:
            # Update no face counter
            if not has_landmarks:
                self.no_face_counter += 1
            else:
                self.no_face_counter = 0

        # Reset logic if no face detected for extended time
        if self.no_face_counter > SYSTEM_CONFIG['no_face_reset_frames']:
            self.reset_system_state()

    def handle_unknown_person_detection(self):
        """Handle unknown person detection and verification"""
        if self.last_detections:
            self.unknown_person_detected = any(d['name'] == "Unknown Face" for d in self.last_detections)

            if self.unknown_person_detected:
                verification_status = self.verification_system.handle_unknown_person_verification()

                # Reset recognition for retry attempts
                if verification_status == "retry_attempt":
                    self.recognition_done = False
                    self.last_detections.clear()
                    self.processed_faces.clear()
            else:
                # Known face detected - reset verification if needed
                self.verification_system.check_for_known_person(self.last_detections)
        else:
            # No detections - face has moved away
            if self.verification_system.is_in_progress():
                self.verification_system.handle_face_disappeared()
            self.unknown_person_detected = False

    def update_voice_recognition(self):
        """Update voice recognition state"""
        self.voice_manager.update_listening_state(
            self.gaze_detected,
            self.last_detections,
            self.unknown_person_detected,
            self.verification_system.is_in_progress()
        )

    def render_frame(self, frame):
        """Render the frame with all UI elements"""
        # Draw face detections
        if self.last_detections:
            self.ui_manager.draw_face_detections(frame, self.last_detections, self.debug_mode)

        # Generate and draw main message
        main_message = self.ui_manager.generate_main_message(
            self.last_detections,
            self.verification_system,
            self.unknown_person_detected
        )
        self.ui_manager.draw_main_message(frame, main_message)

        # Draw voice status
        is_authorized_detected = (
                self.last_detections and
                any(d['name'] == VERIFICATION_CONFIG['authorized_name'] for d in self.last_detections) and
                not self.verification_system.is_in_progress()
        )

        self.ui_manager.draw_voice_status(
            frame,
            self.voice_manager.is_listening,
            self.verification_system.is_in_progress(),
            self.unknown_person_detected
        )

        # Draw listening indicator for authorized users
        if is_authorized_detected:
            self.ui_manager.draw_listening_indicator(frame, True)

        # Draw voice input
        voice_input = self.voice_manager.get_last_input()
        if not voice_input and self.verification_system.get_verification_message():
            voice_input = self.verification_system.get_verification_message()
        elif not voice_input and not self.gaze_detected:
            voice_input = "Please look at the camera to activate"

        self.ui_manager.draw_voice_input(frame, voice_input)

        # Draw debug information
        if self.debug_mode:
            debug_info = {
                "Gaze": "YES" if self.gaze_detected else "NO",
                "Recognition": "DONE" if self.recognition_done else "PENDING",
                "Verification": "YES" if self.verification_system.is_in_progress() else "NO"
            }
            attempt_info = self.verification_system.get_attempt_info()
            debug_info["Attempts"] = f"{attempt_info['current_attempt']}/{attempt_info['max_attempts']}"

            self.ui_manager.draw_debug_info(frame, debug_info)

        # Draw help text
        self.ui_manager.draw_help_text(frame)

        # Display frame
        self.ui_manager.display_frame(frame)

    def handle_keyboard_input(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False  # Exit
        elif key == ord('d'):
            self.debug_mode = not self.debug_mode
            print(f"[INFO] Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        elif key == ord('r'):  # Manual reset
            self.verification_system.reset_verification_system()
            self.reset_system_state()
            print("[INFO] Manual system reset")

        return True  # Continue

    def run(self):
        """Main application loop"""
        try:
            print("[INFO] Starting Face Recognition System...")
            print("[CONTROLS] Press 'q' to quit, 'd' to toggle debug, 'r' to reset")

            while True:
                # Read frame from camera
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Process frame
                self.process_frame(frame)

                # Handle unknown person detection and verification
                self.handle_unknown_person_detection()

                # Update voice recognition state
                self.update_voice_recognition()

                # Render frame with UI
                self.render_frame(frame)

                # Handle keyboard input
                key = self.ui_manager.wait_for_key()
                if not self.handle_keyboard_input(key):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down gracefully...")
        except Exception as e:
            print(f"[ERROR] {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up all resources"""
        self.camera_manager.release()
        self.ui_manager.cleanup()
        print("[INFO] All resources released")


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()