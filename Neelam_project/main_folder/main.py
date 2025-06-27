"""
Main Face Recognition Application
Entry point for the face recognition system
"""
import cv2
import numpy as np
import time

# Import our custom modules
from config import *
from face_utils import load_known_faces, recognize_faces, save_unknown_face
from gaze_detection import GazeDetector
from voice_recognition import VoiceRecognizer  
from verification_system import VerificationSystem
from camera_handler import VideoCaptureThreaded
from display_utils import DisplayManager

class FaceRecognitionApp:
    def __init__(self):
        # Initialize components
        self.known_encodings, self.known_names = load_known_faces()
        self.camera = VideoCaptureThreaded()
        self.gaze_detector = GazeDetector()
        self.voice_recognizer = VoiceRecognizer()
        self.verification_system = VerificationSystem()
        self.display_manager = DisplayManager()
        
        # Set OpenCV threads
        cv2.setNumThreads(1)
        
        # State variables
        self.recognition_done = False
        self.recognition_locked = False
        self.last_detections = []
        self.processed_faces = set()
        self.no_face_counter = 0
        self.unknown_face_img = None
        
        print("Face Recognition System initialized successfully!")
        print("Camera started. Press 'q' to quit, 'n' to save unknown face.")
    
    def is_known_face_present(self):
        """Check if any known face is present"""
        return any(d["name"] != "Unknown Face" for d in self.last_detections)
    
    def is_unknown_detected(self):
        """Check if unknown person is detected"""
        return any(d['name'] == "Unknown Face" for d in self.last_detections)
    
    def is_verification_in_progress(self):
        """Check if verification is in progress"""
        return self.verification_system.is_verification_in_progress()
    
    def reset_system(self):
        """Reset the entire system"""
        self.last_detections.clear()
        self.processed_faces.clear()
        self.recognition_done = False
        self.recognition_locked = False
        self.no_face_counter = 0
        self.verification_system.reset_verification_system()
        self.voice_recognizer.clear_voice_input()
        print("[INFO] System reset")
    
    def process_frame(self, frame):
        """Process a single frame for face recognition"""
        if self.recognition_locked:
            return
        
        if not self.recognition_done:
            # Prepare frame for recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            # Calculate brightness for adaptive threshold
            brightness = np.mean(frame)
            
            # Recognize faces
            current_detections = recognize_faces(rgb_small, self.known_encodings, 
                                               self.known_names, brightness)
            
            if current_detections:
                self.last_detections = current_detections
                self.no_face_counter = 0
                self.recognition_done = True
                
                # Check if known person detected during verification
                self.verification_system.check_for_known_person(self.last_detections)
                
                # For known face, lock immediately
                if self.is_known_face_present():
                    self.recognition_locked = True
                    self.verification_system.reset_verification_system()
            else:
                self.no_face_counter += 1
        else:
            # Check if face disappeared
            if not self.gaze_detector.detect_gaze(frame):
                self.no_face_counter += 1
            else:
                self.no_face_counter = 0
    
    def handle_verification(self):
        """Handle unknown person verification"""
        if self.last_detections:
            unknown_person_detected = self.is_unknown_detected()
            
            if unknown_person_detected:
                verification_status = self.verification_system.handle_unknown_person_verification()
                
                # Reset recognition for retry attempts
                if verification_status == "retry_attempt":
                    self.recognition_done = False
                    self.recognition_locked = False
                    self.last_detections.clear()
                    self.processed_faces.clear()
                elif verification_status == "access_denied":
                    self.recognition_locked = True
            else:
                # Known face detected - reset verification if needed
                self.verification_system.check_for_known_person(self.last_detections)
        else:
            # No detections - face disappeared during verification
            if self.verification_system.is_verification_in_progress():
                print("[INFO] Face disappeared during verification - resetting system")
                self.verification_system.reset_verification_system()
    
    def handle_no_face_timeout(self):
        """Handle case when no face is detected for too long"""
        if self.no_face_counter > NO_FACE_RESET_THRESHOLD:
            self.reset_system()
            print("[INFO] No face detected - system reset")
    
    def render_frame(self, frame):
        """Render all UI elements on the frame"""
        # Draw face rectangles and get status
        frame, face_found, unknown_detected, known_detected, unknown_face_img = \
            self.display_manager.draw_face_rectangles(frame, self.last_detections, 
                                                    self.display_manager.is_debug_enabled())
        
        # Store unknown face for saving
        if unknown_face_img is not None:
            self.unknown_face_img = unknown_face_img
        
        # Get status information
        verification_status = self.verification_system.get_verification_status()
        voice_status = self.voice_recognizer.get_voice_status()
        
        # Draw all UI elements
        frame = self.display_manager.draw_status_message(frame, face_found, unknown_detected, 
                                                       known_detected, verification_status)
        
        frame = self.display_manager.draw_microphone_status(frame, unknown_detected, 
                                                          self.is_verification_in_progress(),
                                                          self.last_detections, 
                                                          voice_status['is_listening'])
        
        frame = self.display_manager.draw_voice_input(frame, voice_status['last_input'])
        frame = self.display_manager.draw_debug_info(frame, verification_status)
        frame = self.display_manager.draw_help_text(frame)
        
        return frame
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False  # Quit
        elif key == ord('d'):
            self.display_manager.toggle_debug_mode()
        elif key == ord('r'):
            self.reset_system()
        elif key == ord('n'):
            self.save_unknown_face()
        
        return True  # Continue
    
    def save_unknown_face(self):
        """Save unknown face with user input"""
        for d in self.last_detections:
            if d["name"] == "Unknown Face":
                top, right, bottom, left = d["location"]
                voice_name = input("Enter name for New face: ").strip()
                
                if save_unknown_face(cv2.flip(self.camera.frame, 1), d["location"], voice_name):
                    print("[INFO] Reloading known faces...")
                    self.known_encodings, self.known_names = load_known_faces()
                    self.reset_system()
                break
    
    def run(self):
        """Main application loop"""
        # Start voice recognition
        self.voice_recognizer.start_listening(
            self.is_known_face_present,
            self.is_unknown_detected, 
            self.is_verification_in_progress
        )
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame for face recognition
                self.process_frame(frame)
                
                # Handle verification logic
                self.handle_verification()
                
                # Handle no face timeout
                self.handle_no_face_timeout()
                
                # Render UI
                frame = self.render_frame(frame)
                
                # Show frame
                cv2.imshow("Face Recognition System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        self.voice_recognizer.stop_listening()
        self.camera.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

def main():
    """Main entry point"""
    print("Starting Face Recognition System...")
    print("=" * 50)
    
    try:
        app = FaceRecognitionApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()