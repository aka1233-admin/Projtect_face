import face_recognition
import cv2
import os
import threading
import time

class FaceRecognitionHandler:
    def __init__(self, face_match_threshold=0.6):
        self.face_match_threshold = face_match_threshold
        self.manager_encoding = None
        self.face_recognition_running = False
        self.manager_verified = False
        self.recognition_done = False
        
    def load_manager_face(self, image_path):
        """Load and encode manager's face from image file"""
        try:
            if not os.path.exists(image_path):
                print(f"Error: {image_path} not found in current directory")
                return False
                
            manager_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(manager_image)
            
            if not encodings:
                print(f"Error: No face detected in {image_path}")
                return False
                
            self.manager_encoding = encodings[0]
            print("Manager face encoding loaded successfully.")
            return True
            
        except Exception as e:
            print(f"Error loading manager image: {e}")
            return False
    
    def delayed_recognition_reset(self):
        """Reset recognition after delay"""
        time.sleep(3)
        self.recognition_done = False
        print("Recognition reset after delay")
    
    def run_face_recognition(self, frame_rgb, success_callback=None):
        """Run face recognition in a separate thread"""
        self.face_recognition_running = True
        
        try:
            print("Running face recognition in thread...")
            
            # Use smaller frame for faster processing
            frame_for_recognition = cv2.resize(frame_rgb, (320, 180))
            
            face_locations = face_recognition.face_locations(frame_for_recognition, model="hog")
            
            if face_locations:
                # Scale back the locations for the original frame size
                scaled_locations = []
                for (top, right, bottom, left) in face_locations:
                    scaled_locations.append((
                        int(top * 2), int(right * 2), 
                        int(bottom * 2), int(left * 2)
                    ))
                
                face_encodings = face_recognition.face_encodings(
                    frame_rgb, scaled_locations, num_jitters=1, model="small"
                )
                
                face_matched = False
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        [self.manager_encoding], face_encoding, 
                        tolerance=self.face_match_threshold
                    )
                    
                    if matches[0]:
                        self.manager_verified = True
                        self.recognition_done = True
                        face_matched = True
                        print("Manager verification successful!")
                        
                        if success_callback:
                            success_callback()
                        break
                
                if not face_matched:
                    print("Face not matched - access denied")
                    self.recognition_done = True
                    # Reset recognition after delay to allow retry
                    threading.Thread(target=self.delayed_recognition_reset, daemon=True).start()
            else:
                print("No face detected in recognition frame")
                
        except Exception as e:
            print(f"Face recognition error: {e}")
        finally:
            self.face_recognition_running = False
    
    def start_recognition_thread(self, frame_rgb, success_callback=None):
        """Start face recognition in a new thread"""
        if not self.face_recognition_running:
            threading.Thread(
                target=self.run_face_recognition, 
                args=(frame_rgb.copy(), success_callback), 
                daemon=True
            ).start()
    
    def reset_system(self):
        """Reset face recognition system"""
        self.manager_verified = False
        self.recognition_done = False
        print("Face recognition system reset")
    
    def is_manager_verified(self):
        """Check if manager is verified"""
        return self.manager_verified
    
    def is_recognition_done(self):
        """Check if recognition process is complete"""
        return self.recognition_done
    
    def is_recognition_running(self):
        """Check if face recognition is currently running"""
        return self.face_recognition_running