import mediapipe as mp
import cv2

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.gaze_detected = False
        
    def get_gaze_direction(self, landmarks):
        """Optimized gaze tracking with bounds checking"""
        try:
            # Check if landmarks exist for iris points
            if len(landmarks) <= 473:
                return "unknown"
                
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            avg_x = (left_iris.x + right_iris.x) / 2
            
            if 0.4 < avg_x < 0.6:
                return "center"
            return "right" if avg_x <= 0.4 else "left"
        except (IndexError, AttributeError) as e:
            print(f"Gaze detection error: {e}")
            # cv2.putText(display_frame, status, (20, 40), 
            #        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return "unknown"
    
    def process_frame(self, frame_rgb):
        """Process frame for face mesh and gaze detection"""
        results = self.face_mesh.process(frame_rgb)
        
        self.gaze_detected = False
        person_detected = False
        
        if results.multi_face_landmarks:
            person_detected = True
            
            # Check gaze direction
            for landmarks_obj in results.multi_face_landmarks:
                gaze_direction = self.get_gaze_direction(landmarks_obj.landmark)
                if gaze_direction == "center":
                    self.gaze_detected = True
                    break
        
        return person_detected, self.gaze_detected, results
    
    def is_gaze_detected(self):
        """Check if center gaze is detected"""
        return self.gaze_detected
    
    def reset_gaze(self):
        """Reset gaze detection"""
        self.gaze_detected = False