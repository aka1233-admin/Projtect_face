"""
Gaze detection functionality using MediaPipe
"""
import cv2
import mediapipe as mp
from config import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    def is_gazing_directly(self, frame, landmarks):
        """Check if person is gazing directly at camera"""
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
    
    def is_gazing_directly_face_relative(self, frame, landmarks, face_box):
        """Check gaze relative to face center instead of frame center"""
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
    
    def detect_gaze(self, frame):
        """Main gaze detection function"""
        frame_small = cv2.resize(frame, (640, 360))
        rgb_small_gaze = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_small_gaze)
        
        gaze_detected = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_detected = self.is_gazing_directly(frame, face_landmarks.landmark)
                if gaze_detected:
                    break
        
        return gaze_detected