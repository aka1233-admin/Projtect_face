# gaze_detection.py
"""
Gaze detection functionality using MediaPipe
"""

import mediapipe as mp
from config import LANDMARKS, GAZE_CONFIG


class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

    def process_frame(self, frame):
        """Process frame and return face landmarks"""
        rgb_small = self._resize_frame_for_processing(frame)
        results = self.face_mesh.process(rgb_small)
        return results.multi_face_landmarks if results.multi_face_landmarks else None

    def _resize_frame_for_processing(self, frame):
        """Resize frame for efficient processing"""
        import cv2
        frame_small = cv2.resize(frame, (640, 360))
        return cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    def is_person_looking_at_camera(self, frame, landmarks):
        """
        Improved gaze detection that works when approaching from left/right
        """
        if landmarks is None:
            return False

        ih, iw, _ = frame.shape

        try:
            # Get eye landmarks
            left_eye_inner = landmarks[LANDMARKS['LEFT_EYE_INNER']]
            left_eye_outer = landmarks[LANDMARKS['LEFT_EYE_OUTER']]
            right_eye_inner = landmarks[LANDMARKS['RIGHT_EYE_INNER']]
            right_eye_outer = landmarks[LANDMARKS['RIGHT_EYE_OUTER']]

            # Get iris centers
            left_iris = landmarks[LANDMARKS['LEFT_IRIS_CENTER']]
            right_iris = landmarks[LANDMARKS['RIGHT_IRIS_CENTER']]

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
            left_iris_ratio = abs(left_iris_px[0] - left_eye_center[0]) / max(left_eye_width / 2, 1)
            right_iris_ratio = abs(right_iris_px[0] - right_eye_center[0]) / max(right_eye_width / 2, 1)

            # Check if both eyes are looking relatively forward
            if (left_iris_ratio < GAZE_CONFIG['threshold'] and
                    right_iris_ratio < GAZE_CONFIG['threshold']):

                # Additional check: ensure eyes are roughly horizontal
                eye_line_angle = abs(left_eye_center[1] - right_eye_center[1]) / max(
                    abs(left_eye_center[0] - right_eye_center[0]), 1)

                if eye_line_angle < GAZE_CONFIG['eye_line_threshold']:
                    return True

        except Exception as e:
            # Debug mode handling would be added here if needed
            pass

        return False

    def has_clear_face_view(self, landmarks):
        """
        Check if we have a clear view of the face (not too much profile)
        """
        if landmarks is None:
            return False

        try:
            # Check if we can see both eyes clearly
            left_eye_inner = landmarks[LANDMARKS['LEFT_EYE_INNER']]
            right_eye_inner = landmarks[LANDMARKS['RIGHT_EYE_INNER']]
            nose_tip = landmarks[LANDMARKS['NOSE_TIP']]

            # Calculate face symmetry
            left_dist = abs(nose_tip.x - left_eye_inner.x)
            right_dist = abs(nose_tip.x - right_eye_inner.x)

            symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)

            return symmetry_ratio > GAZE_CONFIG['symmetry_threshold']

        except Exception:
            return True  # If we can't calculate, assume it's fine

    def detect_gaze_and_face_view(self, frame):
        """
        Main method to detect both gaze and clear face view
        Returns tuple: (gaze_detected, has_landmarks)
        """
        face_landmarks_list = self.process_frame(frame)

        if not face_landmarks_list:
            return False, False

        for face_landmarks in face_landmarks_list:
            if (self.is_person_looking_at_camera(frame, face_landmarks.landmark) and
                    self.has_clear_face_view(face_landmarks.landmark)):
                return True, True

        return False, True  # Has landmarks but not looking at camera