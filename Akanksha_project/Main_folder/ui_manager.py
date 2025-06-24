# ui_manager.py
"""
UI and display management for the face recognition system
"""

import cv2
from config import VERIFICATION_CONFIG


class UIManager:
    def __init__(self):
        self.window_name = "Face Recognition"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def draw_face_detections(self, frame, detections, debug_mode=False):
        """Draw face detection rectangles and labels"""
        for detection in detections:
            top, right, bottom, left = detection["location"]
            color = detection["color"]
            name = detection["name"]
            confidence = detection["confidence"]

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label
            if debug_mode:
                label = f"{name} {confidence:.2f}"
            else:
                label = name

            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def draw_main_message(self, frame, message):
        """Draw the main status message"""
        cv2.putText(frame, message, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    def draw_voice_status(self, frame, is_listening, verification_in_progress, unknown_detected):
        """Draw microphone/voice status"""
        frame_height = frame.shape[0]

        if unknown_detected or verification_in_progress:
            text = "Microphone OFF - Verification Required"
            color = (0, 0, 255)
        elif not is_listening:
            text = "Microphone OFF - Look at camera"
            color = (0, 0, 255)
        else:
            text = "Microphone ON - Listening"
            color = (0, 255, 0)

        cv2.putText(frame, text, (10, frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_voice_input(self, frame, voice_input):
        """Draw the last voice input"""
        if voice_input:
            frame_height = frame.shape[0]
            cv2.putText(frame, voice_input, (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_listening_indicator(self, frame, is_authorized_and_detected):
        """Draw listening indicator for authorized users"""
        if is_authorized_and_detected:
            cv2.putText(frame, "Listening for voice commands", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def draw_debug_info(self, frame, debug_info):
        """Draw debug information"""
        y_offset = 120
        for key, value in debug_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20

    def draw_help_text(self, frame):
        """Draw help text"""
        frame_height = frame.shape[0]
        cv2.putText(frame, "Press 'd' to toggle debug mode", (10, frame_height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def generate_main_message(self, last_detections, verification_system, unknown_detected):
        """Generate the main status message"""
        if not last_detections:
            return "Looking for faces..."

        authorized_name = VERIFICATION_CONFIG['authorized_name']

        # Check if authorized person is detected
        if any(d['name'] == authorized_name for d in last_detections) and not verification_system.is_in_progress():
            return f"{authorized_name} verified - Ready for commands"
        elif verification_system.is_in_progress():
            attempt_info = verification_system.get_attempt_info()
            return f"Verification in progress - Attempt {attempt_info['current_attempt']}/{attempt_info['max_attempts']}"
        elif unknown_detected:
            cooldown_remaining = verification_system.get_cooldown_remaining()
            if cooldown_remaining > 0:
                return f"Access denied - Cooldown: {int(cooldown_remaining)}s"
            else:
                return "Unknown person detected - Starting verification"

        return "Face detected"

    def display_frame(self, frame):
        """Display the frame in the window"""
        cv2.imshow(self.window_name, frame)

    def wait_for_key(self):
        """Wait for key press and return the key"""
        return cv2.waitKey(1) & 0xFF

    def cleanup(self):
        """Clean up UI resources"""
        cv2.destroyAllWindows()
        print("[INFO] UI resources cleaned up")