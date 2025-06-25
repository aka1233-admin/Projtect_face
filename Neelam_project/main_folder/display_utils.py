"""
Display and UI utility functions
"""
import cv2
from config import *

class DisplayManager:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.debug_mode = False
    
    def draw_face_rectangles(self, frame, detections, debug_mode=False):
        """Draw rectangles and labels around detected faces"""
        face_found = False
        unknown_detected = False
        known_detected = False
        unknown_face_img = None
        
        for d in detections:
            top, right, bottom, left = d["location"]
            name = d["name"]
            color = d["color"]
            confidence = d.get("confidence", 0.0)
            
            if name == "Unknown Face":
                unknown_detected = True
                label = "Unknown Person"
                unknown_face_img = frame[top:bottom, left:right]
            else:
                known_detected = True
                label = f"{name} Verified" if not debug_mode else f"{name} ({confidence:.2f})"
            
            face_found = True
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), self.font, self.font_scale, color, self.font_thickness)
        
        return frame, face_found, unknown_detected, known_detected, unknown_face_img
    
    def draw_status_message(self, frame, face_found, unknown_detected, known_detected, verification_status):
        """Draw main status message at top of frame"""
        if not face_found:
            message = "Looking for someone..."
            color = COLOR_INFO
        elif unknown_detected:
            if verification_status['in_progress']:
                message = f"Verification in progress - Attempt {verification_status['attempt_count']}/{verification_status['max_attempts']}"
            elif verification_status['message'] and "Access denied" in verification_status['message']:
                message = "Access Denied - Max attempts reached"
            else:
                message = "Unknown Person Detected!"
            color = COLOR_UNKNOWN
        elif known_detected:
            message = "Face Verified - Voice Commands Ready"
            color = COLOR_KNOWN
        else:
            return frame
        
        cv2.putText(frame, message, (50, 50), self.font, 1, color, self.font_thickness)
        return frame
    
    def draw_microphone_status(self, frame, unknown_detected, verification_in_progress, 
                             detections, is_listening):
        """Draw microphone status at bottom of frame"""
        if unknown_detected or verification_in_progress:
            text = "Microphone Disabled - Unknown Person"
            color = COLOR_UNKNOWN
        elif not detections:
            text = "Microphone Disabled - Looking for face"
            color = COLOR_UNKNOWN
        elif is_listening:
            text = "Listening to your command..."
            color = COLOR_KNOWN
        else:
            return frame  # No message when not speaking
        
        cv2.putText(frame, text, (10, frame.shape[0] - 50),
                   self.font, 0.6, color, 2)
        return frame
    
    def draw_voice_input(self, frame, voice_input):
        """Draw voice input text"""
        if voice_input:
            cv2.putText(frame, voice_input, (10, frame.shape[0] - 20), 
                       self.font, 0.6, COLOR_TEXT, 2)
        return frame
    
    def draw_debug_info(self, frame, verification_status):
        """Draw debug information"""
        if not self.debug_mode:
            return frame
        
        cv2.putText(frame, f"Verification: {'YES' if verification_status['in_progress'] else 'NO'}", 
                   (10, 120), self.font, 0.6, COLOR_INFO, 2)
        cv2.putText(frame, f"Attempts: {verification_status['attempt_count']}/{verification_status['max_attempts']}", 
                   (10, 140), self.font, 0.6, COLOR_INFO, 2)
        return frame
    
    def draw_help_text(self, frame):
        """Draw help text at bottom"""
        cv2.putText(frame, "Press 'd' to toggle debug mode, 'r' to reset", 
                   (10, frame.shape[0] - 80), self.font, 0.6, COLOR_WHITE, 2)
        return frame
    
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        print(f"[DEBUG] Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        return self.debug_mode
    
    def is_debug_enabled(self):
        """Check if debug mode is enabled"""
        return self.debug_mode