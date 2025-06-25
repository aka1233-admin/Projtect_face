#!/usr/bin/env python3
"""
Smart Camera System - Main Controller
Integrates all components for face recognition, gaze tracking, and voice commands
"""

import time
import threading

# Import custom modules
from speech_handler import SpeechHandler
from face_recognition_handler import FaceRecognitionHandler
from gaze_tracker import GazeTracker
from whatsapp_handler import WhatsAppHandler
from camera_handler import CameraHandler
from system_controller import SystemController

class SmartCameraSystem:
    def __init__(self):
        # Initialize all components
        self.speech_handler = SpeechHandler()
        self.face_handler = FaceRecognitionHandler()
        self.gaze_tracker = GazeTracker()
        self.whatsapp_handler = WhatsAppHandler()
        self.camera_handler = CameraHandler()
        self.system_controller = SystemController()
        
        # Configuration
        self.manager_image_path = "Shreya.jpg"
        
    def process_voice_command(self, command):
        """Process voice commands"""
        print(f"Processing command: {command}")
        
        if "send message" in command or "notify" in command:
            print("Processing WhatsApp command...")
            self.whatsapp_handler.send_whatsapp_message(
                command, 
                speak_callback=self.speech_handler.speak
            )
        elif "reset" in command:
            self.reset_system()
        elif "quit" in command or "exit" in command:
            self.speech_handler.speak("Shutting down system")
            self.system_controller.shutdown_system()
        elif "stop listening" in command or "pause" in command:
            self.speech_handler.speak("Voice commands paused")
            self.speech_handler.stop_continuous_listening()
        else:
            print(f"Unknown command: {command}")
            self.speech_handler.speak("Command not recognized. Try 'send message', 'reset', or 'quit'")
    
    def on_manager_verified(self):
        """Callback when manager is verified"""
        self.speech_handler.speak("Manager verified. Ready for commands.")
        # Start continuous listening
        self.speech_handler.start_continuous_listening(
            manager_verified=lambda: self.face_handler.is_manager_verified(),
            system_active=lambda: self.system_controller.is_system_active(),
            command_callback=self.process_voice_command
        )
    
    def on_access_denied(self):
        """Callback when access is denied"""
        self.speech_handler.speak("Access denied. You are not authorized.")
    
    def reset_system(self):
        """Reset the entire system"""
        # Stop continuous listening first
        self.speech_handler.stop_continuous_listening()
        
        # Reset all components
        self.face_handler.reset_system()
        self.gaze_tracker.reset_gaze()
        
        self.speech_handler.speak("System reset")
        print("System manually reset")
    
    def initialize_system(self):
        """Initialize the entire system"""
        print("Initializing Smart Camera System...")
        
        # Load manager face
        if not self.face_handler.load_manager_face(self.manager_image_path):
            self.speech_handler.speak("Manager image file not found or invalid.")
            return False
        
        # Initialize camera
        if not self.camera_handler.initialize_camera():
            self.speech_handler.speak("Camera not accessible.")
            return False
        
        # Create display window
        self.camera_handler.create_window()
        
        # System ready
        self.speech_handler.speak("System ready. Looking for manager...")
        print("System initialization complete!")
        return True
    
    def run(self):
        """Main system loop"""
        if not self.initialize_system():
            return
        
        try:
            while self.system_controller.is_system_active():
                # Get camera frames
                ret, frame, frame_rgb, display_frame = self.camera_handler.get_processed_frames()
                if not ret:
                    print("Failed to read frame")
                    continue
                
                # Process gaze tracking
                person_detected, gaze_detected, gaze_results = self.gaze_tracker.process_frame(frame_rgb)
                
                # Update system controller
                self.system_controller.update_person_detection(person_detected)
                
                # Check for face recognition trigger
                if (not self.face_handler.is_recognition_done() and 
                    gaze_detected and 
                    not self.face_handler.is_recognition_running()):
                    
                    print("Starting face recognition thread...")
                    
                    def success_callback():
                        self.on_manager_verified()
                    
                    def failure_callback():
                        self.on_access_denied()
                    
                    # Start face recognition with callbacks
                    self.face_handler.start_recognition_thread(
                        frame_rgb, 
                        success_callback=success_callback
                    )
                
                # Check if system should reset due to no person
                if self.system_controller.should_reset_system(
                    self.face_handler.is_manager_verified(),
                    self.face_handler.is_recognition_done()
                ):
                    print("No person detected - resetting system")
                    self.speech_handler.stop_continuous_listening()
                    self.reset_system()
                    self.speech_handler.speak("System reset due to no person detected")
                    self.system_controller.reset_detection_timer()
                
                # Get status for display
                status_info = self.system_controller.get_status_info(
                    manager_verified=self.face_handler.is_manager_verified(),
                    recognition_done=self.face_handler.is_recognition_done(),
                    person_detected=person_detected,
                    gaze_detected=gaze_detected,
                    face_recognition_running=self.face_handler.is_recognition_running(),
                    continuous_listening=self.speech_handler.is_listening(),
                    listening_for_command=False
                )
                
                # Display frame with status
                self.camera_handler.display_frame_with_status(display_frame, status_info)
                
                # Check for quit key
                if self.camera_handler.check_quit_key():
                    self.system_controller.shutdown_system()
                    
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        print("Cleaning up system...")
        self.system_controller.shutdown_system()
        self.speech_handler.stop_continuous_listening()
        self.camera_handler.release_camera()
        self.camera_handler.destroy_windows()
        self.speech_handler.speak("System shutdown")
        print("System shutdown complete")

def main():
    """Main entry point"""
    smart_camera = SmartCameraSystem()
    smart_camera.run()

if __name__ == "__main__":
    main()