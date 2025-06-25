import time

class SystemController:
    def __init__(self, no_person_timeout=10):
        self.system_active = True
        self.no_person_timeout = no_person_timeout
        self.last_person_detected_time = time.time()
        self.no_face_counter = 0
        
    def update_person_detection(self, person_detected):
        """Update person detection status"""
        if person_detected:
            self.last_person_detected_time = time.time()
            self.no_face_counter = 0
        else:
            self.no_face_counter += 1
    
    def should_reset_system(self, manager_verified, recognition_done):
        """Check if system should be reset due to no person detected"""
        time_since_last_person = time.time() - self.last_person_detected_time
        
        if (time_since_last_person > self.no_person_timeout or 
            self.no_face_counter > 150):
            if manager_verified or recognition_done:
                return True
        return False
    
    def reset_detection_timer(self):
        """Reset the detection timer"""
        self.last_person_detected_time = time.time()
        self.no_face_counter = 0
    
    def shutdown_system(self):
        """Shutdown the system"""
        self.system_active = False
    
    def is_system_active(self):
        """Check if system is active"""
        return self.system_active
    
    def get_status_info(self, manager_verified, recognition_done, person_detected, 
                       gaze_detected, face_recognition_running, continuous_listening, 
                       listening_for_command):
        """Get status information for display"""
        if manager_verified and continuous_listening:
            status = "Verified - Listening for Commands"
            color = (0, 255, 0)
        elif manager_verified:
            status = "Verified - Voice System Starting"
            color = (0, 200, 0)
        elif face_recognition_running:
            status = "Verifying identity..."
            color = (255, 165, 0)  # Orange
        elif person_detected and gaze_detected:
            status = "Gaze detected - Starting verification"
            color = (0, 255, 255)
        elif person_detected:
            status = "Person detected - Look at camera"
            color = (255, 255, 0)
        else:
            status = "Looking for manager..."
            color = (0, 0, 255)
        
        return {
            'status': status,
            'color': color,
            'gaze_detected': gaze_detected,
            'continuous_listening': continuous_listening,
            'listening_for_command': listening_for_command
        }