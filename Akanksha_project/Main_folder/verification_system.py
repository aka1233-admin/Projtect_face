# verification_system.py
"""
3-attempt verification system for unknown persons
"""

import time
from config import VERIFICATION_CONFIG


class VerificationSystem:
    def __init__(self):
        self.unknown_attempt_count = 0
        self.max_attempts = VERIFICATION_CONFIG['max_attempts']
        self.attempt_start_time = None
        self.verification_cooldown = VERIFICATION_CONFIG['cooldown_seconds']
        self.last_verification_time = 0
        self.verification_in_progress = False
        self.verification_message = ""
        self.attempt_delay = VERIFICATION_CONFIG['attempt_delay']

    def reset_verification_system(self):
        """Reset the 3-attempt verification system"""
        self.unknown_attempt_count = 0
        self.attempt_start_time = None
        self.verification_in_progress = False
        self.verification_message = ""
        print("[VERIFICATION] System reset")

    def handle_unknown_person_verification(self):
        """
        Handle the 3-attempt verification process for unknown persons
        Returns verification status string
        """
        current_time = time.time()

        # Check if we're in cooldown period
        if current_time - self.last_verification_time < self.verification_cooldown:
            remaining_cooldown = self.verification_cooldown - (current_time - self.last_verification_time)
            self.verification_message = f"Access denied. Try again in {int(remaining_cooldown)}s"
            return "cooldown"

        # Start new verification cycle if not already in progress
        if not self.verification_in_progress:
            self.verification_in_progress = True
            self.unknown_attempt_count = 1
            self.attempt_start_time = current_time
            self.verification_message = f"Unknown person detected. Attempt {self.unknown_attempt_count}/{self.max_attempts}"
            print(f"[VERIFICATION] Starting attempt {self.unknown_attempt_count}/{self.max_attempts}")
            return "first_attempt"

        # Continue existing verification cycle
        else:
            # Check if enough time has passed for next attempt
            if current_time - self.attempt_start_time >= self.attempt_delay:
                self.unknown_attempt_count += 1
                self.attempt_start_time = current_time

                if self.unknown_attempt_count <= self.max_attempts:
                    self.verification_message = f"Verification failed. Attempt {self.unknown_attempt_count}/{self.max_attempts}"
                    print(f"[VERIFICATION] Attempt {self.unknown_attempt_count}/{self.max_attempts}")
                    return "retry_attempt"
                else:
                    # All attempts exhausted
                    self.verification_message = "Access denied. Maximum attempts reached."
                    self.last_verification_time = current_time
                    self.reset_verification_system()
                    print("[VERIFICATION] All attempts exhausted. Access denied.")
                    return "access_denied"
            else:
                # Still within the same attempt window
                return "waiting"

        return "unknown"

    def check_for_known_person(self, last_detections):
        """
        Check if a known person is now detected and reset verification if so
        Returns True if known person detected during verification
        """
        if self.verification_in_progress and last_detections:
            for detection in last_detections:
                if detection['name'] != "Unknown Face":
                    print(f"[VERIFICATION] Known person '{detection['name']}' detected. Resetting verification.")
                    self.reset_verification_system()
                    return True
        return False

    def is_in_progress(self):
        """Check if verification is currently in progress"""
        return self.verification_in_progress

    def get_verification_message(self):
        """Get the current verification message"""
        return self.verification_message

    def get_attempt_info(self):
        """Get current attempt information"""
        return {
            'current_attempt': self.unknown_attempt_count,
            'max_attempts': self.max_attempts,
            'in_progress': self.verification_in_progress
        }

    def handle_face_disappeared(self):
        """Handle when face disappears during verification"""
        if self.verification_in_progress:
            print("[INFO] Face disappeared during verification — resetting system")
            self.reset_verification_system()

    def get_cooldown_remaining(self):
        """Get remaining cooldown time in seconds"""
        current_time = time.time()
        if current_time - self.last_verification_time < self.verification_cooldown:
            return self.verification_cooldown - (current_time - self.last_verification_time)
        return 0