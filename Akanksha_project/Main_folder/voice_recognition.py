# voice_recognition.py
"""
Voice recognition and command processing functionality
"""

import time
import threading
import webbrowser
import speech_recognition as sr
from config import VOICE_CONFIG, VERIFICATION_CONFIG


class VoiceRecognitionManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = VOICE_CONFIG['energy_threshold']
        self.recognizer.dynamic_energy_threshold = VOICE_CONFIG['dynamic_energy_threshold']

        self.last_voice_input = ""
        self.is_listening = False
        self.microphone = sr.Microphone()

        # Calibrate microphone
        self._calibrate_microphone()

        # Start voice listening thread
        self.voice_thread = threading.Thread(target=self._continuous_voice_listener, daemon=True)
        self.voice_thread.start()

    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[VOICE] Microphone calibrated")
        except Exception as e:
            print(f"[ERROR] Microphone calibration failed: {e}")

    def should_listen(self, gaze_detected, last_detections, unknown_person_detected,
                      verification_in_progress):
        """
        Determine if voice recognition should be active
        """
        return (gaze_detected and
                last_detections and
                not unknown_person_detected and
                not verification_in_progress and
                any(d['name'] == VERIFICATION_CONFIG['authorized_name'] for d in last_detections))

    def _continuous_voice_listener(self):
        """
        Continuous voice listening thread
        This will be controlled by external state variables
        """
        while True:
            try:
                if self.is_listening:
                    with self.microphone as source:
                        print("[VOICE] Listening...")
                        try:
                            audio = self.recognizer.listen(
                                source,
                                timeout=VOICE_CONFIG['timeout'],
                                phrase_time_limit=VOICE_CONFIG['phrase_time_limit']
                            )
                            print("[VOICE] Processing...")

                            word = self.recognizer.recognize_google(
                                audio,
                                language=VOICE_CONFIG['language']
                            )
                            print(f"[WORD] {word}")

                            # Process the recognized word
                            self._process_voice_command(word)

                        except sr.UnknownValueError:
                            print("[ERROR] Could not understand audio")
                            self.last_voice_input = "Sorry, I didn't catch that."
                        except sr.RequestError as e:
                            print(f"[ERROR] API error: {e}")
                            self.last_voice_input = "Voice recognition failed."
                        except sr.WaitTimeoutError:
                            # Timeout is normal, just continue
                            pass
                else:
                    time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Voice thread exception: {e}")
                time.sleep(1)

    def _process_voice_command(self, word):
        """
        Process recognized voice commands
        """
        word_lower = word.lower()

        if "google" in word_lower:
            webbrowser.open("https://www.google.com")
            self.last_voice_input = f"Opening Google... You said: {word}"
        elif "time" in word_lower:
            current_time = time.strftime('%I:%M %p')
            print(f"[TIME] {current_time}")
            self.last_voice_input = f"Time is {current_time}"
        else:
            self.last_voice_input = f"You said: {word}"
            print("[VOICE] No recognized command")

    def update_listening_state(self, gaze_detected, last_detections, unknown_person_detected,
                               verification_in_progress):
        """
        Update whether the voice recognition should be listening
        """
        should_listen = self.should_listen(
            gaze_detected, last_detections, unknown_person_detected, verification_in_progress
        )

        if should_listen != self.is_listening:
            self.is_listening = should_listen
            if should_listen:
                print("[VOICE] Voice recognition activated")
            else:
                print("[VOICE] Voice recognition deactivated")

    def get_last_input(self):
        """Get the last voice input message"""
        return self.last_voice_input

    def clear_last_input(self):
        """Clear the last voice input message"""
        self.last_voice_input = ""