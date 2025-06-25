"""
Voice recognition and command processing module
"""
import speech_recognition as sr
import webbrowser
import time
import threading
from config import *

class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = VOICE_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = VOICE_DYNAMIC_THRESHOLD
        self.last_voice_input = ""
        self.is_listening = False
        self.listener_thread = None
        self.running = False
    
    def process_voice_command(self, word):
        """Process recognized voice commands"""
        word_lower = word.lower()
        
        if "google" in word_lower:
            webbrowser.open("https://www.google.com")
            return f"Opening Google... You said: {word}"
        elif "time" in word_lower:
            current_time = time.strftime('%I:%M %p')
            print(f"[TIME] {current_time}")
            return f"Current time: {current_time}"
        else:
            return f"You said: {word}"
    
    def listen_for_commands(self, is_known_face_present_func, is_unknown_detected_func, is_verification_in_progress_func):
        """Continuous voice listening in background thread"""
        while self.running:
            try:
                if (is_known_face_present_func() and 
                    not is_unknown_detected_func() and 
                    not is_verification_in_progress_func()):
                    
                    with sr.Microphone() as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        
                        while (self.running and 
                               is_known_face_present_func() and 
                               not is_unknown_detected_func() and 
                               not is_verification_in_progress_func()):
                            try:
                                # Start listening
                                self.is_listening = True
                                audio = self.recognizer.listen(source, 
                                                             timeout=VOICE_TIMEOUT, 
                                                             phrase_time_limit=VOICE_PHRASE_TIME_LIMIT)
                                
                                word = self.recognizer.recognize_google(audio, language=VOICE_LANGUAGE)
                                print(f"[WORD] {word}")
                                self.last_voice_input = self.process_voice_command(word)
                                
                            except sr.WaitTimeoutError:
                                # No speech detected
                                self.is_listening = False
                                continue
                            except Exception as e:
                                self.is_listening = False
                                continue
                    
                    self.is_listening = False
                else:
                    self.is_listening = False
                    time.sleep(0.5)
            except:
                self.is_listening = False
                time.sleep(0.5)
    
    def start_listening(self, is_known_face_present_func, is_unknown_detected_func, is_verification_in_progress_func):
        """Start the voice recognition thread"""
        if not self.running:
            self.running = True
            self.listener_thread = threading.Thread(
                target=self.listen_for_commands,
                args=(is_known_face_present_func, is_unknown_detected_func, is_verification_in_progress_func),
                daemon=True
            )
            self.listener_thread.start()
    
    def stop_listening(self):
        """Stop the voice recognition thread"""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
    
    def get_voice_status(self):
        """Get current voice recognition status"""
        return {
            'last_input': self.last_voice_input,
            'is_listening': self.is_listening
        }
    
    def clear_voice_input(self):
        """Clear the last voice input"""
        self.last_voice_input = ""