import pyttsx3
import speech_recognition as sr
import threading
import time

class SpeechHandler:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.continuous_listening_active = False
        self.listening_for_command = False
        self.voice_thread = None
        
    def speak(self, text):
        """Text-to-speech function with error handling"""
        try:
            print("SPEAKING:", text)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    
    def continuous_voice_listener(self, manager_verified, system_active, command_callback):
        """Continuous voice listening function"""
        print("Starting continuous voice listening...")
        
        while self.continuous_listening_active and manager_verified() and system_active():
            try:
                with self.mic as source:
                    print("Ready for voice command...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")
                
                # Call the command callback function
                if command_callback:
                    command_callback(command)
                    
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Could not understand audio - continuing to listen")
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Unexpected error in voice listener: {e}")
                time.sleep(1)
                continue
        
        print("Continuous voice listening stopped")
    
    def start_continuous_listening(self, manager_verified, system_active, command_callback):
        """Start continuous listening thread"""
        if not self.continuous_listening_active:
            self.continuous_listening_active = True
            self.voice_thread = threading.Thread(
                target=self.continuous_voice_listener, 
                args=(manager_verified, system_active, command_callback),
                daemon=True
            )
            self.voice_thread.start()
            self.speak("Voice commands now active. I'm listening continuously.")
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.continuous_listening_active = False
        print("Stopping continuous voice listening...")
    
    def is_listening(self):
        """Check if continuous listening is active"""
        return self.continuous_listening_active