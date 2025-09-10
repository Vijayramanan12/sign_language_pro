#!/usr/bin/env python3
"""
Text-to-Speech functionality
"""

import pyttsx3
import threading


class TextToSpeech:
    """Text-to-speech functionality"""
    
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
            self.speaking = False
            print("Text-to-speech initialized successfully!")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.engine = None
            
    def speak(self, text):
        """Speak text in a separate thread"""
        if self.engine and not self.speaking:
            self.speaking = True
            thread = threading.Thread(target=self._speak_thread, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak_thread(self, text):
        """Internal speaking thread"""
        try:
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error speaking text: {e}")
        finally:
            self.speaking = False