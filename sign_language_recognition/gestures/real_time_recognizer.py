#!/usr/bin/env python3
"""
Real-time Sign Language Recognition System
"""

import cv2
import time
from collections import deque
from system.hand_landmark_extractor import HandLandmarkExtractor
from system.static_sign_classifier import StaticSignClassifier
from dynamic_gesture_classifier import DynamicGestureClassifier
from text_to_speech import TextToSpeech


class SignLanguageRecognizer:
    """Main recognition system"""
    
    def __init__(self):
        self.extractor = HandLandmarkExtractor()
        self.static_classifier = StaticSignClassifier()
        self.dynamic_classifier = DynamicGestureClassifier()
        self.tts = TextToSpeech()
        self.sequence_buffer = deque(maxlen=30)
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # seconds
        
    def load_models(self):
        """Load both trained models"""
        static_loaded = self.static_classifier.load_model()
        dynamic_loaded = self.dynamic_classifier.load_model()
        return static_loaded, dynamic_loaded
    
    def run_recognition(self):
        """Run real-time recognition"""
        static_loaded, dynamic_loaded = self.load_models()
        
        if not static_loaded and not dynamic_loaded:
            print("No models loaded! Train models first.")
            return
        
        print("\n" + "="*50)
        print("REAL-TIME SIGN LANGUAGE RECOGNITION")
        print("="*50)
        print("Controls:")
        print("  's' - Switch to static mode")
        print("  'd' - Switch to dynamic mode")
        print("  'q' - Quit")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        mode = 'static'  # Start with static mode
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera")
                continue
                
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extractor.extract_landmarks(frame)
            frame = self.extractor.draw_landmarks(frame, results)
            
            current_time = time.time()
            
            # Add to sequence buffer for dynamic gestures
            if results.multi_hand_landmarks:
                self.sequence_buffer.append(landmarks)
            
            # Make predictions based on current mode
            prediction_text = ""
            confidence_text = ""
            
            if mode == 'static' and static_loaded and results.multi_hand_landmarks:
                try:
                    sign, confidence = self.static_classifier.predict(landmarks)
                    if confidence > 0.7:  # Confidence threshold
                        prediction_text = f"Static: {sign}"
                        confidence_text = f"Confidence: {confidence:.2f}"
                        
                        # Speak if enough time has passed and prediction is different
                        if (current_time - self.last_prediction_time > self.prediction_cooldown and 
                            sign != self.last_prediction):
                            self.tts.speak(sign)
                            self.last_prediction = sign
                            self.last_prediction_time = current_time
                except Exception as e:
                    print(f"Static prediction error: {e}")
                    
            elif mode == 'dynamic' and dynamic_loaded and len(self.sequence_buffer) == 30:
                try:
                    gesture, confidence = self.dynamic_classifier.predict(list(self.sequence_buffer))
                    if confidence > 0.7:  # Confidence threshold
                        prediction_text = f"Dynamic: {gesture}"
                        confidence_text = f"Confidence: {confidence:.2f}"
                        
                        # Speak if enough time has passed and prediction is different
                        if (current_time - self.last_prediction_time > self.prediction_cooldown and 
                            gesture != self.last_prediction):
                            self.tts.speak(gesture)
                            self.last_prediction = gesture
                            self.last_prediction_time = current_time
                except Exception as e:
                    print(f"Dynamic prediction error: {e}")
            
            # Display information on frame
            cv2.putText(frame, f"Mode: {mode.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if prediction_text:
                cv2.putText(frame, prediction_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, confidence_text, (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if mode == 'static':
                    cv2.putText(frame, "Show hand sign", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                else:
                    buffer_status = f"Buffer: {len(self.sequence_buffer)}/30"
                    cv2.putText(frame, buffer_status, (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
            # Display controls
            cv2.putText(frame, "s: Static, d: Dynamic, q: Quit", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display model status
            model_status = []
            if static_loaded:
                model_status.append("Static: OK")
            if dynamic_loaded:
                model_status.append("Dynamic: OK")
            
            status_text = " | ".join(model_status) if model_status else "No models loaded"
            cv2.putText(frame, status_text, (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and static_loaded:
                mode = 'static'
                print("Switched to static mode")
                self.sequence_buffer.clear()
            elif key == ord('d') and dynamic_loaded:
                mode = 'dynamic'
                print("Switched to dynamic mode")
            elif key == ord('s') and not static_loaded:
                print("Static model not loaded!")
            elif key == ord('d') and not dynamic_loaded:
                print("Dynamic model not loaded!")
                
        cap.release()
        cv2.destroyAllWindows()
        print("Recognition stopped.")
        