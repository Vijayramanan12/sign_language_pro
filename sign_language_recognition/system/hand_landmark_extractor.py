#!/usr/bin/env python3
"""
Hand Landmark Extraction using MediaPipe
"""

import cv2
import mediapipe as mp


class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks (x, y, z) = 63 features per hand
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_data)
        
        # Pad with zeros if no hands detected or only one hand
        while len(landmarks) < 126:  # 2 hands * 63 features
            landmarks.append(0.0)
            
        return landmarks[:126], results
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame