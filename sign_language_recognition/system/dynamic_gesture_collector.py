#!/usr/bin/env python3
"""
Dynamic Gesture Data Collection
"""

import cv2
import numpy as np
import os
from hand_landmark_extractor import HandLandmarkExtractor


class DynamicGestureCollector:
    """Collect training data for dynamic gestures"""
    
    def __init__(self, sequence_length=30):
        self.extractor = HandLandmarkExtractor()
        self.sequence_length = sequence_length
        self.data_dir = "dynamic_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_data(self, gesture_name, num_sequences=50):
        """Collect training data for a specific dynamic gesture"""
        print(f"Collecting data for gesture: {gesture_name}")
        print(f"Press SPACE to start recording sequence")
        print(f"Each sequence is {self.sequence_length} frames")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        collected_sequences = 0
        all_sequences = []
        
        while collected_sequences < num_sequences:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extractor.extract_landmarks(frame)
            frame = self.extractor.draw_landmarks(frame, results)
            
            # Display info
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequences: {collected_sequences}/{num_sequences}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to start sequence", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Dynamic Gesture Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                sequence = self.record_sequence(cap, gesture_name)
                if sequence:
                    all_sequences.append(sequence)
                    collected_sequences += 1
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data
        if all_sequences:
            filename = os.path.join(self.data_dir, f"{gesture_name}_sequences.npy")
            np.save(filename, np.array(all_sequences))
            print(f"Saved {len(all_sequences)} sequences to {filename}")
        
        return len(all_sequences)
    
    def record_sequence(self, cap, gesture_name):
        """Record a single gesture sequence"""
        sequence = []
        frame_count = 0
        
        print(f"Recording sequence... {self.sequence_length} frames to go")
        
        while frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extractor.extract_landmarks(frame)
            frame = self.extractor.draw_landmarks(frame, results)
            
            # Display recording progress
            remaining = self.sequence_length - frame_count
            cv2.putText(frame, f"Recording: {remaining} frames left", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (10, 170), (10 + int(400 * frame_count / self.sequence_length), 200), 
                         (0, 255, 0), -1)
            
            cv2.imshow('Dynamic Gesture Collection', frame)
            
            if results.multi_hand_landmarks:
                sequence.append(landmarks)
                frame_count += 1
            
            cv2.waitKey(50)  # Small delay for consistent timing
        
        print("Sequence recorded successfully!")
        return sequence