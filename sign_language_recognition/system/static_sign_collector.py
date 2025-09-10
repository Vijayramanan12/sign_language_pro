#!/usr/bin/env python3
"""
Static Sign Data Collection
"""

import cv2
import pandas as pd
import os
from hand_landmark_extractor import HandLandmarkExtractor


class StaticSignCollector:
    """Collect training data for static signs"""
    
    def __init__(self):
        self.extractor = HandLandmarkExtractor()
        self.data_dir = "static_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_data(self, sign_name, num_samples=100):
        """Collect training data for a specific static sign"""
        print(f"Collecting data for sign: {sign_name}")
        print(f"Press SPACE to capture samples ({num_samples} needed)")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        collected_samples = 0
        landmarks_data = []
        
        while collected_samples < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extractor.extract_landmarks(frame)
            frame = self.extractor.draw_landmarks(frame, results)
            
            # Display info
            cv2.putText(frame, f"Sign: {sign_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected_samples}/{num_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Static Sign Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if results.multi_hand_landmarks:
                    landmarks_data.append(landmarks + [sign_name])
                    collected_samples += 1
                    print(f"Captured sample {collected_samples}")
                else:
                    print("No hand detected! Try again.")
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data
        if landmarks_data:
            df = pd.DataFrame(landmarks_data)
            filename = os.path.join(self.data_dir, f"{sign_name}_data.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(landmarks_data)} samples to {filename}")
        
        return len(landmarks_data)