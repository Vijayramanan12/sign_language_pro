#!/usr/bin/env python3
"""
Static Sign Classification using Random Forest
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class StaticSignClassifier:
    """Train and use classifier for static signs"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_training_data(self):
        """Load all static sign training data"""
        data_dir = "static_data"
        all_data = []
        
        if not os.path.exists(data_dir):
            print("No static_data directory found!")
            return None, None
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_data.append(df)
                    print(f"Loaded data from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not all_data:
            print("No training data found!")
            return None, None
            
        combined_data = pd.concat(all_data, ignore_index=True)
        X = combined_data.iloc[:, :-1].values
        y = combined_data.iloc[:, -1].values
        
        print(f"Total samples: {len(X)}")
        print(f"Classes: {set(y)}")
        
        return X, y
    
    def train(self):
        """Train the static sign classifier"""
        X, y = self.load_training_data()
        if X is None:
            return False
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        print("Training static sign classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Static classifier accuracy: {accuracy:.2f}")
        
        # Save model and encoder
        model_path = os.path.join(self.model_dir, "static_model.pkl")
        encoder_path = os.path.join(self.model_dir, "static_encoder.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
        return True
    
    def load_model(self):
        """Load trained model and encoder"""
        model_path = os.path.join(self.model_dir, "static_model.pkl")
        encoder_path = os.path.join(self.model_dir, "static_encoder.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Static model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Static model files not found! Train the model first.")
            return False
    
    def predict(self, landmarks):
        """Predict static sign from landmarks"""
        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = self.model.predict(landmarks)[0]
        confidence = np.max(self.model.predict_proba(landmarks))
        sign_name = self.label_encoder.inverse_transform([prediction])[0]
        return sign_name, confidence