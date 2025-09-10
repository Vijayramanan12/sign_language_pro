#!/usr/bin/env python3
"""
Dynamic Gesture Classification using LSTM
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


class DynamicGestureClassifier:
    """Train and use LSTM classifier for dynamic gestures"""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_training_data(self):
        """Load all dynamic gesture training data"""
        data_dir = "dynamic_data"
        all_sequences = []
        all_labels = []
        
        if not os.path.exists(data_dir):
            print("No dynamic_data directory found!")
            return None, None
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_sequences.npy'):
                gesture_name = filename.replace('_sequences.npy', '')
                filepath = os.path.join(data_dir, filename)
                try:
                    sequences = np.load(filepath)
                    all_sequences.extend(sequences)
                    all_labels.extend([gesture_name] * len(sequences))
                    print(f"Loaded {len(sequences)} sequences for {gesture_name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not all_sequences:
            print("No dynamic gesture training data found!")
            return None, None
            
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Gestures: {set(all_labels)}")
        
        return np.array(all_sequences), np.array(all_labels)
    
    def train(self):
        """Train the dynamic gesture classifier"""
        X, y = self.load_training_data()
        if X is None:
            return False
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )
        
        # Create LSTM model
        n_classes = len(self.label_encoder.classes_)
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 126)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Training dynamic gesture classifier...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and encoder
        model_path = os.path.join(self.model_dir, "dynamic_model.h5")
        encoder_path = os.path.join(self.model_dir, "dynamic_encoder.pkl")
        
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
        return True
    
    def load_model(self):
        """Load trained model and encoder"""
        from tensorflow.keras.models import load_model
        
        model_path = os.path.join(self.model_dir, "dynamic_model.h5")
        encoder_path = os.path.join(self.model_dir, "dynamic_encoder.pkl")
        
        try:
            self.model = load_model(model_path)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Dynamic model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Dynamic model files not found! Train the model first.")
            return False
    
    def predict(self, sequence):
        """Predict dynamic gesture from sequence"""
        sequence = np.array(sequence).reshape(1, self.sequence_length, 126)
        prediction = self.model.predict(sequence, verbose=0)[0]
        confidence = np.max(prediction)
        gesture_index = np.argmax(prediction)
        gesture_name = self.label_encoder.inverse_transform([gesture_index])[0]
        return gesture_name, confidence