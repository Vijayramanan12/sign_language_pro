#!/usr/bin/env python3
"""
Sign Language Recognition System - Main Application
Compatible with Raspberry Pi and laptops
"""

from system.static_sign_collector import StaticSignCollector
from system.dynamic_gesture_collector import DynamicGestureCollector
from system.static_sign_classifier import StaticSignClassifier
from staticsigns.dynamic_gesture_classifier import DynamicGestureClassifier
from gestures.real_time_recognizer import SignLanguageRecognizer
import os


def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🤟 SIGN LANGUAGE RECOGNITION SYSTEM 🤟")
    print("="*60)
    print("Features:")
    print("  ✅ Static Sign Recognition (A-Z, words)")
    print("  ✅ Dynamic Gesture Recognition (phrases)")
    print("  ✅ Text-to-Speech Output")
    print("  ✅ Raspberry Pi & Laptop Compatible")
    print("  ✅ Easy Training & Data Collection")
    print("="*60)


def show_data_status():
    """Show current training data status"""
    print("\n📊 CURRENT DATA STATUS:")
    print("-" * 40)
    
    # Static data status
    static_dir = "static_data"
    if os.path.exists(static_dir):
        static_files = [f for f in os.listdir(static_dir) if f.endswith('.csv')]
        print(f"Static Signs: {len(static_files)} classes")
        for file in static_files[:5]:  # Show first 5
            sign_name = file.replace('_data.csv', '')
            print(f"  - {sign_name}")
        if len(static_files) > 5:
            print(f"  ... and {len(static_files) - 5} more")
    else:
        print("Static Signs: 0 classes")
    
    # Dynamic data status
    dynamic_dir = "dynamic_data"
    if os.path.exists(dynamic_dir):
        dynamic_files = [f for f in os.listdir(dynamic_dir) if f.endswith('.npy')]
        print(f"Dynamic Gestures: {len(dynamic_files)} classes")
        for file in dynamic_files[:5]:  # Show first 5
            gesture_name = file.replace('_sequences.npy', '')
            print(f"  - {gesture_name}")
        if len(dynamic_files) > 5:
            print(f"  ... and {len(dynamic_files) - 5} more")
    else:
        print("Dynamic Gestures: 0 classes")
    
    # Model status
    models_dir = "models"
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        static_model_exists = "static_model.pkl" in models
        dynamic_model_exists = "dynamic_model.h5" in models
        print(f"Models: Static {'✅' if static_model_exists else '❌'}, Dynamic {'✅' if dynamic_model_exists else '❌'}")
    else:
        print("Models: Static ❌, Dynamic ❌")


def collect_static_data():
    """Collect static sign data"""
    collector = StaticSignCollector()
    
    print("\n📸 STATIC SIGN DATA COLLECTION")
    print("-" * 40)
    print("Examples: A, B, C, Hello, Thank_You, Please")
    
    sign_name = input("Enter static sign name: ").strip().replace(" ", "_")
    if not sign_name:
        print("Invalid sign name!")
        return
    
    try:
        num_samples = int(input("Number of samples (default 100): ") or 100)
    except ValueError:
        num_samples = 100
    
    print(f"\nCollecting data for: {sign_name}")
    print("Instructions:")
    print("1. Position your hand to show the sign")
    print("2. Press SPACE to capture each sample")
    print("3. Keep the sign steady during capture")
    print("4. Press 'q' to quit early")
    input("\nPress ENTER to start...")
    
    collected = collector.collect_data(sign_name, num_samples)
    if collected > 0:
        print(f"✅ Successfully collected {collected} samples!")
    else:
        print("❌ No samples collected.")


def collect_dynamic_data():
    """Collect dynamic gesture data"""
    collector = DynamicGestureCollector()
    
    print("\n🎬 DYNAMIC GESTURE DATA COLLECTION")
    print("-" * 40)
    print("Examples: Hello, Good_Morning, Thank_You, Please, How_Are_You")
    
    gesture_name = input("Enter dynamic gesture name: ").strip().replace(" ", "_")
    if not gesture_name:
        print("Invalid gesture name!")
        return
    
    try:
        num_sequences = int(input("Number of sequences (default 50): ") or 50)
    except ValueError:
        num_sequences = 50
    
    print(f"\nCollecting data for: {gesture_name}")
    print("Instructions:")
    print("1. Press SPACE to start recording a sequence")
    print("2. Perform the gesture smoothly (30 frames)")
    print("3. Each sequence should show the complete gesture")
    print("4. Press 'q' to quit early")
    input("\nPress ENTER to start...")
    
    collected = collector.collect_data(gesture_name, num_sequences)
    if collected > 0:
        print(f"✅ Successfully collected {collected} sequences!")
    else:
        print("❌ No sequences collected.")


def train_static_model():
    """Train static sign classifier"""
    classifier = StaticSignClassifier()
    
    print("\n🧠 TRAINING STATIC SIGN CLASSIFIER")
    print("-" * 40)
    
    print("Training model...")
    success = classifier.train()
    
    if success:
        print("✅ Static model trained successfully!")
    else:
        print("❌ Failed to train static model. Check if you have training data.")


def train_dynamic_model():
    """Train dynamic gesture classifier"""
    classifier = DynamicGestureClassifier()
    
    print("\n🧠 TRAINING DYNAMIC GESTURE CLASSIFIER")
    print("-" * 40)
    
    print("Training model... (this may take a few minutes)")
    success = classifier.train()
    
    if success:
        print("✅ Dynamic model trained successfully!")
    else:
        print("❌ Failed to train dynamic model. Check if you have training data.")


def run_recognition():
    """Run real-time recognition"""
    recognizer = SignLanguageRecognizer()
    recognizer.run_recognition()


def show_help():
    """Show help information"""
    print("\n❓ HELP & TIPS")
    print("-" * 40)
    print("📋 WORKFLOW:")
    print("1. Collect static sign data (option 1)")
    print("2. Collect dynamic gesture data (option 2)")
    print("3. Train static model (option 3)")
    print("4. Train dynamic model (option 4)")
    print("5. Run recognition (option 5)")
    
    print("\n💡 TIPS FOR BETTER ACCURACY:")
    print("• Ensure good lighting")
    print("• Keep hands clearly visible")
    print("• Collect diverse samples (different angles/positions)")
    print("• Use consistent gestures for dynamic data")
    print("• Collect more data for better accuracy")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• Camera not working: Check camera permissions")
    print("• Low accuracy: Collect more training data")
    print("• No audio: Check system audio settings")
    print("• Slow performance: Reduce camera resolution")


def main():
    """Main application loop"""
    while True:
        print_banner()
        show_data_status()
        
        print("\n📋 MAIN MENU:")
        print("-" * 40)
        print("1. 📸 Collect Static Sign Data")
        print("2. 🎬 Collect Dynamic Gesture Data")
        print("3. 🧠 Train Static Sign Classifier")
        print("4. 🧠 Train Dynamic Gesture Classifier")
        print("5. 🔴 Run Real-time Recognition")
        print("6. 📊 View Data Status")
        print("7. ❓ Help & Tips")
        print("8. 🚪 Exit")
        print("-" * 40)
        
        try:
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                collect_static_data()
                
            elif choice == '2':
                collect_dynamic_data()
                
            elif choice == '3':
                train_static_model()
                
            elif choice == '4':
                train_dynamic_model()
                
            elif choice == '5':
                run_recognition()
                
            elif choice == '6':
                # Data status is shown at the top
                input("\nPress ENTER to continue...")
                
            elif choice == '7':
                show_help()
                input("\nPress ENTER to continue...")
                
            elif choice == '8':
                print("\n👋 Thank you for using Sign Language Recognition System!")
                break
                
            else:
                print("❌ Invalid choice! Please select 1-8.")
                input("Press ENTER to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            input("Press ENTER to continue...")


if __name__ == "__main__":
    main()