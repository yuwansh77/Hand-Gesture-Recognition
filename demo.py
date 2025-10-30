#!/usr/bin/env python3
"""
Demo script for Hand Gesture Recognition App
This script provides a simple way to test the gesture recognition functionality
"""

import cv2
import sys
import os

def test_webcam():
    """Test if webcam is accessible"""
    print("Testing webcam access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam")
        print("Please ensure your webcam is connected and not being used by another application")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from webcam")
        cap.release()
        return False
    
    print("✅ Webcam is working correctly")
    cap.release()
    return True

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy')
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is not installed")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def run_basic_demo():
    """Run a basic demo of the gesture recognition"""
    print("\n" + "="*50)
    print("HAND GESTURE RECOGNITION DEMO")
    print("="*50)
    
    if not test_dependencies():
        return
    
    if not test_webcam():
        return
    
    print("\nStarting gesture recognition demo...")
    print("Press 'q' to quit the demo")
    print("\nTry these gestures:")
    print("👍 Thumbs Up")
    print("✌️ Victory (index + middle finger)")
    print("👌 OK (thumb + index circle)")
    print("1️⃣ One (index finger only)")
    print("2️⃣ Two (index + middle finger)")
    print("3️⃣ Three (index + middle + ring finger)")
    print("4️⃣ Four (index + middle + ring + pinky)")
    print("5️⃣ Five (all fingers extended)")
    print("✊ Fist (all fingers closed)")
    
    try:
        # Import and run the basic app
        from hand_gesture_app import HandGestureApp
        app = HandGestureApp()
        app.start()
    except ImportError as e:
        print(f"❌ Error importing gesture app: {e}")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def run_advanced_demo():
    """Run the advanced gesture controller demo"""
    print("\n" + "="*50)
    print("ADVANCED GESTURE CONTROLLER DEMO")
    print("="*50)
    
    if not test_dependencies():
        return
    
    # Test PyAutoGUI
    try:
        import pyautogui
        print("✅ PyAutoGUI is installed")
    except ImportError:
        print("❌ PyAutoGUI is not installed")
        print("Please install it using: pip install pyautogui")
        return
    
    if not test_webcam():
        return
    
    print("\nStarting advanced gesture controller demo...")
    print("Press 'q' to quit the demo")
    print("\nGesture Controls:")
    print("👍 Thumbs Up: Volume Up")
    print("👎 Thumbs Down: Volume Down")
    print("✌️ Victory: Take Screenshot")
    print("👌 OK: Play Media")
    print("✌️ Peace: Next Track")
    print("✊ Fist: Previous Track")
    print("1️⃣ One: Volume Up")
    print("2️⃣ Two: Volume Down")
    print("3️⃣ Three: Take Screenshot")
    print("4️⃣ Four: Next Track")
    print("5️⃣ Five: Previous Track")
    print("\nNote: Actions have a 1-second cooldown")
    
    try:
        # Import and run the advanced app
        from advanced_gesture_controller import AdvancedGestureApp
        app = AdvancedGestureApp()
        app.start()
    except ImportError as e:
        print(f"❌ Error importing advanced app: {e}")
    except Exception as e:
        print(f"❌ Error running advanced demo: {e}")

def main():
    """Main demo function"""
    print("Hand Gesture Recognition App - Demo")
    print("="*40)
    
    while True:
        print("\nChoose a demo:")
        print("1. Basic Gesture Recognition")
        print("2. Advanced Gesture Controller")
        print("3. Test Dependencies Only")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_basic_demo()
        elif choice == '2':
            run_advanced_demo()
        elif choice == '3':
            test_dependencies()
            test_webcam()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
