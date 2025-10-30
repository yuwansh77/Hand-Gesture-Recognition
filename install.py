#!/usr/bin/env python3
"""
Installation script for Hand Gesture Recognition App
This script helps set up the environment and install dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("Please install Python 3.7 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def test_installation():
    """Test if the installation was successful"""
    print("\nTesting installation...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy")
    ]
    
    all_success = True
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            all_success = False
    
    # Test PyAutoGUI separately (optional)
    try:
        import pyautogui
        print("✅ PyAutoGUI imported successfully (optional)")
    except ImportError:
        print("⚠️ PyAutoGUI not available (optional for advanced features)")
    
    return all_success

def main():
    """Main installation function"""
    print("Hand Gesture Recognition App - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the error messages above.")
        return
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Some dependencies may not be working correctly.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("=" * 50)
    print("\nYou can now run the applications:")
    print("• Basic app: python hand_gesture_app.py")
    print("• Advanced app: python advanced_gesture_controller.py")
    print("• Demo: python demo.py")
    print("\nMake sure your webcam is connected and not being used by other applications.")

if __name__ == "__main__":
    main()
