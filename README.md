# Hand Gesture Recognition App

A Python-based real-time hand gesture recognition application using MediaPipe and OpenCV. The app can detect various hand gestures and optionally control system functions.

## Features

### Basic Gesture Recognition (`hand_gesture_app.py`)
- Real-time hand landmark detection using MediaPipe
- Gesture classification for:
  - 👍 Thumbs Up
  - ✌️ Victory/Peace
  - 👌 OK
  - 1️⃣ One (index finger only)
  - 2️⃣ Two (index + middle finger)
  - 3️⃣ Three (index + middle + ring finger)
  - 4️⃣ Four (index + middle + ring + pinky)
  - 5️⃣ Five (all fingers extended)
  - ✊ Fist (all fingers closed)
- Live video feed with hand outline
- Visual feedback with colors and labels
- Gesture history tracking

### Advanced Gesture Controller (`advanced_gesture_controller.py`)
- All basic gesture recognition features
- System control integration using PyAutoGUI:
  - 👍 Thumbs Up: Volume Up
  - 👎 Thumbs Down: Volume Down
  - ✌️ Victory: Take Screenshot
  - 👌 OK: Play Media
  - ✌️ Peace: Next Track
  - ✊ Fist: Previous Track
  - 1️⃣ One: Volume Up
  - 2️⃣ Two: Volume Down
  - 3️⃣ Three: Take Screenshot
  - 4️⃣ Four: Next Track
  - 5️⃣ Five: Previous Track
- Action cooldown to prevent accidental triggers
- Enhanced visual feedback

## Requirements

- Python 3.7+
- Webcam
- Windows/macOS/Linux

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Gesture Recognition

Run the basic gesture recognition app:

```bash
python hand_gesture_app.py
```

**Controls:**
- `q`: Quit the application
- `r`: Reset gesture history

### Advanced Gesture Controller

Run the advanced controller with system integration:

```bash
python advanced_gesture_controller.py
```

**Controls:**
- `q`: Quit the application
- Gesture actions have a 1-second cooldown to prevent accidental triggers

## Gesture Recognition

The app uses MediaPipe's hand landmark detection to identify 21 key points on each hand. Gesture classification is based on the relative positions of these landmarks:

### Supported Gestures

1. **Thumbs Up** 👍
   - Thumb extended upward
   - Other fingers closed

2. **Thumbs Down** 👎 (Advanced only)
   - Thumb extended downward
   - Other fingers closed

3. **Victory** ✌️
   - Index and middle fingers extended
   - Other fingers closed

4. **OK** 👌
   - Thumb and index finger forming a circle
   - Other fingers extended

5. **Peace** ✌️ (Advanced only)
   - Index and middle fingers extended
   - Other fingers closed
   - Thumb closed

6. **Fist** ✊ (Advanced only)
   - All fingers closed

7. **One** 1️⃣
   - Index finger extended
   - Other fingers closed

8. **Two** 2️⃣
   - Index and middle fingers extended
   - Other fingers closed

9. **Three** 3️⃣
   - Index, middle, and ring fingers extended
   - Other fingers closed

10. **Four** 4️⃣
    - Index, middle, ring, and pinky fingers extended
    - Thumb closed

11. **Five** 5️⃣
    - All fingers extended

## Technical Details

### Architecture

- **MediaPipe**: Hand landmark detection and tracking
- **OpenCV**: Video capture, processing, and display
- **NumPy**: Numerical operations and array handling
- **PyAutoGUI**: System control (advanced version only)

### Performance

- Optimized for real-time processing
- Configurable detection confidence thresholds
- Efficient gesture classification algorithms
- Minimal CPU usage with proper resource management

### Customization

You can easily modify the gesture recognition by:

1. **Adding new gestures**: Implement new gesture detection functions in the `HandGestureRecognizer` class
2. **Changing visual feedback**: Modify colors, fonts, and display elements
3. **Adjusting sensitivity**: Change confidence thresholds in MediaPipe initialization
4. **Adding new actions**: Extend the `gesture_actions` dictionary in the advanced controller

## Troubleshooting

### Common Issues

1. **Webcam not detected**
   - Ensure your webcam is connected and not being used by another application
   - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **Poor gesture recognition**
   - Ensure good lighting conditions
   - Keep your hand clearly visible in the frame
   - Adjust detection confidence thresholds if needed

3. **System control not working** (Advanced version)
   - Ensure PyAutoGUI has proper permissions
   - Check if your system supports the media control keys
   - Verify that media applications are running

### Performance Tips

- Close other applications using the webcam
- Ensure adequate lighting
- Keep your hand steady when performing gestures
- Use the cooldown period effectively in the advanced version

## Future Enhancements

- Custom gesture training with TensorFlow/Keras
- Multi-hand gesture recognition
- Gesture recording and playback
- Integration with smart home devices
- Voice command integration
- Mobile app version

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
