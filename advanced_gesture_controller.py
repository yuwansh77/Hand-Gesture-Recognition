import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
from typing import List, Tuple, Optional

class AdvancedGestureController:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand for better control
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture control mappings
        self.gesture_actions = {
            'thumbs_up': self._volume_up,
            'thumbs_down': self._volume_down,
            'victory': self._take_screenshot,
            'ok': self._play_media,
            'peace': self._next_track,
            'fist': self._previous_track,
            'one': self._volume_up,
            'two': self._volume_down,
            'three': self._take_screenshot,
            'four': self._next_track,
            'five': self._previous_track
        }
        
        # Colors for different gestures
        self.gesture_colors = {
            'thumbs_up': (0, 255, 0),      # Green
            'thumbs_down': (0, 255, 255),  # Yellow
            'victory': (255, 0, 0),        # Blue
            'ok': (255, 0, 255),           # Magenta
            'peace': (255, 255, 0),        # Cyan
            'fist': (128, 0, 128),         # Purple
            'one': (255, 165, 0),          # Orange
            'two': (0, 255, 255),          # Yellow
            'three': (128, 0, 128),        # Purple
            'four': (255, 192, 203),       # Pink
            'five': (0, 128, 0),           # Dark Green
            'unknown': (128, 128, 128)     # Gray
        }
        
        # Control variables
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second cooldown between actions
        self.gesture_hold_time = 0
        self.gesture_hold_threshold = 0.5  # Hold gesture for 0.5 seconds
        
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_thumbs_up(self, landmarks: List) -> bool:
        """Detect thumbs up gesture"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        thumb_extended = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
        
        # Check if other fingers are closed
        other_fingers_closed = True
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                other_fingers_closed = False
                break
        
        return thumb_extended and other_fingers_closed
    
    def _is_thumbs_down(self, landmarks: List) -> bool:
        """Detect thumbs down gesture"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        thumb_extended_down = thumb_tip[1] > thumb_ip[1] > thumb_mcp[1]
        
        # Check if other fingers are closed
        other_fingers_closed = True
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                other_fingers_closed = False
                break
        
        return thumb_extended_down and other_fingers_closed
    
    def _is_victory(self, landmarks: List) -> bool:
        """Detect victory gesture"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_ok(self, landmarks: List) -> bool:
        """Detect OK gesture"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = self._calculate_distance(thumb_tip, index_tip)
        
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        return distance < 50 and middle_extended and ring_extended and pinky_extended
    
    def _is_peace(self, landmarks: List) -> bool:
        """Detect peace gesture"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[2][1]
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_fist(self, landmarks: List) -> bool:
        """Detect fist gesture (all fingers closed)"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:  # Finger is extended
                return False
        return True
    
    def _is_one(self, landmarks: List) -> bool:
        """Detect one gesture (index finger extended, others closed)"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_closed = landmarks[12][1] > landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_closed and ring_closed and pinky_closed and thumb_closed
    
    def _is_two(self, landmarks: List) -> bool:
        """Detect two gesture (index and middle fingers extended, others closed)"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_three(self, landmarks: List) -> bool:
        """Detect three gesture (index, middle, ring fingers extended, others closed)"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_extended and pinky_closed and thumb_closed
    
    def _is_four(self, landmarks: List) -> bool:
        """Detect four gesture (index, middle, ring, pinky fingers extended, thumb closed)"""
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_extended and pinky_extended and thumb_closed
    
    def _is_five(self, landmarks: List) -> bool:
        """Detect five gesture (all fingers extended)"""
        thumb_extended = landmarks[4][1] < landmarks[3][1]
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended
    
    def recognize_gesture(self, landmarks: List) -> str:
        """Recognize gesture from hand landmarks"""
        gestures = {
            'thumbs_up': self._is_thumbs_up,
            'thumbs_down': self._is_thumbs_down,
            'victory': self._is_victory,
            'ok': self._is_ok,
            'peace': self._is_peace,
            'fist': self._is_fist,
            'one': self._is_one,
            'two': self._is_two,
            'three': self._is_three,
            'four': self._is_four,
            'five': self._is_five
        }
        
        for gesture_name, gesture_func in gestures.items():
            if gesture_func(landmarks):
                return gesture_name
        return 'unknown'
    
    def _volume_up(self):
        """Increase system volume"""
        pyautogui.press('volumeup')
        print("Volume increased")
    
    def _volume_down(self):
        """Decrease system volume"""
        pyautogui.press('volumedown')
        print("Volume decreased")
    
    def _take_screenshot(self):
        """Take a screenshot"""
        screenshot = pyautogui.screenshot()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        screenshot.save(filename)
        print(f"Screenshot saved as {filename}")
    
    def _pause_media(self):
        """Pause media playback"""
        pyautogui.press('space')
        print("Media paused")
    
    def _play_media(self):
        """Play media"""
        pyautogui.press('space')
        print("Media played")
    
    def _next_track(self):
        """Skip to next track"""
        pyautogui.press('nexttrack')
        print("Next track")
    
    def _previous_track(self):
        """Go to previous track"""
        pyautogui.press('prevtrack')
        print("Previous track")
    
    def execute_action(self, gesture: str):
        """Execute action based on gesture with cooldown"""
        current_time = time.time()
        
        if gesture in self.gesture_actions and current_time - self.last_action_time > self.action_cooldown:
            self.gesture_actions[gesture]()
            self.last_action_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process a single frame and return annotated frame with gesture"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = 'unknown'
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                # Recognize gesture
                gesture = self.recognize_gesture(landmarks)
                
                # Execute action if gesture is recognized
                if gesture != 'unknown':
                    self.execute_action(gesture)
                
                # Draw gesture label and action info
                if gesture != 'unknown':
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Draw label
                    label = gesture.replace('_', ' ').title()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw rectangle background
                    cv2.rectangle(annotated_frame, 
                                (x_min - 5, y_min - text_height - 10),
                                (x_min + text_width + 5, y_min),
                                self.gesture_colors[gesture], -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label,
                              (x_min, y_min - 5), font, font_scale,
                              (255, 255, 255), thickness)
        
        return annotated_frame, gesture

class AdvancedGestureApp:
    def __init__(self):
        self.controller = AdvancedGestureController()
        self.cap = None
        
    def start(self):
        """Start the advanced gesture control application"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Advanced Hand Gesture Controller Started!")
        print("Press 'q' to quit")
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
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, gesture = self.controller.process_frame(frame)
            
            # Add visual feedback
            annotated_frame = self._add_visual_feedback(annotated_frame, gesture)
            
            # Display frame
            cv2.imshow('Advanced Gesture Controller', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()
    
    def _add_visual_feedback(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        """Add visual feedback to the frame"""
        h, w = frame.shape[:2]
        
        # Add border color based on gesture
        if gesture != 'unknown':
            color = self.controller.gesture_colors[gesture]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 8)
            
            # Add action indicator
            action_text = f"Action: {gesture.replace('_', ' ').title()}"
            cv2.putText(frame, action_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully!")

def main():
    """Main function to run the advanced application"""
    app = AdvancedGestureApp()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()


