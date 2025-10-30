import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Optional

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture definitions
        self.gestures = {
            'thumbs_up': self._is_thumbs_up,
            'victory': self._is_victory,
            'peace': self._is_peace,
            'ok': self._is_ok,
            'one': self._is_one,
            'two': self._is_two,
            'three': self._is_three,
            'four': self._is_four,
            'five': self._is_five,
            'fist': self._is_fist
        }
        
        # Colors for different gestures
        self.gesture_colors = {
            'thumbs_up': (0, 255, 0),      # Green
            'victory': (255, 0, 0),        # Blue
            'peace': (255, 255, 0),        # Cyan
            'ok': (255, 0, 255),           # Magenta
            'one': (255, 165, 0),          # Orange
            'two': (0, 255, 255),          # Yellow
            'three': (128, 0, 128),        # Purple
            'four': (255, 192, 203),       # Pink
            'five': (0, 128, 0),           # Dark Green
            'fist': (64, 64, 64),          # Dark Gray
            'unknown': (128, 128, 128)     # Gray
        }
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_finger_extended(self, landmarks: List, finger_tips: List[int], finger_pips: List[int]) -> bool:
        """Check if a finger is extended based on landmark positions"""
        if len(finger_tips) != len(finger_pips):
            return False
        
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = landmarks[tip][1]
            pip_y = landmarks[pip][1]
            if tip_y > pip_y:  # Finger is bent
                return False
        return True
    
    def _is_thumbs_up(self, landmarks: List) -> bool:
        """Detect thumbs up gesture"""
        # Thumb tip (4) should be higher than thumb IP (3) and thumb MCP (2)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Check if thumb is extended upward
        thumb_extended = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
        
        # Check if other fingers are closed
        other_fingers_closed = True
        finger_tips = [8, 12, 16, 20]  # Tips of index, middle, ring, pinky
        finger_pips = [6, 10, 14, 18]  # PIP joints
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:  # Finger is extended
                other_fingers_closed = False
                break
        
        return thumb_extended and other_fingers_closed
    
    def _is_victory(self, landmarks: List) -> bool:
        """Detect victory gesture (index and middle finger extended)"""
        # Check if index and middle fingers are extended
        index_extended = landmarks[8][1] < landmarks[6][1]  # Index tip above PIP
        middle_extended = landmarks[12][1] < landmarks[10][1]  # Middle tip above PIP
        
        # Check if other fingers are closed
        ring_closed = landmarks[16][1] > landmarks[14][1]  # Ring tip below PIP
        pinky_closed = landmarks[20][1] > landmarks[18][1]  # Pinky tip below PIP
        thumb_closed = landmarks[4][1] > landmarks[3][1]  # Thumb tip below IP
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_peace(self, landmarks: List) -> bool:
        """Detect peace gesture (index and middle finger extended, others closed)"""
        # Similar to victory but with thumb closed
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[2][1]
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_ok(self, landmarks: List) -> bool:
        """Detect OK gesture (thumb and index finger forming a circle)"""
        # Check if thumb and index finger tips are close together
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = self._calculate_distance(thumb_tip, index_tip)
        
        # Check if other fingers are extended
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        return distance < 50 and middle_extended and ring_extended and pinky_extended
    
    def _is_one(self, landmarks: List) -> bool:
        """Detect one gesture (index finger extended, others closed)"""
        # Index finger extended
        index_extended = landmarks[8][1] < landmarks[6][1]
        
        # Other fingers closed
        middle_closed = landmarks[12][1] > landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_closed and ring_closed and pinky_closed and thumb_closed
    
    def _is_two(self, landmarks: List) -> bool:
        """Detect two gesture (index and middle fingers extended, others closed)"""
        # Index and middle fingers extended
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        
        # Other fingers closed
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
    
    def _is_three(self, landmarks: List) -> bool:
        """Detect three gesture (index, middle, ring fingers extended, others closed)"""
        # Index, middle, and ring fingers extended
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        
        # Other fingers closed
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_extended and pinky_closed and thumb_closed
    
    def _is_four(self, landmarks: List) -> bool:
        """Detect four gesture (index, middle, ring, pinky fingers extended, thumb closed)"""
        # Index, middle, ring, and pinky fingers extended
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        # Thumb closed
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        
        return index_extended and middle_extended and ring_extended and pinky_extended and thumb_closed
    
    def _is_five(self, landmarks: List) -> bool:
        """Detect five gesture (all fingers extended)"""
        # All fingers extended
        thumb_extended = landmarks[4][1] < landmarks[3][1]
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended
    
    def _is_fist(self, landmarks: List) -> bool:
        """Detect fist gesture (all fingers closed)"""
        # All fingers closed
        thumb_closed = landmarks[4][1] > landmarks[3][1]
        index_closed = landmarks[8][1] > landmarks[6][1]
        middle_closed = landmarks[12][1] > landmarks[10][1]
        ring_closed = landmarks[16][1] > landmarks[14][1]
        pinky_closed = landmarks[20][1] > landmarks[18][1]
        
        return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed
    
    def recognize_gesture(self, landmarks: List) -> str:
        """Recognize gesture from hand landmarks"""
        for gesture_name, gesture_func in self.gestures.items():
            if gesture_func(landmarks):
                return gesture_name
        return 'unknown'
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process a single frame and return annotated frame with gesture"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
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
                
                # Draw gesture label
                if gesture != 'unknown':
                    # Get hand bounding box
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Draw label background
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

class HandGestureApp:
    def __init__(self):
        self.recognizer = HandGestureRecognizer()
        self.cap = None
        self.gesture_history = []
        self.max_history = 10
        
    def start(self):
        """Start the hand gesture recognition application"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Hand Gesture Recognition App Started!")
        print("Press 'q' to quit, 'r' to reset gesture history")
        print("Supported gestures: Thumbs Up, Victory, Peace, OK, One, Two, Three, Four, Five, Fist")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, gesture = self.recognizer.process_frame(frame)
            
            # Update gesture history
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)
            
            # Add visual feedback based on gesture
            annotated_frame = self._add_visual_feedback(annotated_frame, gesture)
            
            # Display frame
            cv2.imshow('Hand Gesture Recognition', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.gesture_history.clear()
                print("Gesture history reset")
        
        self.cleanup()
    
    def _add_visual_feedback(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        """Add visual feedback to the frame based on detected gesture"""
        h, w = frame.shape[:2]
        
        # Add border color based on gesture
        if gesture != 'unknown':
            color = self.recognizer.gesture_colors[gesture]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 10)
        
        # Add gesture statistics
        if self.gesture_history:
            gesture_counts = {}
            for g in self.gesture_history:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            # Display most frequent gesture
            most_frequent = max(gesture_counts, key=gesture_counts.get)
            if most_frequent != 'unknown':
                cv2.putText(frame, f"Most Frequent: {most_frequent.replace('_', ' ').title()}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully!")

def main():
    """Main function to run the application"""
    app = HandGestureApp()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()


