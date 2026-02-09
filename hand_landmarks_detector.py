"""
Hand Landmarks Detection using MediaPipe
This script captures video from webcam and detects hand landmarks in real-time.
"""

import cv2
import mediapipe as mp
import time


class HandLandmarksDetector:
    """
    A class to detect hand landmarks using MediaPipe.
    """
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence value for hand detection
            min_tracking_confidence: Minimum confidence value for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_landmarks(self, image):
        """
        Detect hand landmarks in an image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            results: MediaPipe detection results
            image_rgb: Image converted to RGB
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        return results, image_rgb
    
    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks on the image.
        
        Args:
            image: Input image
            results: MediaPipe detection results
            
        Returns:
            image: Image with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image
    
    def get_hand_info(self, results, image_width, image_height):
        """
        Get detailed information about detected hands.
        
        Args:
            results: MediaPipe detection results
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            hands_info: List of dictionaries containing hand information
        """
        hands_info = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_info = {
                    'hand_index': idx,
                    'landmarks': []
                }
                
                # Get handedness (left or right hand)
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx]
                    hand_info['label'] = handedness.classification[0].label
                    hand_info['score'] = handedness.classification[0].score
                
                # Get landmark coordinates
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    hand_info['landmarks'].append({
                        'id': landmark_id,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'pixel_x': int(landmark.x * image_width),
                        'pixel_y': int(landmark.y * image_height)
                    })
                
                hands_info.append(hand_info)
        
        return hands_info
    
    def close(self):
        """Close the hand detector."""
        self.hands.close()


def main():
    """
    Main function to run hand landmarks detection on webcam feed.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize hand detector
    detector = HandLandmarksDetector(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Variables for FPS calculation
    prev_time = 0
    
    print("Hand Landmarks Detection Started!")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    screenshot_counter = 0
    
    try:
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            
            if not success:
                print("Error: Failed to capture frame.")
                break
            
            # Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            results, _ = detector.detect_landmarks(frame)
            
            # Draw landmarks on frame
            frame = detector.draw_landmarks(frame, results)
            
            # Get hand information
            h, w, _ = frame.shape
            hands_info = detector.get_hand_info(results, w, h)
            
            # Display number of hands detected
            num_hands = len(hands_info)
            cv2.putText(frame, f"Hands: {num_hands}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display hand labels (Left/Right)
            for idx, hand_info in enumerate(hands_info):
                if 'label' in hand_info:
                    label = hand_info['label']
                    score = hand_info['score']
                    cv2.putText(frame, f"{label} ({score:.2f})", (10, 70 + idx * 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Hand Landmarks Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"hand_detection_screenshot_{screenshot_counter}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
                screenshot_counter += 1
    
    finally:
        # Cleanup
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")


if __name__ == "__main__":
    main()
