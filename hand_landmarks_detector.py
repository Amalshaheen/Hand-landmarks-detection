"""
Hand Landmarks Detection using MediaPipe
This script captures video from webcam and detects hand landmarks in real-time.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import urllib.request
import os


def download_hand_landmarker_model():
    """Download the hand landmarker model if it doesn't exist."""
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print("=" * 70)
        print("Hand Landmarker Model Required")
        print("=" * 70)
        print("\nThe hand_landmarker.task model file is required to run this application.")
        print("\nPlease download it manually:")
        print("1. Visit: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("2. Save the file as 'hand_landmarker.task' in the current directory")
        print(f"   Current directory: {os.getcwd()}")
        print("\nAlternatively, you can use wget or curl:")
        print("   wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("   or")
        print("   curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("\n" + "=" * 70)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


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
        # Download model if needed
        model_path = download_hand_landmarker_model()
        
        # Use the new MediaPipe API
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frame_counter = 0
        self.HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        
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
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image with timestamp (in milliseconds)
        self.frame_counter += 1
        timestamp_ms = int(self.frame_counter * 33)  # Assuming ~30 FPS
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
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
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                h, w, _ = image.shape
                
                # Draw landmarks
                for idx, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                    
                # Draw connections
                for connection in self.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_landmark = hand_landmarks[start_idx]
                    end_landmark = hand_landmarks[end_idx]
                    
                    start_x = int(start_landmark.x * w)
                    start_y = int(start_landmark.y * h)
                    end_x = int(end_landmark.x * w)
                    end_y = int(end_landmark.y * h)
                    
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
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
        
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                hand_info = {
                    'hand_index': idx,
                    'landmarks': []
                }
                
                # Get handedness (left or right hand)
                if results.handedness and idx < len(results.handedness):
                    handedness = results.handedness[idx]
                    if handedness:
                        hand_info['label'] = handedness[0].category_name
                        hand_info['score'] = handedness[0].score
                
                # Get landmark coordinates
                for landmark_id, landmark in enumerate(hand_landmarks):
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
        self.detector.close()


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
