"""
Example script showing how to use the HandLandmarksDetector class.
This is a simplified example that shows the basic usage without camera.
"""

from hand_landmarks_detector import HandLandmarksDetector
import cv2
import numpy as np


def example_basic_usage():
    """Example of basic detector usage."""
    print("=" * 70)
    print("Hand Landmarks Detector - Basic Usage Example")
    print("=" * 70)
    print()
    
    # Initialize the detector
    print("1. Creating detector...")
    detector = HandLandmarksDetector(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    print("   ✓ Detector created successfully")
    print()
    
    # Create a sample image (in real usage, this would come from your camera)
    print("2. Creating sample image...")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    print("   ✓ Image created (480x640)")
    print()
    
    # Detect hand landmarks
    print("3. Detecting hand landmarks...")
    results, image_rgb = detector.detect_landmarks(image)
    print(f"   ✓ Detection complete")
    print(f"   Found {len(results.hand_landmarks) if results.hand_landmarks else 0} hand(s)")
    print()
    
    # Draw landmarks on the image
    print("4. Drawing landmarks...")
    image_with_landmarks = detector.draw_landmarks(image.copy(), results)
    print("   ✓ Landmarks drawn")
    print()
    
    # Get detailed hand information
    print("5. Getting hand information...")
    hands_info = detector.get_hand_info(results, image.shape[1], image.shape[0])
    print(f"   ✓ Retrieved information for {len(hands_info)} hand(s)")
    
    for hand_info in hands_info:
        print(f"\n   Hand #{hand_info['hand_index']}:")
        if 'label' in hand_info:
            print(f"     - Label: {hand_info['label']}")
            print(f"     - Confidence: {hand_info['score']:.2f}")
        print(f"     - Landmarks: {len(hand_info['landmarks'])}")
    print()
    
    # Clean up
    print("6. Closing detector...")
    detector.close()
    print("   ✓ Detector closed")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("To run with actual camera, use:")
    print("  python hand_landmarks_detector.py")
    print()


def example_with_image_file():
    """Example of using detector with an image file."""
    print("=" * 70)
    print("Hand Landmarks Detector - Image File Example")
    print("=" * 70)
    print()
    
    image_path = "hand_image.jpg"  # Replace with your image path
    
    print(f"Note: This example requires an image file: {image_path}")
    print("If you don't have one, use the basic camera script instead:")
    print("  python hand_landmarks_detector.py")
    print()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Initialize detector
        detector = HandLandmarksDetector(max_num_hands=2)
        
        # Detect hands
        results, _ = detector.detect_landmarks(image)
        
        # Draw landmarks
        output_image = detector.draw_landmarks(image.copy(), results)
        
        # Get hand info
        hands_info = detector.get_hand_info(results, image.shape[1], image.shape[0])
        
        print(f"Detected {len(hands_info)} hand(s)")
        
        # Save output
        output_path = "output_" + image_path
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to: {output_path}")
        
        detector.close()
        
    except Exception as e:
        print(f"Error: {e}")


def print_usage_info():
    """Print usage information."""
    print()
    print("=" * 70)
    print("USAGE INFORMATION")
    print("=" * 70)
    print()
    print("Basic Usage:")
    print("  from hand_landmarks_detector import HandLandmarksDetector")
    print()
    print("  # Create detector")
    print("  detector = HandLandmarksDetector(")
    print("      max_num_hands=2,")
    print("      min_detection_confidence=0.7,")
    print("      min_tracking_confidence=0.7")
    print("  )")
    print()
    print("  # Detect landmarks")
    print("  results, image_rgb = detector.detect_landmarks(image)")
    print()
    print("  # Draw landmarks")
    print("  image = detector.draw_landmarks(image, results)")
    print()
    print("  # Get hand information")
    print("  hands_info = detector.get_hand_info(results, width, height)")
    print()
    print("  # Clean up")
    print("  detector.close()")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        example_with_image_file()
    else:
        try:
            example_basic_usage()
            print_usage_info()
        except FileNotFoundError as e:
            print(f"\n{e}")
            print("\nPlease download the model file first:")
            print("  python download_model.py")
            print()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
