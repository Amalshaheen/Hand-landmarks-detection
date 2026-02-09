# Hand Landmarks Detection 👋

A real-time hand landmarks detection system using MediaPipe and OpenCV. This project captures video from your webcam and detects hand landmarks, including 21 key points on each hand, in real-time.

## 🌟 Features

- **Real-time Detection**: Detects hand landmarks in real-time from webcam feed
- **Multi-hand Support**: Can detect and track up to 2 hands simultaneously
- **Hand Classification**: Identifies whether detected hand is left or right
- **Visual Feedback**: Displays hand landmarks and connections with visual overlay
- **Performance Metrics**: Shows FPS (Frames Per Second) for performance monitoring
- **Screenshot Capability**: Save screenshots of detected hands with a keypress
- **Easy to Use**: Simple command-line interface with minimal setup

## 📋 Requirements

- Python 3.7 or higher
- Webcam or camera device
- Operating System: Windows, macOS, or Linux

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amalshaheen/Hand-landmarks-detection.git
   cd Hand-landmarks-detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MediaPipe hand landmarker model**
   
   The application requires a pre-trained model file. Download it using one of these methods:
   
   **Option A: Using wget (Linux/macOS)**
   ```bash
   wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
   
   **Option B: Using curl (Linux/macOS/Windows)**
   ```bash
   curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
   
   **Option C: Manual download**
   - Visit: [Download Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
   - Save the file as `hand_landmarker.task` in the project directory
   
   The model file should be approximately 26-27 MB in size.

## 💻 Usage

### Basic Usage

Run the main script to start hand landmarks detection:

```bash
python hand_landmarks_detector.py
```

### Controls

- **'q'**: Quit the application
- **'s'**: Save a screenshot of the current frame

### What You'll See

The application window displays:
- Live video feed from your webcam
- Hand landmarks (21 points per hand) drawn on detected hands
- Connections between landmarks showing hand structure
- Number of hands currently detected
- Left/Right hand labels with confidence scores
- FPS (Frames Per Second) counter

## 🎯 Hand Landmarks

MediaPipe detects 21 landmarks on each hand:

```
0:  Wrist
1:  Thumb CMC
2:  Thumb MCP
3:  Thumb IP
4:  Thumb Tip
5:  Index Finger MCP
6:  Index Finger PIP
7:  Index Finger DIP
8:  Index Finger Tip
9:  Middle Finger MCP
10: Middle Finger PIP
11: Middle Finger DIP
12: Middle Finger Tip
13: Ring Finger MCP
14: Ring Finger PIP
15: Ring Finger DIP
16: Ring Finger Tip
17: Pinky MCP
18: Pinky PIP
19: Pinky DIP
20: Pinky Tip
```

## 🛠️ Customization

You can customize the detector by modifying parameters in the `HandLandmarksDetector` class:

```python
detector = HandLandmarksDetector(
    max_num_hands=2,                    # Maximum number of hands to detect
    min_detection_confidence=0.7,       # Minimum confidence for detection
    min_tracking_confidence=0.7         # Minimum confidence for tracking
)
```

## 📦 Project Structure

```
Hand-landmarks-detection/
├── hand_landmarks_detector.py    # Main application script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## 🔧 Troubleshooting

### Model File Not Found
- **Issue**: "Model file not found: hand_landmarker.task"
- **Solution**:
  - Make sure you've downloaded the model file as described in the installation steps
  - Ensure the model file is in the same directory as the script
  - Verify the file size is approximately 26-27 MB (if it's much smaller, the download may have failed)

### Camera Not Opening
- **Issue**: "Error: Could not open webcam"
- **Solution**: 
  - Check if your webcam is connected and not being used by another application
  - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or other values
  - On Linux, ensure you have proper permissions to access the camera

### Poor Detection Performance
- **Issue**: Hands not being detected or jittery tracking
- **Solution**:
  - Ensure good lighting conditions
  - Keep your hands within the camera frame
  - Adjust `min_detection_confidence` and `min_tracking_confidence` parameters
  - Make sure your hands are clearly visible against the background

### Module Not Found Error
- **Issue**: "ModuleNotFoundError: No module named 'cv2'"
- **Solution**: 
  - Make sure you've activated your virtual environment
  - Run `pip install -r requirements.txt` again

### Low FPS
- **Issue**: Low frame rate during detection
- **Solution**:
  - Reduce camera resolution in the code
  - Decrease `max_num_hands` to 1
  - Close other resource-intensive applications

## 🎓 How It Works

1. **Camera Capture**: The application captures video frames from your webcam using OpenCV
2. **Hand Detection**: MediaPipe's hand detection model identifies hands in the frame
3. **Landmark Localization**: For each detected hand, 21 landmarks are identified
4. **Visualization**: Landmarks and connections are drawn on the frame
5. **Display**: The processed frame is displayed in real-time

## 📚 Technologies Used

- **[MediaPipe](https://google.github.io/mediapipe/)**: Google's framework for building multimodal ML pipelines
- **[OpenCV](https://opencv.org/)**: Computer vision library for image processing and camera handling
- **[NumPy](https://numpy.org/)**: Numerical computing library (dependency of OpenCV)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MediaPipe team at Google for the hand detection model
- OpenCV community for the computer vision tools

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This application requires a working webcam and processes video in real-time. Make sure your system meets the requirements before running the application.