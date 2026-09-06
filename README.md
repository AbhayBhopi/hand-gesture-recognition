# Hand Gesture Recognition System for Speech-Impaired Assistance

A real-time Computer Vision application using Deep Learning (CNN) and MediaPipe landmark tracking to recognize sign language and hand gestures for touchless HCI and assistive communication.

## 🚀 Features
- **Real-Time Landmark Extraction**: 21-point spatial hand skeleton tracking via MediaPipe.
- **CNN Classification Engine**: Sub-20ms inference latency for gesture classification.
- **Speech-Impaired Assistance**: Maps recognized gestures to text-to-speech output and interactive controls.

## 🛠 Tech Stack
- Python 3.10+
- OpenCV & MediaPipe
- PyTorch (CNN Architecture)
- NumPy & Matplotlib

## 🎮 How to Run

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe torch torchvision numpy
   ```

2. **Launch Application**:
   ```bash
   python app.py
   ```
