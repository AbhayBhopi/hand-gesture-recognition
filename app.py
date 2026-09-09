import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'nothing', 'space']

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SignLanguageCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_trained_model():
    model = SignLanguageCNN(num_classes=len(CLASSES))
    pth_file = 'best_model.pth' if os.path.exists('best_model.pth') else 'sign_language_model.pth'
    if os.path.exists(pth_file):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(pth_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    return None, 'cpu'

model, device = load_trained_model()

# Hand Skeleton Connections
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

if os.path.exists('hand_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hands_detector = vision.HandLandmarker.create_from_options(options)
else:
    hands_detector = None

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict(image, model, device):
    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probabilities, dim=0)
    
    top_pred = CLASSES[top_idx.item()]
    conf = top_prob.item() * 100
    return top_pred, conf

def process_frame(frame):
    if frame is None:
        return None
    if model is None:
        return frame
        
    rgb_frame = frame
    pil_image = Image.fromarray(rgb_frame)
    
    top_pred, conf = predict(pil_image, model, device)
    display_pred = top_pred if conf >= 50.0 and top_pred != 'nothing' else "No hand detected"
    
    img = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    if hands_detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hands_detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, _ = img.shape
                points = []
                for lm in hand_landmarks:
                    px_val, py_val = int(lm.x * w), int(lm.y * h)
                    points.append((px_val, py_val))
                    cv2.circle(img, (px_val, py_val), 5, (255, 0, 255), -1)
                
                for conn in HAND_CONNECTIONS:
                    pt1 = points[conn[0]]
                    pt2 = points[conn[1]]
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    
    cv2.putText(img, f"Prediction: {display_pred} ({conf:.1f}%)", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with gr.Blocks() as demo:
    gr.Markdown("# 🤟 Silent Voice — Hand Gesture Recognition")
    gr.Markdown("Real-time sign language recognition using PyTorch and MediaPipe.")
    
    with gr.Row():
        with gr.Column():
            video_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input")
        with gr.Column():
            video_out = gr.Image(label="Live Prediction & Skeleton")
            
    video_in.stream(fn=process_frame, inputs=video_in, outputs=video_out)

if __name__ == "__main__":
    demo.launch()
