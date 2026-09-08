import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import cv2
from collections import Counter, deque
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import mediapipe as mp

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Hand Gesture Recognition | Silent Voice Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark & Glassmorphism Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .hero-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Prediction Result Highlight Badge */
    .prediction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 2.5rem;
        padding: 12px 32px;
        border-radius: 16px;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        text-align: center;
        margin: 10px 0;
    }

    .confidence-badge {
        font-size: 1.1rem;
        color: #38bdf8;
        font-weight: 600;
    }

    /* Text Display Box for Sentence */
    .sentence-box {
        background: rgba(15, 23, 42, 0.8);
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 18px;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #38bdf8;
        min-height: 70px;
        word-wrap: break-word;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Streamlit Buttons Customization */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Define Classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'nothing', 'space']

# PyTorch Model Architecture
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

# Cache Model Loader
@st.cache_resource
def load_trained_model():
    model = SignLanguageCNN(num_classes=len(CLASSES))
    pth_file = 'best_model.pth' if os.path.exists('best_model.pth') else 'sign_language_model.pth'
    if os.path.exists(pth_file):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(pth_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, pth_file
    else:
        return None, 'cpu', None

# Image Preprocessing Pipeline
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

# Perform Inference
def predict(image, model, device):
    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probabilities, dim=0)
    
    top_pred = CLASSES[top_idx.item()]
    conf = top_prob.item() * 100
    
    # Get top 5 predictions for visual bar chart
    probs_np = probabilities.cpu().numpy()
    top5_indices = np.argsort(probs_np)[::-1][:5]
    top5_data = {CLASSES[idx]: float(probs_np[idx] * 100) for idx in top5_indices}
    
    return top_pred, conf, top5_data

# Initialize Session State
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Load Model
model, device, loaded_file = load_trained_model()

# Header Section
st.markdown('<h1 class="hero-header">🤟 Silent Voice — Hand Gesture Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Empowering Speech-Impaired Communication using Real-Time AI & Computer Vision</p>', unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/sign-language.png", width=70)
    st.title("⚙️ Control Panel")
    
    if model is not None:
        st.success(f"✅ Model Loaded: `{loaded_file}`")
        st.caption(f"Device: `{device.type.upper()}` | Classes: `{len(CLASSES)}`")
    else:
        st.error("❌ Model Checkpoint Not Found! (.pth file)")
    
    st.divider()
    app_mode = st.radio("Choose App Mode:", [
        "📸 Camera / Live Capture",
        "🖼️ Upload Image",
        "💬 Sentence & Speech Builder",
        "📖 ASL Sign Reference Guide",
        "📊 Model Details & Architecture"
    ])
    
    st.divider()
    st.markdown("### 🔤 Sentence Controls")
    if st.button("🧹 Clear Sentence", use_container_width=True):
        st.session_state.sentence = ""
        st.session_state.history = []
        st.rerun()

    st.divider()
    # Sidebar quick reference popup
    if os.path.exists("asl_alphabet_guide.png"):
        with st.expander("📖 Quick ASL Cheat Sheet"):
            st.image("asl_alphabet_guide.png", caption="ASL Alphabet Signs (A-Z)", use_container_width=True)

# APP MODES IMPLEMENTATION

# MODE 1: LIVE CAMERA RECOGNITION
if app_mode == "📸 Camera / Live Capture":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔴 Live Real-Time WebRTC Hand Gesture Input")
    
    col1, col2 = st.columns([1.1, 1], gap="medium")
    
    with col1:
        st.markdown("Grant camera permissions to start streaming.")
        
        # Free Google STUN server to establish connection over the internet
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Initialize MediaPipe Tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        hands_detector = vision.HandLandmarker.create_from_options(options)
        
        # Hand Skeleton Connections
        HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
        
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 1. Predict Gesture using PyTorch
            top_pred, conf, _ = predict(pil_image, model, device)
            display_pred = top_pred if conf >= 50.0 and top_pred != 'nothing' else "No hand detected"
            
            # 2. Extract and Draw MediaPipe Hand Landmarks manually using Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = hands_detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    h, w, _ = img.shape
                    points = []
                    
                    # Get pixel coordinates
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        points.append((cx, cy))
                        cv2.circle(img, (cx, cy), 5, (121, 22, 76), -1)
                        cv2.circle(img, (cx, cy), 2, (250, 44, 250), -1)
                        
                    # Draw connecting skeleton lines
                    for connection in HAND_CONNECTIONS:
                        pt1 = points[connection[0]]
                        pt2 = points[connection[1]]
                        cv2.line(img, pt1, pt2, (121, 22, 76), 2)
            else:
                display_pred = "No hand detected"
            
            # Overlay text prediction on the frame
            box_color = (0, 255, 0) if display_pred != "No hand detected" else (255, 0, 0)
            cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
            cv2.putText(img, f"Gesture: {display_pred}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="gesture-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
        if os.path.exists("asl_alphabet_guide.png"):
            with st.expander("🖐️ Need help with signs? View ASL Reference Guide"):
                st.image("asl_alphabet_guide.png", caption="ASL Hand Gestures Chart", use_container_width=True)
                
    with col2:
        st.subheader("🧠 Live Recognition Result")
        st.info("The AI draws the predicted gesture directly onto your live video feed! Ensure your hand is clearly visible in the camera.")
        st.markdown("### Why WebRTC?")
        st.markdown("Because this app runs in the cloud, standard desktop camera inputs don't work. **WebRTC** establishes a secure, low-latency tunnel directly between your browser and our AI server.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# MODE 2: UPLOAD IMAGE INFERENCE
elif app_mode == "🖼️ Upload Image":
    col1, col2 = st.columns([1.1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Upload Gesture Input")
        
        image_input = None
        uploaded_file = st.file_uploader("Upload a gesture photo (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image_input = Image.open(uploaded_file)
            st.image(image_input, caption="Uploaded Image", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if os.path.exists("asl_alphabet_guide.png"):
            with st.expander("🖐️ Need help with signs? View ASL Reference Guide"):
                st.image("asl_alphabet_guide.png", caption="American Sign Language (ASL)", use_container_width=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧠 Recognition Result")
        
        if image_input and model:
            with st.spinner("Analyzing Hand Gesture..."):
                top_pred, conf, top5_dict = predict(image_input, model, device)
                
                st.markdown(f'''
                    <div style="text-align: center;">
                        <p style="margin:0; font-weight:600; color:#94a3b8;">Predicted Gesture</p>
                        <div class="prediction-badge">{top_pred}</div>
                        <p class="confidence-badge">Confidence Score: {conf:.2f}%</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("➕ Append to Sentence", use_container_width=True):
                        if top_pred == 'space':
                            st.session_state.sentence += " "
                        elif top_pred == 'del':
                            st.session_state.sentence = st.session_state.sentence[:-1]
                        elif top_pred != 'nothing':
                            st.session_state.sentence += top_pred
                        st.session_state.history.append((top_pred, conf))
                        st.success(f"Added '{top_pred}' to sentence!")
                        
                with col_btn2:
                    if st.button("🔊 Speak Symbol", use_container_width=True):
                        word_to_speak = "Space" if top_pred == 'space' else ("Delete" if top_pred == 'del' else top_pred)
                        js_speak = f"""
                        <script>
                            var msg = new SpeechSynthesisUtterance("{word_to_speak}");
                            window.speechSynthesis.speak(msg);
                        </script>
                        """
                        st.components.v1.html(js_speak, height=0)
                
                st.divider()
                st.markdown("#### 📈 Top 5 Class Probabilities")
                df_top5 = pd.DataFrame({
                    'Gesture': list(top5_dict.keys()),
                    'Probability (%)': list(top5_dict.values())
                })
                fig = px.bar(
                    df_top5, 
                    x='Probability (%)', 
                    y='Gesture', 
                    orientation='h', 
                    color='Probability (%)',
                    color_continuous_scale='Purples',
                    text_auto='.1f'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    height=240,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif not model:
            st.warning("Please ensure `best_model.pth` or `sign_language_model.pth` exists in the folder.")
        else:
            st.info("👈 Upload an image to see predictions.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# MODE 3: SENTENCE & SPEECH BUILDER
elif app_mode == "💬 Sentence & Speech Builder":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗣️ Live Sentence Construction & Speech Synthesizer")
    
    st.markdown("<p style='color:#94a3b8;'>Formulate words and sentences from detected gestures, then speak them out loud instantly.</p>", unsafe_allow_html=True)
    
    # Display Current Sentence
    current_text = st.session_state.sentence if st.session_state.sentence else " (Empty sentence — add gestures or type below) "
    st.markdown(f'<div class="sentence-box">{current_text}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔊 Speak Full Sentence", use_container_width=True):
            if st.session_state.sentence.strip():
                clean_text = st.session_state.sentence.replace('"', '\\"')
                js_speak = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance("{clean_text}");
                    msg.rate = 0.9;
                    window.speechSynthesis.speak(msg);
                </script>
                """
                st.components.v1.html(js_speak, height=0)
                st.toast("🔊 Speaking sentence...", icon="🗣️")
            else:
                st.warning("Sentence is empty!")
                
    with col2:
        if st.button("⌫ Backspace (Delete)", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
            
    with col3:
        if st.button("␣ Add Space", use_container_width=True):
            st.session_state.sentence += " "
            st.rerun()
            
    with col4:
        if st.button("🧹 Clear All", use_container_width=True):
            st.session_state.sentence = ""
            st.session_state.history = []
            st.rerun()
            
    st.divider()
    
    # Interactive Quick Letter Keyboard
    st.markdown("#### ⌨️ Quick Sign Virtual Keyboard")
    cols = st.columns(10)
    alphabet_keys = [c for c in CLASSES if c not in ['del', 'nothing', 'space']]
    for idx, char in enumerate(alphabet_keys):
        with cols[idx % 10]:
            if st.button(char, key=f"key_{char}", use_container_width=True):
                st.session_state.sentence += char
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# MODE 4: ASL SIGN REFERENCE GUIDE (FULL SCREEN VIEW)
elif app_mode == "📖 ASL Sign Reference Guide":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📖 American Sign Language (ASL) Alphabet Reference Chart")
    st.markdown("<p style='color:#94a3b8;'>Use this reference guide to learn or verify the hand gesture for each letter (A through Z).</p>", unsafe_allow_html=True)
    
    if os.path.exists("asl_alphabet_guide.png"):
        st.image("asl_alphabet_guide.png", caption="ASL Alphabet Hand Gesture Reference Chart", use_container_width=True)
    else:
        st.error("ASL Guide image not found.")
    st.markdown('</div>', unsafe_allow_html=True)

# MODE 5: MODEL DETAILS & ARCHITECTURE
elif app_mode == "📊 Model Details & Architecture":
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🤖 Neural Network Architecture")
        st.markdown("""
        **CNN Model Architecture (`SignLanguageCNN`)**:
        - **Input Layer**: `(3, 128, 128)` RGB Image Tensor
        - **Conv Block 1**: Conv2D (32 filters, 3x3) + ReLU + MaxPool2d (2x2)
        - **Conv Block 2**: Conv2D (64 filters, 3x3) + ReLU + MaxPool2d (2x2)
        - **Conv Block 3**: Conv2D (128 filters, 3x3) + ReLU + MaxPool2d (2x2)
        - **Dense Layers**: 
          - Flatten (`128 * 16 * 16` = 32,768 features)
          - Dense (256 Neurons) + ReLU
          - Dropout (`p=0.5`)
          - Output Dense (29 Classes: `A-Z`, `del`, `nothing`, `space`)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📜 Supported ASL Classes (29)")
        
        st.markdown("""
        The system recognizes **29 total classes**:
        - **Alphabets**: `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
        - **Special Actions**: 
          - `space` : Inserts a blank space between words
          - `del` : Deletes the previously detected character
          - `nothing` : Background / Idle position (ignored)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><div style='text-align: center; color: #64748b; font-size: 0.9rem;'>AI-Based Hand Gesture Recognition System for Speech-Impaired People • Streamlit App</div>", unsafe_allow_html=True)
