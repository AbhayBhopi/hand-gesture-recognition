# Monkey-patch Python 3.14 asyncio DatagramTransport & aioice timer issues on cloud
try:
    import asyncio.selector_events
    _orig_sendto = asyncio.selector_events._SelectorDatagramTransport.sendto
    def _safe_sendto(self, data, addr=None):
        if getattr(self, '_sock', None) is None:
            return 0
        try:
            return _orig_sendto(self, data, addr)
        except Exception:
            return 0
    asyncio.selector_events._SelectorDatagramTransport.sendto = _safe_sendto

    _orig_fatal = asyncio.selector_events._SelectorDatagramTransport._fatal_error
    def _safe_fatal(self, exc, message='Fatal error on transport'):
        if getattr(self, '_loop', None) is None:
            return
        try:
            return _orig_fatal(self, exc, message)
        except Exception:
            return
    asyncio.selector_events._SelectorDatagramTransport._fatal_error = _safe_fatal

    import aioice.stun
    _orig_retry = aioice.stun.Transaction._Transaction__retry
    def _safe_retry(self):
        try:
            return _orig_retry(self)
        except Exception:
            return
    aioice.stun.Transaction._Transaction__retry = _safe_retry
except Exception:
    pass

import os
import base64
import io
import queue
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter, deque
import cv2
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Two-Way Sign Language Communicator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}

/* ── Animated gradient background ── */
.main,.stApp{
  background: linear-gradient(-45deg,#0f172a,#1e1b4b,#12163a,#0d1117);
  background-size: 400% 400%;
  animation: gradShift 12s ease infinite;
}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

/* ── Glassmorphism cards ── */
.glass-card{
  background:rgba(15,23,42,0.55);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border:1px solid rgba(99,102,241,0.18);
  border-radius:20px;padding:28px;margin-bottom:22px;
  box-shadow:0 8px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover{transform:translateY(-2px);box-shadow:0 12px 48px rgba(99,102,241,0.2);}

/* ── Hero banner ── */
.hero-banner{
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,rgba(99,102,241,0.15) 0%,rgba(168,85,247,0.15) 50%,rgba(236,72,153,0.15) 100%);
  border:1px solid rgba(99,102,241,0.3); border-radius:24px;
  padding:48px 40px 36px; margin-bottom:28px;
  box-shadow:0 0 60px rgba(99,102,241,0.15);
}
.hero-banner::before{
  content:''; position:absolute; top:-80px; right:-80px;
  width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle,rgba(168,85,247,0.2) 0%,transparent 70%);
}
.hero-banner::after{
  content:''; position:absolute; bottom:-60px; left:-60px;
  width:200px; height:200px; border-radius:50%;
  background:radial-gradient(circle,rgba(99,102,241,0.15) 0%,transparent 70%);
}
.hero-title{
  background:linear-gradient(90deg,#818cf8 0%,#c084fc 40%,#f472b6 80%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-weight:800;font-size:3.2rem;letter-spacing:-0.03em;line-height:1.1;
  margin-bottom:10px;
}
.hero-subtitle{color:#94a3b8;font-size:1.15rem;font-weight:400;max-width:600px;}
.hero-tag{
  display:inline-block;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);
  color:#a5b4fc;font-size:0.75rem;font-weight:700;padding:3px 10px;border-radius:20px;
  margin-right:8px;margin-top:12px;text-transform:uppercase;letter-spacing:0.08em;
}

/* ── Neon metric cards ── */
.metric-card{
  background:rgba(15,23,42,0.6);
  border-radius:18px;padding:22px 16px;text-align:center;
  border:1px solid rgba(255,255,255,0.06);
  position:relative;overflow:hidden;
  transition:transform 0.25s ease,box-shadow 0.25s ease;
}
.metric-card:hover{transform:translateY(-4px);}
.metric-card .icon{font-size:2.2rem;margin-bottom:10px;display:block;}
.metric-card .val{font-weight:800;font-size:1.5rem;color:#f8fafc;}
.metric-card .lbl{color:#64748b;font-size:0.78rem;margin-top:4px;text-transform:uppercase;letter-spacing:0.06em;}
.metric-card.purple{box-shadow:0 0 0 1px rgba(168,85,247,0.3),0 8px 32px rgba(168,85,247,0.15);}
.metric-card.blue{box-shadow:0 0 0 1px rgba(56,189,248,0.3),0 8px 32px rgba(56,189,248,0.15);}
.metric-card.pink{box-shadow:0 0 0 1px rgba(236,72,153,0.3),0 8px 32px rgba(236,72,153,0.15);}
.metric-card.green{box-shadow:0 0 0 1px rgba(16,185,129,0.3),0 8px 32px rgba(16,185,129,0.15);}
.metric-card.purple .val{color:#c084fc;}
.metric-card.blue .val{color:#38bdf8;}
.metric-card.pink .val{color:#f472b6;}
.metric-card.green .val{color:#34d399;}

/* ── Feature direction cards ── */
.dir-card{
  background:rgba(15,23,42,0.65);border-radius:20px;padding:28px;
  height:100%;border:1px solid rgba(255,255,255,0.07);
  transition:transform 0.2s ease,box-shadow 0.2s ease;
}
.dir-card:hover{transform:translateY(-4px);}
.dir-card.left{border-top:3px solid #a855f7;box-shadow:0 0 30px rgba(168,85,247,0.12);}
.dir-card.right{border-top:3px solid #ec4899;box-shadow:0 0 30px rgba(236,72,153,0.12);}
.dir-card h3{font-size:1.3rem;font-weight:800;margin-bottom:16px;}
.dir-card .step{
  display:flex;align-items:flex-start;gap:12px;
  margin-bottom:14px;padding:10px 14px;
  background:rgba(255,255,255,0.03);border-radius:12px;
  border:1px solid rgba(255,255,255,0.05);
}
.dir-card .step-num{
  min-width:26px;height:26px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:0.75rem;
  background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;
}
.dir-card.right .step-num{background:linear-gradient(135deg,#db2777,#ec4899);}
.dir-card .step-text{color:#cbd5e1;font-size:0.88rem;line-height:1.5;}

/* ── Flow arrows ── */
.flow-step{
  display:flex;flex-direction:column;align-items:center;text-align:center;
  flex:1;position:relative;
}
.flow-step .f-icon{
  width:52px;height:52px;border-radius:16px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.4rem;margin-bottom:8px;
  background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
}
.flow-step .f-label{color:#94a3b8;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;}
.flow-arrow{color:rgba(99,102,241,0.5);font-size:1.4rem;padding:0 2px;display:flex;align-items:center;margin-bottom:28px;}

/* ── Prediction badges ── */
.prediction-badge{display:inline-block;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;font-weight:800;font-size:2.5rem;padding:12px 32px;border-radius:16px;box-shadow:0 0 20px rgba(99,102,241,0.5),0 0 60px rgba(99,102,241,0.2);text-align:center;margin:10px 0;}
.confirmed-badge{display:inline-block;background:linear-gradient(135deg,#059669,#10b981);color:#fff;font-weight:800;font-size:2.5rem;padding:12px 32px;border-radius:16px;box-shadow:0 0 24px rgba(16,185,129,0.6),0 0 60px rgba(16,185,129,0.2);text-align:center;margin:10px 0;animation:confirmPop 0.3s ease;}
@keyframes confirmPop{0%{transform:scale(0.9);}60%{transform:scale(1.08);}100%{transform:scale(1);}}
.confidence-badge{font-size:1.1rem;color:#38bdf8;font-weight:600;}
.sentence-box{background:rgba(15,23,42,0.8);border:2px solid #6366f1;border-radius:12px;padding:18px;font-size:1.8rem;font-weight:700;letter-spacing:0.05em;color:#38bdf8;min-height:70px;word-wrap:break-word;box-shadow:inset 0 2px 8px rgba(0,0,0,0.5);}
.gesture-letter{text-align:center;font-size:1.3rem;font-weight:800;color:#a855f7;margin-top:6px;}
.progress-bar-container{background:rgba(30,41,59,0.8);border-radius:8px;height:10px;margin:4px 0;overflow:hidden;}
.progress-bar-fill{height:100%;border-radius:8px;transition:width 0.2s ease;}
.status-dot-green{display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:6px;animation:blink 1s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(9,11,24,0.98) 0%,rgba(15,23,42,0.98) 100%)!important;border-right:1px solid rgba(99,102,241,0.15);}

/* ── Buttons ── */
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;font-weight:600;border:none;border-radius:10px;padding:0.6rem 1.2rem;transition:all 0.25s ease;letter-spacing:0.02em;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,102,241,0.45);}

/* ── Shimmer text ── */
.shimmer{background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6,#c084fc,#818cf8);background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 3s linear infinite;}
@keyframes shimmer{0%{background-position:200%;}100%{background-position:-200%;}}
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
           'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
           'del','nothing','space']
ALPHABET = [c for c in CLASSES if c not in ['del','nothing','space']]
GESTURE_DIR = 'gestures'
BUFFER_SIZE = 20
HOLD_FRAMES = 18
COOLDOWN_FRAMES = 30

# ── Webcam component (JS-based, works on Streamlit Cloud) ──
@st.cache_resource
def _get_webcam_comp():
    return components.declare_component("webcam_stream", path="webcam_component")

# ── Model ──
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),nn.Linear(128*16*16,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,num_classes)
        )
    def forward(self,x):
        return self.fc_layers(self.conv_layers(x))

@st.cache_resource
def load_trained_model():
    m = SignLanguageCNN(num_classes=len(CLASSES))
    pth = 'best_model.pth' if os.path.exists('best_model.pth') else 'sign_language_model.pth'
    if os.path.exists(pth):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m.load_state_dict(torch.load(pth,map_location=dev))
        m.to(dev); m.eval()
        return m,dev,pth
    return None,'cpu',None

_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

def predict(image,model,device):
    if image.mode!='RGB': image=image.convert('RGB')
    t=_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs=F.softmax(model(t),dim=1)[0]
        top_prob,top_idx=torch.max(probs,0)
    pred=CLASSES[top_idx.item()]; conf=top_prob.item()*100
    pnp=probs.cpu().numpy()
    top5={CLASSES[i]:float(pnp[i]*100) for i in np.argsort(pnp)[::-1][:5]}
    return pred,conf,top5

model,device,pth_file=load_trained_model()

# ── Session State ──
for k,v in {'sentence':'','history':[],'pred_buffer':deque(maxlen=BUFFER_SIZE),
             'hold_counter':0,'cooldown_counter':0,'last_confirmed':'',
             'auto_append':True,'live_pred':'--','live_conf':0.0,
             'confirmed_letter':'','live_pred_display':'—'}.items():
    if k not in st.session_state: st.session_state[k]=v

# ── Helpers ──
def speak_text(text):
    c=text.replace('"','\\"').replace('\n',' ')
    components.html(f'<script>var u=new SpeechSynthesisUtterance("{c}");u.rate=0.9;window.speechSynthesis.speak(u);</script>',height=0)

def gesture_path(letter):
    p=os.path.join(GESTURE_DIR,f'{letter}.jpg')
    return p if os.path.exists(p) else None

def plotly_bar(top5):
    df=pd.DataFrame({'Gesture':list(top5.keys()),'Probability (%)':list(top5.values())})
    fig=px.bar(df,x='Probability (%)',y='Gesture',orientation='h',color='Probability (%)',
               color_continuous_scale='Purples',text_auto='.1f')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#f8fafc',
                      height=230,margin=dict(l=0,r=0,t=10,b=0),yaxis=dict(autorange='reversed'))
    return fig

# ── Sidebar ──
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:10px 0 20px;'><div style='font-size:2.5rem;'>🤟</div><div style='font-weight:800;font-size:1.1rem;color:#a855f7;'>Sign Language</div><div style='font-size:0.85rem;color:#64748b;'>Communicator</div></div>",unsafe_allow_html=True)
    app_mode=st.selectbox("Nav",[
        "🏠 Dashboard","🎥 Live Communicator","📝 Text → Gesture",
        "📷 Quick Snapshot","🖼️ Upload & Analyze","📖 ASL Reference Guide","📊 Model Info"
    ],label_visibility="collapsed")
    st.divider()
    if st.session_state.sentence:
        st.markdown("**📝 Sentence**")
        st.markdown(f'<div class="sentence-box" style="font-size:1rem;min-height:40px;">{st.session_state.sentence}</div>',unsafe_allow_html=True)
        if st.button("🔊 Speak",key="sb_spk",width='stretch'): speak_text(st.session_state.sentence)
        sc1,sc2=st.columns(2)
        with sc1:
            if st.button("⌫",key="sb_dl",width='stretch'): st.session_state.sentence=st.session_state.sentence[:-1]; st.rerun()
        with sc2:
            if st.button("🧹",key="sb_cl",width='stretch'): st.session_state.sentence=''; st.session_state.history=[]; st.rerun()
        st.divider()
    st.markdown(f"<small style='color:{'#10b981' if model else '#ef4444'};'>{'✅ Model loaded' if model else '⚠️ Model not found'}</small>",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────
if app_mode=="🏠 Dashboard":

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🤟 Sign Language Communicator</div>
        <div class="hero-subtitle">
            AI-powered two-way real-time communication system — translate hand gestures into voice,<br>
            and text into visual sign language guides. Breaking barriers, one sign at a time.
        </div>
        <div style="margin-top:16px;">
            <span class="hero-tag">🧠 PyTorch CNN</span>
            <span class="hero-tag">📷 WebRTC Live</span>
            <span class="hero-tag">🤚 MediaPipe</span>
            <span class="hero-tag">🔊 Web Speech API</span>
            <span class="hero-tag">29 ASL Classes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Neon Metric Cards ──
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("purple","🔤","29 Classes","A – Z + del, space, nothing"),
        ("blue","🧠","CNN Model","3 Conv Blocks · 256 Dense"),
        ("pink","🎥","Live Stream","WebRTC Real-Time"),
        ("green","🔊","Voice Output","Web Speech API"),
    ]
    for col,(color,icon,val,lbl) in zip([c1,c2,c3,c4],cards):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <span class="icon">{icon}</span>
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── System Flow Diagram ──
    st.markdown("""
    <div class="glass-card" style="padding:28px 32px;">
        <div style="text-align:center;margin-bottom:24px;">
            <span class="shimmer" style="font-size:1.1rem;font-weight:700;">⚡ How It Works</span>
        </div>
        <div style="display:flex;align-items:center;justify-content:center;flex-wrap:nowrap;gap:0;overflow-x:auto;">
            <div class="flow-step">
                <div class="f-icon">📷</div>
                <div class="f-label">Camera</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="f-icon">🤚</div>
                <div class="f-label">MediaPipe</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="f-icon">🧠</div>
                <div class="f-label">CNN Model</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="f-icon">🔤</div>
                <div class="f-label">Stabilize</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="f-icon">📝</div>
                <div class="f-label">Sentence</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="f-icon">🔊</div>
                <div class="f-label">Voice</div>
            </div>
        </div>
        <div style="text-align:center;margin-top:10px;">
            <small style="color:#475569;">Direction 1: Gesture → Voice &nbsp;|&nbsp; Direction 2: Text → Gesture Images (reverse flow)</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Direction Cards ──
    da, db = st.columns(2, gap="large")
    with da:
        st.markdown("""
        <div class="dir-card left">
            <h3 style="color:#c084fc;">🎥 Gesture → Voice</h3>
            <div class="step">
                <div class="step-num">1</div>
                <div class="step-text">Open <b>Live Communicator</b> from the sidebar</div>
            </div>
            <div class="step">
                <div class="step-num">2</div>
                <div class="step-text">Click <b>START</b> and allow camera access</div>
            </div>
            <div class="step">
                <div class="step-num">3</div>
                <div class="step-text">Show a hand sign — hold it <b>steady for ~0.7s</b></div>
            </div>
            <div class="step">
                <div class="step-num">4</div>
                <div class="step-text">Letter auto-confirms and builds your sentence</div>
            </div>
            <div class="step">
                <div class="step-num">5</div>
                <div class="step-text">Click <b>🔊 Speak</b> to hear the full sentence aloud</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with db:
        st.markdown("""
        <div class="dir-card right">
            <h3 style="color:#f472b6;">📝 Text → Gesture</h3>
            <div class="step">
                <div class="step-num">1</div>
                <div class="step-text">Open <b>Text → Gesture</b> from the sidebar</div>
            </div>
            <div class="step">
                <div class="step-num">2</div>
                <div class="step-text">Type any word or sentence (e.g. "HELLO")</div>
            </div>
            <div class="step">
                <div class="step-num">3</div>
                <div class="step-text">Click <b>Show Gestures</b> — see ASL hand images</div>
            </div>
            <div class="step">
                <div class="step-num">4</div>
                <div class="step-text">Each letter displays its real hand-sign photo</div>
            </div>
            <div class="step">
                <div class="step-num">5</div>
                <div class="step-text">Use <b>▶ Animate</b> for a guided slideshow walkthrough</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ASL Reference ──
    if os.path.exists('asl_alphabet_guide.png'):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:16px;">
            <span class="shimmer" style="font-size:1.1rem;font-weight:700;">📖 ASL Alphabet Quick Reference</span>
            <div style="color:#64748b;font-size:0.85rem;margin-top:4px;">American Sign Language — A through Z</div>
        </div>""", unsafe_allow_html=True)
        st.image('asl_alphabet_guide.png', width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LIVE COMMUNICATOR  (JS webcam component — works on cloud)
# ─────────────────────────────────────────────────────────
elif app_mode=="🎥 Live Communicator":
    webcam_comp = _get_webcam_comp()

    col_stream, col_panel = st.columns([1.2, 1], gap="medium")

    with col_stream:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📷 Live Camera Feed")
        st.markdown(
            "<small style='color:#94a3b8;'>Allow camera when prompted → hold your hand sign <b>steady for ~0.7s</b> to confirm a letter.</small>",
            unsafe_allow_html=True
        )
        ac, _ = st.columns([1, 2])
        with ac:
            st.session_state.auto_append = st.toggle(
                "Auto-add letters", value=st.session_state.auto_append,
                help="When ON, confirmed gestures are automatically added to the sentence"
            )

        # ── Render JS webcam component ──
        # It sends {frame: base64_jpeg, ts: timestamp} back to Python every 400ms.
        # We pass the current AI prediction back so JS can overlay it on the video.
        cam_result = webcam_comp(
            prediction=st.session_state.live_pred_display,
            confidence=float(st.session_state.live_conf),
            confirmed=st.session_state.confirmed_letter or '',
            key="webcam_main"
        )

        # ── Process received frame ──
        if cam_result and isinstance(cam_result, dict) and cam_result.get('frame'):
            try:
                frame_bytes = base64.b64decode(cam_result['frame'])
                frame_img   = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
                pred, conf, _ = predict(frame_img, model, device)
                st.session_state.live_pred         = pred
                st.session_state.live_conf         = conf
                st.session_state.live_pred_display = pred if pred != 'nothing' else 'No hand'
                st.session_state.pred_buffer.append(pred)
            except Exception:
                pass

        st.markdown('</div>', unsafe_allow_html=True)

    with col_panel:
        # ── Gesture Stabilization (majority vote) ──
        buf      = list(st.session_state.pred_buffer)
        majority = Counter(buf).most_common(1)[0][0] if buf else ''

        if majority and majority == st.session_state.live_pred and majority not in ('nothing',):
            st.session_state.hold_counter = min(st.session_state.hold_counter + 1, HOLD_FRAMES + 5)
        else:
            st.session_state.hold_counter = max(st.session_state.hold_counter - 2, 0)
        if st.session_state.cooldown_counter > 0:
            st.session_state.cooldown_counter -= 1

        just_confirmed = False
        if (st.session_state.hold_counter >= HOLD_FRAMES
                and st.session_state.cooldown_counter == 0
                and majority not in ('nothing', '')):
            st.session_state.confirmed_letter = majority
            just_confirmed = True
            st.session_state.hold_counter      = 0
            st.session_state.cooldown_counter  = COOLDOWN_FRAMES
            if st.session_state.auto_append:
                if majority == 'del':
                    st.session_state.sentence = st.session_state.sentence[:-1]
                elif majority == 'space':
                    st.session_state.sentence += ' '
                else:
                    st.session_state.sentence += majority
                st.session_state.history.append((majority, st.session_state.live_conf))

        # ── Prediction card ──
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Recognition Panel")
        lp  = st.session_state.live_pred
        lc  = st.session_state.live_conf
        bdg = "confirmed-badge" if just_confirmed else "prediction-badge"
        dl  = lp if lp not in ('nothing', '--') else "—"
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="margin:0;color:#94a3b8;font-size:.9rem;">Current Gesture</p>
            <div class="{bdg}">{dl}</div>
            <p class="confidence-badge">Confidence: {lc:.1f}%</p>
        </div>""", unsafe_allow_html=True)

        hp = int((st.session_state.hold_counter / HOLD_FRAMES) * 100)
        if st.session_state.cooldown_counter > 0:
            bc, lbl = "#f59e0b", "⏳ Cooldown"
        elif hp > 0:
            bc, lbl = "#10b981", f"⏱ Hold {hp}%"
        else:
            bc, lbl = "#6366f1", "Waiting..."
        st.markdown(f"""
        <div style="margin:8px 0 4px;">
            <small style="color:#94a3b8;">{lbl}</small>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width:{min(hp,100)}%;background:{bc};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.confirmed_letter:
            st.markdown(f"""
            <div style="text-align:center;margin-bottom:8px;">
                <small style="color:#10b981;font-weight:600;">
                    <span class="status-dot-green"></span>Last confirmed: <b>{st.session_state.confirmed_letter}</b>
                </small>
            </div>""", unsafe_allow_html=True)
        gp = gesture_path(lp) if lp and lp not in ('nothing', '--') else None
        if gp:
            st.image(gp, caption=f"ASL: {lp}", width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Sentence panel ──
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Sentence Builder")
        sent = st.session_state.sentence if st.session_state.sentence else "(empty — start signing)"
        st.markdown(f'<div class="sentence-box">{sent}</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            if st.button("🔊 Speak", key="ls_spk", width='stretch'):
                if st.session_state.sentence.strip(): speak_text(st.session_state.sentence)
        with s2:
            if st.button("␣ Space", key="ls_sp", width='stretch'):
                st.session_state.sentence += ' '; st.rerun()
        with s3:
            if st.button("⌫ Del", key="ls_dl", width='stretch'):
                st.session_state.sentence = st.session_state.sentence[:-1]; st.rerun()
        with s4:
            if st.button("🧹 Clear", key="ls_cl", width='stretch'):
                st.session_state.sentence = ''
                st.session_state.history  = []
                st.session_state.confirmed_letter = ''
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists('asl_alphabet_guide.png'):
        with st.expander("🖐️ ASL Reference Chart"):
            st.image('asl_alphabet_guide.png', width='stretch')

# ─────────────────────────────────────────────────────────
# TEXT → GESTURE
# ─────────────────────────────────────────────────────────
elif app_mode=="📝 Text → Gesture":
    st.markdown('<div class="hero-header" style="font-size:2rem!important;">📝 Text → Gesture Translator</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Type a word or sentence — see the ASL hand sign for every letter.</div>',unsafe_allow_html=True)

    st.markdown('<div class="glass-card">',unsafe_allow_html=True)
    ic,bc=st.columns([3,1])
    with ic:
        user_text=st.text_input("msg","",placeholder="Type HELLO, GOOD MORNING…",label_visibility="collapsed")
    with bc:
        show_btn=st.button("👁 Show Gestures",width='stretch')
    animate_btn=st.button("▶ Animate (Slideshow)",width='stretch')
    st.markdown('</div>',unsafe_allow_html=True)

    if user_text and (show_btn or animate_btn):
        letters=[ch.upper() for ch in user_text if ch.upper() in CLASSES or ch==' ']
        if not letters:
            st.warning("No recognisable ASL letters found.")
        else:
            if show_btn:
                st.markdown('<div class="glass-card">',unsafe_allow_html=True)
                st.markdown(f"**Gestures for:** `{user_text.upper()}`")
                cols_r=7
                for grp in [letters[i:i+cols_r] for i in range(0,len(letters),cols_r)]:
                    rcols=st.columns(len(grp))
                    for col,lt in zip(rcols,grp):
                        with col:
                            if lt==' ':
                                st.markdown('<div style="text-align:center;padding:20px 0;color:#64748b;">SPACE<br>_</div>',unsafe_allow_html=True)
                            else:
                                gp=gesture_path(lt)
                                if gp: st.image(gp,width='stretch')
                                st.markdown(f'<div class="gesture-letter">{lt}</div>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)

            if animate_btn:
                st.markdown('<div class="glass-card">',unsafe_allow_html=True)
                st.markdown(f"### ▶ Animating: `{user_text.upper()}`")
                ph=st.empty(); pp=st.empty()
                for idx,lt in enumerate(letters):
                    prog=int(((idx+1)/len(letters))*100)
                    with ph.container():
                        _,ac2,_=st.columns([1,2,1])
                        with ac2:
                            if lt==' ':
                                st.markdown('<div style="text-align:center;padding:40px;font-size:3rem;">␣<br><span style="font-size:1rem;color:#94a3b8;">SPACE</span></div>',unsafe_allow_html=True)
                            else:
                                gp=gesture_path(lt)
                                if gp: st.image(gp,width='stretch')
                                st.markdown(f'<div style="text-align:center;font-size:2.5rem;font-weight:800;color:#a855f7;margin-top:8px;">{lt}</div><div style="text-align:center;color:#94a3b8;font-size:.9rem;">Letter {idx+1} of {len(letters)}</div>',unsafe_allow_html=True)
                    pp.progress(prog,text=f"Showing: {lt if lt!=' ' else 'SPACE'}")
                    time.sleep(1.1)
                ph.success("✅ Slideshow complete!")
                pp.empty()
                st.markdown('</div>',unsafe_allow_html=True)
    elif not user_text:
        st.markdown('<div class="glass-card" style="text-align:center;padding:40px;"><div style="font-size:3rem;margin-bottom:12px;">✍️</div><div style="color:#94a3b8;font-size:1.1rem;">Type a word above, then click <b>Show Gestures</b> or <b>Animate</b>.</div></div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# QUICK SNAPSHOT
# ─────────────────────────────────────────────────────────
elif app_mode=="📷 Quick Snapshot":
    st.markdown('<div class="glass-card">',unsafe_allow_html=True)
    st.subheader("📷 Camera Snapshot Recognition")
    c1,c2=st.columns([1.1,1],gap="medium")
    with c1:
        captured=st.camera_input("Take a snapshot of your hand gesture")
    with c2:
        st.subheader("🧠 Result")
        if captured and model:
            ip=Image.open(captured)
            with st.spinner("Analysing…"):
                top_pred,conf,top5=predict(ip,model,device)
            st.markdown(f'<div style="text-align:center;"><p style="margin:0;color:#94a3b8;">Predicted Gesture</p><div class="prediction-badge">{top_pred}</div><p class="confidence-badge">Confidence: {conf:.2f}%</p></div>',unsafe_allow_html=True)
            b1,b2=st.columns(2)
            with b1:
                if st.button("➕ Add to Sentence",key="sn_add",width='stretch'):
                    if top_pred=='space': st.session_state.sentence+=' '
                    elif top_pred=='del': st.session_state.sentence=st.session_state.sentence[:-1]
                    elif top_pred!='nothing': st.session_state.sentence+=top_pred
                    st.session_state.history.append((top_pred,conf)); st.success(f"Added '{top_pred}'")
            with b2:
                if st.button("🔊 Speak",key="sn_spk",width='stretch'): speak_text(top_pred)
            st.divider(); st.markdown("#### 📈 Top 5")
            st.plotly_chart(plotly_bar(top5),width='stretch')
        elif not captured: st.info("👆 Click Take Photo to classify a gesture.")
        elif not model: st.warning("Model not loaded.")
    st.markdown('</div>',unsafe_allow_html=True)
    if os.path.exists('asl_alphabet_guide.png'):
        with st.expander("🖐️ ASL Reference Chart"): st.image('asl_alphabet_guide.png',width='stretch')

# ─────────────────────────────────────────────────────────
# UPLOAD & ANALYZE
# ─────────────────────────────────────────────────────────
elif app_mode=="🖼️ Upload & Analyze":
    c1,c2=st.columns([1.1,1],gap="medium")
    with c1:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("🖼️ Upload Gesture Image")
        up=st.file_uploader("Upload gesture photo",type=['jpg','jpeg','png'])
        if up:
            ip=Image.open(up)
            st.image(ip,caption="Uploaded Image",width='stretch')
        st.markdown('</div>',unsafe_allow_html=True)
        if os.path.exists('asl_alphabet_guide.png'):
            with st.expander("🖐️ ASL Reference Chart"): st.image('asl_alphabet_guide.png',width='stretch')
    with c2:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("🧠 Recognition Result")
        if up and model:
            with st.spinner("Analysing…"):
                top_pred,conf,top5=predict(ip,model,device)
            st.markdown(f'<div style="text-align:center;"><p style="margin:0;color:#94a3b8;">Predicted Gesture</p><div class="prediction-badge">{top_pred}</div><p class="confidence-badge">Confidence: {conf:.2f}%</p></div>',unsafe_allow_html=True)
            b1,b2=st.columns(2)
            with b1:
                if st.button("➕ Add to Sentence",key="up_add",width='stretch'):
                    if top_pred=='space': st.session_state.sentence+=' '
                    elif top_pred=='del': st.session_state.sentence=st.session_state.sentence[:-1]
                    elif top_pred!='nothing': st.session_state.sentence+=top_pred
                    st.session_state.history.append((top_pred,conf)); st.success(f"Added '{top_pred}'")
            with b2:
                if st.button("🔊 Speak",key="up_spk",width='stretch'): speak_text(top_pred)
            st.divider(); st.markdown("#### 📈 Top 5")
            st.plotly_chart(plotly_bar(top5),width='stretch')
        elif not model: st.warning("Model not loaded.")
        else: st.info("👈 Upload an image to start.")
        st.markdown('</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ASL REFERENCE GUIDE
# ─────────────────────────────────────────────────────────
elif app_mode=="📖 ASL Reference Guide":
    st.markdown('<div class="glass-card">',unsafe_allow_html=True)
    st.subheader("📖 ASL Alphabet Reference Chart")
    if os.path.exists('asl_alphabet_guide.png'):
        st.image('asl_alphabet_guide.png',caption="ASL Alphabet",width='stretch')
    else: st.error("asl_alphabet_guide.png not found.")
    st.divider()
    st.subheader("🔤 Individual Gesture Cards (A–Z)")
    rows_=[ALPHABET[i:i+9] for i in range(0,len(ALPHABET),9)]
    for row_ in rows_:
        rcols=st.columns(len(row_))
        for col,lt in zip(rcols,row_):
            with col:
                gp=gesture_path(lt)
                if gp: st.image(gp,width='stretch')
                st.markdown(f'<div style="text-align:center;font-weight:700;color:#a855f7;">{lt}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────
elif app_mode=="📊 Model Info":
    c1,c2=st.columns(2,gap="medium")
    with c1:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("🤖 Neural Network Architecture")
        st.markdown("""
**CNN Model (`SignLanguageCNN`)**
- **Input**: `(3, 128, 128)` RGB Tensor
- **Conv Block 1**: Conv2D 32 filters → ReLU → MaxPool2D
- **Conv Block 2**: Conv2D 64 filters → ReLU → MaxPool2D
- **Conv Block 3**: Conv2D 128 filters → ReLU → MaxPool2D
- **Flatten**: 128 × 16 × 16 = 32,768 features
- **Dense**: 256 neurons + ReLU + Dropout(0.5)
- **Output**: 29 classes (A–Z, del, nothing, space)
        """)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("📜 Classes (29) & Stats")
        st.markdown("**Alphabets (26):** `A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`\n\n**Special:** `space` · `del` · `nothing`")
        if model:
            total=sum(p.numel() for p in model.parameters())
            train=sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.metric("Total Parameters",f"{total:,}")
            st.metric("Trainable Parameters",f"{train:,}")
            st.metric("Device",str(device).upper())
        st.markdown('</div>',unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("📋 Recognition History")
        df=pd.DataFrame(st.session_state.history,columns=['Gesture','Confidence (%)'])
        df.index+=1
        st.dataframe(df,width='stretch')
        st.markdown('</div>',unsafe_allow_html=True)

# ── Footer ──
st.markdown("<br><hr><div style='text-align:center;color:#64748b;font-size:.85rem;'>🤟 AI-Based Two-Way Sign Language Communication System &nbsp;|&nbsp; Streamlit + PyTorch + MediaPipe</div>",unsafe_allow_html=True)

