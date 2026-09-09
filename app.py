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
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
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
.main,.stApp{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);}
.glass-card{background:rgba(30,41,59,0.7);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.37);}
.hero-header{background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;font-size:2.8rem!important;letter-spacing:-0.02em;margin-bottom:0.2rem;}
.sub-header{color:#94a3b8;font-size:1.1rem;font-weight:400;margin-bottom:1.5rem;}
.prediction-badge{display:inline-block;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;font-weight:800;font-size:2.5rem;padding:12px 32px;border-radius:16px;box-shadow:0 0 20px rgba(99,102,241,0.5);text-align:center;margin:10px 0;}
.confirmed-badge{display:inline-block;background:linear-gradient(135deg,#059669,#10b981);color:#fff;font-weight:800;font-size:2.5rem;padding:12px 32px;border-radius:16px;box-shadow:0 0 24px rgba(16,185,129,0.6);text-align:center;margin:10px 0;}
.confidence-badge{font-size:1.1rem;color:#38bdf8;font-weight:600;}
.sentence-box{background:rgba(15,23,42,0.8);border:2px solid #6366f1;border-radius:12px;padding:18px;font-size:1.8rem;font-weight:700;letter-spacing:0.05em;color:#38bdf8;min-height:70px;word-wrap:break-word;box-shadow:inset 0 2px 8px rgba(0,0,0,0.5);}
.gesture-letter{text-align:center;font-size:1.3rem;font-weight:800;color:#a855f7;margin-top:6px;}
.progress-bar-container{background:rgba(30,41,59,0.8);border-radius:8px;height:10px;margin:4px 0;overflow:hidden;}
.progress-bar-fill{height:100%;border-radius:8px;transition:width 0.2s ease;}
.status-dot-green{display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:6px;animation:blink 1s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}
section[data-testid="stSidebar"]{background-color:rgba(15,23,42,0.95)!important;border-right:1px solid rgba(255,255,255,0.08);}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;font-weight:600;border:none;border-radius:10px;padding:0.6rem 1.2rem;transition:all 0.3s ease;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(99,102,241,0.4);}
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
_pred_queue: queue.Queue = queue.Queue(maxsize=2)

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
             'auto_append':True,'live_pred':'--','live_conf':0.0,'confirmed_letter':''}.items():
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
    st.markdown('<div class="hero-header">🤟 Sign Language Communicator</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Two-Way Real-Time Sign Language Communication System</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],[
        ("🔤","29 Classes","A–Z + del, space, nothing"),("🧠","CNN Model","3 Conv Blocks + Dense"),
        ("🎥","Live Stream","WebRTC Real-Time"),("🔊","Voice Output","Web Speech API")]):
        with col:
            st.markdown(f'<div class="glass-card" style="text-align:center;padding:16px;"><div style="font-size:2rem;">{icon}</div><div style="font-weight:700;color:#f8fafc;font-size:1rem;margin-top:6px;">{title}</div><div style="color:#94a3b8;font-size:0.8rem;">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    ca,cb=st.columns(2,gap="large")
    with ca:
        st.markdown('<div class="glass-card"><h3 style="color:#a855f7;">🎥 Gesture → Voice</h3><ol style="color:#cbd5e1;line-height:2.2;"><li>Open <b>Live Communicator</b></li><li>Click <b>START</b> and allow camera</li><li>Show a hand sign — hold it steady</li><li>Letter auto-confirms after ~0.7s</li><li>Click <b>🔊 Speak</b> to hear sentence</li></ol></div>',unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="glass-card"><h3 style="color:#ec4899;">📝 Text → Gesture</h3><ol style="color:#cbd5e1;line-height:2.2;"><li>Open <b>Text → Gesture</b></li><li>Type any word or sentence</li><li>Click <b>Show Gestures</b></li><li>Each letter shows its ASL hand sign</li><li>Use <b>Animate</b> for a walkthrough</li></ol></div>',unsafe_allow_html=True)
    if os.path.exists('asl_alphabet_guide.png'):
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("📖 ASL Quick Reference")
        st.image('asl_alphabet_guide.png',width='stretch')
        st.markdown('</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LIVE COMMUNICATOR
# ─────────────────────────────────────────────────────────
elif app_mode=="🎥 Live Communicator":
    hands_detector=None
    try:
        from mediapipe.tasks import python as _mpp
        from mediapipe.tasks.python import vision as _mpv
        if os.path.exists('hand_landmarker.task'):
            hands_detector=_mpv.HandLandmarker.create_from_options(
                _mpv.HandLandmarkerOptions(base_options=_mpp.BaseOptions(model_asset_path='hand_landmarker.task'),num_hands=1))
    except Exception: pass

    HAND_CONN=[(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
               (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),(0,5),(5,9),(9,13),(13,17),(0,17)]

    RTC_CONFIGURATION={
        "iceServers":[
            {"urls":["turns:global.relay.metered.ca:443?transport=tcp"],"username":"4d3a01f2d43c261926a6ca28","credential":"5FMXSsQM6ms0faRT"},
            {"urls":["turn:global.relay.metered.ca:80?transport=tcp"],"username":"4d3a01f2d43c261926a6ca28","credential":"5FMXSsQM6ms0faRT"},
            {"urls":["turn:global.relay.metered.ca:80"],"username":"4d3a01f2d43c261926a6ca28","credential":"5FMXSsQM6ms0faRT"},
            {"urls":["stun:stun.relay.metered.ca:80"]},
            {"urls":["stun:stun.l.google.com:19302"]},
        ]
    }

    col_stream,col_panel=st.columns([1.2,1],gap="medium")

    with col_stream:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.markdown("### 📷 Live Camera Feed")
        st.markdown("<small style='color:#94a3b8;'>Click <b>START</b> → allow camera → hold your hand sign steady.</small>",unsafe_allow_html=True)
        ac,_=st.columns([1,2])
        with ac:
            st.session_state.auto_append=st.toggle("Auto-add letters",value=st.session_state.auto_append,
                help="Confirmed gestures auto-append to sentence after holding ~0.7s")

        def video_frame_callback(frame):
            if model is None: return frame
            img=frame.to_ndarray(format="bgr24")
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            pred,conf,_=predict(Image.fromarray(rgb),model,device)
            if hands_detector is not None:
                try:
                    res=hands_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                    if res.hand_landmarks:
                        for lms in res.hand_landmarks:
                            h,w=img.shape[:2]
                            pts=[(int(l.x*w),int(l.y*h)) for l in lms]
                            for a,b in HAND_CONN: cv2.line(img,pts[a],pts[b],(99,102,241),2)
                            for pt in pts: cv2.circle(img,pt,4,(168,85,247),-1)
                except Exception: pass
            try: _pred_queue.put_nowait((pred,conf))
            except queue.Full: pass
            disp=pred if pred not in('nothing',) else "No hand"
            col=(0,255,128) if pred not in('nothing','del','space') else (255,120,0)
            cv2.rectangle(img,(0,0),(img.shape[1],80),(0,0,0),-1)
            cv2.putText(img,f"{disp}  {conf:.0f}%",(16,54),cv2.FONT_HERSHEY_DUPLEX,1.4,col,2,cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img,format="bgr24")

        webrtc_streamer(key="live-comm",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,
                        video_frame_callback=video_frame_callback,
                        media_stream_constraints={"video":True,"audio":False},async_processing=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with col_panel:
        # Drain queue
        while not _pred_queue.empty():
            try:
                p,c=_pred_queue.get_nowait()
                st.session_state.live_pred=p; st.session_state.live_conf=c
                st.session_state.pred_buffer.append(p)
            except queue.Empty: break

        buf=list(st.session_state.pred_buffer)
        majority=Counter(buf).most_common(1)[0][0] if buf else ''

        if majority and majority==st.session_state.live_pred and majority not in('nothing',):
            st.session_state.hold_counter=min(st.session_state.hold_counter+1,HOLD_FRAMES+5)
        else:
            st.session_state.hold_counter=max(st.session_state.hold_counter-2,0)
        if st.session_state.cooldown_counter>0: st.session_state.cooldown_counter-=1

        just_confirmed=False
        if(st.session_state.hold_counter>=HOLD_FRAMES
           and st.session_state.cooldown_counter==0
           and majority not in('nothing','')):
            st.session_state.confirmed_letter=majority
            just_confirmed=True
            st.session_state.hold_counter=0
            st.session_state.cooldown_counter=COOLDOWN_FRAMES
            if st.session_state.auto_append:
                if majority=='del': st.session_state.sentence=st.session_state.sentence[:-1]
                elif majority=='space': st.session_state.sentence+=' '
                else: st.session_state.sentence+=majority
                st.session_state.history.append((majority,st.session_state.live_conf))

        # Prediction card
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.markdown("### 🧠 Recognition Panel")
        lp=st.session_state.live_pred; lc=st.session_state.live_conf
        bdg="confirmed-badge" if just_confirmed else "prediction-badge"
        dl=lp if lp not in('nothing',) else "—"
        st.markdown(f'<div style="text-align:center;"><p style="margin:0;color:#94a3b8;font-size:.9rem;">Current Gesture</p><div class="{bdg}">{dl}</div><p class="confidence-badge">Confidence: {lc:.1f}%</p></div>',unsafe_allow_html=True)

        hp=int((st.session_state.hold_counter/HOLD_FRAMES)*100)
        if st.session_state.cooldown_counter>0: bc,lbl="#f59e0b","⏳ Cooldown"
        elif hp>0: bc,lbl="#10b981",f"⏱ Hold {hp}%"
        else: bc,lbl="#6366f1","Waiting..."
        st.markdown(f'<div style="margin:8px 0 4px;"><small style="color:#94a3b8;">{lbl}</small><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{min(hp,100)}%;background:{bc};"></div></div></div>',unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.confirmed_letter:
            st.markdown(f'<div style="text-align:center;margin-bottom:8px;"><small style="color:#10b981;font-weight:600;"><span class="status-dot-green"></span>Last confirmed: <b>{st.session_state.confirmed_letter}</b></small></div>',unsafe_allow_html=True)
        gp=gesture_path(lp) if lp and lp not in('nothing',) else None
        if gp: st.image(gp,caption=f"ASL: {lp}",width='stretch')
        st.markdown('</div>',unsafe_allow_html=True)

        # Sentence panel
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.markdown("### 📝 Sentence Builder")
        sent=st.session_state.sentence if st.session_state.sentence else "(empty — start signing)"
        st.markdown(f'<div class="sentence-box">{sent}</div>',unsafe_allow_html=True)
        s1,s2,s3,s4=st.columns(4)
        with s1:
            if st.button("🔊 Speak",key="ls_spk",width='stretch'):
                if st.session_state.sentence.strip(): speak_text(st.session_state.sentence)
        with s2:
            if st.button("␣ Space",key="ls_sp",width='stretch'): st.session_state.sentence+=' '; st.rerun()
        with s3:
            if st.button("⌫ Del",key="ls_dl",width='stretch'): st.session_state.sentence=st.session_state.sentence[:-1]; st.rerun()
        with s4:
            if st.button("🧹 Clear",key="ls_cl",width='stretch'): st.session_state.sentence=''; st.session_state.history=[]; st.session_state.confirmed_letter=''; st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

    if os.path.exists('asl_alphabet_guide.png'):
        with st.expander("🖐️ ASL Reference Chart"): st.image('asl_alphabet_guide.png',width='stretch')

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

