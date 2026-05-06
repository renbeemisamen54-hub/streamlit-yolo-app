import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import os
import time
import threading

# --- CONFIG & STYLING ---
st.set_page_config(page_title="VisionAI Pro", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4251; }
    </style>
    """, unsafe_allow_html=True)

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- SHARED STATE ---
# We use a Lock and a dictionary to pass data from the video thread to the UI
lock = threading.Lock()
img_container = {"img": None, "count": 0, "alert": False}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Inference
    results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    # Thread-safe data update
    with lock:
        img_container["img"] = annotated_frame
        img_container["count"] = len(results[0].boxes) if results[0].boxes else 0
        
        # Alert Check
        if selected_class != "None":
            names = model.names
            img_container["alert"] = any(names[int(box.cls[0])] == selected_class for box in results[0].boxes)
        else:
            img_container["alert"] = False

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 VisionAI Pro")
    st.subheader("Model Settings")
    conf_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.45)
    selected_class = st.selectbox("Target Alert", ["None"] + list(model.names.values()))
    
    st.divider()
    if st.button("🗑️ Purge Snapshots", use_container_width=True):
        for f in os.listdir("snapshots"): os.remove(os.path.join("snapshots", f))
        st.success("Gallery Cleared!")

# --- MAIN LAYOUT ---
col_vid, col_stats = st.columns([2, 1])

with col_vid:
    webrtc_streamer(
        key="vision-pro",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_stats:
    st.subheader("📊 Live Intelligence")
    
    # High-frequency UI updates
    metric_spot = st.empty()
    alert_spot = st.empty()
    snap_btn = st.button("📸 Capture Moment", use_container_width=True)
    
    # This loop keeps the UI reactive to the video thread
    while True:
        with lock:
            count = img_container["count"]
            is_alert = img_container["alert"]
            current_img = img_container["img"]

        # Update Statistics
        metric_spot.metric("Objects in View", count)
        
        if is_alert:
            alert_spot.error(f"🚨 TARGET DETECTED: {selected_class.upper()}")
        else:
            alert_spot.info("✅ Surveillance Active")

        # Snapshot Logic
        if snap_btn and current_img is not None:
            fn = f"snapshots/snap_{int(time.time())}.jpg"
            cv2.imwrite(fn, current_img)
            st.toast(f"Saved to {fn}!", icon="💾")
            snap_btn = False # Reset pseudo-trigger
            
        time.sleep(0.1) # Prevent CPU spiking
